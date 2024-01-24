from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
import torchvision
from model.dcgan.critic import Critic

from src.model.dcgan.generator import Generator


class WGAN(L.LightningModule):
    def __init__(
        self,
        noise_dim: int = 100,
        image_shape: Tuple[int] = (3, 64, 64),
        lr: float = 0.0001,
        b1: float = 0,
        b2: float = 0.9,
        n_critic: int = 5,
        interpolation_epsilon: float = 0.3,
        gradient_penalty_weight: float = 10.0,
        critic_autoencoder: bool = False,
        logging_interval: int = 100,
        logging_images: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(noise_dim, image_shape)
        self.critic = Critic(noise_dim, image_shape)
        self.automatic_optimization = False  # We need more flexibility with GANs
        self._initialize_weights()

    def on_train_start(self):
        self.fixed_noise = torch.randn(
            self.hparams.logging_images, self.hparams.noise_dim
        )
        self.logging_step = 0

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, noise):
        return self.generator(noise)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(b1, b2))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_ae = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.generator.parameters()),
            lr=lr,
            betas=(b1, b2),
        )
        return [opt_c, opt_g, opt_ae], []

    def _get_critic_gradient_penalty(self, real_imgs, generated_imgs):
        """Enforce Lipschitz-1 continuity."""
        epsilon = self.hparams.interpolation_epsilon
        mixed_images = real_imgs * epsilon + generated_imgs * (1 - epsilon)
        mixed_scores = self.critic(mixed_images)["critic_value"]
        gradient = torch.autograd.grad(
            inputs=mixed_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1)
        penalty = torch.mean((gradient_norm - torch.ones_like(gradient_norm)) ** 2)
        return penalty

    def _train_critic(self, critic_optimizer, real_imgs):
        batch_size = real_imgs.shape[0]
        for _ in range(self.hparams.n_critic):
            noise = torch.randn(batch_size, self.hparams.noise_dim)
            noise = noise.type_as(real_imgs)
            generated_imgs = self(noise)
            # generated
            critic_results = self.critic(generated_imgs.detach())
            noise, generated_pred = (
                critic_results["encoded"],
                critic_results["critic_value"],
            )
            # real
            critic_results = self.critic(real_imgs)
            _, real_pred = critic_results["encoded"], critic_results["critic_value"]

            critic_loss = torch.mean(generated_pred) - torch.mean(real_pred)
            critic_loss += (
                self.hparams.gradient_penalty_weight
                * self._get_critic_gradient_penalty(real_imgs, generated_imgs)
            )
            critic_optimizer.zero_grad()
            self.manual_backward(critic_loss, retain_graph=True)
            critic_optimizer.step()
        return generated_imgs, {"critic_loss": critic_loss}

    def _train_generator(self, generator_optimizer, generated_imgs):
        critic_results = self.critic(generated_imgs)
        _, generated_pred = (
            critic_results["encoded"],
            critic_results["critic_value"],
        )
        generator_loss = -torch.mean(generated_pred)
        generator_optimizer.zero_grad()
        self.manual_backward(generator_loss)
        generator_optimizer.step()
        return {"generator_loss": generator_loss}

    def _train_autoencoder(self, autoencoder_optimizer, real_imgs):
        critic_results = self.critic(real_imgs)
        noise, _ = (
            critic_results["encoded"],
            critic_results["critic_value"],
        )
        reconstructed = self.generator(torch.squeeze(noise))
        autoencoder_loss = torch.nn.functional.mse_loss(real_imgs, reconstructed)
        autoencoder_optimizer.zero_grad()
        self.manual_backward(autoencoder_loss)
        autoencoder_optimizer.step()
        return {"autoencoder_loss": autoencoder_loss}

    def _log(self, real_imgs, batch_idx):
        if batch_idx % self.hparams.logging_interval == 0:
            fixed_generated = self(self.fixed_noise.type_as(real_imgs)).detach()
            sample_imgs = fixed_generated[: self.hparams.logging_images]
            grid = torchvision.utils.make_grid(sample_imgs, normalize=True)
            self.logger.experiment.add_image(
                "generated_images", grid, self.logging_step
            )

            sample_imgs = real_imgs[: self.hparams.logging_images]
            grid = torchvision.utils.make_grid(sample_imgs, normalize=True)
            self.logger.experiment.add_image("real_images", grid, self.logging_step)
            self.logging_step += 1

    def training_step(self, batch, batch_idx):
        real_imgs = batch
        optimizer_c, optimizer_g, optimizer_ae = self.optimizers()
        generated_imgs, critic_loss = self._train_critic(optimizer_c, real_imgs)
        self._log(real_imgs, batch_idx)
        generator_loss = self._train_generator(optimizer_g, generated_imgs)
        if self.hparams.critic_autoencoder:
            autoencoder_loss = self._train_autoencoder(optimizer_ae, real_imgs)
        else:
            autoencoder_loss = {}
        losses = critic_loss | generator_loss | autoencoder_loss
        self.log_dict(losses, prog_bar=True)
