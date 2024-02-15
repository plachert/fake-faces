from typing import Tuple

import lightning as L
import torch
import torch.nn as nn

from model.wgan.critic import Critic
from src.logger.face_logger import FaceLogger
from src.model.wgan.generator import Generator


class WGAN(L.LightningModule):
    def __init__(
        self,
        noise_dim: int = 100,
        image_shape: Tuple[int] = (3, 64, 64),
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        n_critic: int = 5,
        gradient_penalty_weight: float = 10.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.noise_dim = noise_dim
        self.generator = Generator(noise_dim, image_shape)
        self.critic = Critic(noise_dim, image_shape)
        self.automatic_optimization = False  # We need more flexibility with GANs
        self._initialize_weights()

    def setup(self, stage):
        if not isinstance(self.logger, FaceLogger):
            raise TypeError("WGAN module can only work with FaceLogger")
        self.logger.set_fixed_noise(self.noise_dim, device=self.device)

    def on_train_end(self):
        self.logger.save_last_images()

    def training_step(self, batch, batch_idx):
        real_imgs = batch
        optimizer_c, optimizer_g = self.optimizers()
        noise, critic_loss = self._train_critic(optimizer_c, real_imgs)
        self.logger.log_images(real_imgs, self.generator, batch_idx)
        generator_loss = self._train_generator(optimizer_g, noise)
        losses = critic_loss | generator_loss
        self.log_dict(losses, prog_bar=True)

    def forward(self, noise):
        return self.generator(noise)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(b1, b2))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_c, opt_g], []

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def _get_critic_gradient_penalty(self, real_imgs, generated_imgs):
        """Enforce 1-Lipschitz continuity."""
        epsilon = torch.rand(
            len(real_imgs), 1, 1, 1, requires_grad=True, device=self.device
        )
        mixed_images = real_imgs * epsilon + generated_imgs * (1 - epsilon)
        mixed_scores = self.critic(mixed_images)
        gradient = torch.autograd.grad(
            inputs=mixed_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1)
        penalty = torch.mean((gradient_norm - 1.0) ** 2)
        return penalty

    def _train_critic(self, critic_optimizer, real_imgs):
        batch_size = real_imgs.shape[0]
        should_retain = (
            self.hparams.n_critic > 1
        )  # this saves a bit space when we don't have to retain the graph
        noise_tensor = torch.randn(
            batch_size * self.hparams.n_critic,
            self.hparams.noise_dim,
            device=self.device,
        )
        generated_imgs_tensor = self(noise_tensor).detach()
        for step in range(self.hparams.n_critic):
            generated_imgs = generated_imgs_tensor[
                step * batch_size : (step + 1) * batch_size, ...
            ]
            generated_pred = self.critic(generated_imgs)
            # real
            real_pred = self.critic(real_imgs)

            critic_loss = torch.mean(generated_pred) - torch.mean(real_pred)
            if self.hparams.gradient_penalty_weight > 0:
                critic_loss += (
                    self.hparams.gradient_penalty_weight
                    * self._get_critic_gradient_penalty(real_imgs, generated_imgs)
                )
            critic_optimizer.zero_grad(set_to_none=True)
            self.manual_backward(critic_loss, retain_graph=should_retain)
            critic_optimizer.step()
        noise = noise_tensor[:batch_size, ...]  # just to feed the generator
        return noise, {"critic_loss": critic_loss}

    def _train_generator(self, generator_optimizer, noise):
        generated_imgs = self.generator(noise)
        generated_pred = self.critic(generated_imgs)
        generator_loss = -torch.mean(generated_pred)
        generator_optimizer.zero_grad(set_to_none=True)
        self.manual_backward(generator_loss)
        generator_optimizer.step()
        return {"generator_loss": generator_loss}
