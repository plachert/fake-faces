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
        critic_autoencoder: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(noise_dim, image_shape)
        self.critic = Critic(noise_dim, image_shape)
        self.automatic_optimization = False  # We need more flexibility with GANs
        self.fixed_noise = torch.randn(128, self.hparams.noise_dim)
        self._initialize_weights()
        self.k = 0

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, noise):
        return self.generator(noise)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(b1, b2))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_c, opt_g], []

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
            critic_optimizer.zero_grad()
            self.manual_backward(critic_loss)
            critic_optimizer.step()
            for parameter in self.critic.parameters():
                parameter.data.clamp_(-0.01, 0.01)  # grad clipping
        return generated_imgs, {"critic_loss": critic_loss}

    def _train_generator(self, generator_optimizer, generated_imgs, real_imgs):
        critic_results = self.critic(generated_imgs)
        noise, generated_pred = (
            critic_results["encoded"],
            critic_results["critic_value"],
        )
        generator_loss_gan = -torch.mean(generated_pred)
        generator_loss = generator_loss_gan
        generator_losses = {"generator_loss": generator_loss}

        if self.hparams.critic_autoencoder:
            critic_results = self.critic(real_imgs)
            noise, _ = critic_results["encoded"], critic_results["critic_value"]
            reconstructed = self.generator(torch.squeeze(noise))
            generator_loss_ae = torch.nn.functional.mse_loss(reconstructed, real_imgs)
            generator_losses["generator_loss"] += generator_loss_ae
            generator_losses["autoencoder_loss"] = generator_loss_ae

        generator_optimizer.zero_grad()
        self.manual_backward(generator_losses["generator_loss"])
        generator_optimizer.step()
        return generator_losses

    def training_step(self, batch, batch_idx):
        real_imgs = batch
        optimizer_c, optimizer_g = self.optimizers()
        generated_imgs, critic_loss = self._train_critic(optimizer_c, real_imgs)

        ## Logging
        interval = 100
        if batch_idx % interval == 0:
            fixed_generated = self(self.fixed_noise.type_as(real_imgs)).detach()
            sample_imgs = fixed_generated[:32]
            grid = torchvision.utils.make_grid(sample_imgs, normalize=True)
            self.logger.experiment.add_image("generated_images", grid, self.k)

            sample_imgs = real_imgs[:32]
            grid = torchvision.utils.make_grid(sample_imgs, normalize=True)
            self.logger.experiment.add_image("real_images", grid, self.k)
            self.k += 1
        ##

        generator_losses = self._train_generator(optimizer_g, generated_imgs, real_imgs)

        losses = critic_loss | generator_losses

        self.log_dict(losses, prog_bar=True)


if __name__ == "__main__":
    gan = WGAN()
