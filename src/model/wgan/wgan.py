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
        self.logger.set_fixed_noise(self.noise_dim, self.device)

    def on_train_end(self):
        self.logger.save_last_images()

    def training_step(self, batch, batch_idx):
        real_imgs = batch
        optimizer_c, optimizer_g = self.optimizers()
        generated_imgs, critic_loss = self._train_critic(optimizer_c, real_imgs)
        self.logger.log_images(real_imgs, self.generator, batch_idx)
        generator_loss = self._train_generator(optimizer_g, generated_imgs)
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
        epsilon = torch.rand(len(real_imgs), 1, 1, 1, requires_grad=True).type_as(
            real_imgs
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
        for _ in range(self.hparams.n_critic):
            noise = torch.randn(batch_size, self.hparams.noise_dim)
            noise = noise.type_as(real_imgs)
            generated_imgs = self(noise)
            # generated
            generated_pred = self.critic(generated_imgs.detach())
            # real
            real_pred = self.critic(real_imgs)

            critic_loss = torch.mean(generated_pred) - torch.mean(real_pred)
            if self.hparams.gradient_penalty_weight > 0:
                critic_loss += (
                    self.hparams.gradient_penalty_weight
                    * self._get_critic_gradient_penalty(real_imgs, generated_imgs)
                )
            critic_optimizer.zero_grad()
            self.manual_backward(critic_loss, retain_graph=True)
            critic_optimizer.step()
        return generated_imgs, {"critic_loss": critic_loss}

    def _train_generator(self, generator_optimizer, generated_imgs):
        generated_pred = self.critic(generated_imgs)
        generator_loss = -torch.mean(generated_pred)
        generator_optimizer.zero_grad()
        self.manual_backward(generator_loss)
        generator_optimizer.step()
        return {"generator_loss": generator_loss}
