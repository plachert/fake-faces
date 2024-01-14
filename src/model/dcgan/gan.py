from typing import Tuple

import lightning as L
import torch
import torchvision

from src.model.dcgan.discriminator import Discriminator
from src.model.dcgan.generator import Generator


class GAN(L.LightningModule):
    def __init__(
        self,
        noise_dim: int = 100,
        image_shape: Tuple[int] = (3, 512, 512),
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(noise_dim, image_shape)
        self.discriminator = Discriminator()
        self.automatic_optimization = False  # We need more flexibility with GANs

    def forward(self, noise):
        return self.generator(noise)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_d, opt_g], []

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        batch_size = real_imgs.shape[0]

        optimizer_d, optimizer_g = self.optimizers()

        # sample noise
        noise = torch.randn(batch_size, self.hparams.latent_dim)
        noise = noise.type_as(real_imgs)
        generated_imgs = self(noise)

        ## Logging
        sample_imgs = self.generated_imgs[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, 0)
        ##

        ### Discriminator (Critic) ###
        generated_pred = self.discriminator(generated_imgs.detach())
        real_pred = self.discriminator(real_imgs)
        d_loss = torch.mean(generated_pred) - torch.mean(real_pred)
        optimizer_d.zero_grad()
        self.manual_backward()
        optimizer_d.step()
        ###

        ### Generator ###
        generated_pred = self.discriminator(generated_imgs)
        g_loss = -torch.mean(generated_pred)
        optimizer_g.zero_grad()
        self.manual_backward()
        optimizer_g.step()
        ###

        self.log_dict({"g_loss": g_loss, "d_loss": d_loss}, prog_bar=True)


if __name__ == "__main__":
    gan = GAN()
