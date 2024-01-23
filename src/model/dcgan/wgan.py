from typing import Tuple
import torch.nn as nn
import lightning as L
import torch
import torchvision
from model.dcgan.critic import Critic
from src.model.dcgan.generator import Generator


class WGAN(L.LightningModule):
    def __init__(
        self,
        noise_dim: int = 100,
        image_shape: Tuple[int] = (3, 512, 512),
        lr: float = 0.0001,
        b1: float = 0,
        b2: float = 0.9,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(noise_dim, image_shape)
        self.critic = Critic(noise_dim, image_shape)
        self.automatic_optimization = False  # We need more flexibility with GANs
        self.fixed_noise = torch.randn(128, self.hparams.noise_dim)
        self._initialize_weights(self.generator)
        self._initialize_weights(self.discriminator)
        
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
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        # opt_g = torch.optim.Adam(list(self.generator.parameters()) + list(self.encoder.parameters()), lr=lr, betas=(b1, b2))
        return [opt_d, opt_g], []

    def training_step(self, batch, batch_idx): # fmt: off

        real_imgs = batch
        batch_size = real_imgs.shape[0]

        optimizer_d, optimizer_g = self.optimizers()

        # sample noise
        # noise = torch.randn(batch_size, self.hparams.noise_dim)
        # noise = noise.type_as(real_imgs)
        # generated_imgs = self(noise)


        ### Discriminator (Critic) ###
        for _ in range(5):
            noise = torch.randn(batch_size, self.hparams.noise_dim)
            noise = noise.type_as(real_imgs)
            generated_imgs = self(noise)
            noise, generated_pred = self.discriminator(generated_imgs.detach())
            _, real_pred = self.discriminator(real_imgs)
            d_loss = torch.mean(generated_pred) - torch.mean(real_pred)
            optimizer_d.zero_grad()
            self.manual_backward(d_loss)
            optimizer_d.step()
            for parameter in self.discriminator.parameters():
                parameter.data.clamp_(-0.01, 0.01)  # grad clipping
            ###
            
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

        ### Generator (fom GAN) ###
        noise, generated_pred = self.discriminator(generated_imgs)
        g_loss_gan = -torch.mean(generated_pred)
        
        noise, _ = self.discriminator(real_imgs)
        reconstructed = self.generator(torch.squeeze(noise))
        g_loss_ae = torch.nn.functional.mse_loss(reconstructed, real_imgs)
        
        # reconstructed_tv = torch.log(tv(reconstructed))
        # real_images_tv = torch.log(tv(real_imgs))
        # tv_loss = torch.nn.functional.mse_loss(reconstructed_tv, real_images_tv)
        
        # g_loss_ae = 0
        tv_loss = 0
        
        g_loss = g_loss_gan + g_loss_ae + tv_loss
        optimizer_g.zero_grad()
        self.manual_backward(g_loss)
        optimizer_g.step()

        self.log_dict({"g_loss": g_loss, "d_loss": d_loss,"g_loss_gan": g_loss_gan, "reconstruction_loss": g_loss_ae, "tv_loss": tv_loss}, prog_bar=True)


if __name__ == "__main__":
    gan = GAN()
