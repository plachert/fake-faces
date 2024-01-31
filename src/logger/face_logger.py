import torch
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger


class FaceLogger(TensorBoardLogger):
    def __init__(
        self,
        n_images: int = 32,
        interval: int = 100,
        log_real_once: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_images = n_images
        self.interval = interval
        self.step = 0
        self.fixed_noise = None
        self.log_real_once = log_real_once
        self.last_images = {"Fake": None, "Real": None}

    def set_fixed_noise(self, noise_dim: int, device: torch.device):
        self.fixed_noise = torch.randn(self.n_images, noise_dim).to(device)

    def log_images(self, real_images, generator, batch_idx):
        if batch_idx % self.interval == 0:
            fake_images = generator(self.fixed_noise).detach()
            self._log_fake(fake_images)
            if self.log_real_once:
                if self.step == 0:
                    self._log_real(real_images)
            else:
                self._log_real(real_images)
            self.step += 1

    def save_last_images(self):
        real = torchvision.transforms.ToPILImage()(self.last_images["Real"])
        fake = torchvision.transforms.ToPILImage()(self.last_images["Fake"])
        real.save(f"{self.log_dir}/real.jpg")
        fake.save(f"{self.log_dir}/fake.jpg")

    def _log_real(self, real_images):
        grid = torchvision.utils.make_grid(
            real_images[: self.n_images, ...], normalize=True
        )
        self.experiment.add_image("Real images", grid, self.step)
        self.last_images["Real"] = grid

    def _log_fake(self, fake_images):
        grid = torchvision.utils.make_grid(
            fake_images[: self.n_images, ...], normalize=True
        )
        self.experiment.add_image("Fake images", grid, self.step)
        self.last_images["Fake"] = grid
