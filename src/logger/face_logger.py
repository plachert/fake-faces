import torchvision
from pytorch_lightning.loggers import TensorBoardLogger


class FaceLogger(TensorBoardLogger):
    def __init__(
        self,
        n_images: int = 32,
        interval: int = 100,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.logged_images = n_images
        self.interval = interval
        self.step = 0

    def log_images(self, real_images, fake_images, batch_idx):
        if batch_idx % self.interval == 0:
            self._log_real(real_images)
            self._log_fake(fake_images)
            self.step += 1

    def _log_real(self, real_images):
        grid = torchvision.utils.make_grid(
            real_images[: self.logged_images, ...], normalize=True
        )
        self.add_image("Real images", grid, self.step)

    def _log_fake(self, fake_images):
        grid = torchvision.utils.make_grid(
            fake_images[: self.logged_images, ...], normalize=True
        )
        self.add_image("Fake images", grid, self.step)
