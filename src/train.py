import pathlib

import lightning as L
import torch

from model.wgan.wgan import WGAN
from src.data.datamodule import ThisPersonDoesNotExistDataModule
from src.logger.face_logger import FaceLogger

torch.set_float32_matmul_precision("medium")


def main():
    image_shape = (3, 512, 512)
    noise_dim = 100
    dm = ThisPersonDoesNotExistDataModule(
        data_dir=pathlib.Path("/home/piotr/datasets/vision/fake_faces"),
        batch_size=2,
        img_shape=image_shape[-2:],
    )
    log_dir = "logs/wgan"
    model = WGAN(image_shape=image_shape, noise_dim=noise_dim)
    tb_logger = FaceLogger(save_dir=log_dir, interval=50)
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        every_n_train_steps=1000, filename="{epoch}"
    )
    trainer = L.Trainer(
        logger=tb_logger, max_epochs=1, precision=16, callbacks=[checkpoint_callback]
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
