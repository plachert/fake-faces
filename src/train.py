import pathlib

import lightning as L
from lightning.pytorch import loggers as pl_loggers

from model.wgan.wgan import WGAN
from src.data.datamodule import ThisPersonDoesNotExistDataModule


def main():
    image_shape = (3, 64, 64)
    dm = ThisPersonDoesNotExistDataModule(
        data_dir=pathlib.Path("/home/piotr/datasets/vision/fake_faces"),
        batch_size=32,
        img_shape=image_shape[-2:],
    )
    model = WGAN(image_shape=image_shape, critic_autoencoder=True)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/wgan", log_graph=True)
    trainer = L.Trainer(logger=tb_logger)
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
