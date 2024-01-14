import pathlib

import lightning as L
from lightning.pytorch import loggers as pl_loggers

from src.data.datamodule import ThisPersonDoesNotExistDataModule
from src.model.dcgan.gan import GAN


def main():
    dm = ThisPersonDoesNotExistDataModule(
        data_dir=pathlib.Path("/home/piotr/datasets/vision/fake_faces")
    )
    model = GAN()
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    trainer = L.Trainer(logger=tb_logger)
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
