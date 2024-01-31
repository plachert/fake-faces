import pathlib

import lightning as L
import torch

from model.wgan.wgan import WGAN
from src.data.datamodule import ThisPersonDoesNotExistDataModule
from src.logger.face_logger import FaceLogger

torch.set_float32_matmul_precision("medium")


def main():
    image_shape = (3, 32, 32)
    dm = ThisPersonDoesNotExistDataModule(
        data_dir=pathlib.Path("/home/piotr/datasets/vision/fake_faces"),
        batch_size=2,
        img_shape=image_shape[-2:],
    )
    log_dir = "logs/wgan"
    model = WGAN(image_shape=image_shape)
    tb_logger = FaceLogger(save_dir=log_dir, interval=10)
    trainer = L.Trainer(logger=tb_logger, max_epochs=3)
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
