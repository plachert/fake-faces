import math
from typing import Tuple

import torch.nn as nn


class InvalidImageShape(Exception):
    pass


class Generator(nn.Module):
    FIRST_BLOCK_CHANNELS = 1024

    def __init__(self, noise_dim: int, image_shape: Tuple[int]) -> None:
        self.noise_dim = noise_dim
        self.out_channels, self.out_height, self.out_width = image_shape[-3:]
        self._check_image_shape()
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(
                noise_dim, self.FIRST_BLOCK_CHANNELS, 4, 1, 0, bias=False
            ),
            *self._make_layers(),
        )

    def _check_image_shape(self):
        if not self.out_height == self.out_width:
            raise InvalidImageShape("Only square images are supported")
        if not (self.out_height > 0 and (self.out_height & (self.out_height - 1)) == 0):
            raise InvalidImageShape("Image width and height should be a power of two")
        if math.log(self.out_height, 2) - math.log(self.FIRST_BLOCK_CHANNELS, 2) > 2:
            raise InvalidImageShape(
                f"Image width and height should not be greater than {self.FIRST_BLOCK_CHANNELS}^4)"
            )

    def _make_layers(self):
        """Generate all stride 2 blocks."""
        layers = []
        channels = self.FIRST_BLOCK_CHANNELS
        image_width = 4
        while image_width < self.out_width / 2:
            block = [
                nn.ConvTranspose2d(channels, channels // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels // 2),
                nn.ReLU(inplace=True),
            ]
            layers.extend(block)
            channels = channels // 2
            image_width = 2 * image_width

        # Final block
        block = [
            nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        ]
        layers.extend(block)
        return layers

    def _prepare_input(self, noise):
        """Transform noise shape to fit ConvTranspose2d."""
        n_samples, noise_dim = noise.shape
        return noise.view(n_samples, noise_dim, 1, 1)

    def forward(self, noise):
        input_ = self._prepare_input(noise)
        return self.gen(input_)


if __name__ == "__main__":
    m = Generator(100, (3, 512, 512))
    breakpoint()
