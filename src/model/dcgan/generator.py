from typing import Tuple

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim: int, image_shape: Tuple[int]) -> None:
        self.noise_dim = noise_dim
        self.out_channels, self.out_height, self.out_width = image_shape[-3:]
        assert self.out_height == self.out_width  # only square images are supported
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 1024, 4, 1, 0, bias=False),
        )

    def _make_inter_block(self):
        pass

    def _prepare_input(self, noise):
        """Transform noise shape to fit ConvTranspose2d."""
        n_samples, noise_dim = noise.shape
        return noise.view(n_samples, noise_dim, 1, 1)

    def forward(self, noise):
        input_ = self._prepare_input(noise)
        return self.gen(input_)


if __name__ == "__main__":
    g = Generator(100, (3, 64, 64))
    print(g(torch.rand(1, 100)).shape)
