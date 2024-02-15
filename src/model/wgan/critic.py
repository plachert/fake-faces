import math
from typing import Tuple

import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, hidden_channels: int, image_shape: Tuple[int]):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.in_channels, self.in_height, self.in_width = image_shape[-3:]
        self.number_of_blocks = int(
            math.log(self.in_width, 2) - 3
        )  # number of blocks except the first one
        self.final_hidden_channels = hidden_channels * (2**self.number_of_blocks)
        self.feat_extractor = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
            *self._make_extractor_layers(),
        )
        self.critic = nn.Conv2d(
            self.final_hidden_channels, 1, kernel_size=4, stride=2, padding=0
        )

    def _make_extractor_layers(self):
        """Generate all encoding blocks."""
        layers = []
        channels = self.hidden_channels
        for _ in range(self.number_of_blocks):
            block = [
                nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(channels * 2, affine=True),
                nn.LeakyReLU(0.2),
            ]
            layers.extend(block)
            channels = channels * 2
        return layers

    def forward(self, img):
        feats = self.feat_extractor(img)
        critic_value = self.critic(feats)
        return critic_value
