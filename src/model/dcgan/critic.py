from typing import Tuple

import torch.nn as nn


class Critic(nn.Module):
    def __init__(self,  hidden_channels: int, image_shape: Tuple[int]):
        super().__init__()
        self.in_channels, self.in_height, self.in_width = image_shape[-3:]
        self.encode = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
            *self._make_layers(),
        )
        self.critic = nn.Conv2d(
            self.hidden_channels, 1, kernel_size=1, stride=1, padding=0
        )
        self._initialize_weights()

    def _make_layers(self):
        """Generate all encoding blocks."""
        layers = []
        channels = self.hidden_channels
        image_width = self.in_width
        while image_width > 4:
            block = [
                nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(channels * 2, affine=True),
                nn.LeakyReLU(0.2),
            ]
            layers.extend(block)
            channels = channels * 2
            image_width = image_width / 2

        # Final block
        block = [
            nn.Conv2d(channels, self.hidden_channels, 4, 2, 1, bias=False),
        ]
        layers.extend(block)
        return layers

    def forward(self, img):
        encoded = self.encode(img)
        critic_value = self.critic(encoded)
        return {"encoded": encoded, "critic_value": critic_value}
