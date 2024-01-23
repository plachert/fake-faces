import torch.nn as nn
import torch

import torch.nn as nn
from typing import Tuple


class Critic(nn.Module):
    
    HIDDEN_CHANNELS = 100
    
    def __init__(self, image_shape: Tuple[int]):
        super().__init__()
        self.in_channels, self.in_height, self.in_width = image_shape[-3:]
        self.encode = nn.Sequential(
            nn.Conv2d(self.in_channels, self.HIDDEN_CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            *self._make_layers(),
        )
        self.critic = nn.Conv2d(self.HIDDEN_CHANNELS, 1, kernel_size=1, stride=1, padding=0)
        
            
    def _make_layers(self):
        """Generate all encoding blocks."""
        layers = []
        channels = self.HIDDEN_CHANNELS
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
            nn.Conv2d(channels, self.HIDDEN_CHANNELS, 4, 2, 1, bias=False),
        ]
        layers.extend(block)
        return layers

    def forward(self, img):
        encoded = self.encode(img)
        critic_value = self.critic(encoded)
        return {"encoded": encoded, "critic_value": critic_value}

if __name__ == "__main__":
    m = Critic((3, 64, 64))
    print(m(torch.rand(3, 3, 64, 64)))