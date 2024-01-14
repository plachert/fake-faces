import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        weights = ResNet50_Weights.DEFAULT
        self.backbone = nn.Sequential(*list(resnet50(weights).children())[:-2])
        self.backbone_transform = weights.transforms()
        self.critic = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 1),
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, img):
        transformed = self.backbone_transform(img)
        feats = self.backbone(transformed)
        return self.critic(feats)
