"""Task: implement the baseline and transfer-learning models.

Requirements:
1. Implement a manual ResNet18 baseline from PyTorch layers and residual blocks.
2. The baseline config must use `model="manual_resnet18"` and train from scratch.
3. The transfer config must use `model="resnet18"` and load pretrained torchvision ResNet18
   weights before replacing the final layer.
4. Both models must output four class logits.
5. Implement the frozen or fine-tuned backbone behavior used in your experiments.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision import models


def get_model_name(run_id: str) -> str:
    return f"{run_id}_best.pt"


class BasicBlock(nn.Module):
    """Basic residual block used by the manual ResNet18 baseline."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs: Tensor) -> Tensor:
        identity = self.downsample(inputs)
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return self.relu(outputs + identity)


class BaselineCNN(nn.Module):
    """Manual ResNet18 baseline trained from scratch."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, blocks=2)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 添加 Dropout 减少过拟合
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers: list[nn.Module] = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        layers.extend(BasicBlock(self.in_channels, out_channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.stem(inputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = torch.flatten(self.pool(outputs), 1)
        outputs = self.dropout(outputs)
        return self.classifier(outputs)


def build_model(
    name: str,
    num_classes: int = 4,
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Construct a supported model and replace its classification head.

    Required names are `manual_resnet18` for the baseline and `resnet18` for transfer learning.
    """
    if name == "manual_resnet18":
        if pretrained or freeze_backbone:
            raise ValueError(
                "manual_resnet18 must be trained from scratch without a frozen backbone"
            )
        return BaselineCNN(num_classes=num_classes)
    if name != "resnet18":
        raise ValueError(f"Unsupported model: {name}")

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    # 为迁移学习模型也添加 Dropout
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_features, num_classes),
    )
    return model
