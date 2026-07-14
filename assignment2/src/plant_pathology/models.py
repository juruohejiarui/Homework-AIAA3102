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

from torch import nn


class BaselineCNN(nn.Module):
    """Optional helper class.

    TODO: replace this starter with a manual ResNet18 implementation using residual blocks.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, inputs):  # type annotation is part of the TODO
        raise NotImplementedError


def build_model(
    name: str,
    num_classes: int = 4,
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Construct a supported model and replace its classification head.

    Required names are `manual_resnet18` for the baseline and `resnet18` for transfer learning.
    For transfer learning, use `models.ResNet18_Weights.DEFAULT` as the pretrained weights.
    TODO: implement model construction and the frozen/fine-tuned policy used in your experiments.
    """
    raise NotImplementedError
