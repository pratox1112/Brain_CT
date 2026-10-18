"""ResNet-50 with 6 independent outputs (any + 5 hemorrhage subtypes)."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

NUM_OUTPUTS = 6


def build_model(pretrained: bool = True) -> nn.Module:
    """torchvision ResNet-50, ImageNet-pretrained if requested, with a 6-output final layer."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, NUM_OUTPUTS)
    return model
