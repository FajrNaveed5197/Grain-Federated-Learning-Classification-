from __future__ import annotations

import torch.nn as nn
from torchvision import models


def create_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a ResNet18 classifier for federated/distributed experiments."""

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
