from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_mobilenet_v2(
    num_classes: int = 8,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Creates a MobileNetV2 classifier for the 8 grain classes.
    """

    weights = (
        models.MobileNet_V2_Weights.IMAGENET1K_V1
        if pretrained
        else None
    )

    model = models.mobilenet_v2(weights=weights)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features=input_features,
        out_features=num_classes,
    )

    return model
