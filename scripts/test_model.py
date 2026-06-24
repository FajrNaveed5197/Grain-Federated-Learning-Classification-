import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import build_mobilenet_v2


model = build_mobilenet_v2(
    num_classes=8,
    pretrained=False,
    freeze_backbone=True,
)

sample_batch = torch.rand(2, 3, 256, 256)
output = model(sample_batch)

print("Model:", model.__class__.__name__)
print("Input shape:", tuple(sample_batch.shape))
print("Output shape:", tuple(output.shape))
print("Trainable parameters:", sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
))
print("Model test passed.")
