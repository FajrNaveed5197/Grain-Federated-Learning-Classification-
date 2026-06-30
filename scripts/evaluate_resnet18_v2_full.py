import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

CLASS_NAMES = [
    "Black Germ", "Broken", "Fusarium", "Insect",
    "Moldy", "Sound", "Spotted", "Sprouted"
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

ROOT = Path("/scratch/project_2019649/grain_research")
CHECKPOINT = ROOT / "results/additional_finetuning_v2_resnet18/best_resnet18_v2_gpu.pt"

class GrainDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        return self.transform(image), CLASS_TO_ID[row["label"]]

device = torch.device("cuda")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model = model.to(device)
model.eval()

validation_df = pd.read_csv(ROOT / "manifests/validation.csv")

loader = DataLoader(
    GrainDataset(validation_df, transform),
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

targets, predictions = [], []

with torch.no_grad():
    for images, labels in loader:
        outputs = model(images.to(device, non_blocking=True))
        predictions.extend(outputs.argmax(dim=1).cpu().tolist())
        targets.extend(labels.tolist())

targets = np.array(targets)
predictions = np.array(predictions)

print("\nRESNET18 VERSION 2 — FULL VALIDATION RESULTS")
print("-" * 68)

f1_scores = []

for class_id, class_name in enumerate(CLASS_NAMES):
    tp = np.sum((predictions == class_id) & (targets == class_id))
    fp = np.sum((predictions == class_id) & (targets != class_id))
    fn = np.sum((predictions != class_id) & (targets == class_id))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    f1_scores.append(f1)

    print(
        f"{class_name:<12} | "
        f"Precision: {precision * 100:6.2f}% | "
        f"Recall: {recall * 100:6.2f}% | "
        f"F1: {f1 * 100:6.2f}%"
    )

accuracy = (targets == predictions).mean() * 100

print("-" * 68)
print(f"Full validation accuracy: {accuracy:.2f}%")
print(f"Full validation Macro F1: {np.mean(f1_scores) * 100:.2f}%")
