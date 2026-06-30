import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASS_NAMES = [
    "Black Germ", "Broken", "Fusarium", "Insect",
    "Moldy", "Sound", "Spotted", "Sprouted",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

ROOT = Path("/scratch/project_2019649/grain_research")

CHECKPOINTS = {
    "v2": ROOT / "results/additional_finetuning_v2_resnet18/best_resnet18_v2_gpu.pt",
    "v3": ROOT / "results/version3_dynamic_balanced_resnet18/best_resnet18_v3_gpu.pt",
}

MANIFESTS = {
    "validation": ROOT / "manifests/validation.csv",
    "test_set_1": ROOT / "manifests/test.csv",
    "test_set_2": ROOT / "manifests/test_07.csv",
}

OUTPUT = ROOT / "results/version3_dynamic_balanced_resnet18/fair_v2_v3_comparison.json"


class GrainDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        return self.transform(image), CLASS_TO_ID[row["label"]]


def v2_fixed_balanced_subset(csv_path, per_class, seed=42):
    df = pd.read_csv(csv_path)
    groups = []

    for label in CLASS_NAMES:
        groups.append(
            df[df["label"] == label].sample(
                n=per_class,
                random_state=seed,
            )
        )

    return (
        pd.concat(groups)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def calculate_metrics(targets, predictions):
    targets = np.array(targets)
    predictions = np.array(predictions)

    per_class_f1 = []
    per_class_recall = []

    for class_id in range(len(CLASS_NAMES)):
        tp = np.sum((predictions == class_id) & (targets == class_id))
        fp = np.sum((predictions == class_id) & (targets != class_id))
        fn = np.sum((predictions != class_id) & (targets == class_id))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        per_class_recall.append(recall)
        per_class_f1.append(f1)

    return {
        "accuracy": round(float((targets == predictions).mean()) * 100, 2),
        "balanced_accuracy": round(float(np.mean(per_class_recall)) * 100, 2),
        "macro_f1": round(float(np.mean(per_class_f1)) * 100, 2),
    }


def evaluate(model, loader, device):
    model.eval()
    targets = []
    predictions = []

    with torch.no_grad():
        for images, labels in loader:
            output = model(images.to(device, non_blocking=True))
            predictions.extend(output.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    return calculate_metrics(targets, predictions)


def load_resnet18(checkpoint, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device)


def main():
    device = torch.device("cuda")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    datasets = {
        "validation_fixed_v2_protocol": v2_fixed_balanced_subset(
            MANIFESTS["validation"], 194
        ),
        "test_set_1_fixed_v2_protocol": v2_fixed_balanced_subset(
            MANIFESTS["test_set_1"], 200
        ),
        "test_set_2_fixed_v2_protocol": v2_fixed_balanced_subset(
            MANIFESTS["test_set_2"], 200
        ),
        "full_validation": pd.read_csv(MANIFESTS["validation"]),
    }

    loaders = {
        name: DataLoader(
            GrainDataset(df, transform),
            batch_size=128,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        for name, df in datasets.items()
    }

    results = {
        "protocol": (
            "Both checkpoints evaluated on the exact fixed balanced "
            "samples used by Version 2, plus the complete validation set."
        ),
        "datasets": {name: len(df) for name, df in datasets.items()},
        "checkpoints": {},
    }

    for version, checkpoint in CHECKPOINTS.items():
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        model = load_resnet18(checkpoint, device)
        print(f"\nEvaluating {version}: {checkpoint}")

        results["checkpoints"][version] = {
            "checkpoint": str(checkpoint),
            **{
                dataset_name: evaluate(model, loader, device)
                for dataset_name, loader in loaders.items()
            },
        }

        print(json.dumps(results["checkpoints"][version], indent=2))

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved fair comparison: {OUTPUT}")


if __name__ == "__main__":
    main()
