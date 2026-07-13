from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image


CLASS_NAMES = [
    "Black Germ", "Broken", "Fusarium", "Insect",
    "Moldy", "Sound", "Spotted", "Sprouted",
]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a federated ResNet18 checkpoint on the fixed V2/V3 protocol."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


class GrainDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
        return self.transform(image), CLASS_TO_ID[row["label"]]


def fixed_subset(csv_path: Path, per_class: int, seed: int = 42) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    parts = []

    for label in CLASS_NAMES:
        parts.append(
            dataframe[dataframe["label"] == label].sample(
                n=per_class,
                random_state=seed,
            )
        )

    return (
        pd.concat(parts)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def calculate_metrics(targets: list[int], predictions: list[int]) -> dict:
    targets_array = np.array(targets)
    predictions_array = np.array(predictions)

    recalls = []
    f1_scores = []

    for class_id in range(len(CLASS_NAMES)):
        true_positive = np.sum(
            (predictions_array == class_id) & (targets_array == class_id)
        )
        false_positive = np.sum(
            (predictions_array == class_id) & (targets_array != class_id)
        )
        false_negative = np.sum(
            (predictions_array != class_id) & (targets_array == class_id)
        )

        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        recalls.append(float(recall))
        f1_scores.append(float(f1))

    return {
        "accuracy": round(float((targets_array == predictions_array).mean()) * 100, 2),
        "balanced_accuracy": round(float(np.mean(recalls)) * 100, 2),
        "macro_f1": round(float(np.mean(f1_scores)) * 100, 2),
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, name: str) -> dict:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []

    print(f"\nEvaluating {name}: {len(loader.dataset)} images", flush=True)

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            outputs = model(images.to(device, non_blocking=True))
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())

            if batch_index % 5 == 0 or batch_index == len(loader):
                print(f"{name}: batch {batch_index}/{len(loader)}", flush=True)

    return calculate_metrics(targets, predictions)


def main() -> None:
    args = parse_args()

    root = Path(args.root)
    checkpoint = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    manifests = {
        "validation": (root / "manifests/validation.csv", 194),
        "test_set_1": (root / "manifests/test.csv", 200),
        "test_set_2": (root / "manifests/test_07.csv", 200),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {checkpoint}", flush=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device)

    results = {
        "experiment": "federated_iid_resnet18_v3",
        "checkpoint": str(checkpoint),
        "protocol": "Exact fixed balanced samples previously used for V2 and V3 comparison.",
        "results": {},
    }

    for dataset_name, (manifest_path, per_class) in manifests.items():
        dataframe = fixed_subset(manifest_path, per_class)

        loader = DataLoader(
            GrainDataset(dataframe, transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        metrics = evaluate(model, loader, device, dataset_name)
        results["results"][dataset_name] = metrics
        print(f"{dataset_name}: {metrics}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\nFINAL RESULTS", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print(f"\nSaved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
