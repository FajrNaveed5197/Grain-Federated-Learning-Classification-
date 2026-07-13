from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASS_NAMES = [
    "Black Germ",
    "Broken",
    "Fusarium",
    "Insect",
    "Moldy",
    "Sound",
    "Spotted",
    "Sprouted",
]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

ROOT = Path("/scratch/project_2019765/grain_research")

CHECKPOINTS = {
    "centralized_resnet18_v3": ROOT / "results/version3_dynamic_balanced_resnet18/best_resnet18_v3_gpu.pt",
    "fedavg_iid_round3": ROOT / "federated_results/fl_full_iid_v3/global_model_round_3.pt",
    "fedavg_noniid_round3": ROOT / "federated_results/fl_full_noniid_v3/global_model_round_3.pt",
    "ddp_resnet18_v3": ROOT / "distributed_results/ddp_resnet18_v3/best_ddp_resnet18_v3.pt",
}

MANIFESTS = {
    "validation": (ROOT / "manifests/validation.csv", 194),
    "test_set_1": (ROOT / "manifests/test.csv", 200),
    "test_set_2": (ROOT / "manifests/test_07.csv", 200),
}

OUTPUT = ROOT / "final_results/final_model_comparison.json"


class GrainDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
        return self.transform(image), CLASS_TO_ID[row["label"]]


def fixed_subset(csv_path: Path, per_class: int, seed: int = 42) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    parts = []

    for label in CLASS_NAMES:
        class_df = dataframe[dataframe["label"] == label]
        parts.append(class_df.sample(n=per_class, random_state=seed))

    return (
        pd.concat(parts)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def calculate_metrics(targets: list[int], predictions: list[int]) -> dict:
    targets_array = np.asarray(targets)
    predictions_array = np.asarray(predictions)

    recalls = []
    f1_scores = []
    per_class = {}

    for class_id, class_name in enumerate(CLASS_NAMES):
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

        per_class[class_name] = {
            "precision": round(float(precision) * 100, 2),
            "recall": round(float(recall) * 100, 2),
            "f1": round(float(f1) * 100, 2),
        }

    return {
        "accuracy": round(float((targets_array == predictions_array).mean()) * 100, 2),
        "balanced_accuracy": round(float(np.mean(recalls)) * 100, 2),
        "macro_f1": round(float(np.mean(f1_scores)) * 100, 2),
        "per_class": per_class,
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    targets = []
    predictions = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device, non_blocking=True))
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    return calculate_metrics(targets, predictions)


def load_model(checkpoint: Path, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    state = torch.load(checkpoint, map_location="cpu")

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    return model.to(device)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    loaders = {}

    for dataset_name, (manifest, per_class) in MANIFESTS.items():
        dataframe = fixed_subset(manifest, per_class)
        loaders[dataset_name] = DataLoader(
            GrainDataset(dataframe, transform),
            batch_size=256,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

    results = {
        "protocol": "Same fixed balanced evaluation samples for all final checkpoints.",
        "root": str(ROOT),
        "datasets": {
            "validation": "194 images per class",
            "test_set_1": "200 images per class",
            "test_set_2": "200 images per class",
        },
        "results": {},
    }

    print(f"Device: {device}", flush=True)

    for model_name, checkpoint in CHECKPOINTS.items():
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

        print(f"\nEvaluating {model_name}", flush=True)
        print(f"Checkpoint: {checkpoint}", flush=True)

        model = load_model(checkpoint, device)

        results["results"][model_name] = {
            "checkpoint": str(checkpoint),
            "metrics": {},
        }

        for dataset_name, loader in loaders.items():
            metrics = evaluate(model, loader, device)
            results["results"][model_name]["metrics"][dataset_name] = metrics
            print(f"{model_name} | {dataset_name}: {metrics}", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\nFINAL COMPARISON SAVED")
    print(OUTPUT)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
