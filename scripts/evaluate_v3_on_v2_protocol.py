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
CHECKPOINT = ROOT / "results/version3_dynamic_balanced_resnet18/best_resnet18_v3_gpu.pt"

MANIFESTS = {
    "validation": (ROOT / "manifests/validation.csv", 194),
    "test_set_1": (ROOT / "manifests/test.csv", 200),
    "test_set_2": (ROOT / "manifests/test_07.csv", 200),
}

V2_BASELINE = {
    "validation": {"macro_f1": 78.55},
    "test_set_1": {"macro_f1": 85.29},
    "test_set_2": {"macro_f1": 72.57},
}

OUTPUT = ROOT / "results/version3_dynamic_balanced_resnet18/v3_on_v2_fixed_protocol.json"


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


def fixed_v2_subset(csv_path, per_class, seed=42):
    df = pd.read_csv(csv_path)
    pieces = []

    for label in CLASS_NAMES:
        pieces.append(
            df[df["label"] == label].sample(
                n=per_class,
                random_state=seed,
            )
        )

    return (
        pd.concat(pieces)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def calculate_metrics(targets, predictions):
    targets = np.array(targets)
    predictions = np.array(predictions)

    recalls = []
    f1_scores = []

    for class_id in range(len(CLASS_NAMES)):
        tp = np.sum((predictions == class_id) & (targets == class_id))
        fp = np.sum((predictions == class_id) & (targets != class_id))
        fn = np.sum((predictions != class_id) & (targets == class_id))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "accuracy": round(float((targets == predictions).mean()) * 100, 2),
        "balanced_accuracy": round(float(np.mean(recalls)) * 100, 2),
        "macro_f1": round(float(np.mean(f1_scores)) * 100, 2),
    }


def evaluate(model, loader, device, label):
    model.eval()
    targets = []
    predictions = []

    print(f"\nEvaluating {label}: {len(loader.dataset)} images", flush=True)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            output = model(images.to(device, non_blocking=True))
            predictions.extend(output.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())

            if batch_idx % 10 == 0 or batch_idx == len(loader):
                print(
                    f"{label}: batch {batch_idx}/{len(loader)}",
                    flush=True,
                )

    return calculate_metrics(targets, predictions)


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    device = torch.device("cuda")

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
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model = model.to(device)

    results = {
        "protocol": "V3 evaluated on exact fixed balanced samples used by V2.",
        "checkpoint": str(CHECKPOINT),
        "v2_reference_macro_f1": V2_BASELINE,
        "v3_results": {},
        "macro_f1_delta_v3_minus_v2": {},
    }

    for dataset_name, (manifest, per_class) in MANIFESTS.items():
        dataframe = fixed_v2_subset(manifest, per_class)
        loader = DataLoader(
            GrainDataset(dataframe, transform),
            batch_size=256,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )

        metrics = evaluate(model, loader, device, dataset_name)
        results["v3_results"][dataset_name] = metrics
        results["macro_f1_delta_v3_minus_v2"][dataset_name] = round(
            metrics["macro_f1"] - V2_BASELINE[dataset_name]["macro_f1"],
            2,
        )

        print(f"{dataset_name} result: {metrics}", flush=True)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nFINAL FAIR V3 VS V2 COMPARISON", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print(f"\nSaved: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
