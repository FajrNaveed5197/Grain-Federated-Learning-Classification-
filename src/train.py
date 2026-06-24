from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from dataset import GrainDataset
from metrics import calculate_metrics, format_metrics
from models import build_mobilenet_v2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balanced_subset(dataset: GrainDataset, per_class: int | None, seed: int) -> Subset:
    if per_class is None:
        return Subset(dataset, list(range(len(dataset))))

    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []

    for class_id in sorted(dataset.df["target"].unique()):
        class_indices = dataset.df.index[
            dataset.df["target"] == class_id
        ].to_numpy()

        take = min(per_class, len(class_indices))
        chosen = rng.choice(class_indices, size=take, replace=False)
        selected_indices.extend(chosen.tolist())

    rng.shuffle(selected_indices)
    return Subset(dataset, selected_indices)


def subset_targets(dataset: GrainDataset, subset: Subset) -> list[int]:
    return dataset.df.iloc[subset.indices]["target"].astype(int).tolist()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict:
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    return calculate_metrics(y_true, y_pred, class_names)


def write_experiment_log(log_path: Path, row: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = log_path.exists()

    with open(log_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    run_id = config["run_id"]
    set_seed(int(config["seed"]))

    output_dir = Path(config["output_dir"]) / run_id
    checkpoint_dir = Path(config["checkpoint_dir"]) / run_id

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with open(config["class_mapping"], "r", encoding="utf-8") as handle:
        class_to_idx = json.load(handle)

    class_names = [
        class_name
        for class_name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])
    ]

    image_size = int(config["image_size"])

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = GrainDataset(
        config["train_manifest"],
        config["class_mapping"],
        transform=train_transform,
    )

    val_dataset = GrainDataset(
        config["val_manifest"],
        config["class_mapping"],
        transform=eval_transform,
    )

    test1_dataset = GrainDataset(
        config["test1_manifest"],
        config["class_mapping"],
        transform=eval_transform,
    )

    test2_dataset = GrainDataset(
        config["test2_manifest"],
        config["class_mapping"],
        transform=eval_transform,
    )

    train_subset = balanced_subset(
        train_dataset,
        config.get("train_per_class"),
        int(config["seed"]),
    )

    val_subset = balanced_subset(
        val_dataset,
        config.get("val_per_class"),
        int(config["seed"]) + 1,
    )

    test1_subset = balanced_subset(
        test1_dataset,
        config.get("test1_per_class"),
        int(config["seed"]) + 2,
    )

    test2_subset = balanced_subset(
        test2_dataset,
        config.get("test2_per_class"),
        int(config["seed"]) + 3,
    )

    batch_size = int(config["batch_size"])
    num_workers = int(config["num_workers"])

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test1_loader = DataLoader(
        test1_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test2_loader = DataLoader(
        test2_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Run ID: {run_id}")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Test Set 1 samples: {len(test1_subset)}")
    print(f"Test Set 2 samples: {len(test2_subset)}")

    model = build_mobilenet_v2(
        num_classes=int(config["num_classes"]),
        pretrained=bool(config["pretrained"]),
        freeze_backbone=bool(config["freeze_backbone"]),
    ).to(device)

    train_targets = subset_targets(train_dataset, train_subset)

    if config.get("use_class_weights", False):
        classes = np.arange(len(class_names))

        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=np.array(train_targets),
        )

        class_weights = torch.tensor(
            weights,
            dtype=torch.float32,
            device=device,
        )

        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    best_val_macro_f1 = -1.0
    best_checkpoint = checkpoint_dir / "best_model.pt"

    start_time = time.time()

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()

        total_loss = 0.0
        total_items = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_items += images.size(0)

        train_loss = total_loss / max(total_items, 1)
        val_metrics = evaluate(model, val_loader, device, class_names)

        print(
            f"Epoch {epoch}/{config['epochs']} | "
            f"Train loss: {train_loss:.4f} | "
            f"{format_metrics(val_metrics)}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "class_to_idx": class_to_idx,
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                },
                best_checkpoint,
            )

    elapsed_minutes = (time.time() - start_time) / 60

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_val_metrics = evaluate(model, val_loader, device, class_names)
    test1_metrics = evaluate(model, test1_loader, device, class_names)
    test2_metrics = evaluate(model, test2_loader, device, class_names)

    all_metrics = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "train_samples": len(train_subset),
        "validation_samples": len(val_subset),
        "test1_samples": len(test1_subset),
        "test2_samples": len(test2_subset),
        "training_time_minutes": elapsed_minutes,
        "validation": final_val_metrics,
        "test1": test1_metrics,
        "test2": test2_metrics,
        "checkpoint": str(best_checkpoint),
    }

    metrics_path = output_dir / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)

    with open(output_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    log_row = {
        "run_id": run_id,
        "date": datetime.now().isoformat(timespec="seconds"),
        "stage": "centralized_smoke_test",
        "model": config["model"],
        "framework": "PyTorch",
        "seed": config["seed"],
        "train_size": len(train_subset),
        "val_size": len(val_subset),
        "test1_size": len(test1_subset),
        "test2_size": len(test2_subset),
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "device": str(device),
        "training_time_minutes": round(elapsed_minutes, 2),
        "val_macro_f1": round(final_val_metrics["macro_f1"], 4),
        "test1_accuracy": round(test1_metrics["accuracy"], 4),
        "test1_balanced_accuracy": round(test1_metrics["balanced_accuracy"], 4),
        "test1_macro_f1": round(test1_metrics["macro_f1"], 4),
        "test2_accuracy": round(test2_metrics["accuracy"], 4),
        "test2_balanced_accuracy": round(test2_metrics["balanced_accuracy"], 4),
        "test2_macro_f1": round(test2_metrics["macro_f1"], 4),
        "checkpoint_path": str(best_checkpoint),
        "config_path": str(config_path),
        "notes": "Local CPU smoke test",
    }

    if config.get("track_experiment", True):
        write_experiment_log(Path(config["experiment_log"]), log_row)

    print("\n=== Final validation metrics ===")
    print(format_metrics(final_val_metrics))

    print("\n=== Test Set 1 metrics ===")
    print(format_metrics(test1_metrics))

    print("\n=== Test Set 2 metrics ===")
    print(format_metrics(test2_metrics))

    print(f"\nSaved checkpoint: {best_checkpoint}")
    print(f"Saved metrics: {metrics_path}")
    print("Training run completed successfully.")


if __name__ == "__main__":
    main()
