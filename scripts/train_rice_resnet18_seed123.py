from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


SEED = 123

DATA_ROOT = Path(
    "/scratch/project_2019765/fnaveed/datasets/rice_grouped"
)

MANIFEST_DIR = DATA_ROOT / "grouped_split"

TRAIN_CSV = MANIFEST_DIR / "train.csv"
VAL_CSV = MANIFEST_DIR / "validation.csv"
TEST_CSV = MANIFEST_DIR / "test.csv"

OUTPUT_DIR = Path(
    "/scratch/project_2019765/fnaveed/results/"
    "rice_resnet18_grouped_seed123"
)

CLASS_NAMES = [
    "0_NOR",
    "1_F&S",
    "2_SD",
    "3_MY",
    "4_AP",
    "5_BN",
    "6_UN",
    "7_IM",
]

CLASS_TO_ID = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

IMAGE_SIZE = 224
BATCH_SIZE = 128
NUM_WORKERS = 6

WARMUP_EPOCHS = 2
FINETUNE_EPOCHS = 8

WARMUP_LR = 1e-3
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4


class RiceDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        transform,
    ) -> None:
        self.dataframe = pd.read_csv(manifest_path)
        self.transform = transform

        required_columns = {
            "image_path",
            "class_name",
            "capture_group",
        }

        missing = required_columns - set(self.dataframe.columns)

        if missing:
            raise ValueError(
                f"{manifest_path} is missing columns: {sorted(missing)}"
            )

        unknown_classes = (
            set(self.dataframe["class_name"].unique())
            - set(CLASS_NAMES)
        )

        if unknown_classes:
            raise ValueError(
                f"Unknown classes: {sorted(unknown_classes)}"
            )

        self.targets = (
            self.dataframe["class_name"]
            .map(CLASS_TO_ID)
            .astype(int)
            .tolist()
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]

        image_path = DATA_ROOT / row["image_path"]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image at row {index}: {image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        image = self.transform(image)
        label = self.targets[index]

        return image, label


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(num_classes: int) -> nn.Module:
    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


def freeze_backbone(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def unfreeze_model(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def calculate_class_weights(
    targets: list[int],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    counts = np.bincount(
        np.asarray(targets),
        minlength=num_classes,
    )

    total = len(targets)

    weights = total / (
        num_classes * np.maximum(counts, 1)
    )

    print("\nTraining class counts and weights:")

    for class_id, class_name in enumerate(CLASS_NAMES):
        print(
            f"  {class_name}: "
            f"count={counts[class_id]}, "
            f"weight={weights[class_id]:.4f}"
        )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    running_loss = 0.0
    sample_count = 0

    for images, labels in loader:
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size
        sample_count += batch_size

    return running_loss / max(sample_count, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict, list[int], list[int]]:
    model.eval()

    total_loss = 0.0
    sample_count = 0

    targets: list[int] = []
    predictions: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(
                device,
                non_blocking=True,
            )
            labels = labels.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)
            loss = criterion(outputs, labels)

            predicted = outputs.argmax(dim=1)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            sample_count += batch_size

            targets.extend(labels.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())

    targets_array = np.asarray(targets)
    predictions_array = np.asarray(predictions)

    accuracy = float(
        np.mean(targets_array == predictions_array)
    )

    metrics = {
        "loss": total_loss / max(sample_count, 1),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy_score(
            targets,
            predictions,
        ),
        "macro_f1": f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            targets,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }

    return metrics, targets, predictions


def save_confusion_matrix(
    targets: list[int],
    predictions: list[int],
    output_path: Path,
) -> None:
    matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
    )

    figure, axis = plt.subplots(figsize=(10, 8))

    image = axis.imshow(matrix)

    figure.colorbar(image, ax=axis)

    axis.set_xticks(range(len(CLASS_NAMES)))
    axis.set_yticks(range(len(CLASS_NAMES)))

    axis.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )
    axis.set_yticklabels(CLASS_NAMES)

    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Rice ResNet18 Test Confusion Matrix")

    threshold = matrix.max() / 2 if matrix.size else 0

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row, column] > threshold
                    else "black"
                ),
            )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "GPU was not detected. Run this through a GPU SLURM job."
        )

    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset root: {DATA_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.85, 1.0),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.20),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.08,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    evaluation_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = RiceDataset(
        TRAIN_CSV,
        train_transform,
    )
    validation_dataset = RiceDataset(
        VAL_CSV,
        evaluation_transform,
    )
    test_dataset = RiceDataset(
        TEST_CSV,
        evaluation_transform,
    )

    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")

    loader_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": NUM_WORKERS > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_options,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_options,
    )

    model = create_model(
        num_classes=len(CLASS_NAMES)
    ).to(device)

    class_weights = calculate_class_weights(
        train_dataset.targets,
        len(CLASS_NAMES),
        device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    best_macro_f1 = -1.0
    best_epoch = -1
    history: list[dict] = []

    checkpoint_path = (
        OUTPUT_DIR / "best_resnet18_rice_grouped.pt"
    )

    start_time = time.time()

    freeze_backbone(model)

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=WARMUP_LR,
        weight_decay=WEIGHT_DECAY,
    )

    total_epochs = WARMUP_EPOCHS + FINETUNE_EPOCHS

    for epoch_index in range(total_epochs):
        epoch_number = epoch_index + 1

        if epoch_index == WARMUP_EPOCHS:
            print("\nUnfreezing complete ResNet18 model.")

            unfreeze_model(model)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=FINETUNE_LR,
                weight_decay=WEIGHT_DECAY,
            )

        stage = (
            "warmup"
            if epoch_index < WARMUP_EPOCHS
            else "full_finetuning"
        )

        epoch_start = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_metrics, _, _ = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        epoch_result = {
            "epoch": epoch_number,
            "stage": stage,
            "train_loss": round(train_loss, 6),
            "validation_loss": round(
                validation_metrics["loss"],
                6,
            ),
            "validation_accuracy": round(
                validation_metrics["accuracy"] * 100,
                2,
            ),
            "validation_balanced_accuracy": round(
                validation_metrics["balanced_accuracy"] * 100,
                2,
            ),
            "validation_macro_f1": round(
                validation_metrics["macro_f1"] * 100,
                2,
            ),
            "epoch_seconds": round(
                time.time() - epoch_start,
                2,
            ),
        }

        history.append(epoch_result)

        print(
            f"Epoch {epoch_number:02d}/{total_epochs} "
            f"[{stage}] | "
            f"train loss={train_loss:.4f} | "
            f"val loss={validation_metrics['loss']:.4f} | "
            f"val acc={validation_metrics['accuracy'] * 100:.2f}% | "
            f"val balanced acc="
            f"{validation_metrics['balanced_accuracy'] * 100:.2f}% | "
            f"val macro-F1="
            f"{validation_metrics['macro_f1'] * 100:.2f}%"
        )

        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch_number

            torch.save(
                {
                    "epoch": epoch_number,
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "validation_metrics": validation_metrics,
                },
                checkpoint_path,
            )

            print(
                f"Saved new best checkpoint: {checkpoint_path}"
            )

        with (
            OUTPUT_DIR / "history.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_metrics, test_targets, test_predictions = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    report = classification_report(
        test_targets,
        test_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    save_confusion_matrix(
        test_targets,
        test_predictions,
        OUTPUT_DIR / "confusion_matrix.png",
    )

    predictions_dataframe = pd.DataFrame({
        "true_id": test_targets,
        "predicted_id": test_predictions,
        "true_class": [
            CLASS_NAMES[class_id]
            for class_id in test_targets
        ],
        "predicted_class": [
            CLASS_NAMES[class_id]
            for class_id in test_predictions
        ],
    })

    predictions_dataframe.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    final_results = {
        "experiment": "rice_resnet18_grouped_seed123",
        "dataset_root": str(DATA_ROOT),
        "seed": SEED,
        "classes": CLASS_NAMES,
        "split_sizes": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
            "test": len(test_dataset),
        },
        "training": {
            "warmup_epochs": WARMUP_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": round(
                best_macro_f1 * 100,
                2,
            ),
            "total_seconds": round(
                time.time() - start_time,
                2,
            ),
        },
        "test": {
            "loss": round(test_metrics["loss"], 6),
            "accuracy": round(
                test_metrics["accuracy"] * 100,
                2,
            ),
            "balanced_accuracy": round(
                test_metrics["balanced_accuracy"] * 100,
                2,
            ),
            "macro_f1": round(
                test_metrics["macro_f1"] * 100,
                2,
            ),
            "weighted_f1": round(
                test_metrics["weighted_f1"] * 100,
                2,
            ),
        },
        "classification_report": report,
    }

    with (
        OUTPUT_DIR / "metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(final_results, file, indent=2)

    print("\nFinal test results")
    print(
        f"Accuracy: "
        f"{test_metrics['accuracy'] * 100:.2f}%"
    )
    print(
        f"Balanced accuracy: "
        f"{test_metrics['balanced_accuracy'] * 100:.2f}%"
    )
    print(
        f"Macro-F1: "
        f"{test_metrics['macro_f1'] * 100:.2f}%"
    )
    print(
        f"Weighted-F1: "
        f"{test_metrics['weighted_f1'] * 100:.2f}%"
    )
    print(f"Best epoch: {best_epoch}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
