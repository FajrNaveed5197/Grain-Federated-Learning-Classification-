from __future__ import annotations

import json
import random
import time
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


SEED = 42

MANIFEST_DIR = Path(
    "/scratch/project_2019765/fnaveed/datasets/"
    "wheat_grouped_v1/grouped_split_v2"
)

TRAIN_CSV = MANIFEST_DIR / "train.csv"
VALIDATION_CSV = MANIFEST_DIR / "validation.csv"
TEST_CSV = MANIFEST_DIR / "test.csv"
TEST_07_CSV = MANIFEST_DIR / "test_07.csv"

OUTPUT_DIR = Path(
    "/scratch/project_2019765/fnaveed/results/"
    "wheat_resnet18_grouped_v3_sqrt_weights"
)

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


class WheatDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        transform,
    ) -> None:
        self.manifest_path = manifest_path
        self.dataframe = pd.read_csv(manifest_path)
        self.transform = transform

        required_columns = {
            "path",
            "label",
            "capture_group",
        }

        missing = (
            required_columns
            - set(self.dataframe.columns)
        )

        if missing:
            raise ValueError(
                f"{manifest_path} is missing columns: "
                f"{sorted(missing)}"
            )

        unknown_classes = (
            set(self.dataframe["label"].unique())
            - set(CLASS_NAMES)
        )

        if unknown_classes:
            raise ValueError(
                f"Unknown classes in {manifest_path}: "
                f"{sorted(unknown_classes)}"
            )

        self.targets = (
            self.dataframe["label"]
            .map(CLASS_TO_ID)
            .astype(int)
            .tolist()
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image_path = Path(row["path"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image at row {index}: "
                f"{image_path}"
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


def create_model(
    num_classes: int,
) -> nn.Module:
    model = models.resnet18(
        weights=(
            models.ResNet18_Weights.IMAGENET1K_V1
        )
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


def freeze_backbone(
    model: nn.Module,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def unfreeze_model(
    model: nn.Module,
) -> None:
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

    weights = np.sqrt(
        total / (
            num_classes * np.maximum(counts, 1)
        )
    )

    weights = weights / weights.mean()

    print(
        "\nTraining class counts and weights:",
        flush=True,
    )

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        print(
            f"  {class_name}: "
            f"count={counts[class_id]}, "
            f"weight={weights[class_id]:.4f}",
            flush=True,
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

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        current_batch_size = labels.size(0)

        running_loss += (
            loss.item()
            * current_batch_size
        )

        sample_count += current_batch_size

    return (
        running_loss
        / max(sample_count, 1)
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[
    dict,
    list[int],
    list[int],
]:
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

            predicted = outputs.argmax(
                dim=1
            )

            current_batch_size = (
                labels.size(0)
            )

            total_loss += (
                loss.item()
                * current_batch_size
            )

            sample_count += (
                current_batch_size
            )

            targets.extend(
                labels.cpu().tolist()
            )

            predictions.extend(
                predicted.cpu().tolist()
            )

    accuracy = float(
        np.mean(
            np.asarray(targets)
            == np.asarray(predictions)
        )
    )

    metrics = {
        "loss": (
            total_loss
            / max(sample_count, 1)
        ),
        "accuracy": accuracy,
        "balanced_accuracy": (
            balanced_accuracy_score(
                targets,
                predictions,
            )
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

    return (
        metrics,
        targets,
        predictions,
    )


def save_confusion_matrix(
    targets: list[int],
    predictions: list[int],
    output_path: Path,
    split_name: str,
) -> None:
    matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(
            range(len(CLASS_NAMES))
        ),
    )

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    matrix_dataframe.to_csv(
        output_path.with_suffix(".csv")
    )

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    image = axis.imshow(matrix)
    figure.colorbar(image, ax=axis)

    axis.set_xticks(
        range(len(CLASS_NAMES))
    )

    axis.set_yticks(
        range(len(CLASS_NAMES))
    )

    axis.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(
        CLASS_NAMES
    )

    axis.set_xlabel(
        "Predicted class"
    )

    axis.set_ylabel(
        "True class"
    )

    axis.set_title(
        f"Wheat ResNet18 "
        f"{split_name} Confusion Matrix"
    )

    threshold = (
        matrix.max() / 2
        if matrix.size
        else 0
    )

    for row in range(matrix.shape[0]):
        for column in range(
            matrix.shape[1]
        ):
            axis.text(
                column,
                row,
                str(
                    matrix[row, column]
                ),
                ha="center",
                va="center",
                color=(
                    "white"
                    if (
                        matrix[row, column]
                        > threshold
                    )
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


def save_split_results(
    split_name: str,
    dataset: WheatDataset,
    metrics: dict,
    targets: list[int],
    predictions: list[int],
) -> dict:
    report = classification_report(
        targets,
        predictions,
        labels=list(
            range(len(CLASS_NAMES))
        ),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    save_confusion_matrix(
        targets,
        predictions,
        OUTPUT_DIR
        / f"{split_name}_confusion_matrix.png",
        split_name=split_name,
    )

    predictions_dataframe = (
        dataset.dataframe.copy()
    )

    predictions_dataframe[
        "true_id"
    ] = targets

    predictions_dataframe[
        "predicted_id"
    ] = predictions

    predictions_dataframe[
        "true_class"
    ] = [
        CLASS_NAMES[class_id]
        for class_id in targets
    ]

    predictions_dataframe[
        "predicted_class"
    ] = [
        CLASS_NAMES[class_id]
        for class_id in predictions
    ]

    predictions_dataframe[
        "correct"
    ] = (
        np.asarray(targets)
        == np.asarray(predictions)
    )

    predictions_dataframe.to_csv(
        OUTPUT_DIR
        / f"{split_name}_predictions.csv",
        index=False,
    )

    per_class_rows = []

    for class_name in CLASS_NAMES:
        class_metrics = report[
            class_name
        ]

        per_class_rows.append({
            "class_name": class_name,
            "precision": (
                class_metrics["precision"]
            ),
            "recall": (
                class_metrics["recall"]
            ),
            "f1_score": (
                class_metrics["f1-score"]
            ),
            "support": int(
                class_metrics["support"]
            ),
        })

    pd.DataFrame(
        per_class_rows
    ).to_csv(
        OUTPUT_DIR
        / f"{split_name}_per_class_metrics.csv",
        index=False,
    )

    return {
        "loss": round(
            metrics["loss"],
            6,
        ),
        "accuracy": round(
            metrics["accuracy"] * 100,
            4,
        ),
        "balanced_accuracy": round(
            metrics[
                "balanced_accuracy"
            ] * 100,
            4,
        ),
        "macro_f1": round(
            metrics["macro_f1"] * 100,
            4,
        ),
        "weighted_f1": round(
            metrics["weighted_f1"] * 100,
            4,
        ),
        "classification_report": (
            report
        ),
    }


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
            "GPU was not detected. "
            "Run through a GPU SLURM job."
        )

    print(
        f"Device: {device}",
        flush=True,
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}",
        flush=True,
    )

    print(
        f"Manifest directory: "
        f"{MANIFEST_DIR}",
        flush=True,
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}",
        flush=True,
    )

    train_transform = (
        transforms.Compose([
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.85, 1.0),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(
                p=0.20
            ),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.08,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ])
    )

    evaluation_transform = (
        transforms.Compose([
            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ])
    )

    train_dataset = WheatDataset(
        TRAIN_CSV,
        train_transform,
    )

    validation_dataset = WheatDataset(
        VALIDATION_CSV,
        evaluation_transform,
    )

    test_dataset = WheatDataset(
        TEST_CSV,
        evaluation_transform,
    )

    test_07_dataset = WheatDataset(
        TEST_07_CSV,
        evaluation_transform,
    )

    print(
        f"Training samples:   "
        f"{len(train_dataset)}",
        flush=True,
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}",
        flush=True,
    )

    print(
        f"Test samples:       "
        f"{len(test_dataset)}",
        flush=True,
    )

    print(
        f"Test_07 samples:    "
        f"{len(test_07_dataset)}",
        flush=True,
    )

    loader_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": (
            NUM_WORKERS > 0
        ),
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

    test_07_loader = DataLoader(
        test_07_dataset,
        shuffle=False,
        **loader_options,
    )

    model = create_model(
        num_classes=len(CLASS_NAMES)
    ).to(device)

    class_weights = (
        calculate_class_weights(
            train_dataset.targets,
            len(CLASS_NAMES),
            device,
        )
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    best_macro_f1 = -1.0
    best_epoch = -1
    history: list[dict] = []

    checkpoint_path = (
        OUTPUT_DIR
        / "best_resnet18_wheat_grouped_v3_sqrt_weights.pt"
    )

    start_time = time.time()

    freeze_backbone(model)

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: (
                parameter.requires_grad
            ),
            model.parameters(),
        ),
        lr=WARMUP_LR,
        weight_decay=WEIGHT_DECAY,
    )

    total_epochs = (
        WARMUP_EPOCHS
        + FINETUNE_EPOCHS
    )

    for epoch_index in range(
        total_epochs
    ):
        epoch_number = (
            epoch_index + 1
        )

        if (
            epoch_index
            == WARMUP_EPOCHS
        ):
            print(
                "\nUnfreezing complete "
                "ResNet18 model.",
                flush=True,
            )

            unfreeze_model(model)

            optimizer = (
                torch.optim.AdamW(
                    model.parameters(),
                    lr=FINETUNE_LR,
                    weight_decay=(
                        WEIGHT_DECAY
                    ),
                )
            )

        stage = (
            "warmup"
            if (
                epoch_index
                < WARMUP_EPOCHS
            )
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

        (
            validation_metrics,
            _,
            _,
        ) = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        epoch_result = {
            "epoch": epoch_number,
            "stage": stage,
            "train_loss": round(
                train_loss,
                6,
            ),
            "validation_loss": round(
                validation_metrics[
                    "loss"
                ],
                6,
            ),
            "validation_accuracy": round(
                validation_metrics[
                    "accuracy"
                ] * 100,
                4,
            ),
            "validation_balanced_accuracy": round(
                validation_metrics[
                    "balanced_accuracy"
                ] * 100,
                4,
            ),
            "validation_macro_f1": round(
                validation_metrics[
                    "macro_f1"
                ] * 100,
                4,
            ),
            "validation_weighted_f1": round(
                validation_metrics[
                    "weighted_f1"
                ] * 100,
                4,
            ),
            "epoch_seconds": round(
                time.time()
                - epoch_start,
                2,
            ),
        }

        history.append(epoch_result)

        print(
            f"Epoch "
            f"{epoch_number:02d}/"
            f"{total_epochs} "
            f"[{stage}] | "
            f"train loss="
            f"{train_loss:.4f} | "
            f"val loss="
            f"{validation_metrics['loss']:.4f} | "
            f"val acc="
            f"{validation_metrics['accuracy'] * 100:.2f}% | "
            f"val balanced acc="
            f"{validation_metrics['balanced_accuracy'] * 100:.2f}% | "
            f"val Macro-F1="
            f"{validation_metrics['macro_f1'] * 100:.2f}%",
            flush=True,
        )

        if (
            validation_metrics[
                "macro_f1"
            ]
            > best_macro_f1
        ):
            best_macro_f1 = (
                validation_metrics[
                    "macro_f1"
                ]
            )

            best_epoch = (
                epoch_number
            )

            torch.save(
                {
                    "epoch": (
                        epoch_number
                    ),
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "class_names": (
                        CLASS_NAMES
                    ),
                    "validation_metrics": (
                        validation_metrics
                    ),
                    "manifest_dir": str(
                        MANIFEST_DIR
                    ),
                    "seed": SEED,
                },
                checkpoint_path,
            )

            print(
                "Saved new best "
                f"checkpoint: "
                f"{checkpoint_path}",
                flush=True,
            )

        with (
            OUTPUT_DIR
            / "history.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                history,
                handle,
                indent=2,
            )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    split_results = {}

    evaluation_sets = [
        (
            "validation",
            validation_dataset,
            validation_loader,
        ),
        (
            "test",
            test_dataset,
            test_loader,
        ),
        (
            "test_07",
            test_07_dataset,
            test_07_loader,
        ),
    ]

    for (
        split_name,
        dataset,
        loader,
    ) in evaluation_sets:
        print(
            f"\nEvaluating "
            f"{split_name}...",
            flush=True,
        )

        (
            metrics,
            targets,
            predictions,
        ) = evaluate(
            model,
            loader,
            criterion,
            device,
        )

        split_results[
            split_name
        ] = save_split_results(
            split_name=split_name,
            dataset=dataset,
            metrics=metrics,
            targets=targets,
            predictions=predictions,
        )

        result = split_results[
            split_name
        ]

        print(
            f"{split_name} | "
            f"Accuracy="
            f"{result['accuracy']:.4f}% | "
            f"Balanced Accuracy="
            f"{result['balanced_accuracy']:.4f}% | "
            f"Macro-F1="
            f"{result['macro_f1']:.4f}% | "
            f"Weighted-F1="
            f"{result['weighted_f1']:.4f}%",
            flush=True,
        )

    total_seconds = (
        time.time() - start_time
    )

    final_results = {
        "experiment": (
            "wheat_resnet18_grouped_v3_sqrt_weights"
        ),
        "manifest_dir": str(
            MANIFEST_DIR
        ),
        "seed": SEED,
        "classes": CLASS_NAMES,
        "split_sizes": {
            "train": len(
                train_dataset
            ),
            "validation": len(
                validation_dataset
            ),
            "test": len(
                test_dataset
            ),
            "test_07": len(
                test_07_dataset
            ),
        },
        "capture_groups": {
            "train": int(
                train_dataset.dataframe[
                    "capture_group"
                ].nunique()
            ),
            "validation": int(
                validation_dataset.dataframe[
                    "capture_group"
                ].nunique()
            ),
            "test": int(
                test_dataset.dataframe[
                    "capture_group"
                ].nunique()
            ),
            "test_07": int(
                test_07_dataset.dataframe[
                    "capture_group"
                ].nunique()
            ),
        },
        "training": {
            "warmup_epochs": (
                WARMUP_EPOCHS
            ),
            "finetune_epochs": (
                FINETUNE_EPOCHS
            ),
            "best_epoch": (
                best_epoch
            ),
            "best_validation_macro_f1": round(
                best_macro_f1 * 100,
                4,
            ),
            "total_seconds": round(
                total_seconds,
                2,
            ),
            "batch_size": (
                BATCH_SIZE
            ),
            "warmup_lr": (
                WARMUP_LR
            ),
            "finetune_lr": (
                FINETUNE_LR
            ),
            "weight_decay": (
                WEIGHT_DECAY
            ),
        },
        "results": split_results,
    }

    with (
        OUTPUT_DIR
        / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            final_results,
            handle,
            indent=2,
        )

    summary_rows = []

    for (
        split_name,
        result,
    ) in split_results.items():
        summary_rows.append({
            "experiment": (
                "wheat_resnet18_grouped_v3_sqrt_weights"
            ),
            "split": split_name,
            "num_images": (
                final_results[
                    "split_sizes"
                ][split_name]
            ),
            "accuracy": (
                result["accuracy"]
            ),
            "balanced_accuracy": (
                result[
                    "balanced_accuracy"
                ]
            ),
            "macro_f1": (
                result["macro_f1"]
            ),
            "weighted_f1": (
                result["weighted_f1"]
            ),
        })

    pd.DataFrame(
        summary_rows
    ).to_csv(
        OUTPUT_DIR
        / "evaluation_summary.csv",
        index=False,
    )

    print(
        "\nTraining completed.",
        flush=True,
    )

    print(
        f"Best epoch: "
        f"{best_epoch}",
        flush=True,
    )

    print(
        f"Best validation "
        f"Macro-F1: "
        f"{best_macro_f1 * 100:.4f}%",
        flush=True,
    )

    print(
        f"Total time: "
        f"{total_seconds:.2f} seconds",
        flush=True,
    )

    print(
        f"Results saved to: "
        f"{OUTPUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
