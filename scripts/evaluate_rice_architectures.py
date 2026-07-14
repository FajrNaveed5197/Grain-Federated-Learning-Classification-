from __future__ import annotations

import json
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


DATA_ROOT = Path(
    "/scratch/project_2019765/fnaveed/datasets/rice_grouped"
)

MANIFEST_DIR = DATA_ROOT / "grouped_split"

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
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

IMAGE_SIZE = 224
BATCH_SIZE = 128
NUM_WORKERS = 6

EXPERIMENTS = {
    "ResNet18": {
        "architecture": "resnet18",
        "result_dir": Path(
            "/scratch/project_2019765/fnaveed/results/"
            "rice_resnet18_grouped_v1"
        ),
        "checkpoint": "best_resnet18_rice_grouped.pt",
    },
    "MobileNetV2": {
        "architecture": "mobilenetv2",
        "result_dir": Path(
            "/scratch/project_2019765/fnaveed/results/"
            "rice_mobilenetv2_grouped_v1"
        ),
        "checkpoint": "best_mobilenetv2_rice_grouped.pt",
    },
    "EfficientNetB0": {
        "architecture": "efficientnetb0",
        "result_dir": Path(
            "/scratch/project_2019765/fnaveed/results/"
            "rice_efficientnetb0_grouped_v1"
        ),
        "checkpoint": "best_efficientnetb0_rice_grouped.pt",
    },
}


class RiceDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        transform,
    ) -> None:
        self.dataframe = pd.read_csv(manifest_path)
        self.transform = transform

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

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        return (
            self.transform(image),
            self.targets[index],
        )


def create_model(
    architecture: str,
) -> nn.Module:
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(
            model.fc.in_features,
            len(CLASS_NAMES),
        )
        return model

    if architecture == "mobilenetv2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            len(CLASS_NAMES),
        )
        return model

    if architecture == "efficientnetb0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            len(CLASS_NAMES),
        )
        return model

    raise ValueError(
        f"Unknown architecture: {architecture}"
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict, list[int], list[int]]:
    model.eval()

    targets = []
    predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            targets.extend(labels.tolist())
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
        "accuracy": accuracy * 100,
        "balanced_accuracy": (
            balanced_accuracy_score(
                targets,
                predictions,
            ) * 100
        ),
        "macro_f1": (
            f1_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            ) * 100
        ),
        "weighted_f1": (
            f1_score(
                targets,
                predictions,
                average="weighted",
                zero_division=0,
            ) * 100
        ),
    }

    return metrics, targets, predictions


def save_outputs(
    split_name: str,
    dataset: RiceDataset,
    metrics: dict,
    targets: list[int],
    predictions: list[int],
    output_dir: Path,
    model_name: str,
) -> None:
    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    rows = []

    for class_name in CLASS_NAMES:
        item = report[class_name]

        rows.append({
            "class_name": class_name,
            "precision": item["precision"],
            "recall": item["recall"],
            "f1_score": item["f1-score"],
            "support": int(item["support"]),
        })

    pd.DataFrame(rows).to_csv(
        output_dir
        / f"{split_name}_per_class_metrics.csv",
        index=False,
    )

    predictions_df = dataset.dataframe.copy()

    predictions_df["true_id"] = targets
    predictions_df["predicted_id"] = predictions

    predictions_df["true_class"] = [
        CLASS_NAMES[index]
        for index in targets
    ]

    predictions_df["predicted_class"] = [
        CLASS_NAMES[index]
        for index in predictions
    ]

    predictions_df["correct"] = (
        np.asarray(targets)
        == np.asarray(predictions)
    )

    predictions_df.to_csv(
        output_dir
        / f"{split_name}_predictions.csv",
        index=False,
    )

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
    )

    pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(
        output_dir
        / f"{split_name}_confusion_matrix.csv"
    )

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

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

    axis.set_title(
        f"Rice {model_name} "
        f"{split_name} Confusion Matrix"
    )

    threshold = matrix.max() / 2

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
        output_dir
        / f"{split_name}_confusion_matrix.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    with (
        output_dir
        / f"{split_name}_evaluation.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": model_name,
                "split": split_name,
                "metrics": {
                    key: round(value, 4)
                    for key, value in metrics.items()
                },
                "classification_report": report,
            },
            handle,
            indent=2,
        )


def main() -> None:
    device = torch.device("cuda")

    evaluation_transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    datasets = {
        "validation": RiceDataset(
            MANIFEST_DIR / "validation.csv",
            evaluation_transform,
        ),
        "test": RiceDataset(
            MANIFEST_DIR / "test.csv",
            evaluation_transform,
        ),
    }

    loaders = {
        name: DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        )
        for name, dataset in datasets.items()
    }

    comparison_rows = []

    for model_name, config in EXPERIMENTS.items():
        print(
            f"\nEvaluating {model_name}",
            flush=True,
        )

        output_dir = config["result_dir"]
        checkpoint_path = (
            output_dir / config["checkpoint"]
        )

        model = create_model(
            config["architecture"]
        ).to(device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            state_dict = checkpoint[
                "model_state_dict"
            ]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)

        for split_name in [
            "validation",
            "test",
        ]:
            metrics, targets, predictions = evaluate(
                model,
                loaders[split_name],
                device,
            )

            save_outputs(
                split_name=split_name,
                dataset=datasets[split_name],
                metrics=metrics,
                targets=targets,
                predictions=predictions,
                output_dir=output_dir,
                model_name=model_name,
            )

            comparison_rows.append({
                "model": model_name,
                "split": split_name,
                "num_images": len(
                    datasets[split_name]
                ),
                **{
                    key: round(value, 4)
                    for key, value in metrics.items()
                },
            })

            print(
                f"{split_name}: "
                f"accuracy={metrics['accuracy']:.4f}% | "
                f"balanced_accuracy="
                f"{metrics['balanced_accuracy']:.4f}% | "
                f"macro_f1={metrics['macro_f1']:.4f}% | "
                f"weighted_f1="
                f"{metrics['weighted_f1']:.4f}%",
                flush=True,
            )

    archive_dir = Path(
        "/scratch/project_2019765/fnaveed/results/"
        "final_report_archive/tables"
    )

    archive_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        comparison_rows
    ).to_csv(
        archive_dir
        / "rice_architecture_evaluation.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
