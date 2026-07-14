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
    "FedAvg IID MobileNetV2": Path(
        "/scratch/project_2019765/fnaveed/results/"
        "rice_fedavg_iid_mobilenetv2"
    ),
    "FedAvg non-IID MobileNetV2": Path(
        "/scratch/project_2019765/fnaveed/results/"
        "rice_fedavg_noniid_mobilenetv2"
    ),
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

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image: {image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        return (
            self.transform(image),
            self.targets[index],
        )


def create_model() -> nn.Module:
    model = models.mobilenet_v2(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(CLASS_NAMES),
    )

    return model


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
    experiment_name: str,
) -> None:
    evaluation_dir = (
        output_dir / "evaluation"
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    per_class_rows = []

    for class_name in CLASS_NAMES:
        item = report[class_name]

        per_class_rows.append({
            "class_name": class_name,
            "precision": item["precision"],
            "recall": item["recall"],
            "f1_score": item["f1-score"],
            "support": int(item["support"]),
        })

    pd.DataFrame(
        per_class_rows
    ).to_csv(
        evaluation_dir
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
        evaluation_dir
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
        evaluation_dir
        / f"{split_name}_confusion_matrix.csv"
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
        f"{experiment_name} "
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
        evaluation_dir
        / f"{split_name}_confusion_matrix.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    with (
        evaluation_dir
        / f"{split_name}_evaluation.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "experiment": experiment_name,
                "split": split_name,
                "metrics": {
                    key: round(value, 4)
                    for key, value
                    in metrics.items()
                },
                "classification_report": report,
            },
            handle,
            indent=2,
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is required."
        )

    device = torch.device("cuda")

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
        split_name: DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        )
        for split_name, dataset
        in datasets.items()
    }

    comparison_rows = []

    for (
        experiment_name,
        output_dir,
    ) in EXPERIMENTS.items():
        checkpoint_path = (
            output_dir
            / "best_global_model.pt"
        )

        print(
            f"\nEvaluating "
            f"{experiment_name}",
            flush=True,
        )

        print(
            f"Checkpoint: "
            f"{checkpoint_path}",
            flush=True,
        )

        model = create_model().to(device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        state_dict = (
            checkpoint["model_state_dict"]
            if (
                isinstance(checkpoint, dict)
                and "model_state_dict"
                in checkpoint
            )
            else checkpoint
        )

        model.load_state_dict(state_dict)

        experiment_results = {}

        for split_name in [
            "validation",
            "test",
        ]:
            (
                metrics,
                targets,
                predictions,
            ) = evaluate(
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
                experiment_name=experiment_name,
            )

            experiment_results[
                split_name
            ] = {
                key: round(value, 4)
                for key, value
                in metrics.items()
            }

            comparison_rows.append({
                "experiment": experiment_name,
                "split": split_name,
                "num_images": len(
                    datasets[split_name]
                ),
                **experiment_results[
                    split_name
                ],
            })

            print(
                f"{split_name} | "
                f"accuracy="
                f"{metrics['accuracy']:.4f}% | "
                f"balanced_accuracy="
                f"{metrics['balanced_accuracy']:.4f}% | "
                f"macro_f1="
                f"{metrics['macro_f1']:.4f}% | "
                f"weighted_f1="
                f"{metrics['weighted_f1']:.4f}%",
                flush=True,
            )

        with (
            output_dir
            / "evaluation"
            / "evaluation_metrics.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "experiment": experiment_name,
                    "checkpoint": str(
                        checkpoint_path
                    ),
                    "results": experiment_results,
                },
                handle,
                indent=2,
            )

        pd.DataFrame([
            {
                "experiment": experiment_name,
                "split": split_name,
                **results,
            }
            for split_name, results
            in experiment_results.items()
        ]).to_csv(
            output_dir
            / "evaluation"
            / "evaluation_summary.csv",
            index=False,
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
        / "rice_fedavg_mobilenetv2_comparison.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
