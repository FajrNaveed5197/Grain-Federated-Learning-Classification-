from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from federated_pipeline.common.model import create_resnet18
from federated_pipeline.data.dataset import ManifestImageDataset


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a rice ResNet18 checkpoint."
    )

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)

    return parser.parse_args()


def load_checkpoint(
    checkpoint_path: Path,
) -> tuple[dict[str, torch.Tensor], dict]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    metadata = {}

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]

        metadata = {
            key: value
            for key, value in checkpoint.items()
            if key != "model_state_dict"
        }
    else:
        state_dict = checkpoint

    return state_dict, metadata


def evaluate_split(
    model,
    loader,
    dataframe,
    device,
    split_name,
    output_dir,
):
    model.eval()

    targets = []
    predictions = []
    probabilities = []

    started_at = time.time()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            targets.extend(labels.numpy().tolist())
            predictions.extend(preds.cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())

    elapsed_seconds = time.time() - started_at

    targets_array = np.asarray(targets)
    predictions_array = np.asarray(predictions)
    probability_array = np.asarray(probabilities)

    accuracy = accuracy_score(
        targets_array,
        predictions_array,
    )

    balanced_accuracy = balanced_accuracy_score(
        targets_array,
        predictions_array,
    )

    macro_f1 = f1_score(
        targets_array,
        predictions_array,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        targets_array,
        predictions_array,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        targets_array,
        predictions_array,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        targets_array,
        predictions_array,
        labels=list(range(len(CLASS_NAMES))),
    )

    per_class_rows = []

    for class_name in CLASS_NAMES:
        values = report[class_name]

        per_class_rows.append(
            {
                "class_name": class_name,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1_score": values["f1-score"],
                "support": int(values["support"]),
            }
        )

    per_class_df = pd.DataFrame(per_class_rows)

    per_class_df.to_csv(
        output_dir / f"{split_name}_per_class_metrics.csv",
        index=False,
    )

    predictions_df = dataframe.copy()

    predictions_df["true_class_id"] = targets_array
    predictions_df["true_class_name"] = [
        CLASS_NAMES[index]
        for index in targets_array
    ]

    predictions_df["predicted_class_id"] = predictions_array
    predictions_df["predicted_class_name"] = [
        CLASS_NAMES[index]
        for index in predictions_array
    ]

    predictions_df["correct"] = (
        targets_array == predictions_array
    )

    for class_id, class_name in enumerate(CLASS_NAMES):
        predictions_df[
            f"probability_{class_name}"
        ] = probability_array[:, class_id]

    predictions_df.to_csv(
        output_dir / f"{split_name}_predictions.csv",
        index=False,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    matrix_df.to_csv(
        output_dir / f"{split_name}_confusion_matrix.csv"
    )

    figure, axis = plt.subplots(figsize=(10, 8))

    image = axis.imshow(matrix, interpolation="nearest")
    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title=f"{split_name.title()} confusion matrix",
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = matrix.max() / 2.0

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
        output_dir / f"{split_name}_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return {
        "num_images": int(len(targets_array)),
        "accuracy": round(float(accuracy * 100), 4),
        "balanced_accuracy": round(
            float(balanced_accuracy * 100),
            4,
        ),
        "macro_f1": round(float(macro_f1 * 100), 4),
        "weighted_f1": round(
            float(weighted_f1 * 100),
            4,
        ),
        "evaluation_seconds": round(
            elapsed_seconds,
            2,
        ),
        "per_class": per_class_rows,
    }


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}", flush=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    model = create_resnet18(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
    )

    state_dict, checkpoint_metadata = load_checkpoint(
        args.checkpoint
    )

    model.load_state_dict(state_dict)
    model = model.to(device)

    manifests = {
        "validation": args.validation_manifest,
        "test": args.test_manifest,
    }

    results = {
        "experiment": args.experiment_name,
        "checkpoint": str(args.checkpoint),
        "class_names": CLASS_NAMES,
        "checkpoint_metadata": checkpoint_metadata,
        "results": {},
    }

    for split_name, manifest_path in manifests.items():
        dataset = ManifestImageDataset(
            manifest_path=manifest_path,
            class_to_id=CLASS_TO_ID,
            transform=transform,
            dataset_root=args.dataset_root,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        print(
            f"Evaluating {split_name}: "
            f"{len(dataset)} images",
            flush=True,
        )

        metrics = evaluate_split(
            model=model,
            loader=loader,
            dataframe=dataset.dataframe.copy(),
            device=device,
            split_name=split_name,
            output_dir=args.output_dir,
        )

        results["results"][split_name] = metrics

        print(
            f"{split_name} | "
            f"Accuracy={metrics['accuracy']:.4f}% | "
            f"Balanced Accuracy="
            f"{metrics['balanced_accuracy']:.4f}% | "
            f"Macro-F1={metrics['macro_f1']:.4f}% | "
            f"Weighted-F1={metrics['weighted_f1']:.4f}%",
            flush=True,
        )

    with (
        args.output_dir / "evaluation_metrics.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            results,
            handle,
            indent=2,
            default=str,
        )

    summary_rows = []

    for split_name, metrics in results["results"].items():
        summary_rows.append(
            {
                "experiment": args.experiment_name,
                "split": split_name,
                "num_images": metrics["num_images"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": (
                    metrics["balanced_accuracy"]
                ),
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "evaluation_seconds": (
                    metrics["evaluation_seconds"]
                ),
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "evaluation_summary.csv",
        index=False,
    )

    print(
        f"Saved results to: {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
