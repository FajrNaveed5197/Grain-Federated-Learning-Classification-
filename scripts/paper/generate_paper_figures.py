#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Run this script from the repository root.
REPO_ROOT = Path.cwd()
RESULTS = REPO_ROOT / "experiments" / "results"
OUTPUT = REPO_ROOT / "paper" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def extract_metrics(path: Path, split: str = "test"):
    """
    Extract metrics from centralized, federated, and DDP JSON layouts.
    The requested split is searched first.
    """

    data = read_json(path)

    required = {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    }

    def find_metric_dict(value):
        if isinstance(value, dict):
            if required.issubset(value.keys()):
                return value

            for child in value.values():
                found = find_metric_dict(child)
                if found is not None:
                    return found

        elif isinstance(value, list):
            for child in value:
                found = find_metric_dict(child)
                if found is not None:
                    return found

        return None

    def find_requested_split(value):
        if isinstance(value, dict):
            if split in value:
                found = find_metric_dict(value[split])
                if found is not None:
                    return found

            for child in value.values():
                found = find_requested_split(child)
                if found is not None:
                    return found

        elif isinstance(value, list):
            for child in value:
                found = find_requested_split(child)
                if found is not None:
                    return found

        return None

    metrics = find_requested_split(data)

    if metrics is None:
        metrics = find_metric_dict(data)

    if metrics is None:
        top_keys = list(data.keys()) if isinstance(data, dict) else []
        raise KeyError(
            f"Could not find {split} metrics in {path}. "
            f"Top-level keys: {top_keys}"
        )

    return {
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(
            metrics.get("weighted_f1", np.nan)
        ),
    }


def save_figure(fig, filename: str):
    png_path = OUTPUT / f"{filename}.png"
    pdf_path = OUTPUT / f"{filename}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Created: {png_path}")
    print(f"Created: {pdf_path}")


# ============================================================
# FIGURE 1: RICE TEST MACRO-F1 COMPARISON
# ============================================================

def generate_rice_comparison():
    experiments = [
        (
            "Centralized ResNet18",
            RESULTS
            / "rice/rice_resnet18_grouped_v1/metrics_final.json",
        ),
        (
            "Centralized MobileNetV2",
            RESULTS
            / "rice/rice_mobilenetv2_grouped_v1/metrics.json",
        ),
        (
            "Centralized EfficientNetB0",
            RESULTS
            / "rice/rice_efficientnetb0_grouped_v1/metrics.json",
        ),
        (
            "FedAvg IID ResNet18",
            RESULTS
            / "rice/rice_fedavg_iid_resnet18/evaluation/evaluation_metrics.json",
        ),
        (
            "FedAvg non-IID ResNet18",
            RESULTS
            / "rice/rice_fedavg_noniid_resnet18/evaluation/evaluation_metrics.json",
        ),
        (
            "FedAvg IID MobileNetV2",
            RESULTS
            / "rice/rice_fedavg_iid_mobilenetv2/evaluation/evaluation_metrics.json",
        ),
        (
            "FedAvg non-IID MobileNetV2",
            RESULTS
            / "rice/rice_fedavg_noniid_mobilenetv2/evaluation/evaluation_metrics.json",
        ),
        (
            "DDP ResNet18",
            RESULTS
            / "rice/rice_ddp_resnet18/evaluation/evaluation_metrics.json",
        ),
    ]

    labels = []
    macro_f1_values = []

    for label, path in experiments:
        metrics = extract_metrics(path, split="test")
        labels.append(label)
        macro_f1_values.append(metrics["macro_f1"])

    # Sort from lowest to highest for easier comparison.
    order = np.argsort(macro_f1_values)

    sorted_labels = [labels[index] for index in order]
    sorted_values = [macro_f1_values[index] for index in order]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    y_positions = np.arange(len(sorted_labels))

    ax.scatter(
        sorted_values,
        y_positions,
        s=55,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_labels)

    ax.set_xlabel("Test Macro-F1 (%)")
    ax.set_title("Rice Test Macro-F1 Across Training Approaches")

    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.6,
        alpha=0.6,
    )

    minimum = min(sorted_values)
    maximum = max(sorted_values)
    margin = max((maximum - minimum) * 0.4, 0.08)

    ax.set_xlim(
        minimum - margin,
        maximum + margin,
    )

    for y_position, value in zip(y_positions, sorted_values):
        ax.annotate(
            f"{value:.4f}",
            (value, y_position),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()

    save_figure(
        fig,
        "rice_test_macro_f1_comparison",
    )


# ============================================================
# FIGURE 2: WHEAT WEIGHTING COMPARISON
# ============================================================

def generate_wheat_weighting_comparison():
    v2_path = (
        RESULTS
        / "wheat/wheat_resnet18_grouped_v2/evaluation_summary.csv"
    )

    v3_path = (
        RESULTS
        / "wheat/wheat_resnet18_grouped_v3_sqrt_weights/"
        "evaluation_summary.csv"
    )

    v2_rows = {
        row["split"]: row
        for row in read_csv(v2_path)
    }

    v3_rows = {
        row["split"]: row
        for row in read_csv(v3_path)
    }

    splits = [
        "validation",
        "test",
        "test_07",
    ]

    split_labels = [
        "Validation",
        "Test",
        "Test 07",
    ]

    full_inverse = [
        float(v2_rows[split]["macro_f1"])
        for split in splits
    ]

    sqrt_inverse = [
        float(v3_rows[split]["macro_f1"])
        for split in splits
    ]

    x_positions = np.arange(len(splits))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    bars_v2 = ax.bar(
        x_positions - width / 2,
        full_inverse,
        width,
        label="Full inverse",
    )

    bars_v3 = ax.bar(
        x_positions + width / 2,
        sqrt_inverse,
        width,
        label="Square-root inverse",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(split_labels)

    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title(
        "Wheat Weighting Comparison Under the Corrected Grouped Split"
    )

    ax.legend()

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.6,
    )

    all_values = full_inverse + sqrt_inverse
    minimum = min(all_values)
    maximum = max(all_values)
    margin = max((maximum - minimum) * 0.18, 1.0)

    ax.set_ylim(
        minimum - margin,
        maximum + margin,
    )

    for bars in (bars_v2, bars_v3):
        for bar in bars:
            value = bar.get_height()

            ax.annotate(
                f"{value:.2f}",
                (
                    bar.get_x() + bar.get_width() / 2,
                    value,
                ),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()

    save_figure(
        fig,
        "wheat_weighting_macro_f1_comparison",
    )


# ============================================================
# FIGURE 3: NORMALIZED WHEAT CONFUSION MATRIX
# ============================================================

def read_confusion_matrix(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))

    class_names = rows[0][1:]

    matrix = np.array(
        [
            [int(value) for value in row[1:]]
            for row in rows[1:]
        ],
        dtype=float,
    )

    row_names = [
        row[0]
        for row in rows[1:]
    ]

    if class_names != row_names:
        raise ValueError(
            "Confusion-matrix row labels and column labels do not match."
        )

    return class_names, matrix


def generate_wheat_confusion_matrix():
    matrix_path = (
        RESULTS
        / "wheat/wheat_resnet18_grouped_v3_sqrt_weights/"
        "test_07_confusion_matrix.csv"
    )

    class_names, matrix = read_confusion_matrix(matrix_path)

    row_totals = matrix.sum(
        axis=1,
        keepdims=True,
    )

    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix),
        where=row_totals != 0,
    ) * 100

    fig, ax = plt.subplots(figsize=(7.5, 6.3))

    image = ax.imshow(
        normalized,
        aspect="auto",
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Predictions within each true class (%)"
    )

    ax.set_xticks(
        np.arange(len(class_names))
    )

    ax.set_yticks(
        np.arange(len(class_names))
    )

    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        class_names
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    ax.set_title(
        "Wheat V3 Test 07 Row-Normalized Confusion Matrix"
    )

    for row_index in range(normalized.shape[0]):
        for column_index in range(normalized.shape[1]):
            value = normalized[row_index, column_index]

            # Show important cells and all diagonal cells.
            if value >= 1.0 or row_index == column_index:
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    fig.tight_layout()

    save_figure(
        fig,
        "wheat_v3_test07_confusion_matrix_normalized",
    )


# ============================================================
# FIGURE 4: WHEAT V3 TRAINING HISTORY
# ============================================================

def generate_wheat_training_history():
    history_path = (
        RESULTS
        / "wheat/wheat_resnet18_grouped_v3_sqrt_weights/"
        "history.json"
    )

    history = read_json(history_path)

    epochs = [
        int(row["epoch"])
        for row in history
    ]

    macro_f1 = [
        float(row["validation_macro_f1"])
        for row in history
    ]

    stages = [
        row["stage"]
        for row in history
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.3))

    ax.plot(
        epochs,
        macro_f1,
        marker="o",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Macro-F1 (%)")

    ax.set_title(
        "Wheat V3 Validation Macro-F1 During Training"
    )

    ax.set_xticks(epochs)

    ax.grid(
        linestyle="--",
        linewidth=0.6,
        alpha=0.6,
    )

    full_finetuning_epochs = [
        epoch
        for epoch, stage in zip(epochs, stages)
        if stage == "full_finetuning"
    ]

    if full_finetuning_epochs:
        first_full_epoch = min(full_finetuning_epochs)

        ax.axvline(
            first_full_epoch - 0.5,
            linestyle="--",
            linewidth=1,
        )

        ax.text(
            first_full_epoch - 0.35,
            min(macro_f1),
            "Full fine-tuning",
            rotation=90,
            va="bottom",
            fontsize=8,
        )

    best_index = int(np.argmax(macro_f1))

    ax.annotate(
        (
            f"Best epoch: {epochs[best_index]}\n"
            f"Macro-F1: {macro_f1[best_index]:.4f}%"
        ),
        (
            epochs[best_index],
            macro_f1[best_index],
        ),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "wheat_v3_validation_macro_f1_history",
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("Reading result files from:")
    print(RESULTS)

    generate_rice_comparison()
    generate_wheat_weighting_comparison()
    generate_wheat_confusion_matrix()
    generate_wheat_training_history()

    print()
    print("All figures were generated from repository artifacts.")
    print(f"Output directory: {OUTPUT}")


if __name__ == "__main__":
    main()