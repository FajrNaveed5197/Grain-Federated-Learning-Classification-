#!/usr/bin/env python3
"""
Generate verified static charts for the grain-classification Streamlit dashboard.

Run from the repository root:

    source .venv/bin/activate
    python generate_dashboard_static_charts.py --root .

Output:

    streamlit-ui/assets/generated_charts/
        *.png
        *.pdf
        static_chart_index.csv

The script reads authoritative result artifacts directly. It does not use the
manually entered Streamlit result templates.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# AUTHORITATIVE PROJECT SOURCES
# =============================================================================

RICE_SPLIT_SOURCE = Path(
    "experiments/results/provenance/rice_grouped_split/split_summary.json"
)

WHEAT_CLASS_SOURCE = Path(
    "experiments/results/provenance/wheat_grouped_split_v2/class_allocation.csv"
)

CENTRALIZED_ARCHITECTURE_SOURCE = Path(
    "experiments/results/tables/rice_architecture_evaluation.csv"
)

FINAL_COMPARISON_SOURCE = Path(
    "results/Rice/Federated/Summaries/Presentation_tables/"
    "table_1_alpha0p5_comprehensive.csv"
)

FEDAVG_HISTORY_SOURCES = {
    "MobileNetV2 · IID": Path(
        "experiments/results/rice/rice_fedavg_iid_mobilenetv2/metrics.json"
    ),
    "MobileNetV2 · Non-IID α=0.5": Path(
        "experiments/results/rice/rice_fedavg_noniid_mobilenetv2/metrics.json"
    ),
    "ResNet18 · IID": Path(
        "experiments/results/rice/rice_fedavg_iid_resnet18/metrics.json"
    ),
    "ResNet18 · Non-IID α=0.5": Path(
        "experiments/results/rice/rice_fedavg_noniid_resnet18/metrics.json"
    ),
}

# Use MobileNetV2 histories for the compact IID/non-IID client-count chart.
IID_CLIENT_SOURCE = FEDAVG_HISTORY_SOURCES["MobileNetV2 · IID"]
NONIID_CLIENT_SOURCE = FEDAVG_HISTORY_SOURCES[
    "MobileNetV2 · Non-IID α=0.5"
]


# =============================================================================
# OUTPUT INDEX
# =============================================================================

@dataclass
class ChartRecord:
    chart_id: str
    title: str
    png_path: str
    pdf_path: str
    source_files: str
    dashboard_page: str
    description: str


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate static dashboard charts from verified artifacts."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "<root>/streamlit-ui/assets/generated_charts"
        ),
    )
    return parser.parse_args()


def clean_token(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("non-iid", "non_iid")
    text = text.replace("α", "alpha")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def require_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required source does not exist: {path}")


def read_json(path: Path) -> Any:
    require_file(path)
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse JSON {path}: {exc}") from exc


def read_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    require_file(path)

    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not parse CSV {path}: {exc}") from exc

    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {', '.join(missing)}"
        )

    return frame.dropna(how="all").reset_index(drop=True)


def add_value_labels_vertical(
    axis: plt.Axes,
    bars: Any,
    decimals: int = 2,
    suffix: str = "",
) -> None:
    for bar in bars:
        height = bar.get_height()
        if pd.isna(height):
            continue
        axis.annotate(
            f"{height:.{decimals}f}{suffix}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def add_value_labels_horizontal(
    axis: plt.Axes,
    bars: Any,
    decimals: int = 2,
    suffix: str = "",
) -> None:
    for bar in bars:
        width = bar.get_width()
        if pd.isna(width):
            continue
        axis.annotate(
            f"{width:.{decimals}f}{suffix}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
        )


def finish_axis(
    axis: plt.Axes,
    *,
    title: str,
    x_label: str = "",
    y_label: str = "",
    grid_axis: str = "y",
) -> None:
    axis.set_title(title, pad=14, fontweight="bold")
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, alpha=0.18)
    axis.set_axisbelow(True)


def save_chart(
    figure: plt.Figure,
    *,
    output_dir: Path,
    chart_id: str,
    title: str,
    source_files: list[Path],
    dashboard_page: str,
    description: str,
) -> ChartRecord:
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{chart_id}.png"
    pdf_path = output_dir / f"{chart_id}.pdf"

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)

    print(f"[CREATED] {png_path}")
    print(f"[CREATED] {pdf_path}")

    return ChartRecord(
        chart_id=chart_id,
        title=title,
        png_path=str(png_path),
        pdf_path=str(pdf_path),
        source_files="; ".join(str(path) for path in source_files),
        dashboard_page=dashboard_page,
        description=description,
    )


# =============================================================================
# CHART 1: RICE SPLIT
# =============================================================================

def generate_rice_split(
    root: Path,
    output_dir: Path,
) -> ChartRecord:
    source = root / RICE_SPLIT_SOURCE
    data = read_json(source)

    try:
        counts = {
            "Train": int(data["splits"]["train"]["images"]),
            "Validation": int(data["splits"]["validation"]["images"]),
            "Test": int(data["splits"]["test"]["images"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Unexpected rice split structure in {source}"
        ) from exc

    total = sum(counts.values())
    labels = list(counts)
    values = [counts[label] for label in labels]
    percentages = [100 * value / total for value in values]

    figure, axis = plt.subplots(figsize=(8.5, 5.3))
    bars = axis.bar(labels, values)

    for bar, count, percentage in zip(bars, values, percentages):
        axis.annotate(
            f"{count:,}\n({percentage:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    finish_axis(
        axis,
        title="Rice capture-group-aware classification split",
        y_label="Number of images",
    )
    axis.set_ylim(0, max(values) * 1.18)

    figure.text(
        0.5,
        0.01,
        f"Total classification images: {total:,} · Seed: {data.get('seed', '—')}",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))

    return save_chart(
        figure,
        output_dir=output_dir,
        chart_id="dataset_rice_group_aware_split",
        title="Rice capture-group-aware classification split",
        source_files=[RICE_SPLIT_SOURCE],
        dashboard_page="Dataset Explorer",
        description=(
            "Training, validation and test image counts from the verified "
            "group-aware rice split."
        ),
    )


# =============================================================================
# CHART 2: WHEAT CATEGORY ALLOCATION
# =============================================================================

def generate_wheat_category_allocation(
    root: Path,
    output_dir: Path,
) -> ChartRecord:
    source = root / WHEAT_CLASS_SOURCE
    frame = read_csv(
        source,
        required_columns=[
            "label",
            "total_images",
            "validation_images",
        ],
    )

    frame["total_images"] = pd.to_numeric(
        frame["total_images"],
        errors="coerce",
    )
    frame = frame.dropna(subset=["label", "total_images"]).copy()
    frame = frame[frame["total_images"] > 0]
    frame = frame.sort_values("total_images", ascending=True)

    total = frame["total_images"].sum()
    frame["percentage"] = 100 * frame["total_images"] / total

    figure, axis = plt.subplots(
        figsize=(10, max(5.2, 0.62 * len(frame)))
    )
    bars = axis.barh(frame["label"], frame["percentage"])

    for bar, count, percentage in zip(
        bars,
        frame["total_images"],
        frame["percentage"],
    ):
        axis.annotate(
            f"{percentage:.2f}%  ({int(count):,})",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
        )

    finish_axis(
        axis,
        title="Wheat category allocation in the development pool",
        x_label="Share of images (%)",
        grid_axis="x",
    )
    axis.set_xlim(0, max(frame["percentage"]) * 1.18)

    figure.text(
        0.5,
        0.01,
        (
            f"Uses the source column 'total_images' · "
            f"Total represented: {int(total):,} images · "
            "Independent held-out test sets are not included unless present "
            "in this source file."
        ),
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))

    return save_chart(
        figure,
        output_dir=output_dir,
        chart_id="dataset_wheat_category_allocation",
        title="Wheat category allocation in the development pool",
        source_files=[WHEAT_CLASS_SOURCE],
        dashboard_page="Dataset Explorer",
        description=(
            "Wheat category percentages and counts using the verified "
            "class-allocation file."
        ),
    )


# =============================================================================
# CHART 3: CENTRALIZED ARCHITECTURE COMPARISON
# =============================================================================

def generate_centralized_architecture_comparison(
    root: Path,
    output_dir: Path,
) -> ChartRecord:
    source = root / CENTRALIZED_ARCHITECTURE_SOURCE
    frame = read_csv(
        source,
        required_columns=[
            "model",
            "split",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ],
    )

    test = frame[
        frame["split"].astype(str).str.lower().eq("test")
    ].copy()

    if test.empty:
        raise ValueError(f"No test rows found in {source}")

    metrics = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro-F1"),
    ]

    for column, _ in metrics:
        test[column] = pd.to_numeric(test[column], errors="coerce")

    test = test.dropna(subset=[column for column, _ in metrics])

    x = np.arange(len(test))
    width = 0.24

    figure, axis = plt.subplots(figsize=(10.5, 5.8))

    for index, (column, label) in enumerate(metrics):
        offset = (index - 1) * width
        bars = axis.bar(
            x + offset,
            test[column],
            width,
            label=label,
        )
        add_value_labels_vertical(axis, bars, decimals=2, suffix="%")

    axis.set_xticks(x)
    axis.set_xticklabels(test["model"])
    axis.legend(frameon=False, ncol=3, loc="upper center")
    axis.set_ylim(95, 100)

    finish_axis(
        axis,
        title="Centralized rice architecture comparison on the test set",
        y_label="Score (%)",
    )

    figure.text(
        0.5,
        0.01,
        "The y-axis is shown from 95% to 100% so small performance differences remain visible.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))

    return save_chart(
        figure,
        output_dir=output_dir,
        chart_id="results_centralized_architecture_test_metrics",
        title="Centralized rice architecture comparison on the test set",
        source_files=[CENTRALIZED_ARCHITECTURE_SOURCE],
        dashboard_page="Experiment Comparison",
        description=(
            "Test Accuracy, Balanced Accuracy and Macro-F1 for ResNet18, "
            "MobileNetV2 and EfficientNetB0."
        ),
    )


# =============================================================================
# CHARTS 4-8: FINAL COMPARISON TABLE
# =============================================================================

def load_final_comparison(root: Path) -> pd.DataFrame:
    source = root / FINAL_COMPARISON_SOURCE
    frame = read_csv(
        source,
        required_columns=[
            "Approach",
            "Data setting",
            "Backbone",
            "Accuracy",
            "Balanced Acc.",
            "Macro-F1",
            "Evaluation basis",
        ],
    )

    for column in ["Accuracy", "Balanced Acc.", "Macro-F1"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame.dropna(subset=["Approach", "Macro-F1"]).copy()


def horizontal_macro_f1_chart(
    *,
    frame: pd.DataFrame,
    output_dir: Path,
    chart_id: str,
    title: str,
    source: Path,
    dashboard_page: str,
    description: str,
) -> ChartRecord:
    plot_frame = frame.copy()
    plot_frame["Label"] = (
        plot_frame["Approach"].astype(str)
        + " · "
        + plot_frame["Data setting"].astype(str)
        + " · "
        + plot_frame["Backbone"].astype(str)
    )
    plot_frame = plot_frame.sort_values("Macro-F1", ascending=True)

    figure, axis = plt.subplots(
        figsize=(11, max(4.8, 0.58 * len(plot_frame)))
    )
    bars = axis.barh(plot_frame["Label"], plot_frame["Macro-F1"])
    add_value_labels_horizontal(
        axis,
        bars,
        decimals=2,
        suffix="%",
    )

    axis.set_xlim(95, 100)
    finish_axis(
        axis,
        title=title,
        x_label="Macro-F1 (%)",
        grid_axis="x",
    )

    figure.text(
        0.5,
        0.01,
        "The y-axis is shown from 95% to 100%. Evaluation basis is stated in the accompanying results table.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))

    return save_chart(
        figure,
        output_dir=output_dir,
        chart_id=chart_id,
        title=title,
        source_files=[source],
        dashboard_page=dashboard_page,
        description=description,
    )


def generate_final_comparison_charts(
    root: Path,
    output_dir: Path,
) -> list[ChartRecord]:
    source = FINAL_COMPARISON_SOURCE
    frame = load_final_comparison(root)

    records: list[ChartRecord] = []

    global_frame = frame[
        ~frame["Evaluation basis"]
        .astype(str)
        .str.lower()
        .str.contains("mean of 3 clients")
    ].copy()

    personalized_frame = frame[
        frame["Evaluation basis"]
        .astype(str)
        .str.lower()
        .str.contains("mean of 3 clients")
    ].copy()

    if not global_frame.empty:
        records.append(
            horizontal_macro_f1_chart(
                frame=global_frame,
                output_dir=output_dir,
                chart_id="results_global_models_macro_f1",
                title="Global-model rice performance",
                source=source,
                dashboard_page="Experiment Comparison",
                description=(
                    "Centralized, DDP and FedAvg Macro-F1 results. "
                    "Personalized client means are intentionally excluded."
                ),
            )
        )

    if not personalized_frame.empty:
        records.append(
            horizontal_macro_f1_chart(
                frame=personalized_frame,
                output_dir=output_dir,
                chart_id="results_personalized_methods_macro_f1",
                title="Personalized federated performance",
                source=source,
                dashboard_page="Experiment Comparison",
                description=(
                    "FedPer and FedRep Macro-F1 values reported as means of "
                    "three personalized clients."
                ),
            )
        )

    for method in ["FedAvg", "FedPer", "FedRep"]:
        method_frame = frame[
            frame["Approach"].astype(str).str.casefold().eq(method.casefold())
        ].copy()

        if method_frame.empty:
            continue

        records.append(
            horizontal_macro_f1_chart(
                frame=method_frame,
                output_dir=output_dir,
                chart_id=f"results_{clean_token(method)}_macro_f1",
                title=f"{method} rice comparison",
                source=source,
                dashboard_page="Experiment Comparison",
                description=(
                    f"{method} Macro-F1 comparison across the recorded "
                    "data settings and backbones."
                ),
            )
        )

    # Export the verified presentation table for direct Streamlit display.
    table_output = output_dir / "verified_final_comparison_table.csv"
    frame.to_csv(table_output, index=False)
    print(f"[CREATED] {table_output}")

    return records


# =============================================================================
# CHART 9: FEDAVG CONVERGENCE
# =============================================================================

def extract_history(path: Path) -> pd.DataFrame:
    data = read_json(path)
    history = data.get("history")

    if not isinstance(history, list) or not history:
        raise ValueError(f"No non-empty history list found in {path}")

    rows: list[dict[str, Any]] = []

    for entry in history:
        if not isinstance(entry, dict):
            continue

        validation = entry.get("validation", {})
        if not isinstance(validation, dict):
            validation = {}

        rows.append(
            {
                "round": entry.get("round"),
                "accuracy": validation.get("accuracy"),
                "balanced_accuracy": validation.get("balanced_accuracy"),
                "macro_f1": validation.get("macro_f1"),
            }
        )

    frame = pd.DataFrame(rows)

    for column in [
        "round",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame.dropna(subset=["round", "macro_f1"])


def generate_fedavg_convergence(
    root: Path,
    output_dir: Path,
) -> ChartRecord:
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    source_files: list[Path] = []
    all_values: list[float] = []

    for label, relative_source in FEDAVG_HISTORY_SOURCES.items():
        source = root / relative_source
        history = extract_history(source)
        source_files.append(relative_source)
        all_values.extend(history["macro_f1"].tolist())

        axis.plot(
            history["round"],
            history["macro_f1"],
            marker="o",
            linewidth=2,
            label=label,
        )

    axis.set_xticks([1, 2, 3, 4, 5])
    axis.legend(frameon=False, ncol=2, loc="best")

    if all_values:
        lower = math.floor((min(all_values) - 0.15) * 10) / 10
        upper = math.ceil((max(all_values) + 0.15) * 10) / 10
        axis.set_ylim(lower, upper)

    finish_axis(
        axis,
        title="FedAvg validation Macro-F1 over five rounds",
        x_label="Federated round",
        y_label="Validation Macro-F1 (%)",
    )

    figure.text(
        0.5,
        0.01,
        "Validation metrics are shown; final test results should be taken from the evaluation artifacts.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))

    return save_chart(
        figure,
        output_dir=output_dir,
        chart_id="clients_fedavg_validation_macro_f1_by_round",
        title="FedAvg validation Macro-F1 over five rounds",
        source_files=source_files,
        dashboard_page="Federated Client Analysis",
        description=(
            "Validation convergence for ResNet18 and MobileNetV2 under IID "
            "and non-IID alpha=0.5 partitions."
        ),
    )


# =============================================================================
# CHART 10: CLIENT SAMPLE DISTRIBUTION
# =============================================================================

def extract_client_counts(path: Path) -> dict[str, int]:
    data = read_json(path)
    history = data.get("history")

    if not isinstance(history, list) or not history:
        raise ValueError(f"No non-empty history list found in {path}")

    first_round = history[0]
    clients = first_round.get("clients")

    if not isinstance(clients, list):
        raise ValueError(f"No client list found in first round of {path}")

    result: dict[str, int] = {}

    for client in clients:
        if not isinstance(client, dict):
            continue

        client_name = str(client.get("client", "")).strip()
        samples = client.get("num_samples")

        if not client_name:
            continue

        try:
            result[client_name] = int(samples)
        except (TypeError, ValueError):
            continue

    if not result:
        raise ValueError(f"No client sample counts found in {path}")

    return result


def generate_client_sample_distribution(
    root: Path,
    output_dir: Path,
) -> ChartRecord:
    iid_source = root / IID_CLIENT_SOURCE
    noniid_source = root / NONIID_CLIENT_SOURCE

    iid = extract_client_counts(iid_source)
    noniid = extract_client_counts(noniid_source)

    clients = sorted(set(iid) | set(noniid))
    x = np.arange(len(clients))
    width = 0.36

    iid_values = [iid.get(client, 0) for client in clients]
    noniid_values = [noniid.get(client, 0) for client in clients]

    figure, axis = plt.subplots(figsize=(9.5, 5.5))

    iid_bars = axis.bar(
        x - width / 2,
        iid_values,
        width,
        label="IID",
    )
    noniid_bars = axis.bar(
        x + width / 2,
        noniid_values,
        width,
        label="Non-IID α=0.5",
    )

    add_value_labels_vertical(axis, iid_bars, decimals=0)
    add_value_labels_vertical(axis, noniid_bars, decimals=0)

    axis.set_xticks(x)
    axis.set_xticklabels(
        [client.replace("_", " ").title() for client in clients]
    )
    axis.legend(frameon=False)

    maximum = max(iid_values + noniid_values)
    axis.set_ylim(0, maximum * 1.16)

    finish_axis(
        axis,
        title="Images assigned to federated clients",
        y_label="Number of training images",
    )

    figure.text(
        0.5,
        0.01,
        (
            "This chart shows client sample counts, not per-class client "
            "composition. A class-distribution chart requires client × class counts."
        ),
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))

    return save_chart(
        figure,
        output_dir=output_dir,
        chart_id="clients_iid_vs_noniid_sample_counts",
        title="Images assigned to federated clients",
        source_files=[IID_CLIENT_SOURCE, NONIID_CLIENT_SOURCE],
        dashboard_page="Federated Client Analysis",
        description=(
            "Client training-image totals for IID and non-IID alpha=0.5 "
            "MobileNetV2 FedAvg partitions."
        ),
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()

    if not root.exists() or not root.is_dir():
        print(f"ERROR: Invalid repository root: {root}", file=sys.stderr)
        return 2

    output_dir = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "streamlit-ui" / "assets" / "generated_charts"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository root: {root}")
    print(f"Chart output:    {output_dir}")
    print()

    generators = [
        ("Rice split", lambda: [generate_rice_split(root, output_dir)]),
        (
            "Wheat category allocation",
            lambda: [generate_wheat_category_allocation(root, output_dir)],
        ),
        (
            "Centralized architecture comparison",
            lambda: [
                generate_centralized_architecture_comparison(
                    root,
                    output_dir,
                )
            ],
        ),
        (
            "Final comparison charts",
            lambda: generate_final_comparison_charts(root, output_dir),
        ),
        (
            "FedAvg convergence",
            lambda: [generate_fedavg_convergence(root, output_dir)],
        ),
        (
            "Client sample distribution",
            lambda: [
                generate_client_sample_distribution(root, output_dir)
            ],
        ),
    ]

    records: list[ChartRecord] = []
    failures: list[str] = []

    for name, generator in generators:
        try:
            records.extend(generator())
        except Exception as exc:
            message = f"{name}: {exc}"
            failures.append(message)
            print(f"[FAILED] {message}", file=sys.stderr)

    index_path = output_dir / "static_chart_index.csv"
    pd.DataFrame(asdict(record) for record in records).to_csv(
        index_path,
        index=False,
    )
    print(f"[CREATED] {index_path}")

    print()
    print(f"Charts generated successfully: {len(records)}")

    if failures:
        print(f"Chart groups that failed: {len(failures)}")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("All chart groups completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
