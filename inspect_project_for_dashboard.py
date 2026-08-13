#!/usr/bin/env python3
"""
Inspect a machine-learning repository and identify files and chart opportunities
for a Streamlit research dashboard.

The script is read-only with respect to the project. It writes its audit outputs
only to the selected output directory.

Typical use from the repository root:

    python inspect_project_for_dashboard.py --root .

Outputs:
    project_audit/
        audit_report.md
        files_inventory.csv
        top_level_summary.csv
        extension_summary.csv
        tabular_files_summary.csv
        image_files_summary.csv
        text_metric_files_summary.csv
        candidate_graphs.csv
        audit_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    Image = None

    class UnidentifiedImageError(Exception):
        pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".idea",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    ".venv-paper",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".cache",
}

TABULAR_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".parquet"}
STRUCTURED_SUFFIXES = {".json", ".jsonl", ".ndjson", ".yaml", ".yml"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".h5", ".keras", ".pkl", ".joblib"}
TEXT_SUFFIXES = {
    ".txt",
    ".log",
    ".out",
    ".err",
    ".md",
    ".rst",
    ".tex",
}
SCRIPT_SUFFIXES = {".py", ".sh", ".bash", ".slurm", ".sbatch", ".ipynb"}
DOCUMENT_SUFFIXES = {".pdf", ".docx", ".pptx"}

MAX_TABULAR_ROWS = 2500
MAX_JSON_BYTES = 25 * 1024 * 1024
MAX_TEXT_SCAN_BYTES = 2 * 1024 * 1024
MAX_IMAGE_METADATA_PER_DIRECTORY = 8

METRIC_TERMS = {
    "accuracy",
    "acc",
    "macro_f1",
    "macro-f1",
    "f1",
    "precision",
    "recall",
    "loss",
    "val_loss",
    "train_loss",
    "auc",
    "roc_auc",
    "support",
    "entropy",
    "latency",
    "bandwidth",
    "packet_loss",
    "runtime",
    "duration",
    "seconds",
    "communication",
    "bytes",
    "weight",
    "client_weight",
}

ROUND_TERMS = {"round", "round_id", "fl_round", "server_round", "epoch", "step"}
METHOD_TERMS = {"method", "algorithm", "approach", "strategy", "training_method"}
MODEL_TERMS = {"model", "architecture", "backbone", "network"}
DATASET_TERMS = {"dataset", "grain", "data_name"}
DISTRIBUTION_TERMS = {"distribution", "partition", "iid", "non_iid", "non-iid"}
ALPHA_TERMS = {"alpha", "dirichlet_alpha"}
CLIENT_TERMS = {"client", "client_id", "site", "node"}
CLASS_TERMS = {"class", "category", "label", "disease", "fault", "defect"}
COUNT_TERMS = {"count", "images", "image_count", "samples", "n", "support"}
SPLIT_TERMS = {"split", "subset", "partition_name"}
TRUE_LABEL_TERMS = {"y_true", "true_label", "actual", "target", "ground_truth"}
PRED_LABEL_TERMS = {"y_pred", "pred_label", "prediction", "predicted"}
SEED_TERMS = {"seed", "random_seed"}
TIME_TERMS = {"runtime", "duration", "seconds", "elapsed", "wall_time"}
EFFICIENCY_TERMS = {"params", "parameters", "flops", "model_size", "size_mb"}
NETWORK_TERMS = {"latency", "bandwidth", "packet_loss", "jitter", "throughput"}
WEIGHT_TERMS = {"weight", "client_weight", "aggregation_weight", "adaptive_weight"}

CONFUSION_NAME_MARKERS = (
    "confusion",
    "conf_matrix",
    "conf-matrix",
    "confmatrix",
    "_cm",
    "cm_",
)

VISUAL_NAME_MARKERS = {
    "confusion_matrix": CONFUSION_NAME_MARKERS,
    "training_curve": ("training_curve", "learning_curve", "loss_curve", "convergence"),
    "roc_curve": ("roc_curve", "_roc", "roc_"),
    "precision_recall_curve": ("precision_recall", "pr_curve", "_pr"),
    "class_distribution": ("class_distribution", "category_distribution"),
    "client_distribution": ("client_distribution", "client_partition"),
    "dataset_split": ("dataset_split", "train_val_test", "split_distribution"),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FileRecord:
    relative_path: str
    top_level: str
    filename: str
    extension: str
    category: str
    size_bytes: int
    size_mb: float
    modified_utc: str


@dataclass
class TabularRecord:
    relative_path: str
    file_type: str
    rows_sampled: int
    columns_count: int
    columns: str
    numeric_columns: str
    categorical_columns: str
    detected_concepts: str
    read_status: str
    error: str


@dataclass
class ImageRecord:
    relative_path: str
    extension: str
    width: int | None
    height: int | None
    mode: str
    inferred_visual_type: str
    metadata_status: str
    error: str


@dataclass
class TextMetricRecord:
    relative_path: str
    detected_metric_terms: str
    detected_experiment_terms: str
    scan_status: str
    error: str


@dataclass
class GraphRecommendation:
    priority: int
    graph_name: str
    chart_type: str
    source_file: str
    status: str
    detected_columns_or_asset: str
    reason: str
    suggested_output_filename: str


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory a research repository and recommend dashboard charts "
            "supported by the files that actually exist."
        )
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
        help="Audit output directory. Default: <root>/project_audit.",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden directories except explicitly excluded ones.",
    )
    parser.add_argument(
        "--extra-exclude",
        nargs="*",
        default=[],
        help="Additional directory names to exclude.",
    )
    return parser.parse_args()


def normalize_name(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("%", " percent ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def safe_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return path.as_posix()


def top_level_name(relative_path: str) -> str:
    parts = Path(relative_path).parts
    return parts[0] if len(parts) > 1 else "(repository root)"


def utc_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def classify_file(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()

    if suffix in TABULAR_SUFFIXES:
        return "tabular data"
    if suffix in STRUCTURED_SUFFIXES:
        return "structured data"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in MODEL_SUFFIXES:
        return "model/checkpoint"
    if suffix in TEXT_SUFFIXES:
        if any(term in name for term in ("result", "metric", "report", "eval")):
            return "result/report text"
        return "text/documentation"
    if suffix in SCRIPT_SUFFIXES:
        return "code/script"
    if suffix in DOCUMENT_SUFFIXES:
        return "document"
    if suffix in {".yaml", ".yml", ".toml", ".ini", ".cfg"}:
        return "configuration"
    return "other"


def should_skip_directory(
    directory_name: str,
    excluded: set[str],
    include_hidden: bool,
) -> bool:
    if directory_name in excluded:
        return True
    if not include_hidden and directory_name.startswith("."):
        return True
    return False


def markdown_table(
    rows: list[dict[str, Any]],
    columns: list[str],
    max_rows: int = 30,
) -> str:
    if not rows:
        return "_No rows found._"

    selected = rows[:max_rows]

    def clean(value: Any) -> str:
        text = "" if value is None else str(value)
        text = text.replace("|", r"\|").replace("\n", " ")
        return text

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(clean(row.get(column, "")) for column in columns) + " |"
        for row in selected
    ]

    if len(rows) > max_rows:
        body.append(
            "| "
            + " | ".join(
                [f"… {len(rows) - max_rows} more rows"] + [""] * (len(columns) - 1)
            )
            + " |"
        )

    return "\n".join([header, separator, *body])


def write_dataclass_csv(path: Path, records: Iterable[Any]) -> None:
    records = list(records)

    if not records:
        path.write_text("", encoding="utf-8")
        return

    rows = [asdict(record) for record in records]
    pd.DataFrame(rows).to_csv(path, index=False)


def flatten_json_key_paths(
    value: Any,
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 4,
) -> set[str]:
    if depth > max_depth:
        return set()

    paths: set[str] = set()

    if isinstance(value, dict):
        for key, item in value.items():
            current = f"{prefix}.{key}" if prefix else str(key)
            paths.add(current)
            paths.update(
                flatten_json_key_paths(
                    item,
                    prefix=current,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
            )
    elif isinstance(value, list) and value:
        paths.update(
            flatten_json_key_paths(
                value[0],
                prefix=prefix,
                depth=depth + 1,
                max_depth=max_depth,
            )
        )

    return paths


def infer_visual_type(path: Path) -> str:
    name = path.stem.lower()

    for visual_type, markers in VISUAL_NAME_MARKERS.items():
        if any(marker in name for marker in markers):
            return visual_type

    return "dataset/sample/other image"


def concept_matches(columns: Iterable[str]) -> dict[str, list[str]]:
    normalized_map = {column: normalize_name(column) for column in columns}

    concept_terms = {
        "round_or_epoch": ROUND_TERMS,
        "method": METHOD_TERMS,
        "model": MODEL_TERMS,
        "dataset": DATASET_TERMS,
        "distribution": DISTRIBUTION_TERMS,
        "alpha": ALPHA_TERMS,
        "client": CLIENT_TERMS,
        "class_or_category": CLASS_TERMS,
        "count": COUNT_TERMS,
        "split": SPLIT_TERMS,
        "true_label": TRUE_LABEL_TERMS,
        "predicted_label": PRED_LABEL_TERMS,
        "seed": SEED_TERMS,
        "runtime": TIME_TERMS,
        "efficiency": EFFICIENCY_TERMS,
        "network": NETWORK_TERMS,
        "aggregation_weight": WEIGHT_TERMS,
        "metric": METRIC_TERMS,
    }

    matches: dict[str, list[str]] = {}

    for concept, terms in concept_terms.items():
        matched = []
        normalized_terms = {normalize_name(term) for term in terms}

        for original, normalized in normalized_map.items():
            if normalized in normalized_terms:
                matched.append(original)
                continue

            if any(
                normalized.startswith(f"{term}_")
                or normalized.endswith(f"_{term}")
                for term in normalized_terms
                if len(term) >= 3
            ):
                matched.append(original)

        if matched:
            matches[concept] = sorted(set(matched))

    return matches


# ---------------------------------------------------------------------------
# File inventory
# ---------------------------------------------------------------------------

def inventory_files(
    root: Path,
    excluded: set[str],
    include_hidden: bool,
) -> tuple[list[FileRecord], list[Path]]:
    records: list[FileRecord] = []
    paths: list[Path] = []

    for current_root, directory_names, filenames in os.walk(root):
        current_path = Path(current_root)

        directory_names[:] = [
            directory
            for directory in directory_names
            if not should_skip_directory(directory, excluded, include_hidden)
        ]

        for filename in filenames:
            path = current_path / filename

            if not include_hidden and filename.startswith("."):
                continue

            try:
                stat = path.stat()
            except OSError:
                continue

            relative = safe_relative(path, root)
            suffix = path.suffix.lower() or "(none)"

            records.append(
                FileRecord(
                    relative_path=relative,
                    top_level=top_level_name(relative),
                    filename=path.name,
                    extension=suffix,
                    category=classify_file(path),
                    size_bytes=stat.st_size,
                    size_mb=round(stat.st_size / (1024 * 1024), 4),
                    modified_utc=utc_timestamp(stat.st_mtime),
                )
            )
            paths.append(path)

    records.sort(key=lambda record: record.relative_path.lower())
    paths.sort(key=lambda path: safe_relative(path, root).lower())
    return records, paths


# ---------------------------------------------------------------------------
# Tabular/structured inspection
# ---------------------------------------------------------------------------

def read_table_sample(path: Path) -> tuple[pd.DataFrame | None, str, str]:
    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            try:
                frame = pd.read_csv(path, nrows=MAX_TABULAR_ROWS)
            except (pd.errors.ParserError, UnicodeDecodeError):
                frame = pd.read_csv(
                    path,
                    sep=None,
                    engine="python",
                    nrows=MAX_TABULAR_ROWS,
                )
            return frame, "read", ""

        if suffix == ".tsv":
            frame = pd.read_csv(path, sep="\t", nrows=MAX_TABULAR_ROWS)
            return frame, "read", ""

        if suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(path, nrows=MAX_TABULAR_ROWS)
            return frame, "read", ""

        if suffix == ".parquet":
            frame = pd.read_parquet(path).head(MAX_TABULAR_ROWS)
            return frame, "read", ""

        if suffix in {".jsonl", ".ndjson"}:
            rows = []
            with path.open("r", encoding="utf-8", errors="replace") as file:
                for index, line in enumerate(file):
                    if index >= MAX_TABULAR_ROWS:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
            return pd.json_normalize(rows), "read", ""

        if suffix == ".json":
            if path.stat().st_size > MAX_JSON_BYTES:
                return None, "skipped", "JSON file is larger than the inspection limit."

            with path.open("r", encoding="utf-8", errors="replace") as file:
                value = json.load(file)

            if isinstance(value, list):
                if value and isinstance(value[0], dict):
                    return pd.json_normalize(value[:MAX_TABULAR_ROWS]), "read", ""
                return (
                    pd.DataFrame({"value": value[:MAX_TABULAR_ROWS]}),
                    "read",
                    "",
                )

            if isinstance(value, dict):
                record_lists = [
                    item
                    for item in value.values()
                    if isinstance(item, list)
                    and item
                    and isinstance(item[0], dict)
                ]

                if record_lists:
                    return (
                        pd.json_normalize(record_lists[0][:MAX_TABULAR_ROWS]),
                        "read",
                        "",
                    )

                key_paths = sorted(flatten_json_key_paths(value))
                return (
                    pd.DataFrame(columns=key_paths),
                    "keys-only",
                    "JSON was inspected as nested key paths.",
                )

            return None, "unsupported", "JSON root is not a table-like object."

    except ImportError as exc:
        return None, "dependency missing", str(exc)
    except (
        OSError,
        UnicodeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
        pd.errors.ParserError,
    ) as exc:
        return None, "error", str(exc)

    return None, "unsupported", f"Unsupported file type: {suffix}"


def inspect_tabular_files(
    paths: list[Path],
    root: Path,
) -> tuple[list[TabularRecord], dict[str, dict[str, Any]]]:
    records: list[TabularRecord] = []
    details: dict[str, dict[str, Any]] = {}

    candidates = [
        path
        for path in paths
        if path.suffix.lower() in TABULAR_SUFFIXES | {".json", ".jsonl", ".ndjson"}
    ]

    for path in candidates:
        relative = safe_relative(path, root)
        frame, status, error = read_table_sample(path)

        if frame is None:
            records.append(
                TabularRecord(
                    relative_path=relative,
                    file_type=path.suffix.lower(),
                    rows_sampled=0,
                    columns_count=0,
                    columns="",
                    numeric_columns="",
                    categorical_columns="",
                    detected_concepts="",
                    read_status=status,
                    error=error,
                )
            )
            continue

        columns = [str(column) for column in frame.columns]
        numeric_columns = [
            str(column)
            for column in frame.select_dtypes(include="number").columns
        ]
        categorical_columns = [
            column for column in columns if column not in numeric_columns
        ]
        concepts = concept_matches(columns)

        details[relative] = {
            "columns": columns,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "concepts": concepts,
            "rows_sampled": len(frame),
        }

        concept_text = "; ".join(
            f"{concept}: {', '.join(matches)}"
            for concept, matches in sorted(concepts.items())
        )

        records.append(
            TabularRecord(
                relative_path=relative,
                file_type=path.suffix.lower(),
                rows_sampled=len(frame),
                columns_count=len(columns),
                columns=", ".join(columns),
                numeric_columns=", ".join(numeric_columns),
                categorical_columns=", ".join(categorical_columns),
                detected_concepts=concept_text,
                read_status=status,
                error=error,
            )
        )

    records.sort(key=lambda record: record.relative_path.lower())
    return records, details


# ---------------------------------------------------------------------------
# Image inspection
# ---------------------------------------------------------------------------

def inspect_images(paths: list[Path], root: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    per_directory_count: defaultdict[Path, int] = defaultdict(int)

    for path in paths:
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        relative = safe_relative(path, root)
        inferred = infer_visual_type(path)

        if per_directory_count[path.parent] >= MAX_IMAGE_METADATA_PER_DIRECTORY:
            records.append(
                ImageRecord(
                    relative_path=relative,
                    extension=path.suffix.lower(),
                    width=None,
                    height=None,
                    mode="",
                    inferred_visual_type=inferred,
                    metadata_status="not opened; per-directory sample limit reached",
                    error="",
                )
            )
            continue

        per_directory_count[path.parent] += 1

        if Image is None:
            records.append(
                ImageRecord(
                    relative_path=relative,
                    extension=path.suffix.lower(),
                    width=None,
                    height=None,
                    mode="",
                    inferred_visual_type=inferred,
                    metadata_status="Pillow not installed",
                    error="Install pillow to inspect image dimensions.",
                )
            )
            continue

        try:
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode

            records.append(
                ImageRecord(
                    relative_path=relative,
                    extension=path.suffix.lower(),
                    width=width,
                    height=height,
                    mode=mode,
                    inferred_visual_type=inferred,
                    metadata_status="read",
                    error="",
                )
            )
        except (OSError, UnidentifiedImageError) as exc:
            records.append(
                ImageRecord(
                    relative_path=relative,
                    extension=path.suffix.lower(),
                    width=None,
                    height=None,
                    mode="",
                    inferred_visual_type=inferred,
                    metadata_status="error",
                    error=str(exc),
                )
            )

    records.sort(key=lambda record: record.relative_path.lower())
    return records


# ---------------------------------------------------------------------------
# Text/log metric scan
# ---------------------------------------------------------------------------

def scan_text_metric_files(
    paths: list[Path],
    root: Path,
) -> list[TextMetricRecord]:
    records: list[TextMetricRecord] = []

    experiment_terms = {
        "fedavg",
        "fedper",
        "fedrep",
        "centralized",
        "distributed",
        "ddp",
        "mobilenetv2",
        "resnet18",
        "efficientnet",
        "iid",
        "non-iid",
        "non_iid",
        "alpha",
        "client",
        "round",
        "epoch",
    }

    for path in paths:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue

        name_lower = path.name.lower()
        path_lower = safe_relative(path, root).lower()

        relevant_by_name = any(
            marker in name_lower or marker in path_lower
            for marker in (
                "result",
                "metric",
                "report",
                "eval",
                "slurm",
                "experiment",
                "training",
                "audit",
            )
        )

        if not relevant_by_name:
            continue

        try:
            with path.open("r", encoding="utf-8", errors="replace") as file:
                text = file.read(MAX_TEXT_SCAN_BYTES).lower()

            found_metrics = sorted(
                term for term in METRIC_TERMS if term in text
            )
            found_experiment_terms = sorted(
                term for term in experiment_terms if term in text
            )

            if found_metrics or found_experiment_terms:
                records.append(
                    TextMetricRecord(
                        relative_path=safe_relative(path, root),
                        detected_metric_terms=", ".join(found_metrics),
                        detected_experiment_terms=", ".join(
                            found_experiment_terms
                        ),
                        scan_status="read",
                        error="",
                    )
                )
        except OSError as exc:
            records.append(
                TextMetricRecord(
                    relative_path=safe_relative(path, root),
                    detected_metric_terms="",
                    detected_experiment_terms="",
                    scan_status="error",
                    error=str(exc),
                )
            )

    records.sort(key=lambda record: record.relative_path.lower())
    return records


# ---------------------------------------------------------------------------
# Graph recommendation engine
# ---------------------------------------------------------------------------

def add_recommendation(
    recommendations: list[GraphRecommendation],
    seen: set[tuple[str, str]],
    *,
    priority: int,
    graph_name: str,
    chart_type: str,
    source_file: str,
    status: str,
    detected: str,
    reason: str,
    suggested_output_filename: str,
) -> None:
    key = (source_file, graph_name)

    if key in seen:
        return

    seen.add(key)
    recommendations.append(
        GraphRecommendation(
            priority=priority,
            graph_name=graph_name,
            chart_type=chart_type,
            source_file=source_file,
            status=status,
            detected_columns_or_asset=detected,
            reason=reason,
            suggested_output_filename=suggested_output_filename,
        )
    )


def infer_graphs_from_tabular(
    details: dict[str, dict[str, Any]],
) -> list[GraphRecommendation]:
    recommendations: list[GraphRecommendation] = []
    seen: set[tuple[str, str]] = set()

    for source_file, detail in details.items():
        concepts = detail["concepts"]
        columns = detail["columns"]
        normalized_columns = {normalize_name(column): column for column in columns}

        def matches(concept: str) -> list[str]:
            return concepts.get(concept, [])

        def detected(*concept_names: str) -> str:
            values: list[str] = []
            for concept_name in concept_names:
                values.extend(matches(concept_name))
            return ", ".join(sorted(set(values)))

        source_token = normalize_name(Path(source_file).stem) or "chart"

        # Training/federated convergence
        if matches("round_or_epoch") and matches("metric"):
            metric_columns = [
                column
                for column in matches("metric")
                if normalize_name(column)
                in {
                    "loss",
                    "train_loss",
                    "val_loss",
                    "accuracy",
                    "acc",
                    "macro_f1",
                    "f1",
                    "precision",
                    "recall",
                }
            ]

            for metric_column in metric_columns:
                metric_token = normalize_name(metric_column)
                display_metric = metric_column.replace("_", " ").title()
                add_recommendation(
                    recommendations,
                    seen,
                    priority=1,
                    graph_name=f"{display_metric} by round/epoch",
                    chart_type="line",
                    source_file=source_file,
                    status="Ready from structured data",
                    detected=detected("round_or_epoch", "method", "distribution")
                    + f", {metric_column}",
                    reason=(
                        "Shows convergence and whether FedAvg, FedPer or FedRep "
                        "stabilize differently across training rounds."
                    ),
                    suggested_output_filename=(
                        f"{source_token}_{metric_token}_convergence.png"
                    ),
                )

        # Method/model comparison
        if (matches("method") or matches("model")) and matches("metric"):
            for metric_column in matches("metric"):
                metric_normalized = normalize_name(metric_column)

                if metric_normalized not in {
                    "accuracy",
                    "acc",
                    "macro_f1",
                    "f1",
                    "precision",
                    "recall",
                    "loss",
                    "runtime",
                    "duration",
                    "seconds",
                }:
                    continue

                add_recommendation(
                    recommendations,
                    seen,
                    priority=1,
                    graph_name=(
                        f"{metric_column.replace('_', ' ').title()} "
                        "comparison across experiments"
                    ),
                    chart_type="grouped horizontal bar",
                    source_file=source_file,
                    status="Ready from structured data",
                    detected=detected(
                        "dataset",
                        "method",
                        "model",
                        "distribution",
                    )
                    + f", {metric_column}",
                    reason=(
                        "Provides the clearest final comparison of centralized, "
                        "distributed and federated approaches."
                    ),
                    suggested_output_filename=(
                        f"{source_token}_{metric_normalized}_comparison.png"
                    ),
                )

        # Alpha sensitivity
        if matches("alpha") and matches("metric"):
            for metric_column in matches("metric"):
                if normalize_name(metric_column) not in {
                    "accuracy",
                    "acc",
                    "macro_f1",
                    "f1",
                    "loss",
                    "entropy",
                }:
                    continue

                add_recommendation(
                    recommendations,
                    seen,
                    priority=1,
                    graph_name=(
                        f"Dirichlet alpha vs "
                        f"{metric_column.replace('_', ' ').title()}"
                    ),
                    chart_type="line with markers",
                    source_file=source_file,
                    status="Ready if multiple alpha values exist",
                    detected=detected("alpha", "method", "client")
                    + f", {metric_column}",
                    reason=(
                        "Directly explains how stronger or weaker non-IID "
                        "partitioning changes model performance."
                    ),
                    suggested_output_filename=(
                        f"{source_token}_alpha_vs_{normalize_name(metric_column)}.png"
                    ),
                )

        # Client sample distribution
        if matches("client") and matches("count"):
            add_recommendation(
                recommendations,
                seen,
                priority=1,
                graph_name="Images per federated client",
                chart_type="bar",
                source_file=source_file,
                status="Ready from structured data",
                detected=detected("client", "count", "alpha", "distribution"),
                reason=(
                    "Visually proves that data was partitioned among separate "
                    "federated clients."
                ),
                suggested_output_filename=(
                    f"{source_token}_images_per_client.png"
                ),
            )

        # Client class distribution
        if matches("client") and matches("class_or_category") and matches("count"):
            add_recommendation(
                recommendations,
                seen,
                priority=1,
                graph_name="Class distribution across federated clients",
                chart_type="100% stacked bar or heatmap",
                source_file=source_file,
                status="Ready from structured data",
                detected=detected(
                    "client",
                    "class_or_category",
                    "count",
                    "alpha",
                ),
                reason=(
                    "This is the strongest visual demonstration of IID versus "
                    "non-IID client heterogeneity."
                ),
                suggested_output_filename=(
                    f"{source_token}_client_class_distribution.png"
                ),
            )

        # Dataset category distribution
        if (
            matches("class_or_category")
            and matches("count")
            and not matches("client")
        ):
            add_recommendation(
                recommendations,
                seen,
                priority=1,
                graph_name="Dataset category distribution",
                chart_type="horizontal bar",
                source_file=source_file,
                status="Ready from structured data",
                detected=detected(
                    "dataset",
                    "class_or_category",
                    "count",
                ),
                reason=(
                    "Shows the actual number and percentage of healthy, disease "
                    "or defect categories in the rice and wheat datasets."
                ),
                suggested_output_filename=(
                    f"{source_token}_category_distribution.png"
                ),
            )

        # Split distribution
        if matches("split") and matches("count"):
            add_recommendation(
                recommendations,
                seen,
                priority=1,
                graph_name="Train/validation/test split",
                chart_type="bar",
                source_file=source_file,
                status="Ready from structured data",
                detected=detected("dataset", "split", "count"),
                reason=(
                    "Summarizes the evaluation protocol and can accompany the "
                    "capture-group-overlap value."
                ),
                suggested_output_filename=(
                    f"{source_token}_data_split.png"
                ),
            )

        # Confusion matrix from predictions
        if matches("true_label") and matches("predicted_label"):
            add_recommendation(
                recommendations,
                seen,
                priority=1,
                graph_name="Confusion matrix",
                chart_type="heatmap",
                source_file=source_file,
                status="Ready from prediction labels",
                detected=detected("true_label", "predicted_label"),
                reason=(
                    "Supports class-level error analysis and comparison between "
                    "centralized, IID and non-IID methods."
                ),
                suggested_output_filename=(
                    f"{source_token}_confusion_matrix.png"
                ),
            )

        # Per-class metrics
        if matches("class_or_category") and matches("metric"):
            per_class_metrics = [
                column
                for column in matches("metric")
                if normalize_name(column)
                in {"precision", "recall", "f1", "support"}
            ]

            if per_class_metrics:
                add_recommendation(
                    recommendations,
                    seen,
                    priority=2,
                    graph_name="Per-class precision, recall and F1",
                    chart_type="grouped horizontal bar",
                    source_file=source_file,
                    status="Ready from classification-report data",
                    detected=detected("class_or_category", "metric"),
                    reason=(
                        "Reveals which grain classes remain difficult even when "
                        "overall Accuracy or Macro-F1 is high."
                    ),
                    suggested_output_filename=(
                        f"{source_token}_per_class_metrics.png"
                    ),
                )

        # Seed stability
        if matches("seed") and matches("metric"):
            add_recommendation(
                recommendations,
                seen,
                priority=2,
                graph_name="Performance stability across random seeds",
                chart_type="point/bar with mean and standard deviation",
                source_file=source_file,
                status="Ready if at least two seeds exist",
                detected=detected("seed", "method", "model", "metric"),
                reason=(
                    "Demonstrates reproducibility and prevents one favorable run "
                    "from being treated as the full result."
                ),
                suggested_output_filename=(
                    f"{source_token}_seed_stability.png"
                ),
            )

        # Runtime
        if matches("runtime") and (matches("method") or matches("model")):
            add_recommendation(
                recommendations,
                seen,
                priority=2,
                graph_name="Training runtime comparison",
                chart_type="bar",
                source_file=source_file,
                status="Ready from structured data",
                detected=detected("method", "model", "runtime"),
                reason=(
                    "Adds an efficiency comparison between centralized, DDP and "
                    "federated training."
                ),
                suggested_output_filename=(
                    f"{source_token}_runtime_comparison.png"
                ),
            )

        # Model efficiency
        if matches("efficiency") and (matches("model") or matches("method")):
            add_recommendation(
                recommendations,
                seen,
                priority=2,
                graph_name="Model accuracy-efficiency comparison",
                chart_type="scatter or grouped bar",
                source_file=source_file,
                status="Ready if performance metric is also present",
                detected=detected(
                    "model",
                    "method",
                    "efficiency",
                    "metric",
                ),
                reason=(
                    "Explains why MobileNetV2 or EfficientNetB0 may be suitable "
                    "for resource-constrained federated clients."
                ),
                suggested_output_filename=(
                    f"{source_token}_accuracy_efficiency.png"
                ),
            )

        # Network-aware relationship
        if matches("network") and matches("metric"):
            add_recommendation(
                recommendations,
                seen,
                priority=2,
                graph_name="Network condition vs model performance",
                chart_type="scatter or line",
                source_file=source_file,
                status="Ready from network-aware experiment data",
                detected=detected(
                    "network",
                    "metric",
                    "client",
                    "round_or_epoch",
                ),
                reason=(
                    "Connects the network-aware project claim to measured "
                    "latency, bandwidth or packet-loss effects."
                ),
                suggested_output_filename=(
                    f"{source_token}_network_vs_performance.png"
                ),
            )

        # Adaptive weights
        if matches("aggregation_weight") and (
            matches("client") or matches("round_or_epoch")
        ):
            add_recommendation(
                recommendations,
                seen,
                priority=1,
                graph_name="Adaptive aggregation weights",
                chart_type="multi-line or stacked area",
                source_file=source_file,
                status="Ready from adaptive-weight logs",
                detected=detected(
                    "aggregation_weight",
                    "client",
                    "round_or_epoch",
                ),
                reason=(
                    "Directly visualizes the proposed research contribution: "
                    "how each client's contribution changes over rounds."
                ),
                suggested_output_filename=(
                    f"{source_token}_adaptive_client_weights.png"
                ),
            )

        # Accuracy vs Macro-F1 scatter
        accuracy_column = normalized_columns.get("accuracy") or normalized_columns.get("acc")
        macro_f1_column = normalized_columns.get("macro_f1")

        if accuracy_column and macro_f1_column:
            add_recommendation(
                recommendations,
                seen,
                priority=2,
                graph_name="Accuracy vs Macro-F1",
                chart_type="scatter",
                source_file=source_file,
                status="Ready from structured data",
                detected=f"{accuracy_column}, {macro_f1_column}",
                reason=(
                    "Checks whether high overall accuracy hides weaker balanced "
                    "performance across classes."
                ),
                suggested_output_filename=(
                    f"{source_token}_accuracy_vs_macro_f1.png"
                ),
            )

    recommendations.sort(
        key=lambda item: (
            item.priority,
            item.graph_name.lower(),
            item.source_file.lower(),
        )
    )
    return recommendations


def infer_graphs_from_images(
    image_records: list[ImageRecord],
) -> list[GraphRecommendation]:
    recommendations: list[GraphRecommendation] = []
    seen: set[tuple[str, str]] = set()

    for record in image_records:
        if record.inferred_visual_type == "dataset/sample/other image":
            continue

        visual_name = record.inferred_visual_type.replace("_", " ").title()

        add_recommendation(
            recommendations,
            seen,
            priority=1,
            graph_name=f"Display existing {visual_name}",
            chart_type="existing saved image",
            source_file=record.relative_path,
            status="Existing asset; no regeneration required",
            detected=record.relative_path,
            reason=(
                "The repository already contains a suitable visual asset that "
                "can be displayed directly in Streamlit."
            ),
            suggested_output_filename=Path(record.relative_path).name,
        )

    recommendations.sort(
        key=lambda item: (
            item.priority,
            item.graph_name.lower(),
            item.source_file.lower(),
        )
    )
    return recommendations


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_summary_tables(
    file_records: list[FileRecord],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not file_records:
        return pd.DataFrame(), pd.DataFrame()

    frame = pd.DataFrame(asdict(record) for record in file_records)

    top_level = (
        frame.groupby("top_level", as_index=False)
        .agg(
            file_count=("relative_path", "count"),
            total_size_mb=("size_mb", "sum"),
        )
        .sort_values(["file_count", "top_level"], ascending=[False, True])
    )
    top_level["total_size_mb"] = top_level["total_size_mb"].round(2)

    extension = (
        frame.groupby(["extension", "category"], as_index=False)
        .agg(
            file_count=("relative_path", "count"),
            total_size_mb=("size_mb", "sum"),
        )
        .sort_values(["file_count", "extension"], ascending=[False, True])
    )
    extension["total_size_mb"] = extension["total_size_mb"].round(2)

    return top_level, extension


def create_markdown_report(
    *,
    root: Path,
    output: Path,
    excluded: set[str],
    file_records: list[FileRecord],
    top_level: pd.DataFrame,
    extension: pd.DataFrame,
    tabular_records: list[TabularRecord],
    image_records: list[ImageRecord],
    text_records: list[TextMetricRecord],
    recommendations: list[GraphRecommendation],
) -> str:
    total_size_mb = sum(record.size_mb for record in file_records)
    category_counts = Counter(record.category for record in file_records)

    ready_recommendations = [
        record
        for record in recommendations
        if record.status.startswith("Ready")
        or record.status.startswith("Existing")
    ]

    priority_one = [
        asdict(record)
        for record in recommendations
        if record.priority == 1
    ]

    tabular_rows = [
        asdict(record)
        for record in tabular_records
        if record.read_status in {"read", "keys-only"}
    ]

    existing_visuals = [
        asdict(record)
        for record in image_records
        if record.inferred_visual_type != "dataset/sample/other image"
    ]

    top_level_rows = top_level.to_dict(orient="records") if not top_level.empty else []
    extension_rows = extension.to_dict(orient="records") if not extension.empty else []

    lines = [
        "# Project dashboard audit",
        "",
        f"- **Repository root:** `{root}`",
        f"- **Audit output:** `{output}`",
        f"- **Generated at:** {datetime.now(timezone.utc).isoformat()}",
        f"- **Files inventoried:** {len(file_records):,}",
        f"- **Total inventoried size:** {total_size_mb:,.2f} MB",
        f"- **Readable tabular/structured files:** {len(tabular_rows):,}",
        f"- **Image files:** {len(image_records):,}",
        f"- **Existing recognized visual assets:** {len(existing_visuals):,}",
        f"- **Chart recommendations:** {len(recommendations):,}",
        f"- **Ready/existing recommendations:** {len(ready_recommendations):,}",
        f"- **Excluded directory names:** {', '.join(sorted(excluded))}",
        "",
        "## File categories",
        "",
    ]

    for category, count in sorted(
        category_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lines.append(f"- **{category}:** {count:,}")

    lines.extend(
        [
            "",
            "## Top-level directory summary",
            "",
            markdown_table(
                top_level_rows,
                ["top_level", "file_count", "total_size_mb"],
                max_rows=40,
            ),
            "",
            "## File extension summary",
            "",
            markdown_table(
                extension_rows,
                ["extension", "category", "file_count", "total_size_mb"],
                max_rows=40,
            ),
            "",
            "## Readable result/data files",
            "",
            markdown_table(
                tabular_rows,
                [
                    "relative_path",
                    "rows_sampled",
                    "columns_count",
                    "detected_concepts",
                    "read_status",
                ],
                max_rows=50,
            ),
            "",
            "## Existing recognized charts and figures",
            "",
            markdown_table(
                existing_visuals,
                [
                    "relative_path",
                    "inferred_visual_type",
                    "width",
                    "height",
                    "metadata_status",
                ],
                max_rows=50,
            ),
            "",
            "## Highest-priority chart opportunities",
            "",
            markdown_table(
                priority_one,
                [
                    "graph_name",
                    "chart_type",
                    "source_file",
                    "status",
                    "detected_columns_or_asset",
                    "suggested_output_filename",
                ],
                max_rows=80,
            ),
            "",
            "## Recommended dashboard selection order",
            "",
            "Use this order when deciding what to display:",
            "",
            "1. **Final experiment comparison:** Centralized, FedAvg, FedPer and FedRep using Macro-F1.",
            "2. **Dataset category distribution:** Separate rice and wheat category-count figures.",
            "3. **IID/non-IID client class distribution:** Prefer a 100% stacked bar or heatmap.",
            "4. **Client image counts:** Show how much data each client received.",
            "5. **Round-by-round convergence:** Macro-F1 and loss for FedAvg, FedPer and FedRep.",
            "6. **Confusion matrices:** One selector plus side-by-side comparison mode.",
            "7. **Alpha sensitivity:** Compare α values only when multiple verified α experiments exist.",
            "8. **Seed stability:** Add mean and standard deviation when multiple seeds exist.",
            "9. **Adaptive weights:** Add when the new client-weight logs become available.",
            "10. **Network effects:** Add only when latency/bandwidth/packet-loss data is recorded.",
            "",
            "## Important limitations",
            "",
            "- A recommendation marked **Ready** means the required column pattern was detected; it does not guarantee every row is complete or scientifically verified.",
            "- A recommendation marked **Ready if...** requires multiple values, such as multiple alpha values or random seeds.",
            "- Text/log files are only keyword-scanned. Their contents may need a dedicated parser before plotting.",
            "- The audit does not load model checkpoints or execute training code.",
            "- Raw dataset images are inventoried, but only a small sample per directory is opened for dimensions to keep the scan practical on WSL-mounted drives.",
            "",
            "## Files to share for the next step",
            "",
            "Share these two files after running the audit:",
            "",
            "- `project_audit/audit_report.md`",
            "- `project_audit/candidate_graphs.csv`",
            "",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"ERROR: Repository root does not exist or is not a directory: {root}")
        return 2

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "project_audit"
    )
    output.mkdir(parents=True, exist_ok=True)

    excluded = set(DEFAULT_EXCLUDED_DIRS) | set(args.extra_exclude)

    # Avoid recursively scanning the audit output from a previous run.
    excluded.add(output.name)

    print(f"[1/7] Inventorying repository: {root}")
    file_records, paths = inventory_files(
        root,
        excluded=excluded,
        include_hidden=args.include_hidden,
    )

    print(f"[2/7] Inspecting tabular and structured files...")
    tabular_records, tabular_details = inspect_tabular_files(paths, root)

    print(f"[3/7] Inspecting image assets...")
    image_records = inspect_images(paths, root)

    print(f"[4/7] Scanning result/log text files for metric terms...")
    text_records = scan_text_metric_files(paths, root)

    print(f"[5/7] Inferring supported chart opportunities...")
    recommendations = infer_graphs_from_tabular(tabular_details)
    recommendations.extend(infer_graphs_from_images(image_records))

    # Final de-duplication and ordering.
    unique_recommendations: dict[tuple[str, str], GraphRecommendation] = {}
    for recommendation in recommendations:
        key = (recommendation.source_file, recommendation.graph_name)
        current = unique_recommendations.get(key)
        if current is None or recommendation.priority < current.priority:
            unique_recommendations[key] = recommendation

    recommendations = sorted(
        unique_recommendations.values(),
        key=lambda item: (
            item.priority,
            item.graph_name.lower(),
            item.source_file.lower(),
        ),
    )

    print(f"[6/7] Writing CSV and JSON audit outputs...")
    write_dataclass_csv(output / "files_inventory.csv", file_records)
    write_dataclass_csv(output / "tabular_files_summary.csv", tabular_records)
    write_dataclass_csv(output / "image_files_summary.csv", image_records)
    write_dataclass_csv(output / "text_metric_files_summary.csv", text_records)
    write_dataclass_csv(output / "candidate_graphs.csv", recommendations)

    top_level, extension = build_summary_tables(file_records)
    top_level.to_csv(output / "top_level_summary.csv", index=False)
    extension.to_csv(output / "extension_summary.csv", index=False)

    summary = {
        "repository_root": str(root),
        "output_directory": str(output),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(file_records),
        "total_size_mb": round(sum(record.size_mb for record in file_records), 2),
        "tabular_file_count": len(tabular_records),
        "image_file_count": len(image_records),
        "text_metric_file_count": len(text_records),
        "graph_recommendation_count": len(recommendations),
        "priority_one_graph_count": sum(
            recommendation.priority == 1
            for recommendation in recommendations
        ),
        "excluded_directory_names": sorted(excluded),
    }
    (output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"[7/7] Writing Markdown report...")
    report = create_markdown_report(
        root=root,
        output=output,
        excluded=excluded,
        file_records=file_records,
        top_level=top_level,
        extension=extension,
        tabular_records=tabular_records,
        image_records=image_records,
        text_records=text_records,
        recommendations=recommendations,
    )
    (output / "audit_report.md").write_text(report, encoding="utf-8")

    print()
    print("Audit complete.")
    print(f"Files inventoried: {len(file_records):,}")
    print(f"Tabular/structured files inspected: {len(tabular_records):,}")
    print(f"Images inventoried: {len(image_records):,}")
    print(f"Chart recommendations: {len(recommendations):,}")
    print()
    print(f"Open this report: {output / 'audit_report.md'}")
    print(f"Chart list:       {output / 'candidate_graphs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
