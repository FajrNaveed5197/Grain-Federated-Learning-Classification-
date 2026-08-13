#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


MAX_TABLE_ROWS_IN_REPORT = 40
MAX_JSON_DEPTH = 7

IMPORTANT_JSON_TERMS = {
    "accuracy", "balanced_accuracy", "macro_f1", "mean_macro_f1",
    "best_mean_validation_macro_f1", "loss", "train_loss",
    "validation_loss", "precision", "recall", "support", "round",
    "epoch", "selected_round", "client", "clients", "num_samples",
    "images", "capture_groups", "class_entropy", "alpha", "algorithm",
    "experiment", "communication", "total_parameters", "shared_parameters",
    "private_parameters_per_client", "complete_transfers", "elapsed_seconds",
    "total_seconds", "evaluation_seconds", "seed", "test_summary",
    "validation_summary", "history",
}

EXACT_CANDIDATES = [
    "results/Rice/Federated/Summaries/Presentation_tables/table_1_alpha0p5_comprehensive.csv",
    "experiments/results/tables/rice_architecture_evaluation.csv",
    "results/Reports/FinalReportArchive/tables/rice_architecture_evaluation.csv",
    "experiments/results/provenance/rice_grouped_split/split_summary.json",
    "experiments/results/provenance/wheat_grouped_split_v2/class_allocation.csv",
    "experiments/results/provenance/wheat_grouped_split_v2/metadata.json",
]

SEARCH_PATTERNS = [
    "**/*fedper*/metrics.json",
    "**/*fedper*/test_metrics.json",
    "**/*fedrep*/metrics.json",
    "**/*fedrep*/test_metrics.json",
    "**/*fedavg*/metrics.json",
    "**/*fedavg*/evaluation_summary.csv",
    "**/*client*class*.csv",
    "**/*class*distribution*.csv",
    "**/*partition*.csv",
    "**/*partition*.json",
    "**/*client*.json",
    "**/*split_summary.json",
    "**/*class_allocation.csv",
    "**/*presentation*.csv",
    "**/*comprehensive*.csv",
]

EXCLUDED_PARTS = {
    ".git", ".venv", ".venv-paper", "__pycache__", "project_audit",
    "streamlit-ui",
}


@dataclass
class SourceRecord:
    relative_path: str
    file_type: str
    size_kb: float
    sha256: str
    rows: int | None
    columns: str
    detected_role: str
    chart_use: str
    caution: str
    read_status: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def is_excluded(path: Path, root: Path) -> bool:
    try:
        parts = path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        return True
    return any(part in EXCLUDED_PARTS for part in parts)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def detect_role(path: Path, columns: list[str] | None = None) -> tuple[str, str, str]:
    text = path.as_posix().lower()
    normalized_columns = {normalize(column) for column in (columns or [])}

    if "table_1_alpha0p5_comprehensive" in text:
        return (
            "Final alpha=0.5 comparison table",
            "Primary grouped bar of Accuracy, Balanced Accuracy and Macro-F1",
            "Verify whether FedPer/FedRep rows are client means or individual clients.",
        )
    if "rice_architecture_evaluation" in text:
        return (
            "Centralized architecture comparison",
            "Bar chart comparing ResNet18, MobileNetV2 and EfficientNetB0",
            "Use one canonical copy if duplicate hashes match.",
        )
    if path.name == "split_summary.json" and "rice_grouped_split" in text:
        return (
            "Rice group-aware split metadata",
            "Train/validation/test split chart and capture-group-overlap summary",
            "This is split metadata, not a category distribution.",
        )
    if path.name == "class_allocation.csv" and "wheat" in text:
        return (
            "Wheat class allocation",
            "Wheat full-dataset class-count bar chart",
            "Confirm which columns are full dataset, validation and test allocations.",
        )
    if path.name == "test_metrics.json" and ("fedper" in text or "fedrep" in text):
        return (
            "Personalized federated test results",
            "Per-client Macro-F1/Accuracy chart and client summary table",
            "Do not present one client as a global model.",
        )
    if path.name == "metrics.json" and ("fedper" in text or "fedrep" in text):
        return (
            "Personalized federated round history",
            "Round-by-round mean Macro-F1 and per-client convergence",
            "Parse validation_summary and validation_by_client.",
        )
    if path.name == "metrics.json" and "fedavg" in text:
        return (
            "FedAvg round history",
            "Round-by-round Accuracy, Balanced Accuracy and Macro-F1",
            "Confirm these are validation metrics and label the graph.",
        )
    if path.name == "evaluation_summary.csv":
        return (
            "Final split-level evaluation summary",
            "Final Accuracy/Balanced Accuracy/Macro-F1 table",
            "Use the test row for final comparison.",
        )
    if (
        {"client", "category", "count"}.issubset(normalized_columns)
        or {"client", "class", "count"}.issubset(normalized_columns)
        or {"client_id", "label", "count"}.issubset(normalized_columns)
    ):
        return (
            "Client-by-class distribution",
            "100% stacked client-class bar or heatmap",
            "This can prove IID/non-IID class heterogeneity.",
        )
    if "partition" in text or "client" in text:
        return (
            "Client/partition evidence",
            "Potential client image-count or class-distribution chart",
            "Inspect columns before claiming per-class counts.",
        )
    if "per_class_metrics" in text:
        return (
            "Per-class evaluation metrics",
            "Per-class precision/recall/F1 chart",
            "Support is an evaluation-split count, not necessarily the full dataset.",
        )
    return (
        "Candidate research artifact",
        "Inspect before assigning a chart",
        "No automatic scientific interpretation was applied.",
    )


def read_csv_or_tsv(path: Path) -> tuple[pd.DataFrame | None, str]:
    try:
        if path.suffix.lower() == ".tsv":
            return pd.read_csv(path, sep="\t"), ""
        try:
            return pd.read_csv(path), ""
        except pd.errors.ParserError:
            return pd.read_csv(path, sep=None, engine="python"), ""
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        return None, str(exc)


def important_json_items(value: Any, prefix: str = "", depth: int = 0) -> list[tuple[str, Any]]:
    if depth > MAX_JSON_DEPTH:
        return []

    items: list[tuple[str, Any]] = []

    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            normalized_key = normalize(key)

            if normalized_key in IMPORTANT_JSON_TERMS:
                if isinstance(child, (str, int, float, bool)) or child is None:
                    items.append((path, child))
                elif isinstance(child, list):
                    items.append((path, f"<list length={len(child)}>"))
                elif isinstance(child, dict):
                    items.append((path, f"<object keys={len(child)}>"))

            items.extend(important_json_items(child, path, depth + 1))

    elif isinstance(value, list):
        indexes = range(len(value)) if len(value) <= 10 else ([0, len(value) - 1] if value else [])
        for index in indexes:
            items.extend(important_json_items(value[index], f"{prefix}[{index}]", depth + 1))

    return items


def markdown_table(frame: pd.DataFrame, max_rows: int = MAX_TABLE_ROWS_IN_REPORT) -> str:
    if frame.empty:
        return "_Empty table._"

    shown = frame.head(max_rows).copy().fillna("")
    headers = [str(column) for column in shown.columns]

    def clean(value: Any) -> str:
        return str(value).replace("|", r"\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(clean(row[column]) for column in shown.columns) + " |")

    if len(frame) > max_rows:
        lines.append(f"\n_Only the first {max_rows} of {len(frame)} rows are shown._")

    return "\n".join(lines)


def find_candidates(root: Path) -> list[Path]:
    found: set[Path] = set()

    for candidate in EXACT_CANDIDATES:
        path = root / candidate
        if path.exists() and path.is_file():
            found.add(path.resolve())

    for pattern in SEARCH_PATTERNS:
        for path in root.glob(pattern):
            if not path.is_file() or is_excluded(path, root):
                continue
            rel = relative(path, root).lower()
            if rel.startswith("results/") or rel.startswith("experiments/results/"):
                found.add(path.resolve())

    return sorted(found, key=lambda path: relative(path, root).lower())


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()

    if not root.exists() or not root.is_dir():
        print(f"ERROR: Invalid repository root: {root}")
        return 2

    output = args.output.expanduser().resolve() if args.output else root / "project_audit"
    output.mkdir(parents=True, exist_ok=True)

    candidates = find_candidates(root)
    print(f"Focused candidates found: {len(candidates)}")

    records: list[SourceRecord] = []
    report_sections: list[str] = []
    hashes: dict[str, list[str]] = {}

    for path in candidates:
        rel = relative(path, root)
        suffix = path.suffix.lower()
        size_kb = round(path.stat().st_size / 1024, 2)
        digest = sha256_file(path)
        hashes.setdefault(digest, []).append(rel)

        if suffix in {".csv", ".tsv"}:
            frame, error = read_csv_or_tsv(path)
            role, chart_use, caution = detect_role(path, list(frame.columns) if frame is not None else None)

            if frame is None:
                records.append(SourceRecord(
                    rel, suffix, size_kb, digest, None, "", role, chart_use,
                    caution, "error", error
                ))
                continue

            columns = [str(column) for column in frame.columns]
            records.append(SourceRecord(
                rel, suffix, size_kb, digest, len(frame), ", ".join(columns),
                role, chart_use, caution, "read", ""
            ))
            report_sections.extend([
                f"## `{rel}`", "",
                f"- **Role:** {role}",
                f"- **Recommended use:** {chart_use}",
                f"- **Caution:** {caution}",
                f"- **Rows:** {len(frame)}",
                f"- **Columns:** {', '.join(columns)}", "",
                markdown_table(frame), "",
            ])

        elif suffix == ".json":
            role, chart_use, caution = detect_role(path)
            try:
                with path.open("r", encoding="utf-8") as file:
                    data = json.load(file)

                records.append(SourceRecord(
                    rel, suffix, size_kb, digest, None, "", role, chart_use,
                    caution, "read", ""
                ))
                json_frame = pd.DataFrame(
                    important_json_items(data),
                    columns=["key_path", "value"],
                )
                report_sections.extend([
                    f"## `{rel}`", "",
                    f"- **Role:** {role}",
                    f"- **Recommended use:** {chart_use}",
                    f"- **Caution:** {caution}", "",
                    "### Important metric paths", "",
                    markdown_table(json_frame, max_rows=100), "",
                ])
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                records.append(SourceRecord(
                    rel, suffix, size_kb, digest, None, "", role, chart_use,
                    caution, "error", str(exc)
                ))

    records_frame = pd.DataFrame(asdict(record) for record in records)
    records_frame.to_csv(output / "focused_dashboard_sources.csv", index=False)

    duplicate_rows = []
    for digest, paths in hashes.items():
        if len(paths) < 2:
            continue
        for path in paths:
            duplicate_rows.append({
                "sha256": digest,
                "duplicate_count": len(paths),
                "relative_path": path,
            })

    duplicates = pd.DataFrame(
        duplicate_rows,
        columns=["sha256", "duplicate_count", "relative_path"],
    )
    duplicates.to_csv(output / "duplicate_candidate_files.csv", index=False)

    priority_roles = {
        "Final alpha=0.5 comparison table",
        "Centralized architecture comparison",
        "Rice group-aware split metadata",
        "Wheat class allocation",
        "Personalized federated test results",
        "Personalized federated round history",
        "FedAvg round history",
        "Client-by-class distribution",
    }

    priority_frame = records_frame[
        records_frame["detected_role"].isin(priority_roles)
    ].copy() if not records_frame.empty else pd.DataFrame()

    report_parts = [
        "# Focused dashboard-source inspection", "",
        f"- **Repository:** `{root}`",
        f"- **Candidate files inspected:** {len(records_frame)}",
        f"- **Exact duplicate groups:** {duplicates['sha256'].nunique() if not duplicates.empty else 0}",
        "",
        "## Most useful sources", "",
        markdown_table(
            priority_frame[
                ["relative_path", "detected_role", "chart_use", "caution", "read_status"]
            ],
            max_rows=100,
        ) if not priority_frame.empty else "_No priority sources were found._",
        "",
        "## Interpretation rules", "",
        "1. Use `results/.../Presentation_tables/` for a final overview only after checking its row definitions.",
        "2. Use `test` evaluation rows for final reported performance.",
        "3. Use `validation` values only for checkpoint/model selection or convergence.",
        "4. Do not use per-class `support` as the full dataset distribution unless the file is explicitly a full allocation manifest.",
        "5. Do not claim a client-class distribution from a table containing only client image totals and entropy.",
        "6. For FedPer/FedRep, report client-level results and an explicitly calculated mean; do not call one client the global score.",
        "7. Existing test confusion-matrix PNG files can be displayed directly.",
        "",
        "## Detailed source contents", "",
        *report_sections,
    ]

    (output / "focused_dashboard_sources.md").write_text(
        "\n".join(report_parts),
        encoding="utf-8",
    )

    print()
    print("Focused inspection complete.")
    print(f"Report:     {output / 'focused_dashboard_sources.md'}")
    print(f"Source CSV: {output / 'focused_dashboard_sources.csv'}")
    print(f"Duplicates: {output / 'duplicate_candidate_files.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
