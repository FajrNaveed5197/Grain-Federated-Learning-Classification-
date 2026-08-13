#!/usr/bin/env python3
"""
Generate two static donut/pie charts:

1. Rice test-set class distribution
2. Wheat development-pool category allocation

Run from the repository root:

    source .venv/bin/activate
    python generate_rice_wheat_pie_charts.py --root .

Default output:
    streamlit-ui/streamlit-ui/assets/generated_charts/

Use --output to select another directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Verified sources
# ---------------------------------------------------------------------------

# This file contains class_name and support for the rice test split.
RICE_TEST_CLASS_SOURCE = Path(
    "experiments/results/rice/"
    "rice_mobilenetv2_grouped_v1/test_per_class_metrics.csv"
)

# Fallback copy of the same type of artifact.
RICE_TEST_CLASS_FALLBACK = Path(
    "results/Rice/Centralized/MobileNetV2/seed42/"
    "test_per_class_metrics.csv"
)

# This file contains full development-pool wheat allocation.
WHEAT_CLASS_SOURCE = Path(
    "experiments/results/provenance/"
    "wheat_grouped_split_v2/class_allocation.csv"
)


# Optional readable replacements for rice class codes.
# Unknown codes are left unchanged rather than guessed.
RICE_CLASS_NAMES = {
    "0_NOR": "Normal",
    "1_F&S": "Foreign & strange",
    "2_SD": "Surface damage",
    "3_MY": "Moldy",
    "4_AP": "Abnormal pigmentation",
    "5_BN": "Broken",
    "6_UN": "Unhulled",
    "7_IM": "Immature",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
            "streamlit-ui/streamlit-ui/assets/generated_charts"
        ),
    )
    return parser.parse_args()


def read_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {', '.join(missing)}"
        )

    return frame.dropna(how="all").reset_index(drop=True)


def autopct_with_counts(values: list[float]):
    total = sum(values)

    def formatter(percent: float) -> str:
        count = int(round(percent * total / 100.0))
        return f"{percent:.1f}%\n({count:,})"

    return formatter


def save_donut(
    labels: list[str],
    values: list[float],
    *,
    title: str,
    subtitle: str,
    output_stem: Path,
) -> None:
    if not labels or not values or sum(values) <= 0:
        raise ValueError(f"No positive values available for {title}")

    figure, axis = plt.subplots(figsize=(10, 8))

    wedges, texts, autotexts = axis.pie(
        values,
        labels=None,
        autopct=autopct_with_counts(values),
        startangle=90,
        counterclock=False,
        pctdistance=0.77,
        wedgeprops={
            "width": 0.42,
            "edgecolor": "white",
            "linewidth": 1.2,
        },
        textprops={"fontsize": 9},
    )

    centre_text = f"{int(sum(values)):,}\nimages"
    axis.text(
        0,
        0,
        centre_text,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )

    axis.legend(
        wedges,
        labels,
        title="Classes",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=9,
    )

    axis.set_title(title, fontsize=16, fontweight="bold", pad=18)
    figure.text(
        0.5,
        0.025,
        subtitle,
        ha="center",
        fontsize=9,
    )

    axis.axis("equal")
    figure.tight_layout(rect=(0, 0.05, 0.82, 1))

    output_stem.parent.mkdir(parents=True, exist_ok=True)

    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")

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


def generate_rice_chart(root: Path, output_dir: Path) -> None:
    primary = root / RICE_TEST_CLASS_SOURCE
    fallback = root / RICE_TEST_CLASS_FALLBACK

    source = primary if primary.exists() else fallback

    frame = read_csv(
        source,
        required_columns=["class_name", "support"],
    )

    frame["support"] = pd.to_numeric(frame["support"], errors="coerce")
    frame = frame.dropna(subset=["class_name", "support"])
    frame = frame[frame["support"] > 0].copy()

    frame["display_name"] = frame["class_name"].map(
        lambda value: RICE_CLASS_NAMES.get(str(value), str(value))
    )

    # Sort from largest to smallest for a clearer legend.
    frame = frame.sort_values("support", ascending=False)

    save_donut(
        labels=frame["display_name"].tolist(),
        values=frame["support"].tolist(),
        title="Rice test-set class distribution",
        subtitle=(
            "Counts come from test_per_class_metrics.csv. "
            "This is the held-out test split, not the full rice dataset."
        ),
        output_stem=output_dir / "dataset_rice_test_class_distribution",
    )


def generate_wheat_chart(root: Path, output_dir: Path) -> None:
    source = root / WHEAT_CLASS_SOURCE

    frame = read_csv(
        source,
        required_columns=["label", "total_images"],
    )

    frame["total_images"] = pd.to_numeric(
        frame["total_images"],
        errors="coerce",
    )
    frame = frame.dropna(subset=["label", "total_images"])
    frame = frame[frame["total_images"] > 0].copy()
    frame = frame.sort_values("total_images", ascending=False)

    save_donut(
        labels=frame["label"].astype(str).tolist(),
        values=frame["total_images"].tolist(),
        title="Wheat development-pool category allocation",
        subtitle=(
            "Counts come from class_allocation.csv using total_images. "
            "Independent held-out test collections are not included unless "
            "they are represented in that source."
        ),
        output_stem=output_dir / "dataset_wheat_category_distribution",
    )


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()

    if not root.exists() or not root.is_dir():
        print(f"ERROR: Invalid repository root: {root}", file=sys.stderr)
        return 2

    output_dir = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root
        / "streamlit-ui"
        / "streamlit-ui"
        / "assets"
        / "generated_charts"
    )

    failures: list[str] = []

    try:
        generate_rice_chart(root, output_dir)
    except Exception as exc:
        failures.append(f"Rice chart: {exc}")
        print(f"[FAILED] Rice chart: {exc}", file=sys.stderr)

    try:
        generate_wheat_chart(root, output_dir)
    except Exception as exc:
        failures.append(f"Wheat chart: {exc}")
        print(f"[FAILED] Wheat chart: {exc}", file=sys.stderr)

    if failures:
        print()
        print("One or more charts failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print()
    print("Both pie charts were generated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
