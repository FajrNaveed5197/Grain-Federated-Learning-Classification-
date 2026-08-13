from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
OUT_DIR = APP_DIR / "assets" / "generated_charts"

GREEN = "#245C3A"
GOLD = "#D0A646"
MUTED_GREEN = "#6D8F64"
BROWN = "#8C6A3B"
COLORS = [GREEN, GOLD, MUTED_GREEN, BROWN]


def token(value: object) -> str:
    text = str(value).strip().lower().replace("non-iid", "non_iid")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def load(name: str, required: list[str]) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        print(f"[SKIP] Missing {path}")
        return pd.DataFrame(columns=required)
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        print(f"[ERROR] {name}: {exc}")
        return pd.DataFrame(columns=required)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        print(f"[ERROR] {name} missing columns: {missing}")
        return pd.DataFrame(columns=required)
    return frame.dropna(how="all").reset_index(drop=True)


def style(axis: plt.Axes, title: str, grid_axis: str = "x") -> None:
    axis.set_title(title, pad=14, fontweight="bold")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, alpha=.16)
    axis.set_axisbelow(True)


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {png.name}")
    print(f"[OK] {pdf.name}")


def category_charts() -> None:
    cols = ["dataset","category","count","source"]
    frame = load("dataset_categories.csv", cols)
    if frame.empty:
        return
    frame["dataset"] = frame["dataset"].fillna("").astype(str).str.strip()
    frame["category"] = frame["category"].fillna("").astype(str).str.strip()
    frame["count"] = pd.to_numeric(frame["count"], errors="coerce")
    frame = frame[frame["dataset"].ne("") & frame["category"].ne("") & frame["count"].fillna(0).gt(0)]
    if frame.empty:
        print("[SKIP] No positive category counts.")
        return

    for dataset, group in frame.groupby("dataset"):
        plot = group.groupby("category", as_index=False)["count"].sum().sort_values("count")
        fig, ax = plt.subplots(figsize=(10, max(4.8, .55*len(plot))))
        bars = ax.barh(plot["category"], plot["count"], color=GREEN)
        ax.bar_label(bars, fmt="%d", padding=4)
        ax.set_xlabel("Number of images")
        ax.set_ylabel("")
        style(ax, f"{dataset} category distribution")
        save(fig, f"dataset_categories_{token(dataset)}")


def split_charts() -> None:
    cols = ["dataset","split","count","capture_group_overlap","source"]
    frame = load("dataset_splits.csv", cols)
    if frame.empty:
        return
    frame["dataset"] = frame["dataset"].fillna("").astype(str).str.strip()
    frame["split"] = frame["split"].fillna("").astype(str).str.strip()
    frame["count"] = pd.to_numeric(frame["count"], errors="coerce")
    frame = frame[frame["dataset"].ne("") & frame["split"].ne("") & frame["count"].fillna(0).gt(0)]
    order = {"Train":0,"Validation":1,"Test":2}

    for dataset, group in frame.groupby("dataset"):
        plot = group.groupby("split", as_index=False)["count"].sum()
        plot["_order"] = plot["split"].map(order).fillna(99)
        plot = plot.sort_values("_order")
        fig, ax = plt.subplots(figsize=(8,5))
        bars = ax.bar(plot["split"], plot["count"], color=COLORS[:len(plot)])
        ax.bar_label(bars, fmt="%d", padding=4)
        ax.set_ylabel("Number of images")
        ax.set_xlabel("")
        style(ax, f"{dataset} capture-group-aware data split", "y")
        save(fig, f"dataset_split_{token(dataset)}")


def experiment_charts() -> None:
    cols = [
        "dataset","architecture","method","distribution","alpha","client","seed",
        "accuracy","macro_f1","selected_round","status","confusion_matrix_path","notes"
    ]
    frame = load("experiment_results.csv", cols)
    if frame.empty:
        return

    for column in ["dataset","architecture","method","distribution","client"]:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    for column in ["accuracy","macro_f1"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame[frame["dataset"].ne("") & frame["method"].ne("") & frame["distribution"].ne("")]

    for (dataset, method, distribution), group in frame.groupby(["dataset","method","distribution"]):
        for metric in ["macro_f1","accuracy"]:
            rows = group.dropna(subset=[metric]).copy()
            if rows.empty:
                continue

            rows["label"] = rows.apply(
                lambda r: r["client"] if r["client"] and r["client"].lower() != "global"
                else (r["architecture"] or r["method"]),
                axis=1,
            )
            plot = rows.groupby("label", as_index=False)[metric].mean().sort_values(metric)

            fig, ax = plt.subplots(figsize=(9, max(4.5, .7*len(plot))))
            bars = ax.barh(plot["label"], plot[metric], color=GREEN)
            ax.bar_label(bars, labels=[f"{v:.4f}%" for v in plot[metric]], padding=5)
            ax.set_xlabel("Macro-F1 (%)" if metric == "macro_f1" else "Accuracy (%)")
            ax.set_ylabel("")
            style(ax, f"{dataset} — {method} — {distribution}")

            low = max(0, plot[metric].min()-1)
            high = min(100.5, plot[metric].max()+1)
            if high > low:
                ax.set_xlim(low, high)

            save(fig, f"experiment_{token(dataset)}_{token(method)}_{token(distribution)}_{metric}")


def client_charts() -> None:
    cols = ["dataset","distribution","alpha","client","images","capture_groups","class_entropy","source"]
    frame = load("client_statistics.csv", cols)
    if frame.empty:
        return

    for column in ["dataset","distribution","client"]:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    for column in ["alpha","images","capture_groups","class_entropy"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame[frame["dataset"].ne("") & frame["client"].ne("") & frame["alpha"].notna()]

    for (dataset, alpha), group in frame.groupby(["dataset","alpha"]):
        alpha_token = token(f"{alpha:g}")

        images = group.dropna(subset=["images"])
        if not images.empty:
            fig, ax = plt.subplots(figsize=(8,5))
            bars = ax.bar(images["client"], images["images"], color=COLORS[:len(images)])
            ax.bar_label(bars, fmt="%d", padding=4)
            ax.set_ylabel("Images")
            style(ax, f"{dataset} client image distribution (α={alpha:g})", "y")
            save(fig, f"clients_{token(dataset)}_alpha_{alpha_token}_images")

        entropy = group.dropna(subset=["class_entropy"])
        if not entropy.empty:
            fig, ax = plt.subplots(figsize=(8,5))
            bars = ax.bar(entropy["client"], entropy["class_entropy"], color=COLORS[:len(entropy)])
            ax.bar_label(bars, labels=[f"{v:.4f}" for v in entropy["class_entropy"]], padding=4)
            ax.set_ylabel("Class entropy (bits)")
            style(ax, f"{dataset} client class entropy (α={alpha:g})", "y")
            save(fig, f"clients_{token(dataset)}_alpha_{alpha_token}_entropy")


def main() -> None:
    print(f"Generating static charts in {OUT_DIR}")
    category_charts()
    split_charts()
    experiment_charts()
    client_charts()
    print("Done. Streamlit uses PNG files; PDF files are for reports/slides.")


if __name__ == "__main__":
    main()
