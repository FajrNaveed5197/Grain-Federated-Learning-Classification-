from pathlib import Path
import csv
import re


# ============================================================
# FIND THE REAL REPOSITORY ROOT
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for folder in [start, *start.parents]:
        if (
            (folder / "results").exists()
            and (folder / "experiments").exists()
            and (folder / "src").exists()
        ):
            return folder

    raise RuntimeError(
        "Could not locate repository root."
    )


PROJECT_ROOT = find_project_root(SCRIPT_DIR)

# Search slightly broader than the Git repository because the raw
# datasets may be stored beside the repository rather than inside it.
SEARCH_ROOT = PROJECT_ROOT.parent

STREAMLIT_SAMPLE_DIR = (
    PROJECT_ROOT
    / "streamlit-ui"
    / "streamlit-ui"
    / "assets"
    / "dataset_samples"
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# FOLDERS THAT ARE DEFINITELY NOT RAW DATASET IMAGES
# ============================================================

IGNORE_FOLDER_NAMES = {
    ".git",
    ".venv",
    ".venv-paper",
    "__pycache__",
    "generated_charts",
    "confusion_matrices",
    "figures",
    "results",
    "paper",
    "project_audit",
    "streamlit-ui",
    "slurm",
    "docker",
    "apptainer",
}


def should_ignore(path: Path) -> bool:
    lower_parts = {
        part.lower()
        for part in path.parts
    }

    return any(
        ignored.lower() in lower_parts
        for ignored in IGNORE_FOLDER_NAMES
    )


# ============================================================
# DATASET DETECTION
# ============================================================

def detect_dataset(path: Path) -> str:
    text = str(path).lower()

    if "rice" in text:
        return "Rice"

    if "wheat" in text:
        return "Wheat"

    return "Unknown"


def clean_category(value: str) -> str:
    return re.sub(
        r"[_\-]+",
        " ",
        value,
    ).strip()


def detect_category(path: Path) -> str:
    """
    Uses the parent directory as category.

    Examples:

    dataset/Rice/0_NOR/image001.jpg
        -> 0 NOR

    wheat/Sprouted/img55.png
        -> Sprouted
    """

    return clean_category(path.parent.name)


# ============================================================
# MAIN SCAN
# ============================================================

def main():
    print()
    print("=" * 75)
    print("DATASET SAMPLE FINDER")
    print("=" * 75)

    print(f"\nRepository root:")
    print(PROJECT_ROOT)

    print(f"\nSearching for raw dataset images under:")
    print(SEARCH_ROOT)

    print(f"\nCorrect Streamlit sample directory:")
    print(STREAMLIT_SAMPLE_DIR)

    rows = []

    print("\nSearching...\n")

    for path in SEARCH_ROOT.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if should_ignore(path):
            continue

        dataset = detect_dataset(path)

        if dataset == "Unknown":
            continue

        category = detect_category(path)

        rows.append(
            {
                "dataset": dataset,
                "category": category,
                "filename": path.name,
                "full_path": str(path),
            }
        )

    if not rows:
        print("No raw Rice/Wheat images were found.")
        print()
        print(
            "This probably means the original dataset is stored "
            "somewhere outside the GrainClassification folder."
        )
        print()
        print(
            "In that case, tell me the folder where your original "
            "Rice/Wheat dataset is stored and we will scan that directory."
        )
        return

    rows.sort(
        key=lambda row: (
            row["dataset"],
            row["category"],
            row["filename"],
        )
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    grouped = {}

    for row in rows:
        key = (
            row["dataset"],
            row["category"],
        )

        grouped.setdefault(key, [])
        grouped[key].append(row)

    print("\nFOUND DATASET CATEGORIES\n")

    for (dataset, category), items in grouped.items():

        print("=" * 75)
        print(
            f"{dataset}  →  {category}  "
            f"({len(items):,} images)"
        )
        print("=" * 75)

        # Print first five filenames only
        for row in items[:5]:
            print(f"  {row['filename']}")

        if len(items) > 5:
            print(
                f"  ... plus {len(items) - 5:,} more"
            )

        print()

    # ========================================================
    # SAVE FULL INVENTORY
    # ========================================================

    csv_path = PROJECT_ROOT / "dataset_image_inventory.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset",
                "category",
                "filename",
                "full_path",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("=" * 75)
    print("SUMMARY")
    print("=" * 75)

    print(
        f"Total dataset images found: "
        f"{len(rows):,}"
    )

    print(
        f"Dataset/categories found: "
        f"{len(grouped):,}"
    )

    print("\nInventory saved to:")
    print(csv_path)

    print("\nCorrect sample-image destination:")
    print(STREAMLIT_SAMPLE_DIR)


if __name__ == "__main__":
    main()