from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path(
    "/scratch/project_2019765/fnaveed/datasets/rice_grouped"
)

SPLIT_DIR = DATA_ROOT / "grouped_split"

MANIFESTS = {
    "train": SPLIT_DIR / "train.csv",
    "validation": SPLIT_DIR / "validation.csv",
    "test": SPLIT_DIR / "test.csv",
}

OUTPUT_DIR = Path(
    "/scratch/project_2019765/fnaveed/results/"
    "rice_dataset_validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


records = []

for split_name, manifest_path in MANIFESTS.items():
    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            image_path = DATA_ROOT / row["image_path"]

            if not image_path.exists():
                raise FileNotFoundError(image_path)

            records.append(
                {
                    "split": split_name,
                    "image_path": row["image_path"],
                    "class_name": row["class_name"],
                    "sha256": calculate_sha256(image_path),
                }
            )

hash_groups = defaultdict(list)

for record in records:
    hash_groups[record["sha256"]].append(record)

duplicate_groups = {
    file_hash: rows
    for file_hash, rows in hash_groups.items()
    if len(rows) > 1
}

cross_split_duplicates = {}
same_split_duplicates = {}

for file_hash, rows in duplicate_groups.items():
    involved_splits = {
        row["split"]
        for row in rows
    }

    if len(involved_splits) > 1:
        cross_split_duplicates[file_hash] = rows
    else:
        same_split_duplicates[file_hash] = rows

summary = {
    "total_images": len(records),
    "unique_hashes": len(hash_groups),
    "duplicate_hash_groups": len(duplicate_groups),
    "cross_split_duplicate_groups": len(
        cross_split_duplicates
    ),
    "same_split_duplicate_groups": len(
        same_split_duplicates
    ),
    "cross_split_duplicate_images": sum(
        len(rows)
        for rows in cross_split_duplicates.values()
    ),
}

with (
    OUTPUT_DIR / "exact_duplicate_summary.json"
).open("w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)

with (
    OUTPUT_DIR / "cross_split_exact_duplicates.json"
).open("w", encoding="utf-8") as file:
    json.dump(
        cross_split_duplicates,
        file,
        indent=2,
    )

print("=" * 72)
print("EXACT DUPLICATE CHECK")
print("=" * 72)

for key, value in summary.items():
    print(f"{key}: {value}")

if cross_split_duplicates:
    print("\nWARNING: Exact duplicates exist across splits.")

    for index, (file_hash, rows) in enumerate(
        cross_split_duplicates.items()
    ):
        print(f"\nHash: {file_hash}")

        for row in rows:
            print(
                f"  {row['split']:10s} "
                f"{row['class_name']:8s} "
                f"{row['image_path']}"
            )

        if index >= 9:
            break
else:
    print("\nPASSED: No exact duplicate files across splits.")

print(f"\nResults saved to: {OUTPUT_DIR}")
