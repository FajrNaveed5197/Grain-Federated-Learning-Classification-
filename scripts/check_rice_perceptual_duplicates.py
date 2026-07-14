from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors

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

HASH_SIZE = 16
TOP_K = 5
HAMMING_THRESHOLD = 6


def dhash(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize(
            (HASH_SIZE + 1, HASH_SIZE),
            Image.Resampling.LANCZOS,
        )

        pixels = np.asarray(image, dtype=np.int16)

    return (pixels[:, 1:] > pixels[:, :-1]).astype(np.uint8).flatten()


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

            records.append(
                {
                    "split": split_name,
                    "class_name": row["class_name"],
                    "image_path": row["image_path"],
                    "hash": dhash(image_path),
                }
            )

hash_matrix = np.stack(
    [record["hash"] for record in records]
).astype(np.float32)

model = NearestNeighbors(
    n_neighbors=TOP_K + 1,
    metric="hamming",
    algorithm="brute",
    n_jobs=-1,
)

model.fit(hash_matrix)

distances, indices = model.kneighbors(hash_matrix)

candidates = []
seen_pairs = set()

for source_index, record in enumerate(records):
    for neighbor_position in range(1, TOP_K + 1):
        target_index = int(indices[source_index, neighbor_position])

        if source_index == target_index:
            continue

        first = records[source_index]
        second = records[target_index]

        if first["split"] == second["split"]:
            continue

        pair_key = tuple(
            sorted([
                first["image_path"],
                second["image_path"],
            ])
        )

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)

        normalized_distance = float(
            distances[source_index, neighbor_position]
        )

        bit_distance = int(
            round(normalized_distance * HASH_SIZE * HASH_SIZE)
        )

        if bit_distance <= HAMMING_THRESHOLD:
            candidates.append(
                {
                    "hamming_distance": bit_distance,
                    "first_split": first["split"],
                    "first_class": first["class_name"],
                    "first_path": first["image_path"],
                    "second_split": second["split"],
                    "second_class": second["class_name"],
                    "second_path": second["image_path"],
                }
            )

candidates.sort(
    key=lambda row: (
        row["hamming_distance"],
        row["first_path"],
        row["second_path"],
    )
)

summary = {
    "total_images": len(records),
    "hash_bits": HASH_SIZE * HASH_SIZE,
    "nearest_neighbors_checked": TOP_K,
    "hamming_threshold": HAMMING_THRESHOLD,
    "cross_split_near_duplicate_candidates": len(candidates),
}

with (
    OUTPUT_DIR / "perceptual_duplicate_summary.json"
).open("w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)

with (
    OUTPUT_DIR / "perceptual_duplicate_candidates.json"
).open("w", encoding="utf-8") as file:
    json.dump(candidates, file, indent=2)

print("=" * 72)
print("PERCEPTUAL NEAR-DUPLICATE CHECK")
print("=" * 72)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nFirst candidate pairs:")

for row in candidates[:20]:
    print(
        f"\nDistance: {row['hamming_distance']}\n"
        f"  {row['first_split']:10s} "
        f"{row['first_class']:8s} "
        f"{row['first_path']}\n"
        f"  {row['second_split']:10s} "
        f"{row['second_class']:8s} "
        f"{row['second_path']}"
    )

print(f"\nResults saved to: {OUTPUT_DIR}")
