from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pandas as pd


SEED = 42
VALIDATION_FRACTION = 0.10

MANIFEST_ROOT = Path(
    "/scratch/project_2019765/grain_research/manifests"
)

OUTPUT_ROOT = Path(
    "/scratch/project_2019765/fnaveed/datasets/"
    "wheat_grouped_v1/grouped_split"
)

CLASS_NAMES = [
    "Black Germ",
    "Broken",
    "Fusarium",
    "Insect",
    "Moldy",
    "Sound",
    "Spotted",
    "Sprouted",
]


def derive_capture_group(path_value: str) -> str:
    path = Path(path_value)
    stem = path.stem

    match = re.match(r"^(.*)-(\d+)$", stem)

    if match:
        capture_name = match.group(1)
    else:
        match = re.match(r"^(.*?)[_-](\d+)$", stem)

        if match:
            capture_name = match.group(1)
        else:
            capture_name = stem

    parts = path.parts

    try:
        dataset_index = parts.index(
            "compressed_images_wheat"
        )

        relative_parent = Path(
            *parts[dataset_index + 1:-1]
        )
    except ValueError:
        relative_parent = path.parent

    return str(relative_parent / capture_name)


def assign_class_groups(
    class_dataframe: pd.DataFrame,
    target_validation_images: int,
    seed: int,
) -> tuple[set[str], set[str]]:
    group_sizes = (
        class_dataframe.groupby("capture_group")
        .size()
        .reset_index(name="num_images")
    )

    records = group_sizes.to_dict("records")

    random.Random(seed).shuffle(records)

    # Place larger groups first, with randomized tie ordering.
    records.sort(
        key=lambda record: record["num_images"],
        reverse=True,
    )

    validation_groups: set[str] = set()
    training_groups: set[str] = set()

    validation_images = 0

    for record in records:
        group = record["capture_group"]
        size = int(record["num_images"])

        distance_if_validation = abs(
            target_validation_images
            - (validation_images + size)
        )

        distance_if_training = abs(
            target_validation_images
            - validation_images
        )

        if (
            validation_images < target_validation_images
            and distance_if_validation <= distance_if_training
        ):
            validation_groups.add(group)
            validation_images += size
        else:
            training_groups.add(group)

    # Guarantee that both partitions contain at least one group.
    if not validation_groups:
        candidate = min(
            records,
            key=lambda record: record["num_images"],
        )

        group = candidate["capture_group"]
        training_groups.discard(group)
        validation_groups.add(group)

    if not training_groups:
        candidate = max(
            records,
            key=lambda record: record["num_images"],
        )

        group = candidate["capture_group"]
        validation_groups.discard(group)
        training_groups.add(group)

    return training_groups, validation_groups


def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    old_train = pd.read_csv(
        MANIFEST_ROOT / "train.csv"
    )

    old_validation = pd.read_csv(
        MANIFEST_ROOT / "validation.csv"
    )

    development = pd.concat(
        [old_train, old_validation],
        ignore_index=True,
    )

    development = development.drop_duplicates(
        subset=["path"]
    ).reset_index(drop=True)

    development["capture_group"] = (
        development["path"]
        .astype(str)
        .map(derive_capture_group)
    )

    suspicious = (
        development["path"]
        .astype(str)
        .str.lower()
        .str.contains(
            r"donotuse|do_not|do-not|/duplicates/",
            regex=True,
        )
    )

    if suspicious.any():
        print(
            "Removing suspicious paths:",
            int(suspicious.sum()),
        )

        development = development[
            ~suspicious
        ].reset_index(drop=True)

    unknown_labels = sorted(
        set(development["label"]) - set(CLASS_NAMES)
    )

    if unknown_labels:
        raise ValueError(
            f"Unexpected labels: {unknown_labels}"
        )

    group_label_counts = (
        development.groupby("capture_group")["label"]
        .nunique()
    )

    mixed_groups = group_label_counts[
        group_label_counts > 1
    ]

    if len(mixed_groups):
        raise ValueError(
            "Mixed-label capture groups found: "
            f"{len(mixed_groups)}"
        )

    all_training_groups: set[str] = set()
    all_validation_groups: set[str] = set()

    allocation_rows = []

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):
        class_df = development[
            development["label"] == class_name
        ].copy()

        total_images = len(class_df)

        target_validation_images = round(
            total_images * VALIDATION_FRACTION
        )

        train_groups, validation_groups = (
            assign_class_groups(
                class_dataframe=class_df,
                target_validation_images=(
                    target_validation_images
                ),
                seed=SEED + class_index * 1000,
            )
        )

        all_training_groups.update(train_groups)
        all_validation_groups.update(
            validation_groups
        )

        actual_validation_images = int(
            class_df["capture_group"]
            .isin(validation_groups)
            .sum()
        )

        allocation_rows.append(
            {
                "label": class_name,
                "total_images": total_images,
                "capture_groups": (
                    class_df["capture_group"].nunique()
                ),
                "target_validation_images": (
                    target_validation_images
                ),
                "actual_validation_images": (
                    actual_validation_images
                ),
                "actual_training_images": (
                    total_images
                    - actual_validation_images
                ),
                "validation_fraction": round(
                    actual_validation_images
                    / total_images,
                    6,
                ),
            }
        )

    overlap = (
        all_training_groups
        & all_validation_groups
    )

    if overlap:
        raise RuntimeError(
            f"Group overlap detected: {len(overlap)}"
        )

    new_train = development[
        development["capture_group"].isin(
            all_training_groups
        )
    ].copy()

    new_validation = development[
        development["capture_group"].isin(
            all_validation_groups
        )
    ].copy()

    new_train["split"] = "train"
    new_validation["split"] = "validation"

    new_train = new_train.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    new_validation = new_validation.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    external_test = pd.read_csv(
        MANIFEST_ROOT / "test.csv"
    )

    external_test_07 = pd.read_csv(
        MANIFEST_ROOT / "test_07.csv"
    )

    external_test["capture_group"] = (
        external_test["path"]
        .astype(str)
        .map(derive_capture_group)
    )

    external_test_07["capture_group"] = (
        external_test_07["path"]
        .astype(str)
        .map(derive_capture_group)
    )

    external_test["split"] = "test"
    external_test_07["split"] = "test_07"

    new_train.to_csv(
        OUTPUT_ROOT / "train.csv",
        index=False,
    )

    new_validation.to_csv(
        OUTPUT_ROOT / "validation.csv",
        index=False,
    )

    external_test.to_csv(
        OUTPUT_ROOT / "test.csv",
        index=False,
    )

    external_test_07.to_csv(
        OUTPUT_ROOT / "test_07.csv",
        index=False,
    )

    allocation_df = pd.DataFrame(
        allocation_rows
    )

    allocation_df.to_csv(
        OUTPUT_ROOT / "class_allocation.csv",
        index=False,
    )

    split_summary = []

    for split_name, dataframe in [
        ("train", new_train),
        ("validation", new_validation),
        ("test", external_test),
        ("test_07", external_test_07),
    ]:
        for class_name in CLASS_NAMES:
            class_df = dataframe[
                dataframe["label"] == class_name
            ]

            split_summary.append(
                {
                    "split": split_name,
                    "label": class_name,
                    "images": len(class_df),
                    "capture_groups": (
                        class_df["capture_group"]
                        .nunique()
                    ),
                }
            )

    pd.DataFrame(split_summary).to_csv(
        OUTPUT_ROOT / "split_summary.csv",
        index=False,
    )

    metadata = {
        "seed": SEED,
        "validation_fraction": (
            VALIDATION_FRACTION
        ),
        "classes": CLASS_NAMES,
        "source_manifests": [
            str(MANIFEST_ROOT / "train.csv"),
            str(
                MANIFEST_ROOT
                / "validation.csv"
            ),
        ],
        "external_test_manifests": [
            str(MANIFEST_ROOT / "test.csv"),
            str(MANIFEST_ROOT / "test_07.csv"),
        ],
        "split_sizes": {
            "train": len(new_train),
            "validation": len(new_validation),
            "test": len(external_test),
            "test_07": len(external_test_07),
        },
        "capture_groups": {
            "train": int(
                new_train["capture_group"].nunique()
            ),
            "validation": int(
                new_validation[
                    "capture_group"
                ].nunique()
            ),
            "test": int(
                external_test[
                    "capture_group"
                ].nunique()
            ),
            "test_07": int(
                external_test_07[
                    "capture_group"
                ].nunique()
            ),
        },
        "train_validation_group_overlap": 0,
    }

    with (
        OUTPUT_ROOT / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    print("===== NEW WHEAT GROUPED SPLIT =====")
    print(
        f"Train: {len(new_train)} images, "
        f"{new_train['capture_group'].nunique()} groups"
    )

    print(
        f"Validation: {len(new_validation)} images, "
        f"{new_validation['capture_group'].nunique()} groups"
    )

    print(
        "Train-validation group overlap:",
        len(
            set(new_train["capture_group"])
            & set(
                new_validation["capture_group"]
            )
        ),
    )

    print(
        f"Saved to: {OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()
