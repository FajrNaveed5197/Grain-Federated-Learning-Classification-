from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import pandas as pd


SEED = 42
TARGET_VALIDATION_IMAGE_FRACTION = 0.10
TARGET_VALIDATION_GROUP_FRACTION = 0.15
NUM_SEARCH_TRIALS = 5000

MANIFEST_ROOT = Path(
    "/scratch/project_2019765/grain_research/manifests"
)

OUTPUT_ROOT = Path(
    "/scratch/project_2019765/fnaveed/datasets/"
    "wheat_grouped_v1/grouped_split_v2"
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


def choose_validation_groups(
    class_dataframe: pd.DataFrame,
    class_index: int,
) -> set[str]:
    group_sizes = (
        class_dataframe.groupby("capture_group")
        .size()
        .sort_index()
    )

    groups = list(group_sizes.index)
    num_groups = len(groups)
    total_images = int(group_sizes.sum())

    target_num_groups = max(
        3,
        round(
            num_groups
            * TARGET_VALIDATION_GROUP_FRACTION
        ),
    )

    target_num_groups = min(
        target_num_groups,
        num_groups - 1,
    )

    target_images = round(
        total_images
        * TARGET_VALIDATION_IMAGE_FRACTION
    )

    best_groups = None
    best_score = float("inf")

    rng = random.Random(
        SEED + class_index * 10000
    )

    for _ in range(NUM_SEARCH_TRIALS):
        candidate_groups = set(
            rng.sample(
                groups,
                target_num_groups,
            )
        )

        candidate_images = int(
            group_sizes.loc[
                list(candidate_groups)
            ].sum()
        )

        image_fraction = (
            candidate_images / total_images
        )

        image_error = abs(
            image_fraction
            - TARGET_VALIDATION_IMAGE_FRACTION
        )

        # Penalize extreme deviation strongly.
        score = image_error

        if image_fraction < 0.05:
            score += 1.0

        if image_fraction > 0.20:
            score += 1.0

        if score < best_score:
            best_score = score
            best_groups = candidate_groups

    if best_groups is None:
        raise RuntimeError(
            f"Unable to split class: "
            f"{CLASS_NAMES[class_index]}"
        )

    return best_groups


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

    development = (
        development
        .drop_duplicates(subset=["path"])
        .reset_index(drop=True)
    )

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

    development = development[
        ~suspicious
    ].reset_index(drop=True)

    unknown_labels = sorted(
        set(development["label"])
        - set(CLASS_NAMES)
    )

    if unknown_labels:
        raise ValueError(
            f"Unexpected labels: {unknown_labels}"
        )

    mixed_group_counts = (
        development.groupby("capture_group")["label"]
        .nunique()
    )

    mixed_groups = mixed_group_counts[
        mixed_group_counts > 1
    ]

    if len(mixed_groups):
        raise ValueError(
            f"Mixed-label groups found: "
            f"{len(mixed_groups)}"
        )

    validation_groups: set[str] = set()
    allocation_rows = []

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):
        class_df = development[
            development["label"] == class_name
        ].copy()

        selected_groups = (
            choose_validation_groups(
                class_dataframe=class_df,
                class_index=class_index,
            )
        )

        validation_groups.update(
            selected_groups
        )

        validation_images = int(
            class_df["capture_group"]
            .isin(selected_groups)
            .sum()
        )

        total_images = len(class_df)
        total_groups = (
            class_df["capture_group"].nunique()
        )

        allocation_rows.append(
            {
                "label": class_name,
                "total_images": total_images,
                "total_groups": total_groups,
                "validation_images": (
                    validation_images
                ),
                "validation_groups": len(
                    selected_groups
                ),
                "validation_image_fraction": round(
                    validation_images
                    / total_images,
                    6,
                ),
                "validation_group_fraction": round(
                    len(selected_groups)
                    / total_groups,
                    6,
                ),
            }
        )

    validation = development[
        development["capture_group"].isin(
            validation_groups
        )
    ].copy()

    train = development[
        ~development["capture_group"].isin(
            validation_groups
        )
    ].copy()

    train["split"] = "train"
    validation["split"] = "validation"

    train = train.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    validation = validation.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    train_groups = set(train["capture_group"])
    validation_group_set = set(
        validation["capture_group"]
    )

    overlap = (
        train_groups
        & validation_group_set
    )

    if overlap:
        raise RuntimeError(
            f"Group overlap found: {len(overlap)}"
        )

    external_test = pd.read_csv(
        MANIFEST_ROOT / "test.csv"
    )

    external_test_07 = pd.read_csv(
        MANIFEST_ROOT / "test_07.csv"
    )

    for dataframe, split_name in [
        (external_test, "test"),
        (external_test_07, "test_07"),
    ]:
        dataframe["capture_group"] = (
            dataframe["path"]
            .astype(str)
            .map(derive_capture_group)
        )

        dataframe["split"] = split_name

    train.to_csv(
        OUTPUT_ROOT / "train.csv",
        index=False,
    )

    validation.to_csv(
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

    split_rows = []

    for split_name, dataframe in [
        ("train", train),
        ("validation", validation),
        ("test", external_test),
        ("test_07", external_test_07),
    ]:
        for class_name in CLASS_NAMES:
            class_df = dataframe[
                dataframe["label"] == class_name
            ]

            split_rows.append(
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

    pd.DataFrame(split_rows).to_csv(
        OUTPUT_ROOT / "split_summary.csv",
        index=False,
    )

    metadata = {
        "version": 2,
        "seed": SEED,
        "target_validation_image_fraction": (
            TARGET_VALIDATION_IMAGE_FRACTION
        ),
        "target_validation_group_fraction": (
            TARGET_VALIDATION_GROUP_FRACTION
        ),
        "search_trials_per_class": (
            NUM_SEARCH_TRIALS
        ),
        "classes": CLASS_NAMES,
        "split_sizes": {
            "train": len(train),
            "validation": len(validation),
            "test": len(external_test),
            "test_07": len(external_test_07),
        },
        "capture_groups": {
            "train": int(
                train["capture_group"].nunique()
            ),
            "validation": int(
                validation[
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
        "actual_validation_image_fraction": round(
            len(validation)
            / len(development),
            6,
        ),
        "actual_validation_group_fraction": round(
            validation[
                "capture_group"
            ].nunique()
            / development[
                "capture_group"
            ].nunique(),
            6,
        ),
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

    print(
        "===== WHEAT GROUPED SPLIT V2 ====="
    )

    print(
        f"Train: {len(train)} images, "
        f"{train['capture_group'].nunique()} groups"
    )

    print(
        f"Validation: {len(validation)} images, "
        f"{validation['capture_group'].nunique()} groups"
    )

    print(
        "Validation image fraction:",
        f"{len(validation) / len(development):.4f}",
    )

    validation_group_fraction = (
        validation["capture_group"].nunique()
        / development["capture_group"].nunique()
    )

    print(
        "Validation group fraction:",
        f"{validation_group_fraction:.4f}",
    )

    print(
        "Train-validation overlap:",
        len(overlap),
    )

    print(
        f"Saved to: {OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()
