from pathlib import Path
import re

import pandas as pd


MANIFEST_ROOT = Path(
    "/scratch/project_2019765/grain_research/manifests"
)

OUTPUT_ROOT = Path(
    "/scratch/project_2019765/fnaveed/datasets/"
    "wheat_grouped_v1/audit"
)

SPLITS = [
    "train",
    "validation",
    "test",
    "test_07",
]


def derive_capture_group(path_value: str) -> str:
    path = Path(path_value)
    stem = path.stem

    # Grain images generally end with the segmented-object number.
    match = re.match(r"^(.*)-(\d+)$", stem)

    if match:
        capture_name = match.group(1)
    else:
        match = re.match(r"^(.*?)[_-](\d+)$", stem)

        if match:
            capture_name = match.group(1)
        else:
            capture_name = stem

    # Include acquisition parent directories to avoid treating
    # similarly named captures from separate batches as identical.
    parts = path.parts

    try:
        dataset_index = parts.index("compressed_images_wheat")
        relative_parent = Path(
            *parts[dataset_index + 1:-1]
        )
    except ValueError:
        relative_parent = path.parent

    return str(relative_parent / capture_name)


def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = {}

    for split in SPLITS:
        manifest_path = (
            MANIFEST_ROOT / f"{split}.csv"
        )

        dataframe = pd.read_csv(manifest_path)

        dataframe["split"] = split
        dataframe["capture_group"] = (
            dataframe["path"]
            .astype(str)
            .map(derive_capture_group)
        )

        frames[split] = dataframe

        print(f"\n===== {split.upper()} =====")
        print("Images:", len(dataframe))
        print(
            "Capture groups:",
            dataframe["capture_group"].nunique(),
        )

        group_sizes = (
            dataframe.groupby(
                ["capture_group", "label"]
            )
            .size()
            .reset_index(name="num_images")
        )

        group_sizes.to_csv(
            OUTPUT_ROOT
            / f"{split}_capture_groups.csv",
            index=False,
        )

        print("\nGroups per class:")

        print(
            dataframe.groupby("label")[
                "capture_group"
            ]
            .nunique()
            .sort_index()
            .to_string()
        )

    print("\n===== SPLIT OVERLAP =====")

    overlap_rows = []

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            left_groups = set(
                frames[left]["capture_group"]
            )

            right_groups = set(
                frames[right]["capture_group"]
            )

            overlap = left_groups & right_groups

            row = {
                "left_split": left,
                "right_split": right,
                "overlapping_capture_groups": len(
                    overlap
                ),
            }

            overlap_rows.append(row)

            print(
                f"{left} vs {right}: "
                f"{len(overlap)}"
            )

            if overlap:
                pd.DataFrame(
                    {
                        "capture_group": sorted(
                            overlap
                        )
                    }
                ).to_csv(
                    OUTPUT_ROOT
                    / (
                        f"overlap_{left}_"
                        f"{right}.csv"
                    ),
                    index=False,
                )

    pd.DataFrame(overlap_rows).to_csv(
        OUTPUT_ROOT / "split_overlap_summary.csv",
        index=False,
    )

    combined = pd.concat(
        frames.values(),
        ignore_index=True,
    )

    combined.to_csv(
        OUTPUT_ROOT
        / "all_existing_manifests_with_groups.csv",
        index=False,
    )

    mixed_groups = (
        combined.groupby("capture_group")["label"]
        .nunique()
    )

    mixed_groups = mixed_groups[
        mixed_groups > 1
    ]

    print(
        "\nCapture groups containing "
        f"multiple labels: {len(mixed_groups)}"
    )

    if len(mixed_groups):
        mixed_df = (
            combined[
                combined["capture_group"].isin(
                    mixed_groups.index
                )
            ][
                [
                    "capture_group",
                    "label",
                    "split",
                    "path",
                ]
            ]
            .sort_values(
                [
                    "capture_group",
                    "label",
                    "path",
                ]
            )
        )

        mixed_df.to_csv(
            OUTPUT_ROOT
            / "mixed_label_capture_groups.csv",
            index=False,
        )

    print(
        f"\nAudit saved to: {OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()
