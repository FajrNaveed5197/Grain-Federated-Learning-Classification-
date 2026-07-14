from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_NAMES = [
    "0_NOR",
    "1_F&S",
    "2_SD",
    "3_MY",
    "4_AP",
    "5_BN",
    "6_UN",
    "7_IM",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-client-images", type=int, default=3000)
    return parser.parse_args()


def validate_source(df):
    required = {
        "image_path",
        "mask_path",
        "class_name",
        "capture_group",
        "original_split",
        "split",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if not (df["split"] == "train").all():
        raise ValueError("Manifest contains non-training rows.")

    if df["image_path"].duplicated().any():
        raise ValueError("Duplicate image paths found.")

    unknown = set(df["class_name"]) - set(CLASS_NAMES)

    if unknown:
        raise ValueError(f"Unknown classes: {sorted(unknown)}")


def build_group_table(df):
    counts = pd.crosstab(
        df["capture_group"],
        df["class_name"],
    ).reindex(columns=CLASS_NAMES, fill_value=0)

    result = counts.copy()
    result["total"] = counts.sum(axis=1)
    result["dominant_class"] = counts.idxmax(axis=1)
    result["dominant_count"] = counts.max(axis=1)

    return result.reset_index()


def assign_iid(group_table, num_clients, seed):
    """
    Balanced group-aware IID assignment.

    Groups are divided separately for each dominant class. Within each
    dominant-class bucket, larger groups are assigned first to the client
    with the smallest current total and smallest count for that class.
    """
    rng = np.random.default_rng(seed)

    assignments = {}
    total_loads = np.zeros(num_clients, dtype=np.int64)
    class_loads = np.zeros(
        (num_clients, len(CLASS_NAMES)),
        dtype=np.int64,
    )

    class_to_id = {
        name: index for index, name in enumerate(CLASS_NAMES)
    }

    for dominant_class in CLASS_NAMES:
        subset = group_table[
            group_table["dominant_class"] == dominant_class
        ].copy()

        if subset.empty:
            continue

        subset["_random"] = rng.random(len(subset))

        subset = subset.sort_values(
            ["total", "_random"],
            ascending=[False, True],
        )

        dominant_id = class_to_id[dominant_class]

        for _, row in subset.iterrows():
            candidates = sorted(
                range(num_clients),
                key=lambda client_id: (
                    class_loads[client_id, dominant_id],
                    total_loads[client_id],
                    client_id,
                ),
            )

            selected = candidates[0]

            assignments[row["capture_group"]] = selected

            group_vector = row[CLASS_NAMES].to_numpy(
                dtype=np.int64
            )

            class_loads[selected] += group_vector
            total_loads[selected] += int(row["total"])

    return assignments


def generate_noniid_targets(
    global_counts,
    num_clients,
    alpha,
    seed,
):
    rng = np.random.default_rng(seed)

    targets = np.zeros(
        (num_clients, len(CLASS_NAMES)),
        dtype=np.float64,
    )

    proportions = np.zeros_like(targets)

    for class_id, class_total in enumerate(global_counts):
        p = rng.dirichlet(
            np.full(num_clients, alpha, dtype=np.float64)
        )

        proportions[:, class_id] = p
        targets[:, class_id] = p * class_total

    return targets, proportions


def assign_noniid(
    group_table,
    num_clients,
    alpha,
    seed,
    min_client_images,
):
    """
    Group-aware non-IID partition.

    First gives each client enough groups to avoid tiny clients, then assigns
    remaining groups based on Dirichlet class targets.
    """
    rng = np.random.default_rng(seed)

    global_counts = (
        group_table[CLASS_NAMES]
        .sum(axis=0)
        .to_numpy(dtype=np.float64)
    )

    targets, proportions = generate_noniid_targets(
        global_counts,
        num_clients,
        alpha,
        seed,
    )

    client_counts = np.zeros_like(targets)
    client_totals = np.zeros(num_clients, dtype=np.int64)

    assignments = {}

    working = group_table.copy()
    working["_random"] = rng.random(len(working))

    working = working.sort_values(
        ["total", "_random"],
        ascending=[False, True],
    )

    remaining_indices = list(working.index)

    # Stage 1: enforce a reasonable minimum client size.
    while (
        client_totals.min() < min_client_images
        and remaining_indices
    ):
        smallest_client = int(np.argmin(client_totals))

        best_index = None
        best_score = None

        for index in remaining_indices[:500]:
            row = working.loc[index]
            vector = row[CLASS_NAMES].to_numpy(dtype=np.float64)

            deficit = np.maximum(
                targets[smallest_client] -
                client_counts[smallest_client],
                0.0,
            )

            score = float(
                np.dot(vector, deficit) /
                max(vector.sum(), 1.0)
            )

            if best_score is None or score > best_score:
                best_score = score
                best_index = index

        if best_index is None:
            best_index = remaining_indices[0]

        row = working.loc[best_index]
        vector = row[CLASS_NAMES].to_numpy(dtype=np.float64)

        assignments[row["capture_group"]] = smallest_client
        client_counts[smallest_client] += vector
        client_totals[smallest_client] += int(row["total"])

        remaining_indices.remove(best_index)

    # Stage 2: approximate Dirichlet class targets.
    for index in remaining_indices:
        row = working.loc[index]
        vector = row[CLASS_NAMES].to_numpy(dtype=np.float64)

        scores = []

        for client_id in range(num_clients):
            projected = client_counts[client_id] + vector

            denominator = np.maximum(
                targets[client_id],
                1.0,
            )

            class_error = np.mean(
                ((projected - targets[client_id]) /
                 denominator) ** 2
            )

            size_penalty = (
                client_totals[client_id] /
                max(group_table["total"].sum(), 1)
            )

            scores.append(
                float(class_error + 0.02 * size_penalty)
            )

        selected = int(np.argmin(scores))

        assignments[row["capture_group"]] = selected
        client_counts[selected] += vector
        client_totals[selected] += int(row["total"])

    return assignments, proportions


def write_partition(
    source,
    assignments,
    output_dir,
    partition_name,
    source_manifest,
    num_clients,
    seed,
    alpha=None,
    dirichlet_proportions=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    assigned = source.copy()

    assigned["client_id"] = assigned[
        "capture_group"
    ].map(assignments)

    if assigned["client_id"].isna().any():
        raise RuntimeError("Unassigned rows found.")

    assigned["client_id"] = assigned["client_id"].astype(int)

    group_client_counts = (
        assigned.groupby("capture_group")["client_id"].nunique()
    )

    if group_client_counts.max() != 1:
        raise RuntimeError("Capture groups were split.")

    if len(assigned) != len(source):
        raise RuntimeError("Row count changed.")

    if assigned["image_path"].nunique() != len(source):
        raise RuntimeError("Images are missing or duplicated.")

    summary = pd.crosstab(
        assigned["client_id"],
        assigned["class_name"],
    ).reindex(
        index=range(num_clients),
        columns=CLASS_NAMES,
        fill_value=0,
    )

    summary["total"] = summary.sum(axis=1)

    group_summary = (
        assigned[
            ["capture_group", "client_id"]
        ]
        .drop_duplicates()
        .groupby("client_id")
        .size()
        .reindex(range(num_clients), fill_value=0)
    )

    for client_id in range(num_clients):
        client_df = assigned[
            assigned["client_id"] == client_id
        ].copy()

        client_df = client_df.sort_values(
            ["capture_group", "class_name", "image_path"]
        )

        client_df.to_csv(
            output_dir / f"client_{client_id}.csv",
            index=False,
        )

    assigned.to_csv(
        output_dir / "all_clients.csv",
        index=False,
    )

    summary.to_csv(
        output_dir / "client_class_image_counts.csv"
    )

    proportions = summary[CLASS_NAMES].div(
        summary["total"],
        axis=0,
    )

    proportions.to_csv(
        output_dir / "client_class_proportions.csv"
    )

    group_summary.rename("capture_groups").to_csv(
        output_dir / "client_capture_group_counts.csv"
    )

    metadata = {
        "partition_name": partition_name,
        "source_manifest": str(source_manifest),
        "num_clients": num_clients,
        "num_images": int(len(source)),
        "num_capture_groups": int(
            source["capture_group"].nunique()
        ),
        "seed": seed,
        "dirichlet_alpha": alpha,
        "class_order": CLASS_NAMES,
        "capture_groups_split": 0,
        "client_image_counts": {
            str(client_id): int(summary.loc[client_id, "total"])
            for client_id in range(num_clients)
        },
    }

    with (
        output_dir / "partition_metadata.json"
    ).open("w") as handle:
        json.dump(metadata, handle, indent=2)

    if dirichlet_proportions is not None:
        pd.DataFrame(
            dirichlet_proportions,
            index=[
                f"client_{client_id}"
                for client_id in range(num_clients)
            ],
            columns=CLASS_NAMES,
        ).to_csv(
            output_dir / "dirichlet_target_proportions.csv"
        )

    print(f"\n===== {partition_name.upper()} =====")
    print(summary.to_string())

    print("\nCapture groups:")
    print(group_summary.to_string())

    print("\nClass proportions:")
    print(proportions.round(4).to_string())

    print("\nValidation:")
    print("Every image assigned exactly once: PASS")
    print("Capture groups split across clients: 0")


def main():
    args = parse_args()

    source = pd.read_csv(args.train_manifest)
    validate_source(source)

    group_table = build_group_table(source)

    print("Images:", len(source))
    print("Capture groups:", len(group_table))

    iid_assignments = assign_iid(
        group_table,
        args.num_clients,
        args.seed,
    )

    iid_dir = (
        args.output_root /
        f"iid_{args.num_clients}clients_seed{args.seed}_v2"
    )

    write_partition(
        source,
        iid_assignments,
        iid_dir,
        "iid",
        args.train_manifest,
        args.num_clients,
        args.seed,
    )

    noniid_assignments, targets = assign_noniid(
        group_table,
        args.num_clients,
        args.alpha,
        args.seed,
        args.min_client_images,
    )

    alpha_text = str(args.alpha).replace(".", "p")

    noniid_dir = (
        args.output_root /
        (
            f"noniid_{args.num_clients}clients_"
            f"alpha{alpha_text}_seed{args.seed}_v2"
        )
    )

    write_partition(
        source,
        noniid_assignments,
        noniid_dir,
        "non_iid",
        args.train_manifest,
        args.num_clients,
        args.seed,
        args.alpha,
        targets,
    )


if __name__ == "__main__":
    main()
