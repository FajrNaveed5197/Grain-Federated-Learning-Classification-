from __future__ import annotations

import argparse
import json
import random
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create capture-group-safe IID and non-IID federated "
            "client manifests for the rice dataset."
        )
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_source(df: pd.DataFrame) -> None:
    required = {
        "image_path",
        "mask_path",
        "class_name",
        "capture_group",
        "original_split",
        "split",
    }

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    unknown_classes = sorted(
        set(df["class_name"].unique()) - set(CLASS_NAMES)
    )
    if unknown_classes:
        raise ValueError(
            f"Unexpected classes in source manifest: {unknown_classes}"
        )

    if not (df["split"] == "train").all():
        invalid = df.loc[df["split"] != "train", "split"].value_counts()
        raise ValueError(
            "Source manifest contains non-training rows:\n"
            f"{invalid.to_string()}"
        )

    duplicate_paths = int(df["image_path"].duplicated().sum())
    if duplicate_paths:
        raise ValueError(
            f"Source contains {duplicate_paths} duplicate image paths."
        )

    if df["capture_group"].isna().any():
        raise ValueError("Source contains missing capture_group values.")

    if df["class_name"].isna().any():
        raise ValueError("Source contains missing class_name values.")


def build_group_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    group_counts = pd.crosstab(
        df["capture_group"],
        df["class_name"],
    ).reindex(columns=CLASS_NAMES, fill_value=0)

    group_counts = group_counts.sort_index()
    matrix = group_counts.to_numpy(dtype=np.int64)

    return group_counts, matrix


def normalized_assignment_score(
    projected_counts: np.ndarray,
    target_counts: np.ndarray,
    projected_total: float,
    target_total: float,
) -> float:
    class_denominator = np.maximum(target_counts, 1.0)

    class_error = np.mean(
        ((projected_counts - target_counts) / class_denominator) ** 2
    )

    total_error = (
        (projected_total - target_total) / max(target_total, 1.0)
    ) ** 2

    return float(class_error + 0.20 * total_error)


def assign_groups_iid(
    group_names: list[str],
    group_matrix: np.ndarray,
    num_clients: int,
    seed: int,
) -> dict[str, int]:
    """
    Assign complete capture groups while making each client's full
    class distribution as similar as possible.
    """
    rng = np.random.default_rng(seed)

    global_class_counts = group_matrix.sum(axis=0).astype(np.float64)
    global_total = float(group_matrix.sum())

    target_class_counts = np.tile(
        global_class_counts / num_clients,
        (num_clients, 1),
    )
    target_totals = np.full(
        num_clients,
        global_total / num_clients,
        dtype=np.float64,
    )

    client_class_counts = np.zeros(
        (num_clients, len(CLASS_NAMES)),
        dtype=np.float64,
    )
    client_totals = np.zeros(num_clients, dtype=np.float64)

    group_sizes = group_matrix.sum(axis=1)
    group_diversity = (group_matrix > 0).sum(axis=1)

    random_tie_break = rng.random(len(group_names))

    order = sorted(
        range(len(group_names)),
        key=lambda index: (
            -int(group_sizes[index]),
            -int(group_diversity[index]),
            float(random_tie_break[index]),
        ),
    )

    assignments: dict[str, int] = {}

    # Seed each client with one large group where possible.
    for client_id, group_index in enumerate(order[:num_clients]):
        counts = group_matrix[group_index].astype(np.float64)
        assignments[group_names[group_index]] = client_id
        client_class_counts[client_id] += counts
        client_totals[client_id] += counts.sum()

    for group_index in order[num_clients:]:
        counts = group_matrix[group_index].astype(np.float64)
        size = float(counts.sum())

        scores = []

        for client_id in range(num_clients):
            projected_counts = client_class_counts[client_id] + counts
            projected_total = client_totals[client_id] + size

            score = normalized_assignment_score(
                projected_counts=projected_counts,
                target_counts=target_class_counts[client_id],
                projected_total=projected_total,
                target_total=target_totals[client_id],
            )

            scores.append(score)

        best_score = min(scores)
        candidates = [
            client_id
            for client_id, score in enumerate(scores)
            if np.isclose(score, best_score)
        ]

        selected = min(
            candidates,
            key=lambda cid: client_totals[cid],
        )

        assignments[group_names[group_index]] = selected
        client_class_counts[selected] += counts
        client_totals[selected] += size

    return assignments


def generate_dirichlet_targets(
    global_class_counts: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    target_class_counts = np.zeros(
        (num_clients, len(CLASS_NAMES)),
        dtype=np.float64,
    )

    proportions = np.zeros_like(target_class_counts)

    for class_id, class_total in enumerate(global_class_counts):
        class_proportions = rng.dirichlet(
            np.full(num_clients, alpha, dtype=np.float64)
        )

        proportions[:, class_id] = class_proportions
        target_class_counts[:, class_id] = (
            class_proportions * float(class_total)
        )

    target_totals = target_class_counts.sum(axis=1)

    return target_class_counts, proportions


def assign_groups_noniid(
    group_names: list[str],
    group_matrix: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
) -> tuple[dict[str, int], np.ndarray]:
    """
    Generate class-wise Dirichlet targets, then greedily assign whole
    capture groups to approximate those targets without splitting groups.
    """
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be greater than zero.")

    rng = np.random.default_rng(seed)

    global_class_counts = group_matrix.sum(axis=0).astype(np.float64)

    target_class_counts, proportions = generate_dirichlet_targets(
        global_class_counts=global_class_counts,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
    )

    target_totals = target_class_counts.sum(axis=1)

    client_class_counts = np.zeros_like(target_class_counts)
    client_totals = np.zeros(num_clients, dtype=np.float64)

    group_sizes = group_matrix.sum(axis=1)
    group_dominance = (
        group_matrix.max(axis=1) /
        np.maximum(group_sizes, 1)
    )

    random_tie_break = rng.random(len(group_names))

    order = sorted(
        range(len(group_names)),
        key=lambda index: (
            -float(group_dominance[index]),
            -int(group_sizes[index]),
            float(random_tie_break[index]),
        ),
    )

    assignments: dict[str, int] = {}

    # Ensure no client is empty.
    initial_indices = order[:num_clients]

    for client_id, group_index in enumerate(initial_indices):
        counts = group_matrix[group_index].astype(np.float64)
        assignments[group_names[group_index]] = client_id
        client_class_counts[client_id] += counts
        client_totals[client_id] += counts.sum()

    for group_index in order[num_clients:]:
        counts = group_matrix[group_index].astype(np.float64)
        size = float(counts.sum())

        scores = []

        for client_id in range(num_clients):
            projected_counts = client_class_counts[client_id] + counts
            projected_total = client_totals[client_id] + size

            score = normalized_assignment_score(
                projected_counts=projected_counts,
                target_counts=target_class_counts[client_id],
                projected_total=projected_total,
                target_total=target_totals[client_id],
            )

            scores.append(score)

        best_score = min(scores)
        candidates = [
            client_id
            for client_id, score in enumerate(scores)
            if np.isclose(score, best_score)
        ]

        selected = min(
            candidates,
            key=lambda cid: client_totals[cid],
        )

        assignments[group_names[group_index]] = selected
        client_class_counts[selected] += counts
        client_totals[selected] += size

    return assignments, proportions


def validate_assignments(
    source: pd.DataFrame,
    assigned: pd.DataFrame,
    num_clients: int,
) -> None:
    if len(assigned) != len(source):
        raise RuntimeError(
            f"Assigned rows {len(assigned)} != source rows {len(source)}"
        )

    if assigned["client_id"].isna().any():
        raise RuntimeError("Some rows were not assigned to a client.")

    if assigned["image_path"].duplicated().any():
        raise RuntimeError(
            "Duplicate image paths found after assignment."
        )

    if set(assigned["image_path"]) != set(source["image_path"]):
        raise RuntimeError(
            "Assigned image paths differ from source image paths."
        )

    group_client_counts = (
        assigned.groupby("capture_group")["client_id"].nunique()
    )

    split_groups = group_client_counts[group_client_counts != 1]

    if not split_groups.empty:
        raise RuntimeError(
            f"{len(split_groups)} capture groups were split "
            "across multiple clients."
        )

    present_clients = sorted(assigned["client_id"].unique().tolist())

    expected_clients = list(range(num_clients))

    if present_clients != expected_clients:
        raise RuntimeError(
            f"Expected clients {expected_clients}, "
            f"found {present_clients}."
        )


def write_partition(
    source: pd.DataFrame,
    assignments: dict[str, int],
    output_dir: Path,
    partition_name: str,
    source_manifest: Path,
    num_clients: int,
    seed: int,
    alpha: float | None,
    dirichlet_proportions: np.ndarray | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    assigned = source.copy()
    assigned["client_id"] = (
        assigned["capture_group"]
        .map(assignments)
        .astype("Int64")
    )

    validate_assignments(
        source=source,
        assigned=assigned,
        num_clients=num_clients,
    )

    assigned["client_id"] = assigned["client_id"].astype(int)

    image_summary = pd.crosstab(
        assigned["client_id"],
        assigned["class_name"],
    ).reindex(
        index=range(num_clients),
        columns=CLASS_NAMES,
        fill_value=0,
    )

    unique_groups = assigned[
        ["capture_group", "client_id"]
    ].drop_duplicates()

    group_summary = unique_groups.groupby(
        "client_id"
    ).size().reindex(
        range(num_clients),
        fill_value=0,
    )

    for client_id in range(num_clients):
        client_df = assigned[
            assigned["client_id"] == client_id
        ].copy()

        client_df = client_df.sort_values(
            ["capture_group", "class_name", "image_path"]
        ).reset_index(drop=True)

        client_df.to_csv(
            output_dir / f"client_{client_id}.csv",
            index=False,
        )

    combined = assigned.sort_values(
        ["client_id", "capture_group", "class_name", "image_path"]
    ).reset_index(drop=True)

    combined.to_csv(
        output_dir / "all_clients.csv",
        index=False,
    )

    image_summary_with_total = image_summary.copy()
    image_summary_with_total["total"] = (
        image_summary_with_total.sum(axis=1)
    )

    image_summary_with_total.to_csv(
        output_dir / "client_class_image_counts.csv"
    )

    group_summary.rename("capture_groups").to_csv(
        output_dir / "client_capture_group_counts.csv"
    )

    class_proportions = image_summary.div(
        image_summary.sum(axis=1),
        axis=0,
    ).fillna(0.0)

    class_proportions.to_csv(
        output_dir / "client_class_proportions.csv"
    )

    metadata = {
        "partition_name": partition_name,
        "source_manifest": str(source_manifest.resolve()),
        "num_clients": num_clients,
        "num_images": int(len(source)),
        "num_capture_groups": int(
            source["capture_group"].nunique()
        ),
        "seed": seed,
        "dirichlet_alpha": alpha,
        "class_order": CLASS_NAMES,
        "capture_groups_split_between_clients": 0,
        "client_image_counts": {
            str(client_id): int(
                image_summary_with_total.loc[client_id, "total"]
            )
            for client_id in range(num_clients)
        },
        "client_capture_group_counts": {
            str(client_id): int(group_summary.loc[client_id])
            for client_id in range(num_clients)
        },
    }

    with (
        output_dir / "partition_metadata.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    if dirichlet_proportions is not None:
        proportion_df = pd.DataFrame(
            dirichlet_proportions,
            index=[
                f"client_{client_id}"
                for client_id in range(num_clients)
            ],
            columns=CLASS_NAMES,
        )

        proportion_df.to_csv(
            output_dir / "dirichlet_target_proportions.csv"
        )

    print(f"\n===== {partition_name.upper()} =====")
    print(f"Output directory: {output_dir}")

    print("\nImage distribution:")
    print(image_summary_with_total.to_string())

    print("\nCapture groups per client:")
    print(group_summary.to_string())

    print("\nWithin-client class proportions:")
    print(class_proportions.round(4).to_string())

    print("\nValidation:")
    print("  Every source image assigned once: PASS")
    print("  Capture groups split across clients: 0")
    print("  Empty clients: 0")


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    source = pd.read_csv(args.train_manifest)

    validate_source(source)

    group_counts, group_matrix = build_group_matrix(source)
    group_names = group_counts.index.astype(str).tolist()

    mixed_group_count = int(
        ((group_matrix > 0).sum(axis=1) > 1).sum()
    )

    print("===== SOURCE DATA =====")
    print(f"Images: {len(source)}")
    print(
        f"Capture groups: "
        f"{source['capture_group'].nunique()}"
    )
    print(
        f"Mixed-class capture groups: {mixed_group_count}"
    )
    print(f"Classes: {CLASS_NAMES}")

    iid_assignments = assign_groups_iid(
        group_names=group_names,
        group_matrix=group_matrix,
        num_clients=args.num_clients,
        seed=args.seed,
    )

    iid_dir = (
        args.output_root /
        f"iid_{args.num_clients}clients_seed{args.seed}"
    )

    write_partition(
        source=source,
        assignments=iid_assignments,
        output_dir=iid_dir,
        partition_name="iid",
        source_manifest=args.train_manifest,
        num_clients=args.num_clients,
        seed=args.seed,
        alpha=None,
    )

    noniid_assignments, dirichlet_proportions = (
        assign_groups_noniid(
            group_names=group_names,
            group_matrix=group_matrix,
            num_clients=args.num_clients,
            alpha=args.alpha,
            seed=args.seed,
        )
    )

    alpha_text = str(args.alpha).replace(".", "p")

    noniid_dir = (
        args.output_root /
        (
            f"noniid_{args.num_clients}clients_"
            f"alpha{alpha_text}_seed{args.seed}"
        )
    )

    write_partition(
        source=source,
        assignments=noniid_assignments,
        output_dir=noniid_dir,
        partition_name="non_iid",
        source_manifest=args.train_manifest,
        num_clients=args.num_clients,
        seed=args.seed,
        alpha=args.alpha,
        dirichlet_proportions=dirichlet_proportions,
    )


if __name__ == "__main__":
    main()
