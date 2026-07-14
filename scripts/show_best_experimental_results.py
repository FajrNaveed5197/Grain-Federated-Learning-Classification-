from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


RESULT_ROOT = Path(
    "/scratch/project_2019765/fnaveed/results"
)


def sha256_short(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()[:16]


def file_info(path: Path) -> str:
    modified = datetime.fromtimestamp(
        path.stat().st_mtime
    ).strftime("%Y-%m-%d %H:%M:%S")

    return (
        f"{path}\n"
        f"    modified={modified}, "
        f"sha256={sha256_short(path)}"
    )


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def print_table(
    headers: list[str],
    rows: list[list[str]],
) -> None:
    widths = [
        len(header)
        for header in headers
    ]

    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(
                widths[index],
                len(str(value)),
            )

    separator = (
        "+"
        + "+"
        .join(
            "-" * (width + 2)
            for width in widths
        )
        + "+"
    )

    print(separator)

    print(
        "|"
        + "|".join(
            f" {header:<{widths[index]}} "
            for index, header
            in enumerate(headers)
        )
        + "|"
    )

    print(separator)

    for row in rows:
        print(
            "|"
            + "|".join(
                f" {str(value):<{widths[index]}} "
                for index, value
                in enumerate(row)
            )
            + "|"
        )

    print(separator)


def load_rice_results():
    architecture_csv = (
        RESULT_ROOT
        / "final_report_archive"
        / "tables"
        / "rice_architecture_evaluation.csv"
    )

    mobile_fed_csv = (
        RESULT_ROOT
        / "final_report_archive"
        / "tables"
        / "rice_fedavg_mobilenetv2_comparison.csv"
    )

    resnet_iid_csv = (
        RESULT_ROOT
        / "rice_fedavg_iid_resnet18"
        / "evaluation"
        / "evaluation_summary.csv"
    )

    resnet_noniid_csv = (
        RESULT_ROOT
        / "rice_fedavg_noniid_resnet18"
        / "evaluation"
        / "evaluation_summary.csv"
    )

    ddp_csv = (
        RESULT_ROOT
        / "rice_ddp_resnet18"
        / "evaluation"
        / "evaluation_summary.csv"
    )

    source_files = [
        architecture_csv,
        mobile_fed_csv,
        resnet_iid_csv,
        resnet_noniid_csv,
        ddp_csv,
    ]

    missing = [
        path
        for path in source_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing rice files:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )

    rows = []

    for item in read_csv_rows(
        architecture_csv
    ):
        if item["split"] != "test":
            continue

        rows.append([
            "Centralized",
            item["model"],
            "Group-aware",
            item["accuracy"],
            item["balanced_accuracy"],
            item["macro_f1"],
            item["weighted_f1"],
        ])

    for path, label in [
        (
            resnet_iid_csv,
            "IID",
        ),
        (
            resnet_noniid_csv,
            "non-IID",
        ),
    ]:
        test_row = next(
            row
            for row in read_csv_rows(path)
            if row["split"] == "test"
        )

        rows.append([
            "FedAvg",
            "ResNet18",
            label,
            test_row["accuracy"],
            test_row["balanced_accuracy"],
            test_row["macro_f1"],
            test_row["weighted_f1"],
        ])

    for item in read_csv_rows(
        mobile_fed_csv
    ):
        if item["split"] != "test":
            continue

        rows.append([
            "FedAvg",
            "MobileNetV2",
            (
                "IID"
                if "IID MobileNetV2"
                in item["experiment"]
                and "non-IID"
                not in item["experiment"]
                else "non-IID"
            ),
            item["accuracy"],
            item["balanced_accuracy"],
            item["macro_f1"],
            item["weighted_f1"],
        ])

    test_row = next(
        row
        for row in read_csv_rows(ddp_csv)
        if row["split"] == "test"
    )

    rows.append([
        "DDP",
        "ResNet18",
        "2x GH200",
        test_row["accuracy"],
        test_row["balanced_accuracy"],
        test_row["macro_f1"],
        test_row["weighted_f1"],
    ])

    print("\n===== RICE: BEST AND FINAL TEST RESULTS =====\n")

    print_table(
        [
            "Paradigm",
            "Architecture",
            "Setting",
            "Accuracy",
            "Balanced Acc.",
            "Macro-F1",
            "Weighted-F1",
        ],
        rows,
    )

    print("\nSource files used:")

    for path in source_files:
        print(file_info(path))


def load_wheat_results():
    v2_csv = (
        RESULT_ROOT
        / "wheat_resnet18_grouped_v2"
        / "evaluation_summary.csv"
    )

    v3_csv = (
        RESULT_ROOT
        / "wheat_resnet18_grouped_v3_sqrt_weights"
        / "evaluation_summary.csv"
    )

    v2_metrics = (
        RESULT_ROOT
        / "wheat_resnet18_grouped_v2"
        / "metrics.json"
    )

    v3_metrics = (
        RESULT_ROOT
        / "wheat_resnet18_grouped_v3_sqrt_weights"
        / "metrics.json"
    )

    source_files = [
        v2_csv,
        v3_csv,
        v2_metrics,
        v3_metrics,
    ]

    missing = [
        path
        for path in source_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing wheat files:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )

    rows = []

    for name, path in [
        (
            "V2 full inverse weights",
            v2_csv,
        ),
        (
            "V3 sqrt inverse weights",
            v3_csv,
        ),
    ]:
        for item in read_csv_rows(path):
            rows.append([
                name,
                item["split"],
                item["accuracy"],
                item["balanced_accuracy"],
                item["macro_f1"],
                item["weighted_f1"],
            ])

    print("\n\n===== WHEAT: CLEAN GROUP-AWARE RESULTS =====\n")

    print_table(
        [
            "Experiment",
            "Split",
            "Accuracy",
            "Balanced Acc.",
            "Macro-F1",
            "Weighted-F1",
        ],
        rows,
    )

    print("\nSource files used:")

    for path in source_files:
        print(file_info(path))


def main() -> None:
    load_rice_results()
    load_wheat_results()


if __name__ == "__main__":
    main()
