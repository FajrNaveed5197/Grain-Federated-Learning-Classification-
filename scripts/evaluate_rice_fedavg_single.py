from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one Rice FedAvg MobileNetV2 checkpoint "
            "without modifying shared comparison archives."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    return parser.parse_args()


def load_base_evaluator():
    source_path = (
        Path(__file__).resolve().parent
        / "evaluate_rice_fedavg_mobilenetv2.py"
    )

    spec = importlib.util.spec_from_file_location(
        "rice_fedavg_base_evaluator",
        source_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load evaluator module: {source_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required.")

    for path in [
        args.checkpoint,
        args.validation_manifest,
        args.test_manifest,
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    evaluation_dir = args.output_dir / "evaluation"

    if (
        evaluation_dir.exists()
        and any(evaluation_dir.iterdir())
    ):
        raise FileExistsError(
            "Refusing to overwrite non-empty evaluation directory: "
            f"{evaluation_dir}"
        )

    evaluation_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_evaluator()

    # RiceDataset resolves image paths using this module-level value.
    base.DATA_ROOT = args.dataset_root

    evaluation_transform = transforms.Compose([
        transforms.Resize((base.IMAGE_SIZE, base.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    datasets = {
        "validation": base.RiceDataset(
            args.validation_manifest,
            evaluation_transform,
        ),
        "test": base.RiceDataset(
            args.test_manifest,
            evaluation_transform,
        ),
    }

    loaders = {
        split_name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        for split_name, dataset in datasets.items()
    }

    device = torch.device("cuda")
    model = base.create_model().to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    state_dict = (
        checkpoint["model_state_dict"]
        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        )
        else checkpoint
    )

    model.load_state_dict(state_dict, strict=True)

    checkpoint_round = (
        checkpoint.get("round")
        if isinstance(checkpoint, dict)
        else None
    )
    checkpoint_validation_metrics = (
        checkpoint.get("validation_metrics")
        if isinstance(checkpoint, dict)
        else None
    )

    training_metrics_path = args.output_dir / "metrics.json"
    training_metrics = (
        json.loads(training_metrics_path.read_text())
        if training_metrics_path.is_file()
        else {}
    )

    results = {}
    summary_rows = []

    for split_name in ["validation", "test"]:
        metrics, targets, predictions = base.evaluate(
            model,
            loaders[split_name],
            device,
        )

        rounded_metrics = {
            key: round(float(value), 4)
            for key, value in metrics.items()
        }

        base.save_outputs(
            split_name=split_name,
            dataset=datasets[split_name],
            metrics=metrics,
            targets=targets,
            predictions=predictions,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
        )

        results[split_name] = rounded_metrics

        summary_rows.append({
            "experiment": args.experiment_name,
            "split": split_name,
            "num_images": len(datasets[split_name]),
            **rounded_metrics,
        })

        print(
            f"{split_name} | "
            f"accuracy={metrics['accuracy']:.4f}% | "
            f"balanced_accuracy="
            f"{metrics['balanced_accuracy']:.4f}% | "
            f"macro_f1={metrics['macro_f1']:.4f}% | "
            f"weighted_f1={metrics['weighted_f1']:.4f}%",
            flush=True,
        )

    evaluation_record = {
        "experiment": args.experiment_name,
        "algorithm": "FedAvg",
        "checkpoint": str(args.checkpoint),
        "checkpoint_round": checkpoint_round,
        "checkpoint_validation_metrics": (
            checkpoint_validation_metrics
        ),
        "dataset_root": str(args.dataset_root),
        "validation_manifest": str(
            args.validation_manifest
        ),
        "test_manifest": str(args.test_manifest),
        "results": results,
        "communication": training_metrics.get(
            "communication"
        ),
        "training_best_round": training_metrics.get(
            "best_round"
        ),
        "training_best_validation_macro_f1": (
            training_metrics.get(
                "best_validation_macro_f1"
            )
        ),
        "evaluation_scope": (
            "Single alpha=0.1 FedAvg experiment; "
            "no shared archive files modified."
        ),
    }

    with (
        evaluation_dir / "evaluation_metrics.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            evaluation_record,
            handle,
            indent=2,
        )

    pd.DataFrame(summary_rows).to_csv(
        evaluation_dir / "evaluation_summary.csv",
        index=False,
    )

    with (
        evaluation_dir / "run_metadata.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "device": str(device),
                "torch_version": torch.__version__,
                "shared_archive_modified": False,
            },
            handle,
            indent=2,
        )

    print(
        f"Evaluation saved only to: {evaluation_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
