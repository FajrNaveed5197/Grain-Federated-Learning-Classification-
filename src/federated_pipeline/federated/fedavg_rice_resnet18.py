from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from federated_pipeline.common.model import create_resnet18
from federated_pipeline.data.dataset import ManifestImageDataset


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
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group-aware FedAvg ResNet18 experiment for the rice dataset."
    )
    parser.add_argument("--client-dir", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--experiment-name",
        required=True,
        choices=[
            "rice_fedavg_iid_resnet18",
            "rice_fedavg_noniid_resnet18",
        ],
    )

    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output-dir/training_state.pt if it exists.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(
            brightness=0.08,
            contrast=0.10,
            saturation=0.06,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, eval_transform


def local_class_weights(dataset: ManifestImageDataset) -> torch.Tensor:
    targets = (
        dataset.dataframe[dataset.label_column]
        .map(CLASS_TO_ID)
        .to_numpy()
    )
    counts = np.bincount(targets, minlength=len(CLASS_NAMES)).astype(np.float32)

    # Square-root inverse frequency weighting. This reduces majority-class
    # dominance without making rare-class weights excessively large.
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def train_local_model(
    global_model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    local_epochs: int,
    learning_rate: float,
) -> tuple[OrderedDict[str, torch.Tensor], int, float]:
    model = copy.deepcopy(global_model).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    weights = local_class_weights(train_loader.dataset).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.03,
    )

    total_loss = 0.0
    total_samples = 0

    for epoch in range(1, local_epochs + 1):
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            if batch_index % 250 == 0:
                print(
                    f"    local epoch {epoch}/{local_epochs} | "
                    f"batch {batch_index}/{len(train_loader)}",
                    flush=True,
                )

    average_loss = total_loss / max(total_samples, 1)

    weights_cpu = OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in model.state_dict().items()
    )

    return weights_cpu, len(train_loader.dataset), average_loss


def fedavg(
    client_updates: list[tuple[OrderedDict[str, torch.Tensor], int]],
) -> OrderedDict[str, torch.Tensor]:
    total_samples = sum(num_samples for _, num_samples in client_updates)
    aggregated = OrderedDict()

    for key in client_updates[0][0]:
        reference = client_updates[0][0][key]

        # BatchNorm counters are integer tensors, so they cannot be averaged.
        if not torch.is_floating_point(reference):
            aggregated[key] = max(
                state_dict[key]
                for state_dict, _ in client_updates
            )
            continue

        weighted_sum = torch.zeros_like(reference, dtype=torch.float32)

        for state_dict, num_samples in client_updates:
            weighted_sum += state_dict[key].float() * num_samples

        aggregated[key] = (weighted_sum / total_samples).to(reference.dtype)

    return aggregated


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device, non_blocking=True))
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    targets_array = np.asarray(targets)
    predictions_array = np.asarray(predictions)

    accuracy = float((targets_array == predictions_array).mean())
    recalls = []
    f1_scores = []

    for class_id in range(len(CLASS_NAMES)):
        true_positive = np.sum(
            (predictions_array == class_id) & (targets_array == class_id)
        )
        false_positive = np.sum(
            (predictions_array == class_id) & (targets_array != class_id)
        )
        false_negative = np.sum(
            (predictions_array != class_id) & (targets_array == class_id)
        )

        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        recalls.append(float(recall))
        f1_scores.append(float(f1))

    class_supports = [
        int(np.sum(targets_array == class_id))
        for class_id in range(len(CLASS_NAMES))
    ]

    total_support = max(sum(class_supports), 1)

    weighted_f1 = sum(
        f1 * support
        for f1, support in zip(f1_scores, class_supports)
    ) / total_support

    return {
        "accuracy": round(accuracy * 100, 4),
        "balanced_accuracy": round(
            float(np.mean(recalls)) * 100,
            4,
        ),
        "macro_f1": round(
            float(np.mean(f1_scores)) * 100,
            4,
        ),
        "weighted_f1": round(
            float(weighted_f1) * 100,
            4,
        ),
    }


def save_state(
    output_dir: Path,
    model: nn.Module,
    completed_round: int,
    history: list[dict],
    args: argparse.Namespace,
    best_round: int,
    best_macro_f1: float,
) -> None:
    torch.save(
        {
            "completed_round": completed_round,
            "model_state_dict": model.state_dict(),
            "history": history,
            "arguments": vars(args),
        },
        output_dir / "training_state.pt",
    )

    torch.save(
        model.state_dict(),
        output_dir / f"global_model_round_{completed_round}.pt",
    )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment": args.experiment_name,
                "completed_round": completed_round,
                "best_round": best_round,
                "best_validation_macro_f1": best_macro_f1,
                "class_names": CLASS_NAMES,
                "history": history,
                "arguments": {
                    key: str(value)
                    if isinstance(value, Path)
                    else value
                    for key, value in vars(args).items()
                },
            },
            handle,
            indent=2,
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print(f"Device: {device}", flush=True)

    client_dir = Path(args.client_dir)
    output_dir = Path(args.output_dir)
    initial_checkpoint = Path(args.initial_checkpoint)
    state_path = output_dir / "training_state.pt"

    output_dir.mkdir(parents=True, exist_ok=True)

    client_manifests = sorted(
        manifest
        for manifest in client_dir.glob("client_*.csv")
        if manifest.stem.removeprefix("client_").isdigit()
    )

    if not client_manifests:
        raise FileNotFoundError(
            f"No numbered client manifests found in: {client_dir}"
        )

    print(
        "Client manifests: "
        + ", ".join(manifest.name for manifest in client_manifests),
        flush=True,
    )

    if not initial_checkpoint.exists():
        raise FileNotFoundError(
            f"Initial checkpoint not found: {initial_checkpoint}"
        )

    train_transform, eval_transform = make_transforms()

    client_loaders = []
    for manifest in client_manifests:
        dataset = ManifestImageDataset(
            manifest_path=manifest,
            class_to_id=CLASS_TO_ID,
            transform=train_transform,
            dataset_root=args.dataset_root,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        client_loaders.append((manifest.stem, loader))
        print(f"{manifest.stem}: {len(dataset)} images", flush=True)

    validation_dataset = ManifestImageDataset(
        manifest_path=args.validation_manifest,
        class_to_id=CLASS_TO_ID,
        transform=eval_transform,
        dataset_root=args.dataset_root,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    global_model = create_resnet18(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
    ).to(device)

    history: list[dict] = []
    start_round = 1

    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        global_model.load_state_dict(state["model_state_dict"])
        history = state["history"]
        start_round = int(state["completed_round"]) + 1
        print(
            f"Resuming after completed round {start_round - 1}",
            flush=True,
        )
    else:
        checkpoint = torch.load(
            initial_checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            model_state_dict = checkpoint["model_state_dict"]
        else:
            model_state_dict = checkpoint

        global_model.load_state_dict(model_state_dict)

        print(
            f"Initialized from: {initial_checkpoint}",
            flush=True,
        )

        if isinstance(checkpoint, dict):
            print(
                f"Initial checkpoint epoch: "
                f"{checkpoint.get('epoch', 'unknown')}",
                flush=True,
            )
            print(
                f"Initial checkpoint classes: "
                f"{checkpoint.get('class_names', 'unknown')}",
                flush=True,
            )

    started_at = time.time()

    if history:
        best_entry = max(
            history,
            key=lambda entry: entry["validation"]["macro_f1"],
        )
        best_round = int(best_entry["round"])
        best_macro_f1 = float(
            best_entry["validation"]["macro_f1"]
        )
    else:
        best_round = 0
        best_macro_f1 = float("-inf")

    for round_number in range(start_round, args.rounds + 1):
        print(
            f"\n===== Federated round {round_number}/{args.rounds} =====",
            flush=True,
        )

        client_updates = []
        client_metrics = []

        for client_name, train_loader in client_loaders:
            print(f"Starting {client_name}", flush=True)

            weights, num_samples, local_loss = train_local_model(
                global_model=global_model,
                train_loader=train_loader,
                device=device,
                local_epochs=args.local_epochs,
                learning_rate=args.learning_rate,
            )

            client_updates.append((weights, num_samples))
            client_metrics.append(
                {
                    "client": client_name,
                    "num_samples": num_samples,
                    "local_loss": round(local_loss, 6),
                }
            )

            print(
                f"Finished {client_name}: samples={num_samples}, "
                f"local_loss={local_loss:.4f}",
                flush=True,
            )

        global_model.load_state_dict(fedavg(client_updates))

        print("Evaluating full validation set...", flush=True)
        validation_metrics = evaluate(
            model=global_model,
            loader=validation_loader,
            device=device,
        )

        print(
            "Global validation | "
            f"Accuracy={validation_metrics['accuracy']:.2f}% | "
            f"Balanced Accuracy={validation_metrics['balanced_accuracy']:.2f}% | "
            f"Macro-F1={validation_metrics['macro_f1']:.4f}% | "
            f"Weighted-F1={validation_metrics['weighted_f1']:.4f}%",
            flush=True,
        )

        round_elapsed_seconds = time.time() - started_at

        history.append(
            {
                "round": round_number,
                "clients": client_metrics,
                "validation": validation_metrics,
                "elapsed_seconds_from_start": round(
                    round_elapsed_seconds,
                    2,
                ),
            }
        )

        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(
                validation_metrics["macro_f1"]
            )
            best_round = round_number

            torch.save(
                {
                    "round": round_number,
                    "model_state_dict": global_model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "validation_metrics": validation_metrics,
                    "experiment": args.experiment_name,
                    "arguments": vars(args),
                },
                output_dir / "best_global_model.pt",
            )

            print(
                "New best global model | "
                f"round={best_round} | "
                f"Macro-F1={best_macro_f1:.4f}%",
                flush=True,
            )

        save_state(
            output_dir=output_dir,
            model=global_model,
            completed_round=round_number,
            history=history,
            args=args,
            best_round=best_round,
            best_macro_f1=best_macro_f1,
        )

        print(
            f"Saved checkpoint: {output_dir}/global_model_round_{round_number}.pt",
            flush=True,
        )

    elapsed_hours = (time.time() - started_at) / 3600
    print(f"\nCompleted in {elapsed_hours:.2f} hours", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
