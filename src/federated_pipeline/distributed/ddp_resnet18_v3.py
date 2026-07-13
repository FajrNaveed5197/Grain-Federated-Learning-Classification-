from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import models, transforms


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
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DistributedDataParallel ResNet18 V3 training."
    )
    parser.add_argument(
        "--root",
        default="/scratch/project_2019765/grain_research",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-per-class", type=int, default=1738)
    parser.add_argument("--batch-size-per-gpu", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=(
            "/scratch/project_2019765/grain_research/"
            "distributed_results/ddp_resnet18_v3"
        ),
    )
    parser.add_argument(
        "--initial-checkpoint",
        default=(
            "/scratch/project_2019765/grain_research/results/"
            "additional_finetuning_v2_resnet18/"
            "best_resnet18_v2_gpu.pt"
        ),
    )
    parser.add_argument(
        "--skip-final-evaluation",
        action="store_true",
    )
    return parser.parse_args()


class GrainDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]

        with Image.open(row["path"]) as image:
            image = image.convert("RGB")

        return self.transform(image), CLASS_TO_ID[row["label"]]


def setup_distributed() -> tuple[int, int, int, torch.device]:
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_balanced_epoch(
    dataframe: pd.DataFrame,
    per_class: int,
    epoch_seed: int,
) -> pd.DataFrame:
    parts = []

    for label in CLASS_NAMES:
        class_df = dataframe[dataframe["label"] == label]

        if len(class_df) < per_class:
            raise ValueError(
                f"{label} contains only {len(class_df)} images; "
                f"{per_class} requested."
            )

        parts.append(
            class_df.sample(
                n=per_class,
                random_state=epoch_seed + CLASS_TO_ID[label],
            )
        )

    return (
        pd.concat(parts)
        .sample(frac=1, random_state=epoch_seed)
        .reset_index(drop=True)
    )


def calculate_metrics(targets: list[int], predictions: list[int]) -> dict:
    targets_array = np.asarray(targets)
    predictions_array = np.asarray(predictions)

    accuracy = float((targets_array == predictions_array).mean())
    recalls = []
    f1_scores = []
    per_class = {}

    for class_id, class_name in enumerate(CLASS_NAMES):
        true_positive = np.sum(
            (predictions_array == class_id) &
            (targets_array == class_id)
        )
        false_positive = np.sum(
            (predictions_array == class_id) &
            (targets_array != class_id)
        )
        false_negative = np.sum(
            (predictions_array != class_id) &
            (targets_array == class_id)
        )

        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        recalls.append(float(recall))
        f1_scores.append(float(f1))

        per_class[class_name] = {
            "precision": round(float(precision) * 100, 2),
            "recall": round(float(recall) * 100, 2),
            "f1": round(float(f1) * 100, 2),
        }

    return {
        "accuracy": round(accuracy * 100, 2),
        "balanced_accuracy": round(float(np.mean(recalls)) * 100, 2),
        "macro_f1": round(float(np.mean(f1_scores)) * 100, 2),
        "per_class": per_class,
    }


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

    return calculate_metrics(targets, predictions)


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = setup_distributed()

    try:
        set_seed(args.seed + rank)
        torch.backends.cudnn.benchmark = True

        root = Path(args.root)
        output_dir = Path(args.output_dir)
        initial_checkpoint = Path(args.initial_checkpoint)

        train_manifest = root / "manifests/train.csv"
        validation_manifest = root / "manifests/validation.csv"
        test1_manifest = root / "manifests/test.csv"
        test2_manifest = root / "manifests/test_07.csv"

        if not initial_checkpoint.exists():
            raise FileNotFoundError(
                f"Initial checkpoint not found: {initial_checkpoint}"
            )

        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"DDP world size: {world_size} | "
                f"effective global batch size: "
                f"{args.batch_size_per_gpu * world_size}",
                flush=True,
            )
            print(
                f"GPU 0: {torch.cuda.get_device_name(local_rank)}",
                flush=True,
            )

        dist.barrier()

        full_train_df = pd.read_csv(train_manifest)
        full_val_df = pd.read_csv(validation_manifest)
        full_test1_df = pd.read_csv(test1_manifest)
        full_test2_df = pd.read_csv(test2_manifest)

        balanced_val_df = create_balanced_epoch(
            full_val_df,
            per_class=194,
            epoch_seed=args.seed,
        )
        balanced_test1_df = create_balanced_epoch(
            full_test1_df,
            per_class=200,
            epoch_seed=args.seed,
        )
        balanced_test2_df = create_balanced_epoch(
            full_test2_df,
            per_class=200,
            epoch_seed=args.seed,
        )

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

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        model.load_state_dict(
            torch.load(initial_checkpoint, map_location="cpu")
        )
        model = model.to(device)

        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

        optimizer = torch.optim.AdamW(
            ddp_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=5e-7,
        )

        history = []
        best_validation_f1 = -1.0
        best_epoch = 0
        started_at = time.time()

        for epoch in range(1, args.epochs + 1):
            epoch_df = create_balanced_epoch(
                full_train_df,
                per_class=args.train_per_class,
                epoch_seed=args.seed + epoch * 100,
            )

            train_dataset = GrainDataset(epoch_df, train_transform)

            sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
            )
            sampler.set_epoch(epoch)

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size_per_gpu,
                sampler=sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=args.num_workers > 0,
            )

            ddp_model.train()
            total_loss = 0.0
            total_samples = 0

            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = ddp_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

            loss_tensor = torch.tensor(
                [total_loss, total_samples],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

            global_loss = (
                loss_tensor[0].item() / max(loss_tensor[1].item(), 1)
            )

            scheduler.step()
            dist.barrier()

            if rank == 0:
                val_loader = DataLoader(
                    GrainDataset(balanced_val_df, eval_transform),
                    batch_size=256,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    persistent_workers=args.num_workers > 0,
                )

                validation = evaluate(model, val_loader, device)

                print(
                    f"DDP epoch {epoch}/{args.epochs} | "
                    f"loss={global_loss:.4f} | "
                    f"balanced val macro-F1="
                    f"{validation['macro_f1']:.2f}%",
                    flush=True,
                )

                history.append({
                    "epoch": epoch,
                    "loss": round(global_loss, 6),
                    "validation": validation,
                })

                if validation["macro_f1"] > best_validation_f1:
                    best_validation_f1 = validation["macro_f1"]
                    best_epoch = epoch

                    torch.save(
                        model.state_dict(),
                        output_dir / "best_ddp_resnet18_v3.pt",
                    )

            dist.barrier()

        if rank == 0:
            best_checkpoint = output_dir / "best_ddp_resnet18_v3.pt"

            if best_checkpoint.exists():
                model.load_state_dict(
                    torch.load(best_checkpoint, map_location=device)
                )
            else:
                model.load_state_dict(
                    torch.load(initial_checkpoint, map_location=device)
                )

            results = {
                "experiment": "ddp_resnet18_v3",
                "framework": "PyTorch DistributedDataParallel",
                "world_size": world_size,
                "epochs": args.epochs,
                "train_per_class": args.train_per_class,
                "training_images_per_epoch": (
                    args.train_per_class * len(CLASS_NAMES)
                ),
                "batch_size_per_gpu": args.batch_size_per_gpu,
                "effective_global_batch_size": (
                    args.batch_size_per_gpu * world_size
                ),
                "initial_checkpoint": str(initial_checkpoint),
                "best_epoch": best_epoch,
                "best_validation_macro_f1": best_validation_f1,
                "history": history,
                "elapsed_minutes": round(
                    (time.time() - started_at) / 60,
                    2,
                ),
            }

            if not args.skip_final_evaluation:
                evaluation_loader_kwargs = {
                    "batch_size": 256,
                    "shuffle": False,
                    "num_workers": args.num_workers,
                    "pin_memory": True,
                    "persistent_workers": args.num_workers > 0,
                }

                results["validation"] = evaluate(
                    model,
                    DataLoader(
                        GrainDataset(balanced_val_df, eval_transform),
                        **evaluation_loader_kwargs,
                    ),
                    device,
                )
                results["test_set_1"] = evaluate(
                    model,
                    DataLoader(
                        GrainDataset(balanced_test1_df, eval_transform),
                        **evaluation_loader_kwargs,
                    ),
                    device,
                )
                results["test_set_2"] = evaluate(
                    model,
                    DataLoader(
                        GrainDataset(balanced_test2_df, eval_transform),
                        **evaluation_loader_kwargs,
                    ),
                    device,
                )

            with open(
                output_dir / "metrics.json",
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(results, handle, indent=2)

            print("\nFINAL DDP RESULTS", flush=True)
            print(json.dumps(results, indent=2), flush=True)

        dist.barrier()

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
