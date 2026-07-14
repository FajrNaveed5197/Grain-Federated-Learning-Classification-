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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-GPU DDP ResNet18 training for the rice dataset."
    )

    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size-per-gpu", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def setup_distributed():
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


def calculate_class_weights(
    manifest_path: Path,
) -> torch.Tensor:
    dataframe = pd.read_csv(manifest_path)

    counts = (
        dataframe["class_name"]
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
        .to_numpy(dtype=np.float32)
    )

    weights = np.sqrt(
        counts.sum() / np.maximum(counts, 1.0)
    )

    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()

    targets = []
    predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            targets.extend(labels.numpy().tolist())
            predictions.extend(predicted.cpu().numpy().tolist())

    return {
        "accuracy": round(
            accuracy_score(targets, predictions) * 100,
            4,
        ),
        "balanced_accuracy": round(
            balanced_accuracy_score(
                targets,
                predictions,
            ) * 100,
            4,
        ),
        "macro_f1": round(
            f1_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            ) * 100,
            4,
        ),
        "weighted_f1": round(
            f1_score(
                targets,
                predictions,
                average="weighted",
                zero_division=0,
            ) * 100,
            4,
        ),
    }


def main() -> None:
    args = parse_args()

    rank, world_size, local_rank, device = setup_distributed()

    try:
        set_seed(args.seed + rank)
        torch.backends.cudnn.benchmark = True

        if rank == 0:
            args.output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            print(
                f"DDP world size: {world_size}",
                flush=True,
            )

            print(
                f"Effective global batch size: "
                f"{args.batch_size_per_gpu * world_size}",
                flush=True,
            )

            print(
                f"GPU 0: {torch.cuda.get_device_name(local_rank)}",
                flush=True,
            )

        dist.barrier()

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                224,
                scale=(0.85, 1.0),
            ),
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

        train_dataset = ManifestImageDataset(
            manifest_path=args.train_manifest,
            class_to_id=CLASS_TO_ID,
            transform=train_transform,
            dataset_root=args.dataset_root,
        )

        validation_dataset = ManifestImageDataset(
            manifest_path=args.validation_manifest,
            class_to_id=CLASS_TO_ID,
            transform=eval_transform,
            dataset_root=args.dataset_root,
        )

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size_per_gpu,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

        validation_loader = None

        if rank == 0:
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=256,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=args.num_workers > 0,
            )

            print(
                f"Train images: {len(train_dataset)}",
                flush=True,
            )

            print(
                f"Validation images: {len(validation_dataset)}",
                flush=True,
            )

        checkpoint = torch.load(
            args.initial_checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model = create_resnet18(
            num_classes=len(CLASS_NAMES),
            pretrained=False,
        )

        model.load_state_dict(state_dict)
        model = model.to(device)

        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

        class_weights = calculate_class_weights(
            args.train_manifest
        ).to(device)

        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )

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
        best_epoch = 0
        best_macro_f1 = float("-inf")
        training_started = time.time()

        for epoch in range(1, args.epochs + 1):
            train_sampler.set_epoch(epoch)
            ddp_model.train()

            local_loss_sum = 0.0
            local_sample_count = 0

            epoch_started = time.time()

            for images, labels in train_loader:
                images = images.to(
                    device,
                    non_blocking=True,
                )

                labels = labels.to(
                    device,
                    non_blocking=True,
                )

                optimizer.zero_grad(set_to_none=True)

                outputs = ddp_model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                local_loss_sum += (
                    loss.item() * labels.size(0)
                )

                local_sample_count += labels.size(0)

            loss_tensor = torch.tensor(
                [
                    local_loss_sum,
                    local_sample_count,
                ],
                dtype=torch.float64,
                device=device,
            )

            dist.all_reduce(
                loss_tensor,
                op=dist.ReduceOp.SUM,
            )

            global_loss = (
                loss_tensor[0].item()
                / max(loss_tensor[1].item(), 1)
            )

            scheduler.step()
            dist.barrier()

            if rank == 0:
                validation_metrics = evaluate(
                    model=model,
                    loader=validation_loader,
                    device=device,
                )

                epoch_seconds = (
                    time.time() - epoch_started
                )

                epoch_record = {
                    "epoch": epoch,
                    "train_loss": round(
                        global_loss,
                        6,
                    ),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "validation": validation_metrics,
                    "epoch_seconds": round(
                        epoch_seconds,
                        2,
                    ),
                }

                history.append(epoch_record)

                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"loss={global_loss:.6f} | "
                    f"val accuracy="
                    f"{validation_metrics['accuracy']:.4f}% | "
                    f"val balanced accuracy="
                    f"{validation_metrics['balanced_accuracy']:.4f}% | "
                    f"val Macro-F1="
                    f"{validation_metrics['macro_f1']:.4f}% | "
                    f"time={epoch_seconds:.2f}s",
                    flush=True,
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "class_names": CLASS_NAMES,
                        "validation_metrics": validation_metrics,
                        "world_size": world_size,
                        "arguments": vars(args),
                    },
                    args.output_dir
                    / f"ddp_global_epoch_{epoch}.pt",
                )

                if (
                    validation_metrics["macro_f1"]
                    > best_macro_f1
                ):
                    best_macro_f1 = float(
                        validation_metrics["macro_f1"]
                    )

                    best_epoch = epoch

                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "class_names": CLASS_NAMES,
                            "validation_metrics": validation_metrics,
                            "world_size": world_size,
                            "arguments": vars(args),
                        },
                        args.output_dir
                        / "best_ddp_resnet18_rice.pt",
                    )

                    print(
                        f"New best checkpoint: epoch {epoch}",
                        flush=True,
                    )

                metrics = {
                    "experiment": "rice_ddp_resnet18",
                    "class_names": CLASS_NAMES,
                    "world_size": world_size,
                    "effective_global_batch_size": (
                        args.batch_size_per_gpu
                        * world_size
                    ),
                    "completed_epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_macro_f1": (
                        best_macro_f1
                    ),
                    "history": history,
                    "arguments": {
                        key: str(value)
                        if isinstance(value, Path)
                        else value
                        for key, value
                        in vars(args).items()
                    },
                }

                with (
                    args.output_dir / "metrics.json"
                ).open(
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(
                        metrics,
                        handle,
                        indent=2,
                    )

            dist.barrier()

        if rank == 0:
            total_seconds = (
                time.time() - training_started
            )

            summary = {
                "experiment": "rice_ddp_resnet18",
                "world_size": world_size,
                "epochs": args.epochs,
                "best_epoch": best_epoch,
                "best_validation_macro_f1": (
                    best_macro_f1
                ),
                "total_training_seconds": round(
                    total_seconds,
                    2,
                ),
                "effective_global_batch_size": (
                    args.batch_size_per_gpu
                    * world_size
                ),
                "train_images": len(train_dataset),
                "validation_images": len(
                    validation_dataset
                ),
            }

            with (
                args.output_dir / "training_summary.json"
            ).open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    summary,
                    handle,
                    indent=2,
                )

            print(
                f"DDP training completed in "
                f"{total_seconds:.2f} seconds",
                flush=True,
            )

            print(
                f"Best epoch: {best_epoch}",
                flush=True,
            )

            print(
                f"Best validation Macro-F1: "
                f"{best_macro_f1:.4f}%",
                flush=True,
            )

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
