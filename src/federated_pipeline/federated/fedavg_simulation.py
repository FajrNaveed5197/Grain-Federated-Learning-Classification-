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
    parser = argparse.ArgumentParser(description="FedAvg image-classification simulation")

    parser.add_argument("--client-dir", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

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
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    for _ in range(local_epochs):
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

    average_loss = total_loss / max(total_samples, 1)

    weights = OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in model.state_dict().items()
    )

    return weights, len(train_loader.dataset), average_loss


def fedavg(
    client_weights: list[tuple[OrderedDict[str, torch.Tensor], int]],
) -> OrderedDict[str, torch.Tensor]:
    total_samples = sum(num_samples for _, num_samples in client_weights)

    aggregated = OrderedDict()

    for key in client_weights[0][0]:
        weighted_sum = None

        for state_dict, num_samples in client_weights:
            value = state_dict[key].float() * num_samples

            if weighted_sum is None:
                weighted_sum = value
            else:
                weighted_sum += value

        aggregated[key] = (weighted_sum / total_samples).to(
            client_weights[0][0][key].dtype
        )

    return aggregated


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
            outputs = model(images.to(device, non_blocking=True))
            predicted = outputs.argmax(dim=1).cpu()

            predictions.extend(predicted.tolist())
            targets.extend(labels.tolist())

    targets_array = np.array(targets)
    predictions_array = np.array(predictions)

    accuracy = float((targets_array == predictions_array).mean())

    per_class_f1 = []
    per_class_recall = []

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

        per_class_f1.append(float(f1))
        per_class_recall.append(float(recall))

    return {
        "accuracy": round(accuracy * 100, 2),
        "balanced_accuracy": round(float(np.mean(per_class_recall)) * 100, 2),
        "macro_f1": round(float(np.mean(per_class_f1)) * 100, 2),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    client_dir = Path(args.client_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client_manifests = sorted(client_dir.glob("client_*_train.csv"))

    if not client_manifests:
        raise FileNotFoundError(
            f"No client manifests found in: {client_dir}"
        )

    train_transform, eval_transform = make_transforms()

    client_loaders = []

    for manifest in client_manifests:
        dataset = ManifestImageDataset(
            manifest_path=manifest,
            class_to_id=CLASS_TO_ID,
            transform=train_transform,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
        )

        client_loaders.append((manifest.stem, loader))
        print(f"{manifest.stem}: {len(dataset)} images")

    validation_dataset = ManifestImageDataset(
        manifest_path=args.validation_manifest,
        class_to_id=CLASS_TO_ID,
        transform=eval_transform,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    global_model = create_resnet18(
        num_classes=len(CLASS_NAMES),
        pretrained=True,
    ).to(device)

    history = []
    started_at = time.time()

    for round_number in range(1, args.rounds + 1):
        print(f"\n===== Federated round {round_number}/{args.rounds} =====")

        client_updates = []
        round_client_metrics = []

        for client_name, train_loader in client_loaders:
            weights, num_samples, local_loss = train_local_model(
                global_model=global_model,
                train_loader=train_loader,
                device=device,
                local_epochs=args.local_epochs,
                learning_rate=args.learning_rate,
            )

            client_updates.append((weights, num_samples))

            round_client_metrics.append({
                "client": client_name,
                "num_samples": num_samples,
                "local_loss": round(local_loss, 6),
            })

            print(
                f"{client_name}: samples={num_samples}, "
                f"local_loss={local_loss:.4f}"
            )

        aggregated_weights = fedavg(client_updates)
        global_model.load_state_dict(aggregated_weights)

        validation_metrics = evaluate(
            model=global_model,
            loader=validation_loader,
            device=device,
        )

        round_result = {
            "round": round_number,
            "clients": round_client_metrics,
            "validation": validation_metrics,
        }

        history.append(round_result)

        print(
            "Global validation | "
            f"Accuracy={validation_metrics['accuracy']:.2f}% | "
            f"Balanced Accuracy={validation_metrics['balanced_accuracy']:.2f}% | "
            f"Macro-F1={validation_metrics['macro_f1']:.2f}%"
        )

        with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "experiment": "fedavg_simulation",
                    "rounds": args.rounds,
                    "local_epochs": args.local_epochs,
                    "num_clients": len(client_loaders),
                    "history": history,
                },
                handle,
                indent=2,
            )

        torch.save(
            global_model.state_dict(),
            output_dir / "global_model_latest.pt",
        )

    elapsed_minutes = (time.time() - started_at) / 60

    print(f"\nCompleted in {elapsed_minutes:.2f} minutes")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
