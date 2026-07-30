from __future__ import annotations

import argparse
import json
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from federated_pipeline.data.dataset import ManifestImageDataset
from federated_pipeline.federated.fedavg_rice_mobilenetv2 import (
    CLASS_NAMES,
    CLASS_TO_ID,
    create_mobilenetv2,
    evaluate,
    fedavg,
    local_class_weights,
    make_transforms,
)

PRIVATE_PREFIX = "classifier.1."
PRIVATE_KEYS = {"classifier.1.weight", "classifier.1.bias"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FedRep MobileNetV2 on the three-client rice split. "
            "The classifier head is private and the representation is shared."
        )
    )
    parser.add_argument("--client-dir", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--experiment-name", required=True)

    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--representation-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--representation-learning-rate",
        type=float,
        default=2e-6,
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clone_state(
    state: OrderedDict[str, torch.Tensor] | dict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in state.items()
    )


def is_private(name: str) -> bool:
    return name.startswith(PRIVATE_PREFIX)


def split_state(
    state: OrderedDict[str, torch.Tensor] | dict[str, torch.Tensor],
) -> tuple[
    OrderedDict[str, torch.Tensor],
    OrderedDict[str, torch.Tensor],
]:
    shared: OrderedDict[str, torch.Tensor] = OrderedDict()
    private: OrderedDict[str, torch.Tensor] = OrderedDict()

    for name, tensor in state.items():
        destination = private if is_private(name) else shared
        destination[name] = tensor.detach().cpu().clone()

    if set(private) != PRIVATE_KEYS:
        raise RuntimeError(
            f"Expected private keys {sorted(PRIVATE_KEYS)}, "
            f"found {sorted(private)}"
        )

    return shared, private


def build_model(
    shared: OrderedDict[str, torch.Tensor],
    private: OrderedDict[str, torch.Tensor],
) -> nn.Module:
    model = create_mobilenetv2(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
    )

    complete: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name in model.state_dict():
        source = private if is_private(name) else shared
        complete[name] = source[name]

    model.load_state_dict(complete, strict=True)
    return model


def set_trainable_partition(
    model: nn.Module,
    *,
    train_private: bool,
) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (
            is_private(name) if train_private else not is_private(name)
        )


def run_training_phase(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
    train_private: bool,
) -> tuple[float, int]:
    if epochs < 1:
        return 0.0, 0

    set_trainable_partition(model, train_private=train_private)

    if train_private:
        # Keep the shared feature extractor and its BatchNorm statistics fixed,
        # while allowing the private classifier (including dropout) to train.
        model.train()
        model.features.eval()
        model.classifier.train()
        phase_name = "head"
    else:
        # Train the shared representation with the private classifier fixed.
        model.train()
        phase_name = "representation"

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError(
            f"No trainable parameters found for {phase_name} phase."
        )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss(
        weight=local_class_weights(loader.dataset).to(device),
        label_smoothing=0.03,
    )

    total_loss = 0.0
    total_samples = 0

    for epoch in range(1, epochs + 1):
        for batch_index, (images, labels) in enumerate(loader, start=1):
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
                    f"    {phase_name} epoch {epoch}/{epochs} | "
                    f"batch {batch_index}/{len(loader)}",
                    flush=True,
                )

    return (
        total_loss / max(total_samples, 1),
        total_samples,
    )


def train_client(
    shared: OrderedDict[str, torch.Tensor],
    private: OrderedDict[str, torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    *,
    head_epochs: int,
    representation_epochs: int,
    head_learning_rate: float,
    representation_learning_rate: float,
) -> tuple[
    OrderedDict[str, torch.Tensor],
    OrderedDict[str, torch.Tensor],
    int,
    float,
    float,
]:
    model = build_model(shared, private).to(device)

    head_loss, _ = run_training_phase(
        model,
        loader,
        device,
        epochs=head_epochs,
        learning_rate=head_learning_rate,
        train_private=True,
    )

    representation_loss, _ = run_training_phase(
        model,
        loader,
        device,
        epochs=representation_epochs,
        learning_rate=representation_learning_rate,
        train_private=False,
    )

    new_shared, new_private = split_state(model.state_dict())
    num_samples = len(loader.dataset)

    del model

    return (
        new_shared,
        new_private,
        num_samples,
        head_loss,
        representation_loss,
    )


def evaluate_clients(
    shared: OrderedDict[str, torch.Tensor],
    private_by_client: dict[str, OrderedDict[str, torch.Tensor]],
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, dict], dict[str, float]]:
    by_client: dict[str, dict] = {}

    for client_name in sorted(private_by_client):
        model = build_model(
            shared,
            private_by_client[client_name],
        ).to(device)
        by_client[client_name] = evaluate(
            model,
            loader,
            device,
        )
        del model

    metrics_list = list(by_client.values())
    macro_values = [
        metrics["macro_f1"]
        for metrics in metrics_list
    ]

    summary = {
        "mean_accuracy": round(
            float(np.mean([m["accuracy"] for m in metrics_list])),
            4,
        ),
        "mean_balanced_accuracy": round(
            float(
                np.mean(
                    [m["balanced_accuracy"] for m in metrics_list]
                )
            ),
            4,
        ),
        "mean_macro_f1": round(
            float(np.mean(macro_values)),
            4,
        ),
        "mean_weighted_f1": round(
            float(np.mean([m["weighted_f1"] for m in metrics_list])),
            4,
        ),
        "worst_client_macro_f1": round(
            float(np.min(macro_values)),
            4,
        ),
        "best_client_macro_f1": round(
            float(np.max(macro_values)),
            4,
        ),
        "std_client_macro_f1": round(
            float(np.std(macro_values)),
            4,
        ),
    }

    return by_client, summary


def make_loader(
    manifest: str | Path,
    dataset_root: str | Path,
    transform,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = ManifestImageDataset(
        manifest_path=manifest,
        class_to_id=CLASS_TO_ID,
        transform=transform,
        dataset_root=dataset_root,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client_paths = sorted(
        path
        for path in Path(args.client_dir).glob("client_*.csv")
        if path.stem.removeprefix("client_").isdigit()
    )
    if not client_paths:
        raise FileNotFoundError(
            f"No numbered client manifests found in: {args.client_dir}"
        )

    train_transform, eval_transform = make_transforms()

    client_loaders = {
        path.stem: make_loader(
            path,
            args.dataset_root,
            train_transform,
            args.batch_size,
            args.num_workers,
            True,
        )
        for path in client_paths
    }

    validation_loader = make_loader(
        args.validation_manifest,
        args.dataset_root,
        eval_transform,
        args.batch_size,
        args.num_workers,
        False,
    )
    test_loader = make_loader(
        args.test_manifest,
        args.dataset_root,
        eval_transform,
        args.batch_size,
        args.num_workers,
        False,
    )

    checkpoint = torch.load(
        args.initial_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    initial_state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
        else checkpoint
    )

    initial_model = create_mobilenetv2(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
    )
    initial_model.load_state_dict(
        initial_state,
        strict=True,
    )

    shared, initial_private = split_state(
        initial_model.state_dict()
    )
    private_by_client = {
        name: clone_state(initial_private)
        for name in client_loaders
    }

    total_parameters = sum(
        parameter.numel()
        for parameter in initial_model.parameters()
    )
    private_parameters = sum(
        parameter.numel()
        for name, parameter in initial_model.named_parameters()
        if is_private(name)
    )
    shared_parameters = total_parameters - private_parameters

    communication = {
        "total_parameters": total_parameters,
        "shared_parameters": shared_parameters,
        "private_parameters_per_client": private_parameters,
        "shared_model_transfers": (
            2 * len(client_loaders) * args.rounds
        ),
        "estimated_shared_fp32_mib_per_transfer": round(
            shared_parameters * 4 / (1024 ** 2),
            4,
        ),
        "estimated_cumulative_shared_fp32_mib": round(
            shared_parameters
            * 4
            * 2
            * len(client_loaders)
            * args.rounds
            / (1024 ** 2),
            4,
        ),
    }

    print(f"Device: {device}", flush=True)
    print(f"Experiment: {args.experiment_name}", flush=True)
    for name, loader in client_loaders.items():
        print(
            f"{name}: {len(loader.dataset)} images",
            flush=True,
        )
    print(
        f"Shared parameters: {shared_parameters:,}; "
        f"private parameters/client: {private_parameters:,}",
        flush=True,
    )
    print(
        f"FedRep schedule: "
        f"{args.head_epochs} head epoch(s) at "
        f"{args.head_learning_rate:g}, then "
        f"{args.representation_epochs} representation epoch(s) at "
        f"{args.representation_learning_rate:g} per round",
        flush=True,
    )

    history: list[dict] = []
    best_mean_macro_f1 = float("-inf")
    started_at = time.time()

    for round_number in range(1, args.rounds + 1):
        print(
            f"\n===== FedRep round "
            f"{round_number}/{args.rounds} =====",
            flush=True,
        )

        shared_updates: list[
            tuple[OrderedDict[str, torch.Tensor], int]
        ] = []
        new_private_by_client: dict[
            str,
            OrderedDict[str, torch.Tensor],
        ] = {}
        client_results: list[dict] = []

        for client_name, loader in client_loaders.items():
            print(
                f"Starting {client_name}",
                flush=True,
            )

            (
                local_shared,
                local_private,
                num_samples,
                head_loss,
                representation_loss,
            ) = train_client(
                shared,
                private_by_client[client_name],
                loader,
                device,
                head_epochs=args.head_epochs,
                representation_epochs=args.representation_epochs,
                head_learning_rate=args.head_learning_rate,
                representation_learning_rate=(
                    args.representation_learning_rate
                ),
            )

            shared_updates.append(
                (local_shared, num_samples)
            )
            new_private_by_client[client_name] = (
                local_private
            )
            client_results.append(
                {
                    "client": client_name,
                    "num_samples": num_samples,
                    "head_loss": round(head_loss, 6),
                    "representation_loss": round(
                        representation_loss,
                        6,
                    ),
                }
            )

            print(
                f"Finished {client_name}: "
                f"samples={num_samples}, "
                f"head_loss={head_loss:.4f}, "
                f"representation_loss="
                f"{representation_loss:.4f}",
                flush=True,
            )

        shared = fedavg(shared_updates)
        private_by_client = new_private_by_client

        (
            validation_by_client,
            validation_summary,
        ) = evaluate_clients(
            shared,
            private_by_client,
            validation_loader,
            device,
        )

        print(
            json.dumps(
                validation_by_client,
                indent=2,
            ),
            flush=True,
        )
        print(
            f"Validation summary: "
            f"{validation_summary}",
            flush=True,
        )

        history.append(
            {
                "round": round_number,
                "clients": client_results,
                "validation_by_client": (
                    validation_by_client
                ),
                "validation_summary": (
                    validation_summary
                ),
                "elapsed_seconds": round(
                    time.time() - started_at,
                    2,
                ),
            }
        )

        if (
            validation_summary["mean_macro_f1"]
            > best_mean_macro_f1
        ):
            best_mean_macro_f1 = float(
                validation_summary["mean_macro_f1"]
            )

            torch.save(
                {
                    "round": round_number,
                    "shared_state_dict": clone_state(
                        shared
                    ),
                    "client_private_state_dicts": {
                        name: clone_state(state)
                        for name, state
                        in private_by_client.items()
                    },
                    "validation_by_client": (
                        validation_by_client
                    ),
                    "validation_summary": (
                        validation_summary
                    ),
                    "communication": communication,
                    "arguments": vars(args),
                    "class_names": CLASS_NAMES,
                    "experiment": args.experiment_name,
                    "algorithm": "FedRep",
                },
                output_dir / "best_fedrep_state.pt",
            )

            print(
                f"New best FedRep state: "
                f"round={round_number}, "
                f"mean validation Macro-F1="
                f"{best_mean_macro_f1:.4f}%",
                flush=True,
            )

        torch.save(
            {
                "round": round_number,
                "shared_state_dict": clone_state(shared),
                "client_private_state_dicts": {
                    name: clone_state(state)
                    for name, state in private_by_client.items()
                },
                "validation_by_client": validation_by_client,
                "validation_summary": validation_summary,
                "communication": communication,
                "arguments": vars(args),
                "class_names": CLASS_NAMES,
                "experiment": args.experiment_name,
                "algorithm": "FedRep",
            },
            output_dir / f"fedrep_state_round_{round_number}.pt",
        )

        with open(
            output_dir / "metrics.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "experiment": args.experiment_name,
                    "algorithm": "FedRep",
                    "history": history,
                    "best_mean_validation_macro_f1": (
                        best_mean_macro_f1
                    ),
                    "communication": communication,
                    "arguments": vars(args),
                },
                handle,
                indent=2,
            )

    best = torch.load(
        output_dir / "best_fedrep_state.pt",
        map_location="cpu",
        weights_only=False,
    )

    test_by_client, test_summary = evaluate_clients(
        clone_state(best["shared_state_dict"]),
        {
            name: clone_state(state)
            for name, state
            in best[
                "client_private_state_dicts"
            ].items()
        },
        test_loader,
        device,
    )

    test_result = {
        "experiment": args.experiment_name,
        "algorithm": "FedRep",
        "selected_round": int(best["round"]),
        "protocol": (
            "Each personalized client model is evaluated on the same "
            "capture-group-disjoint rice test manifest."
        ),
        "training_schedule": {
            "head_epochs_per_round": args.head_epochs,
            "representation_epochs_per_round": (
                args.representation_epochs
            ),
            "head_learning_rate": (
                args.head_learning_rate
            ),
            "representation_learning_rate": (
                args.representation_learning_rate
            ),
        },
        "test_by_client": test_by_client,
        "test_summary": test_summary,
        "communication": communication,
    }

    with open(
        output_dir / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            test_result,
            handle,
            indent=2,
        )

    print("\n===== Test results =====", flush=True)
    print(
        json.dumps(test_result, indent=2),
        flush=True,
    )
    print(
        f"Completed in "
        f"{(time.time() - started_at) / 3600:.2f} hours",
        flush=True,
    )


if __name__ == "__main__":
    main()
