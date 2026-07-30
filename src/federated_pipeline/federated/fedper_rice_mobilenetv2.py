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


def parse_args():
    parser = argparse.ArgumentParser(
        description="FedPer MobileNetV2 on the three-client non-IID rice split."
    )
    parser.add_argument("--client-dir", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clone_state(state):
    return OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in state.items()
    )


def is_private(name: str) -> bool:
    return name.startswith(PRIVATE_PREFIX)


def split_state(state):
    shared, private = OrderedDict(), OrderedDict()
    for name, tensor in state.items():
        (private if is_private(name) else shared)[name] = (
            tensor.detach().cpu().clone()
        )
    if set(private) != PRIVATE_KEYS:
        raise RuntimeError(
            f"Expected private keys {sorted(PRIVATE_KEYS)}, found {sorted(private)}"
        )
    return shared, private


def build_model(shared, private):
    model = create_mobilenetv2(len(CLASS_NAMES), pretrained=False)
    complete = OrderedDict()
    for name in model.state_dict():
        source = private if is_private(name) else shared
        complete[name] = source[name]
    model.load_state_dict(complete, strict=True)
    return model


def train_client(shared, private, loader, device, epochs, lr):
    model = build_model(shared, private).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss(
        weight=local_class_weights(loader.dataset).to(device),
        label_smoothing=0.03,
    )

    total_loss = 0.0
    total_samples = 0
    for _ in range(epochs):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

    new_shared, new_private = split_state(model.state_dict())
    del model
    return (
        new_shared,
        new_private,
        len(loader.dataset),
        total_loss / max(total_samples, 1),
    )


def evaluate_clients(shared, private_by_client, loader, device):
    by_client = {}
    for client_name in sorted(private_by_client):
        model = build_model(shared, private_by_client[client_name]).to(device)
        by_client[client_name] = evaluate(model, loader, device)
        del model

    macro_values = [metrics["macro_f1"] for metrics in by_client.values()]
    summary = {
        "mean_accuracy": round(
            float(np.mean([m["accuracy"] for m in by_client.values()])), 4
        ),
        "mean_balanced_accuracy": round(
            float(np.mean([m["balanced_accuracy"] for m in by_client.values()])), 4
        ),
        "mean_macro_f1": round(float(np.mean(macro_values)), 4),
        "mean_weighted_f1": round(
            float(np.mean([m["weighted_f1"] for m in by_client.values()])), 4
        ),
        "worst_client_macro_f1": round(float(np.min(macro_values)), 4),
        "best_client_macro_f1": round(float(np.max(macro_values)), 4),
        "std_client_macro_f1": round(float(np.std(macro_values)), 4),
    }
    return by_client, summary


def make_loader(manifest, root, transform, batch_size, workers, shuffle):
    dataset = ManifestImageDataset(
        manifest_path=manifest,
        class_to_id=CLASS_TO_ID,
        transform=transform,
        dataset_root=root,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client_paths = sorted(
        path
        for path in Path(args.client_dir).glob("client_*.csv")
        if path.stem.removeprefix("client_").isdigit()
    )
    if not client_paths:
        raise FileNotFoundError(f"No client manifests in {args.client_dir}")

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
        args.initial_checkpoint, map_location="cpu", weights_only=False
    )
    initial_state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    initial_model = create_mobilenetv2(len(CLASS_NAMES), pretrained=False)
    initial_model.load_state_dict(initial_state)
    shared, initial_private = split_state(initial_model.state_dict())
    private_by_client = {
        name: clone_state(initial_private) for name in client_loaders
    }

    total_params = sum(p.numel() for p in initial_model.parameters())
    private_params = sum(
        p.numel()
        for name, p in initial_model.named_parameters()
        if is_private(name)
    )
    shared_parameters = total_params - private_params
    shared_model_transfers = (
        2 * len(client_loaders) * args.rounds
    )
    communication = {
        "total_parameters": total_params,
        "shared_parameters": shared_parameters,
        "private_parameters_per_client": private_params,
        "complete_transfers": shared_model_transfers,
        "shared_model_transfers": shared_model_transfers,
        "estimated_shared_fp32_mib_per_transfer": round(
            shared_parameters * 4 / (1024 ** 2),
            4,
        ),
        "estimated_cumulative_shared_fp32_mib": round(
            shared_parameters
            * 4
            * shared_model_transfers
            / (1024 ** 2),
            4,
        ),
    }

    print(f"Device: {device}")
    for name, loader in client_loaders.items():
        print(f"{name}: {len(loader.dataset)} images")
    print(
        f"Shared parameters: {communication['shared_parameters']:,}; "
        f"private parameters/client: {private_params:,}"
    )

    history = []
    best_mean_macro_f1 = float("-inf")
    started = time.time()

    for round_number in range(1, args.rounds + 1):
        print(f"\n===== FedPer round {round_number}/{args.rounds} =====")
        shared_updates = []
        new_private_by_client = {}
        local_results = []

        for client_name, loader in client_loaders.items():
            local_shared, local_private, samples, loss = train_client(
                shared,
                private_by_client[client_name],
                loader,
                device,
                args.local_epochs,
                args.learning_rate,
            )
            shared_updates.append((local_shared, samples))
            new_private_by_client[client_name] = local_private
            local_results.append(
                {
                    "client": client_name,
                    "num_samples": samples,
                    "local_loss": round(loss, 6),
                }
            )
            print(f"{client_name}: samples={samples}, loss={loss:.4f}")

        shared = fedavg(shared_updates)
        private_by_client = new_private_by_client
        validation_by_client, validation_summary = evaluate_clients(
            shared, private_by_client, validation_loader, device
        )
        print(json.dumps(validation_by_client, indent=2))
        print(f"Validation summary: {validation_summary}")

        history.append(
            {
                "round": round_number,
                "clients": local_results,
                "validation_by_client": validation_by_client,
                "validation_summary": validation_summary,
                "elapsed_seconds": round(time.time() - started, 2),
            }
        )

        if validation_summary["mean_macro_f1"] > best_mean_macro_f1:
            best_mean_macro_f1 = validation_summary["mean_macro_f1"]
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
                    "algorithm": "FedPer",
                },
                output_dir / "best_fedper_state.pt",
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
                "algorithm": "FedPer",
            },
            output_dir / f"fedper_state_round_{round_number}.pt",
        )

        with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "experiment": args.experiment_name,
                    "algorithm": "FedPer",
                    "history": history,
                    "best_mean_validation_macro_f1": best_mean_macro_f1,
                    "communication": communication,
                    "arguments": vars(args),
                },
                handle,
                indent=2,
            )

    best = torch.load(
        output_dir / "best_fedper_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    test_by_client, test_summary = evaluate_clients(
        clone_state(best["shared_state_dict"]),
        {
            name: clone_state(state)
            for name, state in best["client_private_state_dicts"].items()
        },
        test_loader,
        device,
    )
    test_result = {
        "experiment": args.experiment_name,
        "algorithm": "FedPer",
        "selected_round": int(best["round"]),
        "protocol": (
            "Each personalized client model is evaluated on the same "
            "capture-group-disjoint rice test manifest."
        ),
        "test_by_client": test_by_client,
        "test_summary": test_summary,
        "communication": communication,
    }
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(test_result, handle, indent=2)

    print("\n===== Test results =====")
    print(json.dumps(test_result, indent=2))
    print(f"Completed in {(time.time() - started) / 3600:.2f} hours")


if __name__ == "__main__":
    main()
