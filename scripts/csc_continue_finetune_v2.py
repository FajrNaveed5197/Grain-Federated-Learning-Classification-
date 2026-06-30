import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
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

ROOT = Path("/scratch/project_2019649/grain_research")
RESULTS_ROOT = ROOT / "results"

TRAIN_MANIFEST = ROOT / "manifests/train.csv"
VAL_MANIFEST = ROOT / "manifests/validation.csv"
TEST1_MANIFEST = ROOT / "manifests/test.csv"
TEST2_MANIFEST = ROOT / "manifests/test_07.csv"


CONFIG = {
    "mobilenetv2": {
        "display_name": "MobileNetV2",
        "v1_dir": "first_time_finetuning_v1_mobilenetv2",
        "v1_checkpoint": "best_mobilenetv2_gpu.pt",
        "v2_dir": "additional_finetuning_v2_mobilenetv2",
        "v2_checkpoint": "best_mobilenetv2_v2_gpu.pt",
    },
    "resnet18": {
        "display_name": "ResNet18",
        "v1_dir": "first_time_finetuning_v1_resnet18",
        "v1_checkpoint": "best_resnet18_gpu.pt",
        "v2_dir": "additional_finetuning_v2_resnet18",
        "v2_checkpoint": "best_resnet18_v2_gpu.pt",
    },
    "efficientnetb0": {
        "display_name": "EfficientNet-B0",
        "v1_dir": "first_time_finetuning_v1_efficientnetb0",
        "v1_checkpoint": "best_efficientnetb0_gpu.pt",
        "v2_dir": "additional_finetuning_v2_efficientnetb0",
        "v2_checkpoint": "best_efficientnetb0_v2_gpu.pt",
    },
}


class GrainDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        label = CLASS_TO_ID[row["label"]]
        return self.transform(image), label


def balanced_subset(manifest_path, per_class, seed):
    dataframe = pd.read_csv(manifest_path)
    groups = []

    for label in CLASS_NAMES:
        group = dataframe[dataframe["label"] == label]

        if len(group) < per_class:
            raise ValueError(
                f"{manifest_path}: {label} has {len(group)} images, "
                f"but {per_class} requested."
            )

        groups.append(group.sample(n=per_class, random_state=seed))

    return (
        pd.concat(groups)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def calculate_metrics(targets, predictions):
    targets = np.array(targets)
    predictions = np.array(predictions)

    accuracy = float((targets == predictions).mean())
    per_class_recall = []
    per_class_f1 = []

    for class_id in range(len(CLASS_NAMES)):
        true_positive = np.sum((predictions == class_id) & (targets == class_id))
        false_positive = np.sum((predictions == class_id) & (targets != class_id))
        false_negative = np.sum((predictions != class_id) & (targets == class_id))

        recall = true_positive / max(true_positive + false_negative, 1)
        precision = true_positive / max(true_positive + false_positive, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        per_class_recall.append(recall)
        per_class_f1.append(f1)

    return {
        "accuracy": round(accuracy * 100, 2),
        "balanced_accuracy": round(float(np.mean(per_class_recall)) * 100, 2),
        "macro_f1": round(float(np.mean(per_class_f1)) * 100, 2),
    }


def evaluate(model, loader, device):
    model.eval()
    targets = []
    predictions = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device, non_blocking=True))
            predicted = outputs.argmax(dim=1).cpu().tolist()
            predictions.extend(predicted)
            targets.extend(labels.tolist())

    return calculate_metrics(targets, predictions)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_examples += labels.size(0)

    return total_loss / total_examples


def build_model(model_name):
    if model_name == "mobilenetv2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            len(CLASS_NAMES),
        )

    elif model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(
            model.fc.in_features,
            len(CLASS_NAMES),
        )

    elif model_name == "efficientnetb0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            len(CLASS_NAMES),
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["mobilenetv2", "resnet18", "efficientnetb0"],
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    args = parser.parse_args()

    cfg = CONFIG[args.model]
    v1_checkpoint = RESULTS_ROOT / cfg["v1_dir"] / cfg["v1_checkpoint"]
    output_dir = RESULTS_ROOT / cfg["v2_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    v2_checkpoint = output_dir / cfg["v2_checkpoint"]

    if not v1_checkpoint.exists():
        raise FileNotFoundError(
            f"Version 1 checkpoint not found: {v1_checkpoint}"
        )

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    train_df = balanced_subset(TRAIN_MANIFEST, per_class=1738, seed=seed)
    val_df = balanced_subset(VAL_MANIFEST, per_class=194, seed=seed)
    test1_df = balanced_subset(TEST1_MANIFEST, per_class=200, seed=seed)
    test2_df = balanced_subset(TEST2_MANIFEST, per_class=200, seed=seed)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.80, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.12,
            contrast=0.12,
            saturation=0.08,
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

    train_loader = DataLoader(
        GrainDataset(train_df, train_transform),
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        GrainDataset(val_df, eval_transform),
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    test1_loader = DataLoader(
        GrainDataset(test1_df, eval_transform),
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    test2_loader = DataLoader(
        GrainDataset(test2_df, eval_transform),
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(v1_checkpoint, map_location=device))

    for parameter in model.parameters():
        parameter.requires_grad = True

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    starting_validation = evaluate(model, val_loader, device)
    best_validation_f1 = starting_validation["macro_f1"]
    best_epoch = 0
    selected_checkpoint = "Version 1 checkpoint retained"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    history = []
    started = time.time()

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", cfg["display_name"])
    print("Version 1 checkpoint:", v1_checkpoint)
    print("Starting validation:", starting_validation)
    print(
        f"\nVERSION 2: continuation fine-tuning, "
        f"{args.epochs} epochs, learning rate={args.learning_rate}"
    )

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        scheduler.step()

        validation = evaluate(model, val_loader, device)

        print(
            f"Version 2 epoch {epoch}/{args.epochs} | "
            f"loss={loss:.4f} | "
            f"val macro-F1={validation['macro_f1']:.2f}%"
        )

        history.append({
            "epoch": epoch,
            "loss": loss,
            "validation": validation,
        })

        if validation["macro_f1"] > best_validation_f1:
            best_validation_f1 = validation["macro_f1"]
            best_epoch = epoch
            selected_checkpoint = f"Version 2 epoch {epoch}"
            torch.save(model.state_dict(), v2_checkpoint)

    if best_epoch > 0:
        model.load_state_dict(torch.load(v2_checkpoint, map_location=device))
    else:
        model.load_state_dict(torch.load(v1_checkpoint, map_location=device))

    results = {
        "run_type": "additional_finetuning_v2",
        "model": cfg["display_name"],
        "device": torch.cuda.get_device_name(0),
        "base_checkpoint": str(v1_checkpoint),
        "training": (
            f"Continuation from Version 1 best checkpoint; "
            f"{args.epochs} full-model epochs; "
            f"learning rate {args.learning_rate}"
        ),
        "starting_validation": starting_validation,
        "best_validation_macro_f1": best_validation_f1,
        "best_v2_epoch": best_epoch,
        "selected_checkpoint": selected_checkpoint,
        "validation": evaluate(model, val_loader, device),
        "test_set_1": evaluate(model, test1_loader, device),
        "test_set_2": evaluate(model, test2_loader, device),
        "elapsed_minutes": round((time.time() - started) / 60, 2),
        "history": history,
    }

    result_path = output_dir / "metrics.json"

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nFINAL VERSION 2 RESULTS")
    print(json.dumps(results, indent=2))
    print(f"\nSaved results: {result_path}")


if __name__ == "__main__":
    main()
