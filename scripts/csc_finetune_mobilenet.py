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

TRAIN_MANIFEST = Path("/scratch/project_2019649/grain_research/manifests/train.csv")
VAL_MANIFEST = Path("/scratch/project_2019649/grain_research/manifests/validation.csv")
TEST1_MANIFEST = Path("/scratch/project_2019649/grain_research/manifests/test.csv")
TEST2_MANIFEST = Path("/scratch/project_2019649/grain_research/manifests/test_07.csv")

OUTPUT_DIR = Path("/scratch/project_2019649/grain_research/results/presentation_finetune_mobilenet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

        f1 = (
            2 * precision * recall / max(precision + recall, 1e-12)
        )

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


def main():
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

    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(CLASS_NAMES),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_validation_f1 = -1.0
    best_checkpoint = OUTPUT_DIR / "best_mobilenetv2_gpu.pt"
    history = []
    started = time.time()

    print("GPU:", torch.cuda.get_device_name(0))
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test Set 1:", len(test1_df))
    print("Test Set 2:", len(test2_df))

    print("\nSTAGE 1: classifier-head warm-up, 2 epochs")

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    warmup_optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    for epoch in range(1, 3):
        loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            warmup_optimizer,
            device,
        )
        validation = evaluate(model, val_loader, device)

        print(
            f"Warm-up {epoch}/2 | loss={loss:.4f} | "
            f"val macro-F1={validation['macro_f1']:.2f}%"
        )

        history.append({
            "stage": "warmup",
            "epoch": epoch,
            "loss": loss,
            "validation": validation,
        })

        if validation["macro_f1"] > best_validation_f1:
            best_validation_f1 = validation["macro_f1"]
            torch.save(model.state_dict(), best_checkpoint)

    print("\nSTAGE 2: full end-to-end fine-tuning, 8 epochs")

    for parameter in model.features.parameters():
        parameter.requires_grad = True

    finetune_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-5,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        finetune_optimizer,
        T_max=8,
    )

    for epoch in range(1, 9):
        loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            finetune_optimizer,
            device,
        )
        scheduler.step()

        validation = evaluate(model, val_loader, device)

        print(
            f"Fine-tune {epoch}/8 | loss={loss:.4f} | "
            f"val macro-F1={validation['macro_f1']:.2f}%"
        )

        history.append({
            "stage": "finetune",
            "epoch": epoch,
            "loss": loss,
            "validation": validation,
        })

        if validation["macro_f1"] > best_validation_f1:
            best_validation_f1 = validation["macro_f1"]
            torch.save(model.state_dict(), best_checkpoint)

    model.load_state_dict(
        torch.load(best_checkpoint, map_location=device)
    )

    results = {
        "run_type": "presentation_preliminary_gpu_finetuning",
        "official_experiment": False,
        "model": "MobileNetV2",
        "device": torch.cuda.get_device_name(0),
        "training": "2 frozen-head warm-up epochs + 8 full fine-tuning epochs",
        "train_images": len(train_df),
        "validation_images": len(val_df),
        "test1_images": len(test1_df),
        "test2_images": len(test2_df),
        "best_validation_macro_f1": best_validation_f1,
        "validation": evaluate(model, val_loader, device),
        "test_set_1": evaluate(model, test1_loader, device),
        "test_set_2": evaluate(model, test2_loader, device),
        "elapsed_minutes": round((time.time() - started) / 60, 2),
        "history": history,
    }

    result_path = OUTPUT_DIR / "metrics.json"

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nFINAL RESULTS")
    print(json.dumps(results, indent=2))
    print(f"\nSaved results: {result_path}")


if __name__ == "__main__":
    main()
