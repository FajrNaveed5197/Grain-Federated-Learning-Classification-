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
TRAIN_MANIFEST = ROOT / "manifests/train.csv"
VAL_MANIFEST = ROOT / "manifests/validation.csv"
TEST1_MANIFEST = ROOT / "manifests/test.csv"
TEST2_MANIFEST = ROOT / "manifests/test_07.csv"

V2_CHECKPOINT = (
    ROOT
    / "results/additional_finetuning_v2_resnet18"
    / "best_resnet18_v2_gpu.pt"
)

OUTPUT_DIR = ROOT / "results/version3_dynamic_balanced_resnet18"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = OUTPUT_DIR / "best_resnet18_v3_gpu.pt"
RESULT_PATH = OUTPUT_DIR / "metrics.json"


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


def create_balanced_epoch(dataframe, per_class, epoch_seed):
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


def create_balanced_fixed(dataframe, per_class, seed):
    return create_balanced_epoch(dataframe, per_class, seed)


def calculate_metrics(targets, predictions, include_per_class=False):
    targets = np.array(targets)
    predictions = np.array(predictions)

    accuracy = float((targets == predictions).mean())
    recalls = []
    f1_scores = []
    per_class = {}

    for class_id, class_name in enumerate(CLASS_NAMES):
        tp = np.sum((predictions == class_id) & (targets == class_id))
        fp = np.sum((predictions == class_id) & (targets != class_id))
        fn = np.sum((predictions != class_id) & (targets == class_id))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        recalls.append(recall)
        f1_scores.append(f1)

        per_class[class_name] = {
            "precision": round(float(precision) * 100, 2),
            "recall": round(float(recall) * 100, 2),
            "f1": round(float(f1) * 100, 2),
        }

    result = {
        "accuracy": round(accuracy * 100, 2),
        "balanced_accuracy": round(float(np.mean(recalls)) * 100, 2),
        "macro_f1": round(float(np.mean(f1_scores)) * 100, 2),
    }

    if include_per_class:
        result["per_class"] = per_class

    return result


def evaluate(model, loader, device, include_per_class=False):
    model.eval()
    targets = []
    predictions = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device, non_blocking=True))
            predicted = outputs.argmax(dim=1).cpu().tolist()

            predictions.extend(predicted)
            targets.extend(labels.tolist())

    return calculate_metrics(
        targets,
        predictions,
        include_per_class=include_per_class,
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    return total_loss / total_samples


def main():
    seed = 42
    epochs = 8
    per_class_train = 1738

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    if not V2_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing V2 checkpoint: {V2_CHECKPOINT}")

    full_train_df = pd.read_csv(TRAIN_MANIFEST)
    full_val_df = pd.read_csv(VAL_MANIFEST)
    test1_full_df = pd.read_csv(TEST1_MANIFEST)
    test2_full_df = pd.read_csv(TEST2_MANIFEST)

    balanced_val_df = create_balanced_fixed(
        full_val_df,
        per_class=194,
        seed=seed,
    )
    balanced_test1_df = create_balanced_fixed(
        test1_full_df,
        per_class=200,
        seed=seed,
    )
    balanced_test2_df = create_balanced_fixed(
        test2_full_df,
        per_class=200,
        seed=seed,
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

    val_loader = DataLoader(
        GrainDataset(balanced_val_df, eval_transform),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    full_val_loader = DataLoader(
        GrainDataset(full_val_df, eval_transform),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    test1_loader = DataLoader(
        GrainDataset(balanced_test1_df, eval_transform),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    test2_loader = DataLoader(
        GrainDataset(balanced_test2_df, eval_transform),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model = model.to(device)

    model.load_state_dict(torch.load(V2_CHECKPOINT, map_location=device))

    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-6,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=5e-7,
    )

    starting_validation = evaluate(model, val_loader, device)
    best_validation_f1 = starting_validation["macro_f1"]
    best_epoch = 0
    selected_checkpoint = "Version 2 checkpoint retained"

    print("GPU:", torch.cuda.get_device_name(0))
    print("Base checkpoint:", V2_CHECKPOINT)
    print("Training method: fresh balanced sample every epoch")
    print("Training images per epoch:", per_class_train * len(CLASS_NAMES))
    print("Starting balanced validation:", starting_validation)

    history = []
    started = time.time()

    for epoch in range(1, epochs + 1):
        epoch_df = create_balanced_epoch(
            full_train_df,
            per_class=per_class_train,
            epoch_seed=seed + epoch * 100,
        )

        train_loader = DataLoader(
            GrainDataset(epoch_df, train_transform),
            batch_size=128,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

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
            f"Version 3 epoch {epoch}/{epochs} | "
            f"loss={loss:.4f} | "
            f"balanced val macro-F1={validation['macro_f1']:.2f}%"
        )

        history.append({
            "epoch": epoch,
            "loss": loss,
            "validation": validation,
        })

        if validation["macro_f1"] > best_validation_f1:
            best_validation_f1 = validation["macro_f1"]
            best_epoch = epoch
            selected_checkpoint = f"Version 3 epoch {epoch}"
            torch.save(model.state_dict(), BEST_CHECKPOINT)

    if best_epoch > 0:
        model.load_state_dict(torch.load(BEST_CHECKPOINT, map_location=device))
    else:
        model.load_state_dict(torch.load(V2_CHECKPOINT, map_location=device))

    results = {
        "run_type": "version3_dynamic_balanced_resampling",
        "model": "ResNet18",
        "device": torch.cuda.get_device_name(0),
        "base_checkpoint": str(V2_CHECKPOINT),
        "training": (
            "8 continuation epochs from best V2 checkpoint; "
            "fresh balanced sample each epoch; "
            "13,904 images per epoch; "
            "learning rate 2e-6"
        ),
        "starting_balanced_validation": starting_validation,
        "best_balanced_validation_macro_f1": best_validation_f1,
        "best_v3_epoch": best_epoch,
        "selected_checkpoint": selected_checkpoint,
        "balanced_validation": evaluate(
            model,
            val_loader,
            device,
            include_per_class=True,
        ),
        "full_validation": evaluate(
            model,
            full_val_loader,
            device,
            include_per_class=True,
        ),
        "test_set_1": evaluate(
            model,
            test1_loader,
            device,
            include_per_class=True,
        ),
        "test_set_2": evaluate(
            model,
            test2_loader,
            device,
            include_per_class=True,
        ),
        "elapsed_minutes": round((time.time() - started) / 60, 2),
        "history": history,
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nFINAL VERSION 3 RESULTS")
    print(json.dumps(results, indent=2))
    print("\nSaved:", RESULT_PATH)


if __name__ == "__main__":
    main()
