from __future__ import annotations

import os
import time
from collections import OrderedDict
from pathlib import Path

import flwr as fl
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


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class GrainFlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: str,
        manifest_path: str,
        batch_size: int,
        local_epochs: int,
        learning_rate: float,
    ) -> None:
        self.client_id = client_id
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = create_resnet18(
            num_classes=len(CLASS_NAMES),
            pretrained=False,
        ).to(self.device)

        dataset = ManifestImageDataset(
            manifest_path=Path(manifest_path),
            class_to_id=CLASS_TO_ID,
            transform=build_transform(),
        )

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        self.local_epochs = local_epochs
        self.learning_rate = learning_rate

        print(
            f"{self.client_id}: {len(dataset)} local images | "
            f"device={self.device}",
            flush=True,
        )

    def get_parameters(self, config):
        return [
            value.detach().cpu().numpy()
            for value in self.model.state_dict().values()
        ]

    def set_parameters(self, parameters) -> None:
        current_state = self.model.state_dict()

        updated_state = OrderedDict(
            (
                key,
                torch.from_numpy(array).to(dtype=current_state[key].dtype),
            )
            for key, array in zip(current_state.keys(), parameters)
        )

        self.model.load_state_dict(updated_state, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        local_epochs = int(config.get("local_epochs", self.local_epochs))
        learning_rate = float(
            config.get("learning_rate", self.learning_rate)
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for _ in range(local_epochs):
            for images, labels in self.loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                output = self.model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

        average_loss = total_loss / max(total_samples, 1)

        print(
            f"{self.client_id}: local training complete | "
            f"loss={average_loss:.4f}",
            flush=True,
        )

        return (
            self.get_parameters({}),
            len(self.loader.dataset),
            {"local_loss": float(average_loss)},
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.loader.dataset), {}


def main() -> None:
    address = os.environ["FL_SERVER_ADDRESS"]
    client_id = os.environ["FL_CLIENT_ID"]
    manifest_path = os.environ["FL_CLIENT_MANIFEST"]

    client = GrainFlowerClient(
        client_id=client_id,
        manifest_path=manifest_path,
        batch_size=int(os.getenv("FL_BATCH_SIZE", "32")),
        local_epochs=int(os.getenv("FL_LOCAL_EPOCHS", "1")),
        learning_rate=float(os.getenv("FL_LEARNING_RATE", "0.000002")),
    )

    for attempt in range(1, 31):
        try:
            print(
                f"{client_id}: connecting to {address} "
                f"(attempt {attempt}/30)",
                flush=True,
            )

            fl.client.start_client(
                server_address=address,
                client=client.to_client(),
            )
            return

        except Exception as error:
            print(
                f"{client_id}: server not ready: {error}",
                flush=True,
            )
            time.sleep(3)

    raise RuntimeError(f"{client_id}: could not connect to {address}")


if __name__ == "__main__":
    main()
