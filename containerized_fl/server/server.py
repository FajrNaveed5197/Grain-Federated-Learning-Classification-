from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path

import flwr as fl
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from federated_pipeline.common.model import create_resnet18


NUM_CLASSES = 8


def model_parameters() -> object:
    model = create_resnet18(num_classes=NUM_CLASSES, pretrained=False)
    arrays = [
        tensor.detach().cpu().numpy()
        for tensor in model.state_dict().values()
    ]
    return ndarrays_to_parameters(arrays)


class CheckpointingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, run_dir: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_keys = list(
            create_resnet18(
                num_classes=NUM_CLASSES,
                pretrained=False,
            ).state_dict().keys()
        )

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if parameters is None:
            return parameters, metrics

        arrays = parameters_to_ndarrays(parameters)

        model = create_resnet18(
            num_classes=NUM_CLASSES,
            pretrained=False,
        )

        state_dict = OrderedDict(
            (key, torch.from_numpy(array))
            for key, array in zip(self.state_keys, arrays)
        )

        model.load_state_dict(state_dict, strict=True)

        checkpoint = self.run_dir / f"global_model_round_{server_round}.pt"
        torch.save(model.state_dict(), checkpoint)

        with open(
            self.run_dir / "server_rounds.jsonl",
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    {
                        "round": server_round,
                        "successful_clients": len(results),
                        "failed_clients": len(failures),
                        "checkpoint": str(checkpoint),
                    }
                )
                + "\n"
            )

        print(f"Saved {checkpoint}", flush=True)
        return parameters, metrics


def main() -> None:
    address = os.getenv("FL_SERVER_ADDRESS", "0.0.0.0:8080")
    rounds = int(os.getenv("FL_ROUNDS", "3"))
    run_dir = Path(os.getenv("FL_RUNS_DIR", "/app/runs"))

    def fit_config(server_round: int) -> dict[str, object]:
        return {
            "server_round": server_round,
            "local_epochs": int(os.getenv("FL_LOCAL_EPOCHS", "1")),
            "batch_size": int(os.getenv("FL_BATCH_SIZE", "32")),
            "learning_rate": float(os.getenv("FL_LEARNING_RATE", "0.000002")),
        }

    strategy = CheckpointingFedAvg(
        run_dir=run_dir,
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        min_evaluate_clients=0,
        fraction_evaluate=0.0,
        initial_parameters=model_parameters(),
        on_fit_config_fn=fit_config,
    )

    print(
        f"Starting FedAvg server at {address} for {rounds} rounds",
        flush=True,
    )

    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
