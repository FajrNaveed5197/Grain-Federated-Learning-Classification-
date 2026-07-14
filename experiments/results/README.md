# Experimental Results

This directory contains reproducible evaluation outputs for the grain-classification research project.

## Included

- Centralized rice experiments:
  - ResNet18
  - MobileNetV2
  - EfficientNetB0
- Federated rice experiments:
  - ResNet18 IID and non-IID
  - MobileNetV2 IID and non-IID
- Distributed rice experiment:
  - ResNet18 using PyTorch Distributed Data Parallel on 2 GPUs
- Clean group-aware wheat experiments:
  - ResNet18 with full inverse-frequency weighting
  - ResNet18 with square-root inverse-frequency weighting
- Dataset audit summaries
- Confusion matrices
- Per-class metrics
- Predictions
- Training histories
- Final comparison tables
- Checkpoint provenance and SHA-256 inventory

## Excluded

The image datasets, generated train/validation/test manifests, virtual environments, runtime logs, and binary checkpoints are excluded from Git.

The binary checkpoints remain stored on CSC. Their paths, sizes, timestamps, and SHA-256 hashes are recorded in:

`provenance/checkpoint_inventory.csv`

## Reproducibility

Training and evaluation scripts are available under:

- `scripts/`
- `src/federated_pipeline/`
- `slurm/`

The result files in this directory were generated directly from the saved model checkpoints and experiment outputs.
