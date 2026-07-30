# Internship Research Results

This directory contains the organized experiment results for the grain
classification internship research project.

## Structure

- Rice/Centralized
- Rice/Federated/FedAvg
- Rice/Federated/FedPer
- Rice/Federated/FedRep
- Rice/DDP
- Rice/DataValidation
- Wheat
- Development
- Reports
- Logs
- ProjectMetadata

## Federated configurations

- IID, three clients, seed 42
- Non-IID, Dirichlet alpha 0.5, three clients, seed 42

## Model checkpoints

Large PyTorch model checkpoints are not committed to GitHub.

The authoritative models remain on Roihu under:

`/scratch/project_2019765/fnaveed/results`

Their paths and sizes are listed in:

`CHECKPOINTS_ON_ROIHU.tsv`

The repository includes metrics, histories, predictions, confusion matrices,
per-class results, experiment configurations, communication information,
SLURM logs, and validation evidence.
