#!/bin/bash
#SBATCH --job-name=fl_fedavg_smoke
#SBATCH --account=project_2019649
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_2019649/grain_research/logs/fl_fedavg_smoke_%j.out
#SBATCH --error=/scratch/project_2019649/grain_research/logs/fl_fedavg_smoke_%j.err

module load pytorch

cd /projappl/project_2019649/grain_research/code/grain_project

PYTHONPATH=src python -m federated_pipeline.federated.fedavg_simulation \
  --client-dir /scratch/project_2019649/grain_research/partitions/fl_smoke_noniid_3clients \
  --validation-manifest /scratch/project_2019649/grain_research/manifests/validation.csv \
  --output-dir /scratch/project_2019649/grain_research/federated_results/fl_smoke_noniid_3clients \
  --rounds 3 \
  --local-epochs 1 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --num-workers 4 \
  --seed 42
