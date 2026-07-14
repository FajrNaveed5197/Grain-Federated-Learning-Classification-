#!/bin/bash
#SBATCH --job-name=rice_fl_iid_smoke
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/project_2019765/fnaveed/logs/rice_fl_iid_smoke_%j.out
#SBATCH --error=/scratch/project_2019765/fnaveed/logs/rice_fl_iid_smoke_%j.err

set -euo pipefail

module load python-pytorch
source /projappl/project_2019765/grain_research/fl_venv/bin/activate

cd /projappl/project_2019765/grain_research/code/grain_project

mkdir -p \
  /scratch/project_2019765/fnaveed/logs \
  /scratch/project_2019765/fnaveed/results/rice_fedavg_iid_resnet18_smoke

export PYTHONPATH=src
export PYTHONUNBUFFERED=1

python -m federated_pipeline.federated.fedavg_rice_resnet18 \
  --client-dir \
  /scratch/project_2019765/fnaveed/datasets/rice_grouped/federated_partitions/iid_3clients_seed42_v2 \
  --validation-manifest \
  /scratch/project_2019765/fnaveed/datasets/rice_grouped/grouped_split/validation.csv \
  --dataset-root \
  /scratch/project_2019765/fnaveed/datasets/rice_grouped \
  --initial-checkpoint \
  /scratch/project_2019765/fnaveed/results/rice_resnet18_grouped_v1/best_resnet18_rice_grouped.pt \
  --output-dir \
  /scratch/project_2019765/fnaveed/results/rice_fedavg_iid_resnet18_smoke \
  --experiment-name rice_fedavg_iid_resnet18 \
  --rounds 1 \
  --local-epochs 1 \
  --batch-size 64 \
  --learning-rate 0.000002 \
  --num-workers 4 \
  --seed 42
