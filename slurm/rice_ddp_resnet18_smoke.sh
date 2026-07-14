#!/bin/bash
#SBATCH --job-name=rice_ddp_smoke
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_2019765/fnaveed/logs/rice_ddp_smoke_%j.out
#SBATCH --error=/scratch/project_2019765/fnaveed/logs/rice_ddp_smoke_%j.err

set -euo pipefail

module load python-pytorch
source /projappl/project_2019765/grain_research/fl_venv/bin/activate

cd /projappl/project_2019765/grain_research/code/grain_project

mkdir -p \
  /scratch/project_2019765/fnaveed/logs \
  /scratch/project_2019765/fnaveed/results/rice_ddp_resnet18_smoke

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  -m federated_pipeline.distributed.ddp_rice_resnet18 \
  --dataset-root \
  /scratch/project_2019765/fnaveed/datasets/rice_grouped \
  --train-manifest \
  /scratch/project_2019765/fnaveed/datasets/rice_grouped/grouped_split/train.csv \
  --validation-manifest \
  /scratch/project_2019765/fnaveed/datasets/rice_grouped/grouped_split/validation.csv \
  --initial-checkpoint \
  /scratch/project_2019765/fnaveed/results/rice_resnet18_grouped_v1/best_resnet18_rice_grouped.pt \
  --output-dir \
  /scratch/project_2019765/fnaveed/results/rice_ddp_resnet18_smoke \
  --epochs 1 \
  --batch-size-per-gpu 64 \
  --num-workers 4 \
  --learning-rate 0.000002 \
  --weight-decay 0.0001 \
  --seed 42
