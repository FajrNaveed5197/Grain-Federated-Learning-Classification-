#!/bin/bash
#SBATCH --job-name=eval_rice_ddp
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_2019765/fnaveed/logs/eval_rice_ddp_%j.out
#SBATCH --error=/scratch/project_2019765/fnaveed/logs/eval_rice_ddp_%j.err

set -euo pipefail

module load python-pytorch
source /projappl/project_2019765/grain_research/fl_venv/bin/activate

cd /projappl/project_2019765/grain_research/code/grain_project

export PYTHONPATH=src
export PYTHONUNBUFFERED=1

python scripts/evaluate_rice_model.py \
  --checkpoint \
  /scratch/project_2019765/fnaveed/results/rice_ddp_resnet18/best_ddp_resnet18_rice.pt \
  --dataset-root \
  /scratch/project_2019765/fnaveed/datasets/rice_grouped \
  --validation-manifest \
  /scratch/project_2019765/fnaveed/datasets/rice_grouped/grouped_split/validation.csv \
  --test-manifest \
  /scratch/project_2019765/fnaveed/datasets/rice_grouped/grouped_split/test.csv \
  --output-dir \
  /scratch/project_2019765/fnaveed/results/rice_ddp_resnet18/evaluation \
  --experiment-name rice_ddp_resnet18 \
  --batch-size 256 \
  --num-workers 4
