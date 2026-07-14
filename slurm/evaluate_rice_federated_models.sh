#!/bin/bash
#SBATCH --job-name=eval_rice_fl
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_2019765/fnaveed/logs/eval_rice_fl_%j.out
#SBATCH --error=/scratch/project_2019765/fnaveed/logs/eval_rice_fl_%j.err

set -euo pipefail

module load python-pytorch
source /projappl/project_2019765/grain_research/fl_venv/bin/activate

cd /projappl/project_2019765/grain_research/code/grain_project

export PYTHONPATH=src
export PYTHONUNBUFFERED=1

DATASET_ROOT=/scratch/project_2019765/fnaveed/datasets/rice_grouped
RESULT_ROOT=/scratch/project_2019765/fnaveed/results

python scripts/evaluate_rice_model.py \
  --checkpoint \
  ${RESULT_ROOT}/rice_fedavg_iid_resnet18/best_global_model.pt \
  --dataset-root ${DATASET_ROOT} \
  --validation-manifest \
  ${DATASET_ROOT}/grouped_split/validation.csv \
  --test-manifest \
  ${DATASET_ROOT}/grouped_split/test.csv \
  --output-dir \
  ${RESULT_ROOT}/rice_fedavg_iid_resnet18/evaluation \
  --experiment-name rice_fedavg_iid_resnet18 \
  --batch-size 256 \
  --num-workers 4

python scripts/evaluate_rice_model.py \
  --checkpoint \
  ${RESULT_ROOT}/rice_fedavg_noniid_resnet18/best_global_model.pt \
  --dataset-root ${DATASET_ROOT} \
  --validation-manifest \
  ${DATASET_ROOT}/grouped_split/validation.csv \
  --test-manifest \
  ${DATASET_ROOT}/grouped_split/test.csv \
  --output-dir \
  ${RESULT_ROOT}/rice_fedavg_noniid_resnet18/evaluation \
  --experiment-name rice_fedavg_noniid_resnet18 \
  --batch-size 256 \
  --num-workers 4
