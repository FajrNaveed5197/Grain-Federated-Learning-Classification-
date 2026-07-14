#!/bin/bash
#SBATCH --job-name=rice_mob_iid
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_2019765/fnaveed/logs/rice_mob_iid_%j.out
#SBATCH --error=/scratch/project_2019765/fnaveed/logs/rice_mob_iid_%j.err

set -euo pipefail

module load python-pytorch
source /projappl/project_2019765/grain_research/fl_venv/bin/activate

cd /projappl/project_2019765/grain_research/code/grain_project

mkdir -p \
  /scratch/project_2019765/fnaveed/logs \
  /scratch/project_2019765/fnaveed/results/rice_fedavg_iid_mobilenetv2

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

python -m federated_pipeline.federated.fedavg_rice_mobilenetv2 \
  --client-dir \
  /scratch/project_2019765/fnaveed/datasets/rice_grouped/federated_partitions/iid_3clients_seed42_v2 \
  --validation-manifest \
  /scratch/project_2019765/fnaveed/datasets/rice_grouped/grouped_split/validation.csv \
  --initial-checkpoint \
  /scratch/project_2019765/fnaveed/results/rice_mobilenetv2_grouped_v1/best_mobilenetv2_rice_grouped.pt \
  --output-dir \
  /scratch/project_2019765/fnaveed/results/rice_fedavg_iid_mobilenetv2 \
  --dataset-root \
  /scratch/project_2019765/fnaveed/datasets/rice_grouped \
  --experiment-name \
  rice_fedavg_iid_mobilenetv2 \
  --rounds 5 \
  --local-epochs 1 \
  --batch-size 64 \
  --learning-rate 2e-6 \
  --num-workers 4 \
  --seed 42
