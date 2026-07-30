#!/bin/bash
#SBATCH --job-name=rice_fedrep_mob_non
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_2019765/fnaveed/logs/rice_fedrep_mob_noniid_%j.out
#SBATCH --error=/scratch/project_2019765/fnaveed/logs/rice_fedrep_mob_noniid_%j.err

set -euo pipefail

module load python-pytorch
source /projappl/project_2019765/grain_research/fl_venv/bin/activate

cd /projappl/project_2019765/grain_research/code/grain_project

mkdir -p \
  /scratch/project_2019765/fnaveed/logs \
  /scratch/project_2019765/fnaveed/results/rice_fedrep_noniid_mobilenetv2

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

python -m federated_pipeline.federated.fedrep_rice_mobilenetv2 \
  --client-dir \
  /scratch/project_2019765/fnaveed/datasets/rice_grouped/federated_partitions/noniid_3clients_alpha0p5_seed42_v2 \
  --validation-manifest \
  /scratch/project_2019765/fnaveed/datasets/rice_grouped/grouped_split/validation.csv \
  --test-manifest \
  /scratch/project_2019765/fnaveed/datasets/rice_grouped/grouped_split/test.csv \
  --initial-checkpoint \
  /scratch/project_2019765/fnaveed/results/rice_mobilenetv2_grouped_v1/best_mobilenetv2_rice_grouped.pt \
  --output-dir \
  /scratch/project_2019765/fnaveed/results/rice_fedrep_noniid_mobilenetv2 \
  --dataset-root \
  /scratch/project_2019765/fnaveed/datasets/rice_grouped \
  --experiment-name \
  rice_fedrep_noniid_mobilenetv2 \
  --rounds 5 \
  --head-epochs 5 \
  --representation-epochs 1 \
  --batch-size 64 \
  --head-learning-rate 1e-4 \
  --representation-learning-rate 2e-6 \
  --num-workers 4 \
  --seed 42
