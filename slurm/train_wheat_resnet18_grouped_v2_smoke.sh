#!/bin/bash
#SBATCH --job-name=wheat_r18_smoke
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_2019765/fnaveed/logs/wheat_r18_smoke_%j.out
#SBATCH --error=/scratch/project_2019765/fnaveed/logs/wheat_r18_smoke_%j.err

set -euo pipefail

module load python-pytorch
source /projappl/project_2019765/grain_research/fl_venv/bin/activate

cd /projappl/project_2019765/grain_research/code/grain_project

mkdir -p \
  /scratch/project_2019765/fnaveed/logs

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

python \
  scripts/train_wheat_resnet18_grouped_v2_smoke.py
