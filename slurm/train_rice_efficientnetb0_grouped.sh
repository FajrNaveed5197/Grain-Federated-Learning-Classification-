#!/bin/bash
#SBATCH --job-name=rice_effnetb0
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_2019765/fnaveed/logs/rice_effnetb0_%j.out
#SBATCH --error=/scratch/project_2019765/fnaveed/logs/rice_effnetb0_%j.err

set -euo pipefail

module load python-pytorch
source /projappl/project_2019765/grain_research/fl_venv/bin/activate

cd /projappl/project_2019765/grain_research/code/grain_project

mkdir -p \
  /scratch/project_2019765/fnaveed/logs \
  /scratch/project_2019765/fnaveed/results/rice_efficientnetb0_grouped_v1

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

python scripts/train_rice_efficientnetb0_grouped.py
