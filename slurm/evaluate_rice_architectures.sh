#!/bin/bash
#SBATCH --job-name=rice_arch_eval
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_2019765/fnaveed/logs/rice_arch_eval_%j.out
#SBATCH --error=/scratch/project_2019765/fnaveed/logs/rice_arch_eval_%j.err

set -euo pipefail

module load python-pytorch
source /projappl/project_2019765/grain_research/fl_venv/bin/activate

cd /projappl/project_2019765/grain_research/code/grain_project

export PYTHONUNBUFFERED=1

python scripts/evaluate_rice_architectures.py
