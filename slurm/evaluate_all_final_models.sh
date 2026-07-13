#!/bin/bash
#SBATCH --job-name=eval_final_models
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project_2019765/grain_research/logs/eval_final_models_%j.out
#SBATCH --error=/scratch/project_2019765/grain_research/logs/eval_final_models_%j.err

module load python-pytorch
source /projappl/project_2019765/grain_research/fl_venv/bin/activate

cd /projappl/project_2019765/grain_research/code/grain_project

python scripts/evaluate_all_final_models.py
