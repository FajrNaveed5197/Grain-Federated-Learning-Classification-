#!/bin/bash
#SBATCH --job-name=eval_resnet_v2v3
#SBATCH --account=project_2019649
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_2019649/grain_research/logs/eval_resnet_v2v3_%j.out
#SBATCH --error=/scratch/project_2019649/grain_research/logs/eval_resnet_v2v3_%j.err

module load pytorch
cd /projappl/project_2019649/grain_research/code/grain_project
python scripts/evaluate_resnet18_v2_v3_fixed.py
