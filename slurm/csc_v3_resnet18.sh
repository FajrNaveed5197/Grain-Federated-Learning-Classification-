#!/bin/bash
#SBATCH --job-name=grain_resnet18_v3
#SBATCH --account=project_2019649
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_2019649/grain_research/logs/grain_resnet18_v3_%j.out
#SBATCH --error=/scratch/project_2019649/grain_research/logs/grain_resnet18_v3_%j.err

module load pytorch

cd /projappl/project_2019649/grain_research/code/grain_project
python scripts/csc_finetune_v3_resnet18.py
