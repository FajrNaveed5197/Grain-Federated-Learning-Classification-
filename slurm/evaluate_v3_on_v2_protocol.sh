#!/bin/bash
#SBATCH --job-name=eval_v3_fixed
#SBATCH --account=project_2019649
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2019649/grain_research/logs/eval_v3_fixed_%j.out
#SBATCH --error=/scratch/project_2019649/grain_research/logs/eval_v3_fixed_%j.err

module load pytorch
cd /projappl/project_2019649/grain_research/code/grain_project
python -u scripts/evaluate_v3_on_v2_protocol.py
