#!/bin/bash
#SBATCH --job-name=grain_mobilenet_v2
#SBATCH --account=project_2019649
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_2019649/grain_research/logs/grain_mobilenet_v2_%j.out
#SBATCH --error=/scratch/project_2019649/grain_research/logs/grain_mobilenet_v2_%j.err

module load pytorch
cd /projappl/project_2019649/grain_research/code/grain_project
python scripts/csc_continue_finetune_v2.py --model mobilenetv2 --epochs 5 --learning-rate 5e-6
