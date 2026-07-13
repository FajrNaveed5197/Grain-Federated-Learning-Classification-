#!/bin/bash
#SBATCH --job-name=roihu_modcheck
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/project_2019765/grain_research/logs/roihu_modcheck_%j.out
#SBATCH --error=/scratch/project_2019765/grain_research/logs/roihu_modcheck_%j.err

module avail 2>&1
echo "----- MODULE SPIDER PYTORCH -----"
module spider pytorch 2>&1
echo "----- MODULE SPIDER PYTHON-PYTORCH -----"
module spider python-pytorch 2>&1
echo "----- PYTHON -----"
which python
python --version
