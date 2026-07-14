#!/bin/bash
#SBATCH --job-name=rice_resnet18_s123
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project_2019765/fnaveed/logs/rice_resnet18_s123_%j.out
#SBATCH --error=/scratch/project_2019765/fnaveed/logs/rice_resnet18_s123_%j.err

set -euo pipefail

module load python-pytorch/2.10
source /projappl/project_2019765/grain_research/fl_venv/bin/activate

cd /projappl/project_2019765/grain_research/code/grain_project

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Python: $(which python)"

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
"

nvidia-smi

python scripts/train_rice_resnet18_seed123.py

echo "Finished: $(date)"
