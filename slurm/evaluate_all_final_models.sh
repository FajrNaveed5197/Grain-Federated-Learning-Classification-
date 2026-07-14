#!/bin/bash
#SBATCH --job-name=eval_final_models
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project_2019765/grain_research/logs/eval_final_models_%j.out
#SBATCH --error=/scratch/project_2019765/grain_research/logs/eval_final_models_%j.err

set -euo pipefail

module load python-pytorch
source /projappl/project_2019765/grain_research/fl_venv/bin/activate

cd /projappl/project_2019765/grain_research/code/grain_project

echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"

srun python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("GPU is not visible to PyTorch")

print("GPU:", torch.cuda.get_device_name(0))
PY

srun python -u scripts/evaluate_all_final_models.py
