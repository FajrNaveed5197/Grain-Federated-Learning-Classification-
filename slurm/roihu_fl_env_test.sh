#!/bin/bash
#SBATCH --job-name=roihu_fl_env
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/project_2019765/grain_research/logs/roihu_fl_env_%j.out
#SBATCH --error=/scratch/project_2019765/grain_research/logs/roihu_fl_env_%j.err

module load python-pytorch

source /projappl/project_2019765/grain_research/fl_venv/bin/activate

python - <<'PY'
import flwr
import torch
import pandas
from PIL import Image

print("Flower:", flwr.__version__)
print("Torch:", torch.__version__)
print("Pandas:", pandas.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(2048, 2048, device="cuda")
    print("GPU tensor OK:", float(x.mean()))
PY
