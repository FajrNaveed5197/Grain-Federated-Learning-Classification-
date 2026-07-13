#!/bin/bash
#SBATCH --job-name=roihu_gpu_test
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=/scratch/project_2019765/grain_research/logs/roihu_gpu_test_%j.out
#SBATCH --error=/scratch/project_2019765/grain_research/logs/roihu_gpu_test_%j.err

module load python-pytorch

python - <<'PY'
import torch
from pathlib import Path
import pandas as pd
from PIL import Image

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

manifest = Path("/scratch/project_2019765/grain_research/manifests/train.csv")
df = pd.read_csv(manifest)

image_path = Path(df.iloc[0]["path"])
print("Manifest image:", image_path)
print("Image exists:", image_path.exists())

with Image.open(image_path) as image:
    print("Image size:", image.size)

x = torch.randn(2048, 2048, device="cuda")
print("GPU tensor OK:", float(x.mean()))
PY
