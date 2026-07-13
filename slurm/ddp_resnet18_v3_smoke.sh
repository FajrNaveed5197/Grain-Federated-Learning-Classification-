#!/bin/bash
#SBATCH --job-name=ddp_v3_smoke
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project_2019765/grain_research/logs/ddp_v3_smoke_%j.out
#SBATCH --error=/scratch/project_2019765/grain_research/logs/ddp_v3_smoke_%j.err

module load python-pytorch
source /projappl/project_2019765/grain_research/fl_venv/bin/activate

cd /projappl/project_2019765/grain_research/code/grain_project

export PYTHONPATH=src:.
export OMP_NUM_THREADS=4

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  -m federated_pipeline.distributed.ddp_resnet18_v3 \
  --epochs 1 \
  --train-per-class 50 \
  --batch-size-per-gpu 32 \
  --num-workers 2 \
  --skip-final-evaluation \
  --output-dir /scratch/project_2019765/grain_research/distributed_results/ddp_resnet18_v3_smoke
