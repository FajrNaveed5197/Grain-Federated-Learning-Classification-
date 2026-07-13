#!/bin/bash
#SBATCH --job-name=ddp_resnet18_v3
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:2
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_2019765/grain_research/logs/ddp_resnet18_v3_%j.out
#SBATCH --error=/scratch/project_2019765/grain_research/logs/ddp_resnet18_v3_%j.err

module load python-pytorch
source /projappl/project_2019765/grain_research/fl_venv/bin/activate

cd /projappl/project_2019765/grain_research/code/grain_project

export PYTHONPATH=src:.
export OMP_NUM_THREADS=4

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  -m federated_pipeline.distributed.ddp_resnet18_v3 \
  --epochs 8 \
  --train-per-class 1738 \
  --batch-size-per-gpu 64 \
  --num-workers 4 \
  --output-dir /scratch/project_2019765/grain_research/distributed_results/ddp_resnet18_v3
