#!/bin/bash
#SBATCH --job-name=fl_full_iid_v3
#SBATCH --account=project_2019649
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/project_2019649/grain_research/logs/fl_full_iid_v3_%j.out
#SBATCH --error=/scratch/project_2019649/grain_research/logs/fl_full_iid_v3_%j.err

module load pytorch

cd /projappl/project_2019649/grain_research/code/grain_project

PYTHONPATH=src python -m federated_pipeline.federated.fedavg_full_v3 \
  --client-dir /scratch/project_2019649/grain_research/partitions/fl_full_iid_3clients \
  --validation-manifest /scratch/project_2019649/grain_research/manifests/validation.csv \
  --initial-checkpoint /scratch/project_2019649/grain_research/results/version3_dynamic_balanced_resnet18/best_resnet18_v3_gpu.pt \
  --output-dir /scratch/project_2019649/grain_research/federated_results/fl_full_iid_v3 \
  --rounds 3 \
  --local-epochs 1 \
  --batch-size 64 \
  --learning-rate 0.000002 \
  --num-workers 4 \
  --seed 42
