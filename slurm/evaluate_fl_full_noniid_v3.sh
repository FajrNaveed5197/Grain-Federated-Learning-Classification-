#!/bin/bash
#SBATCH --job-name=eval_fl_iid_v3
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_2019765/grain_research/logs/eval_fl_iid_v3_%j.out
#SBATCH --error=/scratch/project_2019765/grain_research/logs/eval_fl_iid_v3_%j.err

module load python-pytorch

cd /projappl/project_2019765/grain_research/code/grain_project

PYTHONPATH=src python -m federated_pipeline.federated.evaluate_federated_checkpoint \
  --checkpoint /scratch/project_2019765/grain_research/federated_results/fl_full_noniid_v3/global_model_round_3.pt \
  --root /scratch/project_2019765/grain_research \
  --output /scratch/project_2019765/grain_research/federated_results/fl_full_noniid_v3/fixed_protocol_results.json \
  --batch-size 256 \
  --num-workers 4
