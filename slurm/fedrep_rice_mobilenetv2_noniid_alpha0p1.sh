#!/bin/bash
#SBATCH --job-name=rice_fr_a01
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_2019765/fnaveed/results/Logs/SLURM/rice_fedrep_mob_alpha0p1_%j.out
#SBATCH --error=/scratch/project_2019765/fnaveed/results/Logs/SLURM/rice_fedrep_mob_alpha0p1_%j.err

set -euo pipefail

module load python-pytorch
source /projappl/project_2019765/grain_research/fl_venv/bin/activate

cd /projappl/project_2019765/grain_research/code/grain_project

CLIENT_DIR=/scratch/project_2019765/fnaveed/datasets/rice_grouped/federated_partitions/noniid_3clients_alpha0p1_seed42_v2
OUTPUT_DIR=/scratch/project_2019765/fnaveed/results/Rice/Federated/FedRep/NonIID_alpha0p1/MobileNetV2
CHECKPOINT=/scratch/project_2019765/fnaveed/results/Rice/Centralized/MobileNetV2/seed42/best_mobilenetv2_rice_grouped.pt

[[ -d "$CLIENT_DIR" ]] || {
  echo "Missing client partition: $CLIENT_DIR"
  exit 1
}

[[ -f "$CHECKPOINT" ]] || {
  echo "Missing initial checkpoint: $CHECKPOINT"
  exit 1
}

if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty output directory: $OUTPUT_DIR"
  exit 1
fi

mkdir -p \
  "$OUTPUT_DIR" \
  /scratch/project_2019765/fnaveed/results/Logs/SLURM

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

python -m federated_pipeline.federated.fedrep_rice_mobilenetv2 \
  --client-dir "$CLIENT_DIR" \
  --validation-manifest /scratch/project_2019765/fnaveed/datasets/rice_grouped/grouped_split/validation.csv \
  --test-manifest /scratch/project_2019765/fnaveed/datasets/rice_grouped/grouped_split/test.csv \
  --initial-checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --dataset-root /scratch/project_2019765/fnaveed/datasets/rice_grouped \
  --experiment-name rice_fedrep_noniid_alpha0p1_mobilenetv2 \
  --rounds 5 \
  --head-epochs 5 \
  --representation-epochs 1 \
  --batch-size 64 \
  --head-learning-rate 1e-4 \
  --representation-learning-rate 2e-6 \
  --num-workers 4 \
  --seed 42
