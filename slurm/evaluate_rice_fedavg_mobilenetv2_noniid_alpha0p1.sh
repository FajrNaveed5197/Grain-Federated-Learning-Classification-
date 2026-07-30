#!/bin/bash
#SBATCH --job-name=rice_faev_a01
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_2019765/fnaveed/results/Rice/Federated/FedAvg/NonIID_alpha0p1/MobileNetV2/SLURM/rice_fedavg_eval_alpha0p1_%j.out
#SBATCH --error=/scratch/project_2019765/fnaveed/results/Rice/Federated/FedAvg/NonIID_alpha0p1/MobileNetV2/SLURM/rice_fedavg_eval_alpha0p1_%j.err

set -euo pipefail

module load python-pytorch
source /projappl/project_2019765/grain_research/fl_venv/bin/activate

cd /projappl/project_2019765/grain_research/code/grain_project

RESULT_DIR=/scratch/project_2019765/fnaveed/results/Rice/Federated/FedAvg/NonIID_alpha0p1/MobileNetV2
CHECKPOINT="$RESULT_DIR/best_global_model.pt"
EVALUATION_DIR="$RESULT_DIR/evaluation"

[[ -f "$CHECKPOINT" ]] || {
  echo "Missing checkpoint: $CHECKPOINT"
  exit 1
}

if [[ -d "$EVALUATION_DIR" ]] && \
   [[ -n "$(find "$EVALUATION_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty evaluation directory: $EVALUATION_DIR"
  exit 1
fi

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MPLBACKEND=Agg

python scripts/evaluate_rice_fedavg_single.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$RESULT_DIR" \
  --experiment-name rice_fedavg_noniid_alpha0p1_mobilenetv2 \
  --dataset-root /scratch/project_2019765/fnaveed/datasets/rice_grouped \
  --validation-manifest /scratch/project_2019765/fnaveed/datasets/rice_grouped/grouped_split/validation.csv \
  --test-manifest /scratch/project_2019765/fnaveed/datasets/rice_grouped/grouped_split/test.csv \
  --batch-size 128 \
  --num-workers 6
