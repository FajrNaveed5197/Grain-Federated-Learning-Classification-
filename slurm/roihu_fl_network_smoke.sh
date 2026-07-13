#!/bin/bash
#SBATCH --job-name=fl_network_smoke
#SBATCH --account=project_2019765
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_2019765/grain_research/logs/fl_network_smoke_%j.out
#SBATCH --error=/scratch/project_2019765/grain_research/logs/fl_network_smoke_%j.err

module load python-pytorch
source /projappl/project_2019765/grain_research/fl_venv/bin/activate

cd /projappl/project_2019765/grain_research/code/grain_project

export PYTHONPATH=src:.
export FL_SERVER_ADDRESS=0.0.0.0:8080
export FL_ROUNDS=1
export FL_RUNS_DIR=/scratch/project_2019765/grain_research/federated_results/container_network_smoke

mkdir -p "$FL_RUNS_DIR"

python -m containerized_fl.server.server \
  > "$FL_RUNS_DIR/server.log" 2>&1 &
SERVER_PID=$!

sleep 5

for CLIENT_ID in 0 1 2; do
  FL_SERVER_ADDRESS=127.0.0.1:8080 \
  FL_CLIENT_ID="client-${CLIENT_ID}" \
  FL_CLIENT_MANIFEST="/scratch/project_2019765/grain_research/partitions/fl_smoke_noniid_3clients/client_${CLIENT_ID}_train.csv" \
  FL_LOCAL_EPOCHS=1 \
  FL_BATCH_SIZE=32 \
  FL_LEARNING_RATE=0.0001 \
  python -m containerized_fl.client.client \
    > "$FL_RUNS_DIR/client_${CLIENT_ID}.log" 2>&1 &
done

wait
wait "$SERVER_PID"

echo "Flower network smoke test completed."
