# Containerized Federated Learning

This setup runs four services:

- fl-server: coordinates FedAvg aggregation
- client-0: trains on Client 0 local data
- client-1: trains on Client 1 local data
- client-2: trains on Client 2 local data

## Local Docker usage

1. Copy docker/.env.example to docker/.env
2. Set the three client manifest paths and data-root paths.
3. Run ./docker/run-local.sh

Each client mounts its own images at /data inside its container.
Each client manifest must therefore contain paths starting with /data/.

## Current grain simulation

Container-ready manifests are stored outside Git here:

/scratch/project_2019765/grain_research/container_manifests/

For real IoT data, each site keeps and mounts only its own local images.

## Output

The federated server saves the aggregated global checkpoint after each round in RUNS_DIR.
