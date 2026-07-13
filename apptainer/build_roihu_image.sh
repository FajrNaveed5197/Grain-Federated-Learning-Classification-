#!/usr/bin/env bash
set -euo pipefail

IMAGE_DIR="/projappl/project_2019765/grain_research/containers"
IMAGE_PATH="$IMAGE_DIR/federated_fl.sif"

mkdir -p "$IMAGE_DIR"

export APPTAINER_CACHEDIR="${TMPDIR:-/tmp}/apptainer-cache"
mkdir -p "$APPTAINER_CACHEDIR"

apptainer build --fakeroot \
  --bind="${TMPDIR:-/tmp}:/tmp" \
  "$IMAGE_PATH" \
  apptainer/federated_fl.def

echo "Built: $IMAGE_PATH"
