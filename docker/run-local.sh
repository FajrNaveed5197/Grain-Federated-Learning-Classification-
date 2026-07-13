#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$SCRIPT_DIR/.env" ]; then
  echo "Missing docker/.env"
  echo "Create it first:"
  echo "  cp docker/.env.example docker/.env"
  echo "Then set the three client manifest and data-root paths."
  exit 1
fi

docker compose \
  --env-file "$SCRIPT_DIR/.env" \
  -f "$SCRIPT_DIR/docker-compose.yml" \
  up --build
