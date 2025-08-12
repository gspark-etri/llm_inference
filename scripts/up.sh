#!/usr/bin/env bash
set -euo pipefail
if [ ! -f .env ]; then
  echo "[ERR] .env not found. Copy .env.example and set your HF token." >&2
  exit 1
fi
docker compose up -d
echo "[OK] Stack is up:"
echo "  - vLLM:       http://localhost:8000/metrics"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana:    http://localhost:3000"
