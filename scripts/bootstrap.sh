#!/usr/bin/env bash
set -euo pipefail
if ! command -v docker >/dev/null 2>&1; then
  echo "[ERR] docker not found." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[WARN] nvidia-smi not found. Ensure NVIDIA drivers are installed."
fi
if [ ! -f .env ]; then
  cp .env.example .env
  echo "[INFO] Created .env from .env.example. Edit HUGGING_FACE_HUB_TOKEN."
fi
mkdir -p hf-cache
echo "[OK] bootstrap done."
