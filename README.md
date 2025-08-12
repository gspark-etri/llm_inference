# LLM Stack: vLLM + Prometheus + Grafana + DCGM Exporter

One-command setup for monitoring LLM inference (TTFT/TPOT/E2E/TPS) and GPU power/utilization.

## Quick start
```bash
git clone <your-repo-url>.git
cd llm-stack

cp .env.example .env
# edit .env and put your Hugging Face token

./scripts/up.sh

# Dashboards
# - vLLM metrics:   http://localhost:8000/metrics
# - Prometheus UI:  http://localhost:9090
# - Grafana UI:     http://localhost:3000 (admin / admin on first login)
```

## Notes
- NVIDIA GPU node required (`nvidia-smi` should work).
- vLLM exposes **colon-style metrics** (e.g., `vllm:time_to_first_token_seconds_*`).
