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

## Traffic replay tool (Azure trace → vLLM)

The repository includes a high-performance traffic replayer that streams an Azure LLM inference trace to a vLLM-compatible endpoint and renders a live terminal dashboard with per-metric visualization (QPS, TX/RX rate, latency p95/p99).

Run example:
```bash
uv run python test/replay_vllm_azure2024.py \
  --csv hf-cache/AzureLLMInferenceTrace_conv_1week.csv \
  --url http://localhost:8000/v1/chat/completions \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speedup 10 --concurrency 16 \
  --log-every 1 --sample-interval 5 \
  --plot-window 60 --plot-rows 2 \
  --plot-height 8 --spark-width 80 --y-width 10
```

Key flags:
- `--csv`: Input CSV path. A compressed cache (`*.norm.csv.gz`) will be built under `--cache-dir` on first run
- `--url`, `--model`: vLLM-compatible chat completions endpoint and model name
- `--speedup`: Time compression factor (replay inter-arrival divided by this)
- `--concurrency`: Max in-flight requests
- `--timeout`: Per-request timeout (s)

Dashboard flags (visualization):
- `--log-every`: UI/stat refresh period (s)
- `--sample-interval`: sampling window for rates and latency percentiles (s). Default 5.0
- `--plot-window`: time window length summarized on the charts (s). Default 60
- `--plot-rows`: 2 to enable multi-row graphs; 1 for single-line fallback
- `--plot-height`: Graph vertical height (rows). Each row uses 8-dot braille resolution
- `--spark-width`: Graph width (columns). Left-padded when history is shorter
- `--y-width`: Fixed width for left Y-axis labels (right-aligned); prevents layout shift

Displayed metrics:
- QPS (curr): requests per second over the last interval
- TX/RX rate (curr): application-layer bytes per second. Each column is color-coded by intensity (green/yellow/red)
- Latency p95 / p99: per-interval percentile latencies (ms)
- Progress: integrated progress bar with ETA

Notes on visualization:
- Charts render with braille (8-dot) resolution to avoid gaps and support smooth vertical scaling
- Graph area, Y-axis labels, and numeric text are separate columns to prevent text from shifting the chart
