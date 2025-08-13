#!/usr/bin/env python3
import time, argparse, requests
from rich.console import Console
from rich.table import Table
from rich.live import Live

console = Console()

QUERIES = {
  "TTFT_avg_s": "rate(vllm:time_to_first_token_seconds_sum[30s]) / rate(vllm:time_to_first_token_seconds_count[30s])",
  "TPOT_avg_s/token": "rate(vllm:time_per_output_token_seconds_sum[30s]) / rate(vllm:time_per_output_token_seconds_count[30s])",
  "TPS_tokens/s": "rate(vllm:generation_tokens_total[30s])",
  "E2E_p95_s": "histogram_quantile(0.95, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket[30s])))",
  "GPU_Power_W_total": "sum(DCGM_FI_DEV_POWER_USAGE)"
}

def query_prom(url, expr):
    r = requests.get(url + "/api/v1/query", params={"query": expr}, timeout=5)
    r.raise_for_status()
    data = r.json().get("data", {}).get("result", [])
    if not data:
        return float("nan")
    try:
        val = float(data[0]["value"][1])
    except Exception:
        val = float("nan")
    return val

def render_table(vals):
    tbl = Table(title="Live Server Metrics (Prometheus)", expand=True)
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value", justify="right")
    for k, v in vals.items():
        if v != v: # NaN
            s = "n/a"
        else:
            s = f"{v:,.4f}" if abs(v) < 1000 else f"{v:,.1f}"
        tbl.add_row(k, s)
    return tbl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prom", default="http://localhost:9090", help="Prometheus base URL")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    with Live(render_table({k: float("nan") for k in QUERIES}), refresh_per_second=4, console=console):
        while True:
            vals = {k: query_prom(args.prom, q) for k,q in QUERIES.items()}
            console.update(render_table(vals))
            time.sleep(args.interval)

if __name__ == "__main__":
    main()
