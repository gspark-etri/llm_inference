#!/usr/bin/env python3
"""
Synthetic LLM load generator (open-loop) for vLLM-compatible chat completions.

Features
- Arrival processes: deterministic, poisson (exponential), uniform, pareto, burst, sinusoidal
- Payload distributions for prompt tokens and max_tokens: const, uniform, normal, lognormal, pareto
- Concurrency limiting, duration or total requests, streaming toggle
- Live metrics table (Rich): totals, recent QPS, latency p50/p95/p99, success rate

Example
  uv run python test/synthetic_load_llm.py \
    --url http://localhost:8000/v1/chat/completions \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --arrival poisson --rps 40 --duration 60 \
    --prompt-tokens uniform:10,200 --max-tokens const:128 --concurrency 64
"""

import argparse, time, math, threading, queue, random, json
from collections import deque
from typing import Callable, Tuple, Optional

import requests
from rich.console import Console
from rich.table import Table
from rich.live import Live


console = Console(force_terminal=True)


# -------------------- helpers --------------------
def clamp_int(n: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(n)))


def make_prompt(num_tokens: int) -> str:
    num_tokens = max(1, int(num_tokens))
    return " ".join(["word"] * num_tokens)


def percentile(values, p: float) -> Optional[float]:
    if not values:
        return None
    data = sorted(values)
    k = (len(data) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(data[int(k)])
    d0 = data[int(f)] * (c - k)
    d1 = data[int(c)] * (k - f)
    return float(d0 + d1)


def parse_kv_pairs(pairs):
    headers = {}
    for item in pairs or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Invalid header: {item}. Expected KEY=VALUE")
        k, v = item.split("=", 1)
        headers[k.strip()] = v.strip()
    return headers


def parse_distribution(spec: str) -> Callable[[], float]:
    """Return a sampler function for the given distribution spec.

    Supported forms:
      - const:X
      - uniform:A,B
      - normal:MU,SIGMA
      - lognormal:MU,SIGMA   (underlying normal)
      - pareto:XM,ALPHA
    """
    if not spec:
        return lambda: 1.0
    try:
        kind, params = spec.split(":", 1)
    except ValueError:
        # shorthand number interpreted as const
        val = float(spec)
        return lambda: val

    kind = kind.strip().lower()
    parts = [p.strip() for p in params.split(",") if p.strip()]

    if kind in ("const", "constant"):
        x = float(parts[0])
        return lambda: x
    if kind == "uniform":
        a, b = map(float, parts[:2])
        return lambda: random.uniform(a, b)
    if kind == "normal":
        mu, sigma = map(float, parts[:2])
        return lambda: max(0.0, random.gauss(mu, sigma))
    if kind == "lognormal":
        mu, sigma = map(float, parts[:2])
        return lambda: random.lognormvariate(mu, sigma)
    if kind == "pareto":
        xm, alpha = map(float, parts[:2])
        return lambda: (random.paretovariate(alpha) * xm)
    if kind == "poisson":
        lam = float(parts[0])
        return lambda: max(0.0, random.poisson(lam) if hasattr(random, "poisson") else _poisson_fallback(lam))

    raise argparse.ArgumentTypeError(f"Unknown distribution: {spec}")


def _poisson_fallback(lam: float) -> int:
    # Knuth algorithm; returns integer k ~ Pois(lam)
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return max(0, k - 1)


def build_arrival_sampler(args) -> Callable[[float], float]:
    """Return function now->delta_t for inter-arrival time."""
    arrival = args.arrival.lower()
    if arrival == "deterministic":
        period = 1.0 / max(1e-6, args.rps)
        return lambda now: period
    if arrival == "uniform":
        period = 1.0 / max(1e-6, args.rps)
        jitter = args.jitter
        low = max(0.0, period * (1.0 - jitter))
        high = period * (1.0 + jitter)
        return lambda now: random.uniform(low, high)
    if arrival == "poisson":
        # exponential with mean 1/rate
        return lambda now: random.expovariate(max(1e-6, args.rps))
    if arrival == "pareto":
        xm = (1.0 / max(1e-6, args.rps)) * args.pareto_xm_scale
        alpha = args.pareto_alpha
        return lambda now: random.paretovariate(alpha) * xm
    if arrival == "burst":
        period = args.burst_interval
        size = max(1, args.burst_size)
        spacing = args.burst_spread / max(size, 1)
        # stateful closure
        state = {"i": 0, "next_burst": 0.0}
        def sampler(now: float) -> float:
            if state["i"] % size == 0:
                # beginning of a new burst window
                if now < state["next_burst"]:
                    return max(0.0, state["next_burst"] - now)
                state["next_burst"] = now + period
            state["i"] += 1
            return max(0.0, random.random() * spacing)
        return sampler
    if arrival == "sinusoidal":
        base = max(1e-6, args.rps)
        amp = max(0.0, min(0.99, args.sine_amp))
        period = max(1e-3, args.sine_period)
        # inhomogeneous Poisson via thinning (Ogata): propose with max rate and accept by r(t)/r_max
        r_max = base * (1.0 + amp)
        def rate(t: float) -> float:
            return base * (1.0 + amp * math.sin(2.0 * math.pi * t / period))
        def sampler(now: float) -> float:
            while True:
                dt = random.expovariate(r_max)
                t = now + dt
                if random.random() < rate(t) / r_max:
                    return dt
        return sampler
    raise argparse.ArgumentTypeError(f"Unknown arrival: {arrival}")


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    # target endpoint
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    # load shape
    ap.add_argument("--arrival", default="poisson", choices=[
        "deterministic", "uniform", "poisson", "pareto", "burst", "sinusoidal"
    ])
    ap.add_argument("--rps", type=float, default=10.0, help="target average requests per second (when applicable)")
    ap.add_argument("--jitter", type=float, default=0.2, help="uniform arrival jitter fraction (uniform mode)")
    ap.add_argument("--pareto-alpha", type=float, default=1.5)
    ap.add_argument("--pareto-xm-scale", type=float, default=0.5)
    ap.add_argument("--burst-size", type=int, default=10)
    ap.add_argument("--burst-interval", type=float, default=5.0)
    ap.add_argument("--burst-spread", type=float, default=0.5, help="seconds to spread within a burst window")
    ap.add_argument("--sine-amp", type=float, default=0.5, help="sinusoidal amplitude in [0,1)")
    ap.add_argument("--sine-period", type=float, default=30.0, help="sinusoidal period in seconds")
    # time bounds
    ap.add_argument("--duration", type=float, default=60.0, help="test duration seconds (ignored if --total given)")
    ap.add_argument("--total", type=int, default=None, help="total requests to send (overrides --duration)")
    # concurrency/timeout
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4, help="number of producer threads (split RPS)")
    ap.add_argument("--timeout", type=float, default=30.0)
    # payload shape
    ap.add_argument("--prompt-tokens", default="const:32", help="tokens distribution (const:x, uniform:a,b, normal:mu,sigma, lognormal:mu,sigma, pareto:xm,alpha)")
    ap.add_argument("--max-tokens", default="const:128", help="max_tokens distribution")
    ap.add_argument("--stream", action="store_true", default=True)
    ap.add_argument("--no-stream", dest="stream", action="store_false")
    ap.add_argument("--header", action="append", default=[], help="extra header KEY=VALUE (repeat)")
    # visualization / reporting
    ap.add_argument("--report-every", type=float, default=1.0)
    ap.add_argument("--sample-interval", type=float, default=5.0)
    ap.add_argument("--plot-window", type=float, default=60.0)
    ap.add_argument("--plot-rows", type=int, default=2)
    ap.add_argument("--plot-height", type=int, default=8)
    ap.add_argument("--spark-width", type=int, default=80)
    ap.add_argument("--y-width", type=int, default=10)
    ap.add_argument("--no-plot", action="store_true")

    args = ap.parse_args()

    headers = {"Authorization": "Bearer dummy", "Content-Type": "application/json"}
    headers.update(parse_kv_pairs(args.header))

    sample_prompt = parse_distribution(args.prompt_tokens)
    sample_max_tokens = parse_distribution(args.max_tokens)
    # share arrival across workers by splitting RPS when applicable
    def make_worker_arrival():
        if args.arrival in ("deterministic", "uniform", "poisson", "sinusoidal") and args.workers > 1:
            sub_args = argparse.Namespace(**vars(args))
            sub_args.rps = max(1e-6, args.rps / args.workers)
            return build_arrival_sampler(sub_args)
        return build_arrival_sampler(args)

    sem = threading.Semaphore(args.concurrency)
    done_q: "queue.Queue[Tuple[bool, float]]" = queue.Queue()
    lock = threading.Lock()

    sent = 0
    succeeded = 0
    failed = 0
    latencies_all: list[float] = []
    recent_latencies = deque(maxlen=1000)
    start_time = time.time()
    next_report = start_time + args.report_every
    next_sample = start_time + args.sample_interval
    # histories for visualization (match replay_vllm_azure2024.py style)
    hist_capacity = max(6, int(args.plot_window / max(args.report_every, 1e-6)))
    rps_hist = deque(maxlen=hist_capacity)          # completed requests per second
    in_tps_hist = deque(maxlen=hist_capacity)       # input tokens per second
    out_tps_hist = deque(maxlen=hist_capacity)      # output tokens per second
    ttft_p99_hist = deque(maxlen=hist_capacity)     # p95 TTFT (ms)
    tpot_p99_hist = deque(maxlen=hist_capacity)     # p95 TPOT (ms)
    # sample accumulators
    sample_sent = 0
    sample_completed = 0
    sample_in_tokens = 0
    sample_out_tokens = 0
    sample_latencies: list[float] = []
    sample_ttft: list[float] = []
    sample_tpot: list[float] = []

    def fire(itok: int, max_tok: int):
        nonlocal sent
        t0 = time.time()
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": make_prompt(itok)}],
            "max_tokens": clamp_int(max_tok, 1, 4096),
            "stream": args.stream,
        }
        ok = True
        ttft_ms = None
        out_tokens = 0
        try:
            with requests.post(args.url, headers=headers, json=payload, timeout=args.timeout, stream=args.stream) as resp:
                if args.stream:
                    for raw in resp.iter_lines(decode_unicode=True):
                        if not raw:
                            continue
                        line = raw.strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                            # OpenAI/vLLM delta formats
                            content = ""
                            if isinstance(obj, dict):
                                ch = obj.get("choices", [{}])
                                if ch:
                                    delta = ch[0].get("delta", {})
                                    if isinstance(delta, dict):
                                        content = delta.get("content", "") or delta.get("text", "")
                            if content is None:
                                content = ""
                        except Exception:
                            content = ""
                        # treat each non-empty chunk as one token for TPOT approx
                        if content != "":
                            out_tokens += 1
                            if ttft_ms is None:
                                ttft_ms = (time.time() - t0) * 1000.0
                if resp.status_code >= 400:
                    ok = False
                if not args.stream:
                    # non-streaming: try to read token usage
                    try:
                        obj = resp.json()
                        usage = obj.get("usage", {}) if isinstance(obj, dict) else {}
                        out_tokens = int(usage.get("completion_tokens", out_tokens or 0))
                    except Exception:
                        pass
        except Exception:
            ok = False
        finally:
            t_end = (time.time() - t0) * 1000.0
            tpot_ms = None
            if ttft_ms is not None and out_tokens > 0:
                tpot_ms = max(0.0, (t_end - ttft_ms) / max(1, out_tokens))
            done_q.put((ok, t_end, itok, out_tokens, ttft_ms, tpot_ms))
            sem.release()

    def render_table() -> Table:
        now = time.time()
        elapsed = max(1e-6, now - start_time)
        total = sent
        succ = succeeded
        err = failed
        rps_actual = total / elapsed
        p50 = percentile(latencies_all, 0.50)
        p95 = percentile(latencies_all, 0.95)
        p99 = percentile(latencies_all, 0.99)
        rp50 = percentile(list(recent_latencies), 0.50)
        rp95 = percentile(list(recent_latencies), 0.95)
        rp99 = percentile(list(recent_latencies), 0.99)
        tbl = Table(title="Synthetic Load - Live", expand=True)
        tbl.add_column("Metric", style="cyan", no_wrap=True)
        tbl.add_column("Value", ratio=1, overflow="crop")
        tbl.add_row("Elapsed (s)", f"{elapsed:,.1f}")
        tbl.add_row("Total / Success / Errors", f"{total:,} / {succ:,} / {err:,}")
        tbl.add_row("Target RPS / Actual", f"{args.rps:,.2f} / {rps_actual:,.2f}")
        tbl.add_row("Latency p50/p95/p99 (ms)", f"{(p50 or 0):,.1f} / {(p95 or 0):,.1f} / {(p99 or 0):,.1f}")
        tbl.add_row("Recent p50/p95/p99 (ms)", f"{(rp50 or 0):,.1f} / {(rp95 or 0):,.1f} / {(rp99 or 0):,.1f}")
        # Visualization using the same panel style as replay_vllm_azure2024.py
        if not args.no_plot and args.plot_rows >= 2:
            from rich.panel import Panel
            height = max(2, args.plot_height)
            spark_w = max(8, args.spark_width)
            y_w = max(0, args.y_width)

            def build_multirow_bars(values, width, height, y_label_fmt=None, y_width=0):
                if width <= 0 or height <= 0:
                    return ["" for _ in range(max(1, height))]
                vals = list(values)
                if not vals:
                    rows = [" " * width for _ in range(height)]
                    if y_width and y_width > 0:
                        rows[0] = (" " * y_width) + rows[0]
                        for r in range(1, height):
                            rows[r] = (" " * y_width) + rows[r]
                    return rows
                vals = vals[-width:]
                vmin = min(vals); vmax = max(vals)
                rng = max(1e-12, vmax - vmin)
                left_pad = max(0, width - len(vals))
                cols = [None] * width
                for i in range(width):
                    cols[i] = None if i < left_pad else vals[i - left_pad]
                rows = [[" "] * width for _ in range(height)]
                FULL = chr(0x28FF)
                order = [7, 8, 3, 6, 2, 5, 1, 4]
                def braille_n(n):
                    n = max(0, min(8, n))
                    if n == 0: return " "
                    if n == 8: return FULL
                    mask = 0
                    for d in order[:n]: mask |= (1 << (d-1))
                    return chr(0x2800 + mask)
                for x in range(width):
                    v = cols[x]
                    if v is None: continue
                    norm = (v - vmin) / rng
                    units = int(round(norm * height * 8))
                    full_rows = units // 8
                    rem = units % 8
                    col = "green" if norm < 0.33 else ("yellow" if norm < 0.66 else "red")
                    for r in range(height):
                        row_from_bottom = height - 1 - r
                        if row_from_bottom < full_rows:
                            rows[r][x] = f"[{col}]{FULL}[/]"
                        elif row_from_bottom == full_rows and rem > 0:
                            rows[r][x] = f"[{col}]{braille_n(rem)}[/]"
                        else:
                            rows[r][x] = " "
                y_w_local = max(0, y_width)
                if y_w_local > 0:
                    if y_label_fmt is None:
                        def y_label_fmt(x):
                            return f"{x:,.2f}"
                    top_label = y_label_fmt(vmax)
                    bot_label = y_label_fmt(vmin)
                    top_label = f"{top_label:>{y_w_local}}"
                    bot_label = f"{bot_label:>{y_w_local}}"
                    rows[0] = [top_label] + rows[0]
                    for r in range(1, height - 1):
                        rows[r] = [" " * y_w_local] + rows[r]
                    rows[-1] = [bot_label] + rows[-1]
                return ["".join(r) for r in rows]

            def build_braille_with_labels(values, y_fmt):
                bars = build_multirow_bars(values, spark_w, height, y_label_fmt=y_fmt, y_width=0)
                vals = list(values)[-spark_w:]
                vmin = vmax = None
                if not vals:
                    y_lines = [" " * y_w for _ in range(height)]
                else:
                    vmin = min(vals); vmax = max(vals)
                    top = y_fmt(vmax); bot = y_fmt(vmin)
                    y_lines = [" " * y_w for _ in range(height)]
                    if y_w > 0:
                        y_lines[0]  = f"{top:>{y_w}}"
                        y_lines[-1] = f"{bot:>{y_w}}"
                return y_lines, bars, vmin, vmax

            def make_metric_panel(title, values, y_fmt, right_text, border_style):
                y_lines_local, bars_local, vmin, vmax = build_braille_with_labels(values, y_fmt)
                header_tbl = Table.grid(expand=False)
                header_tbl.add_column(no_wrap=True)
                if vmax is not None:
                    header_tbl.add_row(f"curr: {right_text}    max: {y_fmt(vmax)}")
                else:
                    header_tbl.add_row(f"curr: {right_text}")
                chart_tbl = Table.grid(expand=True)
                chart_tbl.add_column(no_wrap=True, overflow="crop")
                chart_tbl.add_column(no_wrap=True, ratio=1, overflow="crop")
                for i in range(height):
                    chart_tbl.add_row(y_lines_local[i], bars_local[i])
                container = Table.grid(expand=False)
                container.add_column(no_wrap=True)
                container.add_row(header_tbl)
                container.add_row(chart_tbl)
                return Panel(container, title=title, border_style=border_style, padding=(0,1))

            # Row 1: throughputs
            rps_panel = make_metric_panel(
                "Req Throughput", rps_hist, lambda v: f"{v:,.2f}",
                f"{(rps_hist[-1] if rps_hist else 0):.2f} /s", "cyan")
            in_panel = make_metric_panel(
                "Input Token TPS", in_tps_hist, lambda v: f"{v:,.0f}",
                f"{(in_tps_hist[-1] if in_tps_hist else 0):.0f} t/s", "magenta")
            out_panel = make_metric_panel(
                "Output Token TPS", out_tps_hist, lambda v: f"{v:,.0f}",
                f"{(out_tps_hist[-1] if out_tps_hist else 0):.0f} t/s", "magenta")
            # Row 2: latencies
            ttft_panel = make_metric_panel(
                "TTFT p95", ttft_p99_hist, lambda v: f"{v:,.0f} ms",
                f"{(ttft_p99_hist[-1] if ttft_p99_hist else 0):.0f} ms", "yellow")
            tpot_panel = make_metric_panel(
                "TPOT p95", tpot_p99_hist, lambda v: f"{v:,.1f} ms",
                f"{(tpot_p99_hist[-1] if tpot_p99_hist else 0):.1f} ms", "yellow")

            grid = Table.grid(expand=True)
            grid.add_column(ratio=1, no_wrap=True, overflow="crop")
            grid.add_column(ratio=1, no_wrap=True, overflow="crop")
            grid.add_column(ratio=1, no_wrap=True, overflow="crop")
            grid.add_row(rps_panel, in_panel, out_panel)
            grid.add_row(ttft_panel, tpot_panel, Panel("", border_style="dim"))
            tbl.add_row("Visualize", Panel(grid, border_style="white", padding=(0,1)))
        return tbl

    # Multi-worker producers
    wall0 = time.time()
    end_time = None if args.total else (wall0 + args.duration)
    remaining = [args.total]  # boxed for mutability
    stop_event = threading.Event()

    def producer_loop(worker_id: int):
        nonlocal sent, sample_sent
        arrival = make_worker_arrival()
        next_send = time.time()
        while not stop_event.is_set():
            now = time.time()
            if end_time is not None and now >= end_time:
                break
            if remaining[0] is not None:
                with lock:
                    if remaining[0] is not None and remaining[0] <= 0:
                        break
                    if remaining[0] is not None:
                        remaining[0] -= 1
            if now < next_send:
                time.sleep(min(0.005, next_send - now))
                continue
            dt = arrival(now - wall0)
            next_send = now + max(0.0, dt)
            itok = max(1, int(round(sample_prompt())))
            mtok = max(1, int(round(sample_max_tokens())))
            sem.acquire()
            threading.Thread(target=fire, args=(itok, mtok), daemon=True).start()
            with lock:
                nonlocal sent
                sent += 1
                sample_sent += 1

    workers = []
    for w in range(max(1, args.workers)):
        t = threading.Thread(target=producer_loop, args=(w,), daemon=True)
        t.start()
        workers.append(t)

    with Live(render_table(), refresh_per_second=4, console=console, transient=False) as live:
        while True:
            now = time.time()
            # drain completions
            while True:
                try:
                    ok, lat_ms, itok_done, otok_done, ttft_ms, tpot_ms = done_q.get_nowait()
                except queue.Empty:
                    break
                with lock:
                    if ok:
                        succeeded += 1
                    else:
                        failed += 1
                    latencies_all.append(lat_ms)
                    recent_latencies.append(lat_ms)
                    sample_latencies.append(lat_ms)
                    sample_completed += 1
                    sample_in_tokens += int(itok_done)
                    sample_out_tokens += int(otok_done)
                    if ttft_ms is not None:
                        sample_ttft.append(ttft_ms)
                    if tpot_ms is not None:
                        sample_tpot.append(tpot_ms)

            if now >= next_sample:
                with lock:
                    rps_hist.append(sample_completed / max(1e-6, args.sample_interval))
                    in_tps_hist.append(sample_in_tokens / max(1e-6, args.sample_interval))
                    out_tps_hist.append(sample_out_tokens / max(1e-6, args.sample_interval))
                    ttft_p99_hist.append(percentile(sample_ttft, 0.95) or 0.0)
                    tpot_p99_hist.append(percentile(sample_tpot, 0.95) or 0.0)
                    sample_sent = 0
                    sample_completed = 0
                    sample_in_tokens = 0
                    sample_out_tokens = 0
                    sample_latencies.clear()
                    sample_ttft.clear()
                    sample_tpot.clear()
                next_sample = now + args.sample_interval

            if now >= next_report:
                live.update(render_table())
                next_report = now + args.report_every

            # stopping conditions
            if (end_time is not None and now >= end_time) or (remaining[0] is not None and remaining[0] <= 0):
                stop_event.set()
                # break when in-flight complete
                if sem._value == args.concurrency and done_q.empty():
                    break
            time.sleep(0.01)

    console.print("[green]Test finished[/]")
    console.print(render_table())


if __name__ == "__main__":
    main()



