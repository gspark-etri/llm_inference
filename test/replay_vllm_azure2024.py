#!/usr/bin/env python3
import csv, time, argparse, requests, threading, queue, os, gzip, math, json
from collections import deque
from datetime import datetime, timezone
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live

console = Console(force_terminal=True)

# ---------- helpers ----------
def parse_ts_ms(ts_str: str) -> int:
    # "2024-05-12 00:00:00.001163+00:00" → epoch ms
    dt = datetime.fromisoformat(ts_str.replace(" ", "T"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def make_prompt(n_tokens:int)->str:
    n_tokens = max(1, n_tokens)
    return " ".join(["word"] * n_tokens)

def format_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:,.1f} {unit}"
        size /= 1024.0

def format_rate(bytes_per_second: float) -> str:
    return f"{format_bytes(bytes_per_second)}/s"

def progress_bar(fraction: float, width: int = 36) -> str:
    fraction = 0.0 if math.isnan(fraction) else max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return ("█" * filled) + ("─" * (width - filled))

def format_ms(value_ms: float) -> str:
    return f"{value_ms:,.1f} ms"

def percentile(values, p: float):
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

def format_rate_compact(bytes_per_second: float, max_width: int) -> str:
    s = format_rate(bytes_per_second)
    if len(s) <= max_width:
        return s
    # remove space in unit (e.g., "KB/s" stays, but bytes string becomes like "139.1KB/s")
    num, unit = s.split(" ") if " " in s else (s, "")
    if unit:
        compact = num + unit
        if len(compact) <= max_width:
            return compact
    # reduce decimals
    for decimals in [1, 0]:
        try:
            val = float(num.split("/")[0])
            base = f"{val:.{decimals}f}"
            out = base + (unit if unit else "")
            if len(out) <= max_width:
                return out
        except Exception:
            break
    return s[:max_width]

def format_qps_compact(qps: float, max_width: int) -> str:
    txt = f"{qps:,.2f} r/s"
    if len(txt) <= max_width:
        return txt
    # try fewer decimals
    for decimals in [1, 0]:
        base = f"{qps:.{decimals}f} r/s"
        if len(base) <= max_width:
            return base
    return txt[:max_width]

def need_rebuild_cache(src, cache_path):
    if not os.path.exists(cache_path):
        return True
    return os.path.getmtime(cache_path) < os.path.getmtime(src)

def build_cache(src_csv, cache_path, assume_sorted=True, limit=None, chunk_size=500_000):
    """
    입력 CSV(TIMESTAMP,ContextTokens,GeneratedTokens) → 캐시 CSV(ts_ms,input_tokens,output_tokens) gzip으로 저장
    대용량 대비: pandas 없이 표준 csv로 청크 처리(메모리 절약).
    """
    import itertools
    header_written = False
    written = 0
    with open(src_csv, newline="") as fin, gzip.open(cache_path, "wt", newline="") as fout:
        r = csv.DictReader(fin)
        w = csv.writer(fout)
        if not header_written:
            w.writerow(["ts_ms", "input_tokens", "output_tokens"])
            header_written = True

        buf = []
        for row in r:
            ts_ms = parse_ts_ms(row["TIMESTAMP"])
            itok  = int(row["ContextTokens"])
            otok  = int(row["GeneratedTokens"])
            buf.append((ts_ms, itok, otok))
            if len(buf) >= chunk_size:
                if not assume_sorted:
                    buf.sort(key=lambda x: x[0])
                w.writerows(buf); written += len(buf); buf.clear()
                if limit and written >= limit: break

        if buf:
            if not assume_sorted:
                buf.sort(key=lambda x: x[0])
            if limit:
                remain = max(0, limit - written)
                w.writerows(buf[:remain]); written += min(remain, len(buf))
            else:
                w.writerows(buf); written += len(buf)

    return written

def stream_cached_rows(cache_path):
    # gzip 캐시 CSV를 메모리 사용 최소화로 스트리밍
    with gzip.open(cache_path, "rt", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            yield (int(row["ts_ms"]), int(row["input_tokens"]), int(row["output_tokens"]))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="hf-cache/AzureLLMInferenceTrace_conv_1week.csv")
    ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--speedup", type=float, default=10.0, help="시간 압축 배수(도착 간격/배)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--log-every", type=float, default=2.0)
    ap.add_argument("--plot-window", type=float, default=30.0, help="스파크라인 윈도우(초)")
    ap.add_argument("--no-plot", action="store_true", help="스파크라인 비활성화")
    ap.add_argument("--plot-rows", type=int, default=2, help="스파크라인 라인 수(1 또는 2)")
    ap.add_argument("--bar-width", type=int, default=36, help="진행 바 너비(문자 단위)")
    ap.add_argument("--spark-width", type=int, default=48, help="스파크라인 가로 폭(문자 수, 고정)")
    ap.add_argument("--num-width", type=int, default=12, help="값 표시 폭(고정, 문자 수)")
    ap.add_argument("--plot-height", type=int, default=6, help="시각화 세로 높이(행 수, htop 스타일)")
    ap.add_argument("--y-width", type=int, default=8, help="Y축 라벨 너비(문자 수)")
    ap.add_argument("--cache-dir", default="hf-cache")
    ap.add_argument("--no-cache", action="store_true", help="강제로 원본에서 직접 읽기(캐시 미사용)")
    ap.add_argument("--assume-sorted", action="store_true", default=True, help="입력 CSV가 시간순 정렬됐다고 가정(빠름)")
    ap.add_argument("--sort", dest="assume_sorted", action="store_false", help="정렬 보장 없으면 켜기(느림)")
    ap.add_argument("--limit", type=int, default=None, help="최대 N행만 재생(샘플 테스트용)")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    base = os.path.basename(args.csv)
    cache_path = os.path.join(args.cache_dir, base + ".norm.csv.gz")

    # ----- cache build/check -----
    rows_count = None
    if args.no_cache:
        console.print("[yellow]Caching disabled (--no-cache). Reading source directly.[/]")
        # 직접 읽는 경로(대용량이면 느릴 수 있음): 여기선 캐시 경로를 임시로 사용하지 않음
        # 직접 모드에서도 동일 스트리밍 처리를 하려면 캐시 빌드와 동일 로직을 한 줄씩 yield하도록 구현 필요
        # 간편화를 위해 캐시 사용을 권장
    else:
        if need_rebuild_cache(args.csv, cache_path):
            console.print(f"[cyan]Building cache → {cache_path}[/] (assume_sorted={args.assume_sorted}, limit={args.limit})")
            written = build_cache(args.csv, cache_path, assume_sorted=args.assume_sorted, limit=args.limit)
            console.print(f"[green]Cache built. rows={written}[/]")
        else:
            console.print(f"[green]Using existing cache:[/] {cache_path}")

    # ----- load total count (for progress bar upper bound) -----
    # 캐시 파일의 라인 수를 빠르게 세어 total 계산
    def count_lines_gzip(path):
        cnt = 0
        with gzip.open(path, "rt") as f:
            for _ in f:
                cnt += 1
        return max(0, cnt - 1)  # header 제외
    if not args.no_cache:
        rows_count = count_lines_gzip(cache_path)
    else:
        # no-cache 경로에선 전체 행수를 모르면 진행률 전체값이 커지므로 limit만큼으로 잡음
        rows_count = args.limit if args.limit else 0

    # ----- request worker -----
    done_q = queue.Queue()
    sem = threading.Semaphore(args.concurrency)

    sent = 0
    err = 0
    sum_in = 0
    sum_out = 0
    # Network stats (application-layer bytes)
    tx_total = 0  # bytes sent (request payloads)
    rx_total = 0  # bytes received (streamed responses)
    tx_since = 0  # bytes sent since last UI refresh
    rx_since = 0  # bytes received since last UI refresh
    qps_curr = 0.0
    tx_rate_curr = 0.0
    rx_rate_curr = 0.0
    # Histories for sparklines
    plot_capacity = max(6, int(args.plot_window / max(args.log_every, 1e-6)) * max(1, min(2, args.plot_rows)))
    qps_hist = deque(maxlen=plot_capacity)
    tx_hist = deque(maxlen=plot_capacity)
    rx_hist = deque(maxlen=plot_capacity)
    lat_hist = deque(maxlen=plot_capacity)  # store per-interval p95 or mean if desired
    lat_p95_hist = deque(maxlen=plot_capacity)
    lat_p99_hist = deque(maxlen=plot_capacity)
    # Accumulator for current interval latencies
    latencies_curr = []
    sent_lock = threading.Lock()
    start_wall = time.time()

    def fire(itok, otok):
        payload = {
            "model": args.model,
            "messages":[{"role":"user","content": make_prompt(itok)}],
            "max_tokens": max(8, min(4096, otok)),
            "stream": True
        }
        head={"Authorization":"Bearer dummy","Content-Type":"application/json"}
        ok = True
        # Approximate application-layer tx bytes (JSON body only)
        req_tx = len(json.dumps(payload).encode("utf-8"))
        req_rx = 0
        start_req = time.time()
        latency_ms = None
        try:
            with requests.post(args.url, headers=head, json=payload, stream=True, timeout=args.timeout) as resp:
                for chunk in resp.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    req_rx += len(chunk)
                if resp.status_code >= 400: ok = False
                latency_ms = (time.time() - start_req) * 1000.0
        except Exception:
            ok = False
        finally:
            if latency_ms is None:
                latency_ms = (time.time() - start_req) * 1000.0
            done_q.put((ok, itok, otok, req_tx, req_rx, latency_ms))
            sem.release()

    # ----- UI -----
    def render_table():
        elapsed = max(1e-6, time.time() - start_wall)
        qps = sent / elapsed
        tbl = Table(title="Replay Live Stats", expand=True)
        tbl.add_column("Metric", style="cyan", no_wrap=True)
        tbl.add_column("Value")
        total_disp = f"{sent}" if rows_count == 0 else f"{sent}/{rows_count}"
        tbl.add_row("Total sent", total_disp)
        tbl.add_row("Errors", f"{err}")
        tbl.add_row("Elapsed (s)", f"{elapsed:,.1f}")
        tbl.add_row("QPS (avg)", f"{qps:,.2f}")
        tbl.add_row("Input tokens (sum)", f"{sum_in:,}")
        tbl.add_row("Output tokens (sum)", f"{sum_out:,}")
        # Progress integrated into table
        if rows_count and rows_count > 0:
            frac = sent / max(1, rows_count)
            bar = progress_bar(frac, width=args.bar_width)
            tbl.add_row("Progress", f"{bar}  {frac*100:,.1f}%")
        else:
            tbl.add_row("Progress", "n/a")
        # Network stats
        tbl.add_row("Net TX (total)", format_bytes(tx_total))
        tbl.add_row("Net RX (total)", format_bytes(rx_total))
        # Current rates visualization (two-row vertical bars) and fixed-width values
        def fixed_width(s: str, width: int) -> str:
            if len(s) >= width:
                return s[:width]
            return s + (" " * (width - len(s)))
        def color_for(curr_value: float, hist_values) -> str:
            vals = list(hist_values)
            if not vals:
                return "white"
            vmax = max(vals)
            if vmax <= 1e-12:
                return "white"
            ratio = curr_value / vmax
            if ratio < 0.33:
                return "green"
            elif ratio < 0.66:
                return "yellow"
            else:
                return "red"
        def build_multirow_bars(values, width, height, y_label_fmt=None, y_width=0):
            # Render columns with per-column coloring using 8-dot braille per cell (2x4 pixels per char).
            if width <= 0 or height <= 0:
                return ["" for _ in range(max(1, height))]
            vals = list(values)
            if not vals:
                # return empty box with optional y-axis placeholders
                rows = [" " * width for _ in range(height)]
                if y_width and y_width > 0:
                    rows[0] = (" " * y_width) + rows[0]
                    for r in range(1, height):
                        rows[r] = (" " * y_width) + rows[r]
                return rows
            vals = vals[-width:]
            vmin = min(vals)
            vmax = max(vals)
            rng = max(1e-12, vmax - vmin)
            # Always render exactly 'width' columns: if fewer samples, pad on the left
            left_pad = max(0, width - len(vals))
            cols = [None] * width
            for i in range(width):
                if i < left_pad:
                    cols[i] = None
                else:
                    cols[i] = vals[i - left_pad]
            rows = [[" "] * width for _ in range(height)]
            # helpers for braille composition
            FULL_BRAILLE = chr(0x28FF)  # all 8 dots set
            bottom_fill_order = [7, 8, 3, 6, 2, 5, 1, 4]  # bottom → top in visual order
            def braille_from_dots(dots):
                mask = 0
                for d in dots:
                    if 1 <= d <= 8:
                        mask |= (1 << (d - 1))
                return chr(0x2800 + mask)
            def braille_bottom_n(n):
                n = max(0, min(8, n))
                if n == 0:
                    return " "
                if n == 8:
                    return FULL_BRAILLE
                return braille_from_dots(bottom_fill_order[:n])
            # Map each value to a number of filled braille units (8 per row) and draw bottom-up
            # Draw bottom-up, ensuring continuity (no gaps) within each column
            for x in range(width):
                v = cols[x]
                if v is None:
                    continue
                norm = (v - vmin) / rng  # 0..1
                # total braille dot rows: height * 8 (8-dot resolution per cell)
                units = int(round(norm * height * 8))
                full_rows = units // 8
                rem_units = units % 8
                # column-level color based on value intensity
                if norm < 0.33:
                    col = "green"
                elif norm < 0.66:
                    col = "yellow"
                else:
                    col = "red"
                # bottom-up fill
                for r in range(height):
                    row_from_bottom = height - 1 - r
                    if row_from_bottom < full_rows:
                        rows[r][x] = f"[{col}]{FULL_BRAILLE}[/]"
                    elif row_from_bottom == full_rows and rem_units > 0:
                        rows[r][x] = f"[{col}]{braille_bottom_n(rem_units)}[/]"
                    else:
                        rows[r][x] = " "
            # Y-axis labels (max at top, min at bottom) with proper units
            y_w = max(0, y_width)
            if y_w > 0:
                if y_label_fmt is None:
                    def y_label_fmt(x):
                        return f"{x:,.2f}"
                top_label = y_label_fmt(vmax)
                bot_label = y_label_fmt(vmin)
                # Right align labels to prevent truncation of leading spaces
                top_label = f"{top_label:>{y_w}}"
                bot_label = f"{bot_label:>{y_w}}"
                rows[0] = [top_label] + rows[0]
                for r in range(1, height - 1):
                    rows[r] = [" " * y_w] + rows[r]
                rows[-1] = [bot_label] + rows[-1]
            return ["".join(r) for r in rows]
        # Fallback single-line sparkline
        def simple_spark(values, width):
            if width <= 0:
                return ""
            vals = list(values)
            if not vals:
                return " " * width
            blocks = "▁▂▃▄▅▆▇█"
            vals = vals[-width:]
            vmin = min(vals); vmax = max(vals); rng = max(1e-12, vmax - vmin)
            left_pad = width - len(vals)
            out = [" "] * left_pad
            for v in vals:
                idx = int((v - vmin) / rng * (len(blocks) - 1))
                out.append(blocks[idx])
            return "".join(out)
        spark_w = max(8, args.spark_width)
        num_w = max(6, args.num_width)
        if not args.no_plot and args.plot_rows >= 2:
            height = max(2, args.plot_height)
            # per-metric y-axis formatters
            def fmt_qps(v):
                return f"{v:,.2f} r/s"
            def fmt_rate(v):
                return format_rate(v)
            def fmt_ms(v):
                return format_ms(v)

            def build_braille_with_labels(values, y_fmt):
                # Returns (y_lines, bar_rows, vmin, vmax)
                bar_rows = build_multirow_bars(values, spark_w, height, y_label_fmt=y_fmt, y_width=0)
                vals = list(values)[-spark_w:]
                vmin = None; vmax = None
                if not vals:
                    y_lines = [" " * args.y_width for _ in range(height)]
                else:
                    vmin = min(vals); vmax = max(vals)
                    top = y_fmt(vmax)
                    bot = y_fmt(vmin)
                    y_lines = [" " * args.y_width for _ in range(height)]
                    if args.y_width > 0:
                        y_lines[0]  = f"{top:>{args.y_width}}"
                        y_lines[-1] = f"{bot:>{args.y_width}}"
                return y_lines, bar_rows, vmin, vmax

            def make_metric_panel(title, values, y_fmt, right_text, border_style):
                y_lines, bars, vmin, vmax = build_braille_with_labels(values, y_fmt)
                # Header lines (no scale info to avoid clutter)
                header_tbl = Table.grid(expand=False)
                header_tbl.add_column(no_wrap=True)
                if vmax is not None:
                    header_tbl.add_row(f"curr: {right_text}    max: {y_fmt(vmax)}")
                else:
                    header_tbl.add_row(f"curr: {right_text}")
                # Chart: y-labels + bars (2 columns only to avoid horizontal overflow)
                chart_tbl = Table.grid(expand=False)
                chart_tbl.add_column(no_wrap=True)
                chart_tbl.add_column(no_wrap=True)
                for i in range(height):
                    chart_tbl.add_row(y_lines[i], bars[i])
                # Stack header + chart vertically
                container = Table.grid(expand=False)
                container.add_column(no_wrap=True)
                container.add_row(header_tbl)
                container.add_row(chart_tbl)
                return Panel(container, title=title, border_style=border_style, padding=(0,1))

            # compact right labels of fixed width
            right_w = max(6, num_w)
            qps_right = format_qps_compact(qps_curr, right_w)
            tx_right  = format_rate_compact(tx_rate_curr, right_w)
            rx_right  = format_rate_compact(rx_rate_curr, right_w)
            lat95_right = format_ms(lat_p95_hist[-1]) if len(lat_p95_hist) else "n/a"
            lat99_right = format_ms(lat_p99_hist[-1]) if len(lat_p99_hist) else "n/a"

            qps_panel = make_metric_panel("QPS (curr)", qps_hist, fmt_qps, qps_right, "cyan")
            tx_panel  = make_metric_panel("TX rate (curr)", tx_hist, fmt_rate, tx_right, "magenta")
            rx_panel  = make_metric_panel("RX rate (curr)", rx_hist, fmt_rate, rx_right, "magenta")
            lat95_panel = make_metric_panel("Latency p95", lat_p95_hist, fmt_ms, lat95_right, "yellow")
            lat99_panel = make_metric_panel("Latency p99", lat_p99_hist, fmt_ms, lat99_right, "yellow")
            use_panels = True
        else:
            qps_num = fixed_width(f"{qps_curr:,.2f}", num_w)
            tx_num  = fixed_width(f"{format_rate(tx_rate_curr)}", num_w)
            rx_num  = fixed_width(f"{format_rate(rx_rate_curr)}", num_w)
            qps_color = color_for(qps_curr, qps_hist)
            tx_color  = color_for(tx_rate_curr, tx_hist)
            rx_color  = color_for(rx_rate_curr, rx_hist)
            lat95_color = color_for(lat_p95_hist[-1] if len(lat_p95_hist) else 0, lat_p95_hist)
            lat99_color = color_for(lat_p99_hist[-1] if len(lat_p99_hist) else 0, lat_p99_hist)
            qps_cell = f"[{qps_color}]{simple_spark(qps_hist, spark_w)}[/] {qps_num}"
            tx_cell  = f"[{tx_color}]{simple_spark(tx_hist,  spark_w)}[/] {tx_num}"
            rx_cell  = f"[{rx_color}]{simple_spark(rx_hist,  spark_w)}[/] {rx_num}"
            lat95_cell = f"[{lat95_color}]{simple_spark(lat_p95_hist, spark_w)}[/] {fixed_width(format_ms(lat_p95_hist[-1]) if len(lat_p95_hist) else 'n/a', num_w)}"
            lat99_cell = f"[{lat99_color}]{simple_spark(lat_p99_hist, spark_w)}[/] {fixed_width(format_ms(lat_p99_hist[-1]) if len(lat_p99_hist) else 'n/a', num_w)}"
            use_panels = False
        # Build a dedicated visualization panel separate from the metrics rows
        vis_tbl = Table.grid(expand=True)
        vis_tbl.add_column(ratio=1)
        vis_tbl.add_column(ratio=1)
        vis_tbl.add_column(ratio=1)
        if use_panels:
            qps_render = qps_panel
            tx_render  = tx_panel
            rx_render  = rx_panel
            lat95_render = lat95_panel
            lat99_render = lat99_panel
        else:
            qps_render = Panel(qps_cell, title="QPS (curr)", border_style="cyan", padding=(0,1))
            tx_render  = Panel(tx_cell,  title="TX rate (curr)", border_style="magenta", padding=(0,1))
            rx_render  = Panel(rx_cell,  title="RX rate (curr)", border_style="magenta", padding=(0,1))
            lat95_render = Panel(lat95_cell, title="Latency p95", border_style="yellow", padding=(0,1))
            lat99_render = Panel(lat99_cell, title="Latency p99", border_style="yellow", padding=(0,1))
        vis_tbl.add_row(qps_render, tx_render, rx_render)
        vis_tbl.add_row(lat95_render, lat99_render, Panel("", border_style="dim"))
        tbl.add_row("Visualize", Panel(vis_tbl, border_style="white", padding=(0,1)))
        tbl.add_row("TX rate (avg)", format_rate(tx_total / elapsed))
        tbl.add_row("RX rate (avg)", format_rate(rx_total / elapsed))
        # ETA 추정(rough)
        eta = "n/a"
        if rows_count and sent > 0:
            spd = sent / elapsed
            remain = max(0, rows_count - sent)
            if spd > 0:
                eta = f"{remain/spd:,.1f}s"
        tbl.add_row("ETA", eta)
        return tbl

    # 외부 진행 표시(tqdm) 제거: 통합 테이블에서 표시

    # ----- schedule & replay -----
    # 캐시로부터 스트리밍
    rows_iter = stream_cached_rows(cache_path) if not args.no_cache else None

    # 첫 행의 ts_ms(시간 정렬 가정)
    # 캐시가 비면 빠르게 종료
    first = None
    peeked = []
    if rows_iter:
        try:
            first = next(rows_iter)
            peeked.append(first)
        except StopIteration:
            console.print("[red]Empty cache. Nothing to replay.[/]")
            return

    # 시간 정렬 가정: 첫 ts를 기준으로 도착 간격 재현
    if first:
        t0 = first[0]
    else:
        # no-cache 모드에서 별도 처리하고 싶다면 여기 로직 보강
        console.print("[red]no-cache mode not implemented for live replay. Use cache (default).[/]")
        return

    wall0 = time.time()
    next_stat = time.time() + args.log_every
    last_update = time.time()
    sent_at_last = 0

    def drain_done(non_blocking=True):
        nonlocal sent, err, sum_in, sum_out, tx_total, rx_total, tx_since, rx_since
        while True:
            try:
                ok, i, o, btx, brx, l_ms = done_q.get_nowait() if non_blocking else done_q.get()
            except queue.Empty:
                break
            with sent_lock:
                sent += 1
                sum_in += i
                sum_out += o
                tx_total += btx
                rx_total += brx
                tx_since += btx
                rx_since += brx
                latencies_curr.append(l_ms)
                if not ok: err += 1
            # progress handled by integrated table

    with Live(render_table(), refresh_per_second=4, console=console, transient=False) as live:
        # 먼저 peek한 행 처리
        for row in peeked + list(rows_iter):
            ts_ms, itok, otok = row
            target = (ts_ms - t0)/1000.0/args.speedup
            now = time.time() - wall0
            if target > now:
                time.sleep(target - now)

            sem.acquire()
            threading.Thread(target=fire, args=(itok, otok), daemon=True).start()

            drain_done(non_blocking=True)

            if time.time() >= next_stat:
                now = time.time()
                interval = max(1e-6, now - last_update)
                # Update current rates and histories
                with sent_lock:
                    qps_curr = (sent - sent_at_last) / interval
                    tx_rate_curr = tx_since / interval
                    rx_rate_curr = rx_since / interval
                    # latency percentiles for current interval
                    p95 = percentile(latencies_curr, 0.95) if latencies_curr else None
                    p99 = percentile(latencies_curr, 0.99) if latencies_curr else None
                if not args.no_plot:
                    qps_hist.append(qps_curr)
                    tx_hist.append(tx_rate_curr)
                    rx_hist.append(rx_rate_curr)
                    if p95 is not None:
                        lat_p95_hist.append(p95)
                    if p99 is not None:
                        lat_p99_hist.append(p99)
                live.update(render_table())
                next_stat = time.time() + args.log_every
                last_update = now
                sent_at_last = sent
                tx_since = 0
                rx_since = 0
                latencies_curr.clear()

        # 남은 완료 수집
        while True:
            before = sent
            drain_done(non_blocking=True)
            if sent == before:
                # 더 이상 진행 없으면 한번 블록해서 남은 걸 받아본다
                try:
                    drain_done(non_blocking=False)
                except Exception:
                    pass
            # 모두 끝났다면 탈출
            if rows_count and sent >= rows_count:
                break
            # 약간 쉼
            if time.time() >= next_stat:
                now = time.time()
                interval = max(1e-6, now - last_update)
                with sent_lock:
                    qps_curr = (sent - sent_at_last) / interval
                    tx_rate_curr = tx_since / interval
                    rx_rate_curr = rx_since / interval
                    p95 = percentile(latencies_curr, 0.95) if latencies_curr else None
                    p99 = percentile(latencies_curr, 0.99) if latencies_curr else None
                if not args.no_plot:
                    qps_hist.append(qps_curr)
                    tx_hist.append(tx_rate_curr)
                    rx_hist.append(rx_rate_curr)
                    if p95 is not None:
                        lat_p95_hist.append(p95)
                    if p99 is not None:
                        lat_p99_hist.append(p99)
                live.update(render_table())
                next_stat = time.time() + args.log_every
                last_update = now
                sent_at_last = sent
                tx_since = 0
                rx_since = 0
                latencies_curr.clear()
            time.sleep(0.05)

    console.print("[green]Replay finished.[/]")
    console.print(render_table())

if __name__=="__main__":
    main()
