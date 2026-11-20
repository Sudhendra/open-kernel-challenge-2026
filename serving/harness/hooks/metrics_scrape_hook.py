#!/usr/bin/env python3
"""Lane KFIN: 1 Hz /metrics time series for a proxy run (HOOK for harness/profile_session.sh). CPU only, stdlib only.

Why: lane KWARM measured ~+10 ms of c32 wall per finished request (= per arriving next turn: the proxy sends the next
turn as soon as one ends, so the two are collinear) and could not say where it goes. The server already accounts,
per TP rank, the scheduler loop's wall time by stage (recv_requests, process_input_requests, get_next_batch_to_run,
run_batch, process_batch_result, other, ...) and flushes it to Prometheus about once a second
(metrics_reporter.py:_SCHEDULER_TIME_ACCOUNTING_INTERVAL_NS), plus scheduler CPU seconds per rank, the tokenizer
process's CPU seconds, HiCache backup/eviction counters and pool gauges. Only the end-of-run total is saved today
(metrics_after_proxy.txt). Sampling it every second during the proxy gives, per rank and per second, which stage
grows when requests finish, whether a rank's wall time grows without its CPU time (CPU starvation, e.g. by the
proxy-only tokenizer), and whether backups/evictions line up. No server patch, no GPU cost.

Use as a variant hook (runs after warm-up, before the proxy; returns at once, the sampler keeps running detached):
  --variants "ctl:SRC=full,HOOK=lanes/kfin_tools/metrics_scrape_hook.py,ARGS=..."
or by hand next to any server:  python3 metrics_scrape_hook.py --url http://127.0.0.1:30000 --out <dir> [--foreground]
Writes <out>/metrics_ts.jsonl: one line per sample {"t": wall s, "m": monotonic s, "v": {series: value}}.
The sampler stops after --max-s seconds or after 20 consecutive failed scrapes (server gone)."""

import argparse
import json
import os
import re
import sys
import time
import urllib.request

KEEP = re.compile(
    r"^sglang:(scheduler_stage_seconds_total|scheduler_idle_seconds_total|scheduler_process_cpu_seconds_total"
    r"|process_cpu_seconds_total|hicache_backup_bytes_total|hicache_backup_duration_seconds_(?:sum|count)"
    r"|evicted_tokens_total|eviction_duration_seconds_(?:sum|count)|kv_available_tokens|kv_evictable_tokens"
    r"|kv_used_tokens|num_running_reqs|num_queue_reqs|forward_execution_seconds_total|load_back_tokens_total"
    r"|generation_tokens_total|num_requests_total|spec_verify_calls_total|gen_throughput)(\{[^}]*\})? (\S+)$"
)
LABELS = re.compile(r'(category|forward_overlap|tp_rank|component|mode)="([^"]*)"')


def parse(text):
    out = {}
    for line in text.splitlines():
        m = KEEP.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        key = name + "|" + ",".join(f"{k}={v}" for k, v in LABELS.findall(labels))
        try:
            out[key] = out.get(key, 0.0) + float(val)
        except ValueError:
            continue
    return out


def sample_loop(url, path, interval, max_s):
    fails = 0
    t_end = time.monotonic() + max_s
    with open(path, "a", buffering=1) as f:
        while time.monotonic() < t_end and fails < 20:
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(url + "/metrics", timeout=5) as r:
                    text = r.read().decode("utf-8", "replace")
                f.write(json.dumps({"t": round(time.time(), 3), "m": round(t0, 3), "v": parse(text)}) + "\n")
                fails = 0
            except Exception:  # noqa: BLE001 - the server is booting, busy or gone
                fails += 1
            time.sleep(max(0.0, interval - (time.monotonic() - t0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--max-s", type=float, default=3 * 3600)
    ap.add_argument("--foreground", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "metrics_ts.jsonl")
    if a.foreground:
        sample_loop(a.url.rstrip("/"), path, a.interval, a.max_s)
        return
    # Detach (double fork + new session) so the harness's capped() wrapper and the hook's exit do not stop it.
    if os.fork() > 0:
        print(f"metrics_scrape_hook: sampling {a.url}/metrics every {a.interval}s -> {path}")
        return
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    try:
        sample_loop(a.url.rstrip("/"), path, a.interval, a.max_s)
    finally:
        os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
