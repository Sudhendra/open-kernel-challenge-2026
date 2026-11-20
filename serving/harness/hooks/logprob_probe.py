#!/usr/bin/env python3
"""Peak device memory of one prompt-logprob turn (the shape of the judge's KL turns).

Sends one /generate request with a >= 16,384-token prompt, return_logprob=True and
logprob_start_len=0 (a logprob for every prompt token), max_new_tokens=1, while a
thread samples NVML memory.used per GPU (nvidia-smi, every --interval s). Prints,
per GPU: before, peak during, after, and the deltas. Also prints the memory fields
the server reports (/server_info internal_states[*].memory_usage, /metrics
*memory* gauges); those are the static startup plan, not a live peak, which is
why NVML is the measurement. The caching allocator keeps what the request grew,
so "after - before" is the persistent growth and "peak - before" the transient.

Stdlib only. Always exits 0 and prints whatever it could measure; total wall
time is capped by --budget (default 170 s).

  python3 logprob_probe.py --url http://127.0.0.1:30000 [--tokens 16400]

/generate with input_ids, not chat: the chat API cannot return prompt-token
logprobs, and fixed token ids make the chunk shape exact (first chunk = the
--chunked-prefill-size, 16,384 under the judge's settings).
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


def _http(url, payload=None, timeout=10.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _nvml_used_mib():
    """{gpu index: memory.used MiB} or None when nvidia-smi is unavailable."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return None
    used = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            try:
                used[int(parts[0])] = float(parts[1])
            except ValueError:
                pass
    return used or None


class _Sampler(threading.Thread):
    def __init__(self, interval):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = {}
        self.samples = 0
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            used = _nvml_used_mib()
            if used:
                self.samples += 1
                for gpu, mib in used.items():
                    self.peak[gpu] = max(self.peak.get(gpu, 0.0), mib)
            self.stop.wait(self.interval)


def _server_memory_fields(base):
    lines = []
    try:
        info = json.loads(_http(base + "/server_info", timeout=10))
        for i, state in enumerate(info.get("internal_states") or []):
            mem = state.get("memory_usage") if isinstance(state, dict) else None
            if mem is not None:
                lines.append(f"server_info internal_states[{i}].memory_usage = {json.dumps(mem)}")
        if not lines:
            lines.append("server_info: no memory_usage field")
    except Exception as exc:
        lines.append(f"server_info: unavailable ({exc!r})")
    try:
        text = _http(base + "/metrics", timeout=10)
        mem = [l for l in text.splitlines() if "memory" in l and not l.startswith("#")]
        lines.extend(f"metrics {l}" for l in mem[:40]) if mem else lines.append(
            "metrics: no *memory* gauges"
        )
    except Exception as exc:
        lines.append(f"metrics: unavailable ({exc!r})")
    return lines


def _fmt(d):
    return " ".join(f"gpu{g}={v:.0f}" for g, v in sorted(d.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", default="", help="variant dir (the harness passes it; the probe writes probe.json there if set)")
    ap.add_argument("--tokens", type=int, default=16400, help="prompt length (>= 16384)")
    ap.add_argument("--vocab-lo", type=int, default=1000)
    ap.add_argument("--vocab-hi", type=int, default=30000)
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("--budget", type=float, default=170.0)
    args = ap.parse_args()
    t0 = time.time()
    base = args.url.rstrip("/")
    tokens = max(args.tokens, 16384)
    print(f"[probe] url={base} prompt_tokens={tokens} logprob_start_len=0", flush=True)

    for line in _server_memory_fields(base):
        print("[probe]", line, flush=True)

    before = _nvml_used_mib()
    print(f"[probe] nvml before MiB: {_fmt(before) if before else 'unavailable'}", flush=True)
    sampler = _Sampler(args.interval)
    sampler.start()

    # Fresh random ids every run: no radix-cache prefix hit, the whole prompt prefills.
    rng = random.Random(time.time_ns())
    payload = {
        "input_ids": [rng.randint(args.vocab_lo, args.vocab_hi) for _ in range(tokens)],
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": 0,
    }
    status = "not sent"
    req_s = None
    try:
        timeout = max(5.0, args.budget - (time.time() - t0) - 15.0)
        t1 = time.time()
        body = json.loads(_http(base + "/generate", payload, timeout=timeout))
        req_s = time.time() - t1
        meta = body.get("meta_info", {}) if isinstance(body, dict) else {}
        n_lp = len(meta.get("input_token_logprobs") or [])
        status = f"ok in {req_s:.1f} s, prompt_tokens={meta.get('prompt_tokens')}, input_token_logprobs={n_lp}"
    except urllib.error.HTTPError as exc:
        status = f"HTTP {exc.code}: {exc.read()[:300]!r}"
    except Exception as exc:
        status = f"failed: {exc!r}"
    print(f"[probe] request {status}", flush=True)

    time.sleep(min(1.0, max(0.0, args.budget - (time.time() - t0))))
    sampler.stop.set()
    sampler.join(timeout=5)
    after = _nvml_used_mib()
    print(f"[probe] nvml samples={sampler.samples} interval={args.interval}s", flush=True)
    if before and sampler.peak:
        after = after or {}
        print("[probe] per GPU MiB: before / peak / after | peak-before / after-before", flush=True)
        for gpu in sorted(before):
            b, p, a = before[gpu], sampler.peak.get(gpu, float("nan")), after.get(gpu, float("nan"))
            print(f"[probe] gpu{gpu}: {b:.0f} / {p:.0f} / {a:.0f} | {p - b:+.0f} / {a - b:+.0f}", flush=True)
        worst = max(sampler.peak.get(g, 0) - before[g] for g in before)
        print(f"[probe] max peak-before over GPUs: {worst:+.0f} MiB ({worst / 1024:+.2f} GiB)", flush=True)
    else:
        print("[probe] no NVML readings; only the server fields above are available", flush=True)
    print(f"[probe] done in {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # the hook must never fail the session
        print(f"[probe] error: {exc!r}", flush=True)
    sys.exit(0)
