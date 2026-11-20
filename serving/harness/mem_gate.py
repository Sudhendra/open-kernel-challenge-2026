#!/usr/bin/env python3
"""GPU-memory gate, read the way the judge reads `gpu_memory.over_stock_gb` (limit 2 GB per GPU).

What the judge compares (research/proxy_rebase_2026-11-15.md section 3, lane_audit_2026-11-14.md section 3): the runtime
peak of device memory per GPU over the run, ours vs stock's on the same node. Static allocations are not it: every build
reports kvcache/graph fields BELOW stock (34.0 vs 37.6 GB KV, 13.1 vs 15.8 GB prefill graphs) and still reads 0 / -1.4 /
+2.2 GB, because post-capture KV sizing fills the device to the same budget and what is left over is the caching
allocator's runtime growth (eager transients outside the graph pools). NVML memory.used sees exactly that: the allocator
keeps what it reserved, so a 0.5 s sampler catches the high-water mark even of sub-second transients.

  python3 mem_gate.py peak <mem_run.csv> <out.json>          # per-GPU peak MiB of an `nvidia-smi --query-gpu=index,memory.used
                                                              #  --format=csv,noheader,nounits -lms 500` log
  python3 mem_gate.py compare <stock_peak.json> <cand_peak.json> <out.txt> [--limit 2.0] [--margin 0.3]
                              [--stock-dir <dir> --cand-dir <dir>]   # adds the loaded-probe reading when present

compare writes the per-GPU table and two readings: max over GPUs of (cand - stock) [per-GPU] and max(cand) - max(stock)
[node]. The gate uses the larger. GB here are decimal (MiB * 1.048576 / 1000): if the judge means GiB, our number is
7% larger, i.e. conservative. PASS when the reading <= limit - margin (1.7 GB by default), WARN up to the limit, FAIL above.
Prints one summary line (also the last line of out.txt). Exit 0 always: a gate verdict is data, not a crash.

Second number (MEM_LOADED_PROBE runs only): the "loaded KL transient over stock" = max over loaded probes and GPUs of the
candidate's NVML peak-before during a 16k logprob turn fired mid-c32, minus the same for stock. The validation run
(out/20261115T042354Z) showed why the window peak alone cannot see S5's judge reading: every boot reaches its NVML peak
within ~10 s of /health (warmup) and the c32 rung adds < 50 MiB, so a transient that only exists when a logprob prefill
chunk meets 31 decoding users (MIXED batch, C1-F slices, W(a) SP gathers) never ran on Modal (proxy_rebase section 6.2).
The loaded probes run inside the window, so the first number already includes whatever NVML growth they cause; the second
isolates it (does the candidate's loaded logprob turn grow memory more than stock's?). The verdict stays on the first
number; the second prints as info and flags WARN above 1.0 GB.
"""
import glob
import os
import re
import json
import sys

MIB_TO_GB = 1.048576 / 1000.0


def peak(csv_path, out_path):
    peaks = {}
    n = 0
    for line in open(csv_path, errors="replace"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            gpu, used = int(parts[0]), float(parts[1].split()[0])
        except (ValueError, IndexError):
            continue
        n += 1
        peaks[gpu] = max(peaks.get(gpu, 0.0), used)
    json.dump({"samples": n, "peak_mib": {str(k): v for k, v in sorted(peaks.items())}}, open(out_path, "w"))
    top = max(peaks.values()) if peaks else 0
    print(f"mem peak: {len(peaks)} GPUs, {n} samples, max {top:.0f} MiB ({top * MIB_TO_GB:.2f} GB)")


def loaded_transient(d):
    """max over mem_probe_loaded_*.txt in d and GPUs of (peak - before) MiB, or None when no probe measured."""
    best = None
    for f in glob.glob(os.path.join(d or "", "mem_probe_loaded_*.txt")):
        for m in re.finditer(r"gpu\d+: [\d.]+ / [\d.]+ / [\d.]+ \| ([+-]?\d+) /", open(f, errors="replace").read()):
            v = float(m.group(1))
            best = v if best is None else max(best, v)
    return best


def compare(stock_path, cand_path, out_path, limit=2.0, margin=0.3, stock_dir=None, cand_dir=None):
    try:
        s = json.load(open(stock_path))["peak_mib"]
        c = json.load(open(cand_path))["peak_mib"]
    except (OSError, ValueError, KeyError) as e:
        line = f"mem over stock: UNKNOWN ({type(e).__name__}: {e})"
        open(out_path, "w").write(line + "\n")
        print(line)
        return
    gpus = sorted(set(s) & set(c), key=int)
    if not gpus:
        line = "mem over stock: UNKNOWN (no GPU in both peak files)"
        open(out_path, "w").write(line + "\n")
        print(line)
        return
    rows = ["gpu  stock_peak_MiB  cand_peak_MiB  delta_GB"]
    deltas = []
    for g in gpus:
        d = (c[g] - s[g]) * MIB_TO_GB
        deltas.append(d)
        rows.append(f"{g:>3}  {s[g]:>14.0f}  {c[g]:>13.0f}  {d:>+8.2f}")
    per_gpu = max(deltas)
    node = (max(c[g] for g in gpus) - max(s[g] for g in gpus)) * MIB_TO_GB
    reading = max(per_gpu, node)
    verdict = "PASS" if reading <= limit - margin else "WARN" if reading <= limit else "FAIL"
    rows.append(f"per-GPU max delta {per_gpu:+.2f} GB; node max-minus-max {node:+.2f} GB")
    line = (f"mem over stock: {reading:+.2f} GB {verdict} (limit {limit:.1f}, pass <= {limit - margin:.1f}; "
            f"per-GPU {per_gpu:+.2f}, node {node:+.2f})")
    if stock_dir or cand_dir:
        st, ct = loaded_transient(stock_dir), loaded_transient(cand_dir)
        if (st is None) != (ct is None):
            line += "; loaded KL transient: n/a (a loaded probe on one side only)"
        elif st is not None:
            lt = (ct - st) * MIB_TO_GB
            rows.append(f"loaded KL transient (peak-before during a mid-c32 logprob turn): stock {st:.0f} MiB, cand {ct:.0f} MiB")
            line += f"; loaded KL transient over stock {lt:+.2f} GB{' WARN' if lt > 1.0 else ''}"
    rows.append(line)
    open(out_path, "w").write("\n".join(rows) + "\n")
    print(line)


def main(argv):
    if len(argv) >= 3 and argv[0] == "peak":
        peak(argv[1], argv[2])
    elif len(argv) >= 4 and argv[0] == "compare":
        kw = {}
        rest = argv[4:]
        for flag in ("--limit", "--margin"):
            if flag in rest:
                kw[flag[2:]] = float(rest[rest.index(flag) + 1])
        for flag in ("--stock-dir", "--cand-dir"):
            if flag in rest:
                kw[flag[2:].replace("-", "_")] = rest[rest.index(flag) + 1]
        compare(argv[1], argv[2], argv[3], **kw)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
