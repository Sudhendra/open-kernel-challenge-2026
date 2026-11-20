#!/usr/bin/env python3
"""GPU busy vs idle from a torch-profiler trace: how much of wall time the GPU is executing.

  python trace_busy.py <trace.json.gz> [--steps N]

Prints the trace span, union of kernel busy time, idle fraction, per-step wall if --steps is
given, the count of CUDA graph launches, and the ten longest gaps between kernels with the
CPU op that was running on the host at that moment.
"""
import gzip
import json
import sys
from collections import Counter


def main():
    path = sys.argv[1]
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else None
    with gzip.open(path, "rt") as f:
        ev = json.load(f)["traceEvents"]
    kern = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    cpu = [e for e in ev if e.get("cat") in ("cpu_op", "python_function", "user_annotation") and "dur" in e]
    kern.sort(key=lambda e: e["ts"])
    t0, t1 = kern[0]["ts"], max(e["ts"] + e["dur"] for e in kern)
    busy, cur_s, cur_e, gaps = 0.0, kern[0]["ts"], kern[0]["ts"] + kern[0]["dur"], []
    for e in kern[1:]:
        s, en = e["ts"], e["ts"] + e["dur"]
        if s > cur_e:
            gaps.append((s - cur_e, cur_e, s))
            busy += cur_e - cur_s
            cur_s, cur_e = s, en
        else:
            cur_e = max(cur_e, en)
    busy += cur_e - cur_s
    span = t1 - t0
    launches = Counter(e["name"] for e in cpu if e["name"] in ("cudaGraphLaunch", "cudaLaunchKernel"))
    print(f"span {span / 1e3:.1f} ms  gpu busy {busy / 1e3:.1f} ms  idle {100 * (1 - busy / span):.1f}%  kernels {len(kern)}  {dict(launches)}")
    if steps:
        print(f"per step: wall {span / steps / 1e3:.2f} ms, gpu busy {busy / steps / 1e3:.2f} ms")
    gaps.sort(reverse=True)
    print("longest gaps (ms) and the host ops active then:")
    for g, a, b in gaps[:10]:
        active = [e["name"] for e in cpu if e["ts"] <= a <= e["ts"] + e["dur"] and e.get("cat") == "cpu_op"]
        print(f"  {g / 1e3:6.2f}  {active[-3:]}")
    total_gap = sum(g for g, _, _ in gaps)
    small = sum(g for g, _, _ in gaps if g < 50)
    print(f"gap total {total_gap / 1e3:.1f} ms, of which gaps <50us (launch overhead) {small / 1e3:.1f} ms in {sum(1 for g, _, _ in gaps if g < 50)} gaps")


if __name__ == "__main__":
    main()
