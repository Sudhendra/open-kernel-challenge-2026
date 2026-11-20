#!/usr/bin/env python3
"""Judge-calibrated read of a paired proxy A/B: per-rung % deltas (variant vs control, same node) -> estimated judge score.

  python3 -I harness/stack_read.py <ctl_dir> <var_dir> [--base out/judge/status_186157d12aeb4eba_*.json] [--c32 same|zero|<pct>]

Each dir holds proxy.rungs (one JSON line per rung). Proxy -> judge calibration from METHOD.md: proxy gains overshoot the judge,
log-linear factors c1 1.33x, c4 1.13x, c16 1.05x (a proxy +6% at c1 is read as about +4.5% on the judge). c32 is never on the
proxy: --c32 same (default) uses the calibrated c16 delta, zero assumes none, or give a percent. Score via score_model.py whatif
on the scored run's JSON (the judge's own frontier arithmetic). Noise: the proxy self-floor is about +-3% at c1/c4 (1x1x2, 4x1x2
reps) and +-1% at c16, so deltas inside that are reported as 0 unless --raw.
"""
import argparse, glob, json, math, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
CAL = {1: 1.33, 4: 1.13, 16: 1.05, 32: 1.0}
NOISE = {1: 3.0, 4: 3.0, 16: 1.0, 32: 1.0}
def rungs(d):
    out = {}
    for line in open(os.path.join(d, "proxy.rungs")):
        line = line.strip()
        if line.startswith("{"):
            r = json.loads(line); out[int(r["concurrency"])] = float(r["throughput_tok_s"])
    return out
ap = argparse.ArgumentParser(); ap.add_argument("ctl"); ap.add_argument("var")
ap.add_argument("--base", default=os.path.join(os.path.dirname(HERE), "out/judge/status_186157d12aeb4eba_*.json"))
ap.add_argument("--c32", default="same"); ap.add_argument("--raw", action="store_true")
a = ap.parse_args()
c, v = rungs(a.ctl), rungs(a.var)
moves = {}
print(f"{'rung':>5} {'ctl':>9} {'var':>9} {'proxy %':>8} {'judge %':>8}  note")
for r in sorted(set(c) & set(v)):
    p = 100.0 * (v[r] / c[r] - 1.0)
    j = 100.0 * (math.exp(math.log(v[r] / c[r]) / CAL[r]) - 1.0)
    note = ""
    if not a.raw and abs(p) < NOISE[r]: j = 0.0; note = f"inside the +-{NOISE[r]:.0f}% proxy floor -> 0"
    moves[r] = j; print(f"c{r:<4} {c[r]:9.1f} {v[r]:9.1f} {p:+8.1f} {j:+8.1f}  {note}")
if 32 not in moves:
    moves[32] = moves.get(16, 0.0) if a.c32 == "same" else (0.0 if a.c32 == "zero" else float(a.c32))
    print(f"c32   (not on the proxy) assumed {moves[32]:+.1f}% ({a.c32})")
args = [f"c{r}={m:+.1f}" for r, m in sorted(moves.items())]
base = sorted(glob.glob(a.base))[-1]
print("score_model whatif", os.path.basename(base), " ".join(args))
sys.stdout.flush(); subprocess.run([sys.executable, os.path.join(HERE, "score_model.py"), "whatif", base] + args)
