#!/usr/bin/env python3
"""Raw read of a multi-arm proxy run: per rung, each arm's throughput / per-user decode speed / TTFT and its % against the
control, then the judge score each arm would get if the control were the scored run (--base): the judge's own frontier walk
(log-log interpolation of our (tok_s_per_user_e2e, tok_s_per_gpu) points at stock's per-user speeds, geometric mean of the four
ratios), with each of our rungs scaled by the arm's proxy throughput ratio. The judge's per-user speed is end to end (output
tokens over the conversation's wall time), so it moves with throughput; the proxy's decode-only per-user median does not score.

  python3 -I harness/lad_read.py out/<run> [--ctl ctl]

No proxy-to-judge calibration is applied: with the token-ids client the proxy ranks like the judge but has over-read cold-storm
gains by about 1.5x, so read the 16/32-user rows as an upper bound.
"""
import argparse, glob, json, math, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
def frontier_at(pts, u):
    """Our tok/s/GPU at per-user speed u: log-log along our points, capped at the ends."""
    pts = sorted(pts)  # by per-user speed, ascending = highest concurrency first
    if u <= pts[0][0]:
        return pts[0][1]
    if u >= pts[-1][0]:
        return pts[-1][1]
    for (u0, t0), (u1, t1) in zip(pts, pts[1:]):
        if u0 <= u <= u1:
            w = (math.log(u) - math.log(u0)) / (math.log(u1) - math.log(u0))
            return math.exp(math.log(t0) + w * (math.log(t1) - math.log(t0)))
def judge_score(res, scale):
    """Score of the scored run `res` with our rung c scaled by scale[c] (throughput and end-to-end per-user speed alike)."""
    ours = [(r["tok_s_per_user_e2e"] * scale.get(r["concurrency"], 1.0), r["tok_s_per_gpu"] * scale.get(r["concurrency"], 1.0)) for r in res["rungs"]]
    ratios = [frontier_at(ours, p["tok_s_per_user"]) / p["stock_tok_s_per_gpu"] for p in res["frontier"]["points"]]
    return math.exp(sum(math.log(x) for x in ratios) / len(ratios)), ratios
def rungs(d):
    out = {}
    p = os.path.join(d, "proxy.rungs")
    if not os.path.exists(p):
        return out
    for line in open(p):
        line = line.strip()
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        out.setdefault(int(r["concurrency"]), []).append(r)
    return out
def val(rs, k):
    xs = [float(r[k]) for r in rs if r.get(k) is not None]
    return st.mean(xs) if xs else float("nan")
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--ctl", default="ctl")
ap.add_argument("--base", default=os.path.join(os.path.dirname(HERE), "out/judge/status_788356d87cda4e27_fetched.json"))
a = ap.parse_args()
vd = os.path.join(a.run, "variants")
arms = [n for n in sorted(os.listdir(vd)) if os.path.isdir(os.path.join(vd, n))]
R = {n: rungs(os.path.join(vd, n)) for n in arms}
if a.ctl not in R:
    raise SystemExit(f"no control arm {a.ctl} in {arms}")
arms = [a.ctl] + [n for n in arms if n != a.ctl]
scale = {n: {} for n in arms}
for c in sorted(R[a.ctl]):
    print(f"-- {c} users")
    base = R[a.ctl][c]
    for n in arms:
        rs = R[n].get(c)
        if not rs:
            print(f"   {n:8s} (no data)"); continue
        t, u, f = val(rs, "throughput_tok_s"), val(rs, "interactivity_tok_s_median"), val(rs, "ttft_s_median")
        dt = 100 * (t / val(base, "throughput_tok_s") - 1); du = 100 * (u / val(base, "interactivity_tok_s_median") - 1)
        fails = sum(int(r.get("failed", 0)) for r in rs); wall = val(rs, "wall_s")
        scale[n][c] = 1 + dt / 100
        print(f"   {n:8s} thr {t:8.1f} ({dt:+5.1f}%)  per-user {u:7.1f} ({du:+5.1f}%)  ttft {f:6.3f} s  wall {wall:6.1f} s  lines {len(rs)}  failed {fails}")
res = json.load(open(sorted(glob.glob(a.base))[-1]))
res = res.get("result") or res
s0, r0 = judge_score(res, {})
print(f"-- judge score if {a.ctl} = the scored run ({res['frontier']['score']}; walk reproduces {s0:.4f}); unmeasured rungs unchanged")
for n in arms[1:]:
    s1, r1 = judge_score(res, scale[n])
    print(f"   {n:8s} {s1:.4f}x ({100 * (s1 / s0 - 1):+5.2f}%)  ratios " + " ".join(f"{x:.3f}" for x in r1))
