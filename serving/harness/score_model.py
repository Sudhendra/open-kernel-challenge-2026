#!/usr/bin/env python3
"""Score model for the OKC serving track, fitted to the judge's own JSON.

The judge scores "throughput at stock's per-user speed": for each rung r it takes stock's per-user speed u_r and reads
OUR frontier (per-user speed -> tok/s/GPU over our measured concurrencies) at u_r, interpolating between our points and
capping at our highest concurrency (log-log between points); ratio_r = that / stock_tok_s_r; score = mean of the four ratios. This file:

  python3 harness/score_model.py reproduce out/judge/status_<run>_*.json     # rebuild the score from the rungs (model check)
  python3 harness/score_model.py value      out/judge/status_<run>_*.json     # d(score) per +1% throughput at each measured rung,
                                                                              # and per ms off a prefill chunk / decode step
  python3 harness/score_model.py whatif     out/judge/status_<run>_*.json c32=+8 c16=+3   # score if throughput moves by % per rung
  python3 harness/score_model.py runs_needed 0.02 0.03                        # judge runs per arm to resolve a +3% gain at sd(log)=0.02

Wall split per rung (prefill share) comes from the rung's e2e vs decode-only per-user speed: 1 - e2e/p50.
"""
import glob
import json
import math
import sys

RUNGS = (1, 4, 16, 32)
CHUNK_MS = 620.0        # 16k prefill chunk on the FGHIL stack (NEXT_STEPS 27/37)
DECODE_STEP_MS = {1: 14.0, 4: 16.0, 16: 20.0, 32: 26.0}   # rough per-step wall incl. draft (I_AUDIT c1; others scaled)


def load(path):
    if "*" in path:
        path = sorted(glob.glob(path))[-1]
    d = json.load(open(path))
    return d.get("result", d)


def our_points(res):
    """[(per_user_e2e, tok_s_per_gpu, concurrency)] sorted by per-user speed descending (c1 first)."""
    pts = [(r["tok_s_per_user_e2e"], r["tok_s_per_gpu"], r["concurrency"]) for r in res["rungs"]]
    return sorted(pts, key=lambda p: -p[0])


def frontier_at(pts, u):
    """Our throughput at per-user speed u: linear interpolation along our frontier; capped at the highest-concurrency point
    when u is below its per-user speed (the judge caps at max-running-requests), and at the c1 point above it."""
    if u >= pts[0][0]:
        return pts[0][1]
    for (u1, t1, _), (u2, t2, _) in zip(pts, pts[1:]):
        if u2 <= u < u1:
            # log-log interpolation: reproduces the judge's c1/c4 points within 0.5% (linear is 10% off, Nov 12)
            w = math.log(u1 / u) / math.log(u1 / u2)
            return t1 * (t2 / t1) ** w
    return pts[-1][1]


def stock_points(res):
    """{concurrency: (stock_per_user, stock_tok_s)} from the frontier block of a scored run."""
    fr = res.get("frontier") or {}
    return {p["concurrency"]: (p["tok_s_per_user"], p["stock_tok_s_per_gpu"]) for p in fr.get("points", [])}


def score(res, scale=None):
    """Reproduce the score; scale = {concurrency: factor} multiplies our tok/s at that measured rung (per-user speed too,
    since per-user speed scales with throughput at fixed concurrency)."""
    pts = our_points(res)
    if scale:
        pts = [(u * scale.get(c, 1.0), t * scale.get(c, 1.0), c) for u, t, c in pts]
        pts.sort(key=lambda p: -p[0])
    st = stock_points(res)
    ratios = {}
    for c in RUNGS:
        su, stok = st[c]
        ratios[c] = frontier_at(pts, su) / stok
    return sum(ratios.values()) / len(ratios), ratios


# GPU-level wall split at the saturated rungs (scout_oob_2026-11-12.md §1.2/1.4): the e2e/p50 split counts time a
# user spends queued behind other users' prefill as "decode", which overstates decode-ms value 2-3x at c16/c32. At c1/c4
# the GPU idles between steps, so the e2e split is right there. Decode share of GPU time: c16 ~0.20 (guess), c32 0.17.
GPU_DECODE_SHARE = {16: 0.20, 32: 0.17}


def prefill_share(res, gpu=True):
    out = {r["concurrency"]: 1.0 - r["tok_s_per_user_e2e"] / r["tok_s_per_user_p50"] for r in res["rungs"]}
    if gpu:
        for c, d in GPU_DECODE_SHARE.items():
            if c in out:
                out[c] = 1.0 - d
    return out


def runs_needed(sd_log, gain):
    """Judge runs per arm to see a relative gain at alpha .05, power .8 (two-sided z), same as kar's rule."""
    return max(1, math.ceil(2 * (1.96 + 0.84) ** 2 * sd_log ** 2 / math.log1p(gain) ** 2))


def cmd_reproduce(path):
    res = load(path)
    s, ratios = score(res)
    judge = (res.get("frontier") or {}).get("score")
    print(f"model score {s:.4f}  judge score {judge}  per rung " + " ".join(f"c{c} {r:.4f}" for c, r in ratios.items()))
    fr = {p["concurrency"]: p["ratio"] for p in res["frontier"]["points"]}
    print("judge rungs  " + " ".join(f"c{c} {fr[c]:.4f}" for c in RUNGS))
    err = max(abs(ratios[c] - fr[c]) for c in RUNGS)
    print(f"max per-rung error {err:.4f} -> {'model OK' if err < 0.01 else 'model needs a different interpolation'}")


def cmd_value(path):
    res = load(path)
    base, _ = score(res)
    ps = prefill_share(res)
    print(f"base score {base:.4f}; prefill share of wall (GPU-level at c16/c32): " + " ".join(f"c{c} {ps[c]:.0%}" for c in RUNGS))
    print("\nscore gain per +1% throughput at ONE measured rung (the frontier decides which score rungs it feeds):")
    for c in RUNGS:
        s, _ = score(res, {c: 1.01})
        print(f"  c{c:<3} +1% tok/s -> score {s - base:+.4f} ({(s / base - 1) * 100:+.2f}%)")
    print("\nscore gain for 10 ms off every 16k prefill chunk (applied at every rung via its prefill share):")
    scale = {c: 1.0 / (1.0 - ps[c] * 10.0 / CHUNK_MS) for c in RUNGS}
    s, _ = score(res, scale)
    print(f"  all rungs: score {s - base:+.4f} ({(s / base - 1) * 100:+.2f}%)")
    print("\nscore gain for 1 ms off a decode step (decode share = 1 - prefill share; step ms assumed per rung):")
    scale = {c: 1.0 / (1.0 - (1 - ps[c]) * 1.0 / DECODE_STEP_MS[c]) for c in RUNGS}
    s, _ = score(res, scale)
    print(f"  all rungs: score {s - base:+.4f} ({(s / base - 1) * 100:+.2f}%)")
    for c in RUNGS:
        s, _ = score(res, {c: 1.0 / (1.0 - (1 - ps[c]) * 1.0 / DECODE_STEP_MS[c])})
        print(f"  c{c:<3} only: {(s / base - 1) * 100:+.2f}%")


def cmd_whatif(path, *moves):
    res = load(path)
    base, _ = score(res)
    scale = {}
    for m in moves:
        k, v = m.split("=")
        scale[int(k.lstrip("c"))] = 1.0 + float(v) / 100.0
    s, ratios = score(res, scale)
    print(f"base {base:.4f} -> {s:.4f} ({(s / base - 1) * 100:+.2f}%)  " + " ".join(f"c{c} {r:.3f}" for c, r in ratios.items()))


def cmd_runs_needed(sd, *gains):
    sd = float(sd)
    for g in gains or ("0.01", "0.02", "0.03", "0.05", "0.10"):
        g = float(g)
        print(f"sd(log)={sd:.3f} gain {g:+.0%}: {runs_needed(sd, g)} runs per arm")


if __name__ == "__main__":
    cmds = {"reproduce": cmd_reproduce, "value": cmd_value, "whatif": cmd_whatif, "runs_needed": cmd_runs_needed}
    if len(sys.argv) < 3 or sys.argv[1] not in cmds:
        print(__doc__); sys.exit(1)
    cmds[sys.argv[1]](*sys.argv[2:])
