#!/usr/bin/env python3
"""Lever pricing v2 (Nov 13 evening) on the v2 rung-wall model (harness/ceiling_physics_v2.py).

python3 -I harness/ceiling_levers_v2.py "$PWD/out/judge/status_186157d12aeb4eba_*.json"
V2_WF_NOW=80 python3 -I harness/ceiling_levers_v2.py ...      # sensitivity: warm fixed cost 80 ms instead of E1's 127

Each lever is a parameter change: chunk ms, warm fixed ms, step ms per rung, kb per rung. Base = the scored stack
(FGHIL + SR) in the v2 model (c1 13.8, c4 19.79, c16 24.0, c32 28.9 ms steps; chunk 609; warm 127 + 0.028/token).
Then a cumulative road, in the recommended order, from the scored stack to the practical ceiling.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ceiling_physics as cp  # noqa: E402
import ceiling_physics_v2 as v2  # noqa: E402
import score_model as sm  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "out/judge/status_186157d12aeb4eba_*.json"
res = sm.load(sorted(glob.glob(path))[-1] if "*" in path else path)
acc = {r["concurrency"]: r["acceptance"] for r in res["rungs"]}
CHUNK = 609.0
STEP = {1: 13.8, 4: 19.79, 16: 24.0, 32: 28.9}
KB = {1: 1, 4: 1, 16: 1.5, 32: 2.2}
WF = float(os.environ.get("V2_WF_NOW", 127.0))
COLD_NOW = 5 * 0.166 + cp.FIRST * 0.028e-3
R = (1, 4, 16, 32)


def wall(n, chunk=CHUNK, wf=WF, step=None, kb=None, tok_ms=0.028, hidden=0.0):
    step = step or STEP
    kb = kb or KB
    cold = COLD_NOW * chunk / CHUNK
    warm = (wf + tok_ms * cp.NEW * kb[n]) / 1e3 / kb[n]
    return n * cold + n * 13 * warm + cp.TURNS * cp.OUT / acc[n] * step[n] / 1e3 * (1 - hidden)


BASE = {n: wall(n) for n in R}
S0, _ = sm.score(res)


def price(label, show=True, **kw):
    sc = {n: BASE[n] / wall(n, **kw) for n in R}
    s, r = sm.score(res, sc)
    if show:
        print(f"{label:72s} {s:.3f} ({(s / S0 - 1) * 100:+5.1f}%)  " + " ".join(f"c{k} x{v:.3f}" for k, v in sc.items()))
    return s


def dstep(**d):
    return {n: STEP[n] + d.get(f"c{n}", 0.0) for n in R}


print(f"base model score {S0:.4f}; WF {WF}")
print("-- unit values --")
price("1 ms off every decode step", step=dstep(c1=-1, c4=-1, c16=-1, c32=-1))
price("1 ms off the c1 and c4 steps only", step=dstep(c1=-1, c4=-1))
price("10 ms off every 16k chunk", chunk=CHUNK - 10)
price("10 ms off every warm forward", wf=WF - 10)
print("-- built / measured --")
price("D1 (c1 -0.7 [P], c4 -1.48 [M])", step=dstep(c1=-0.7, c4=-1.48))
price("G5 (-0.31/-0.33 [M ubench], -0.35 c16/c32 [E])", step=dstep(c1=-0.31, c4=-0.33, c16=-0.35, c32=-0.35))
price("G4b built (-0.42/-0.52 mid)", step=dstep(c1=-0.42, c4=-0.52))
price("G6 next (-0.55/-0.62 mid)", step=dstep(c1=-0.55, c4=-0.62))
price("G7 (-0.12 every rung)", step=dstep(c1=-0.12, c4=-0.12, c16=-0.12, c32=-0.12))
price("mfs12 (c32 kb 2.2 -> 7.3)", kb={1: 1, 4: 1, 16: 1.5, 32: 7.3})
price("W(a) on the stack, v2 estimate (-6.5 per warm forward)", wf=WF - 6.5)
price("W(a) lane bound (-16, stock-based)", wf=WF - 16)
price("R2 (-40 net/chunk; my replay gross -43)", chunk=CHUNK - 40)
price("L1 overlap (-40; good case -85)", chunk=CHUNK - 40)
price("T1 (-15/chunk)", chunk=CHUNK - 15)
print("-- new levers (v2) --")
price("P static routing-aware placement: chunk -58 (out-of-sample -61.5)", chunk=CHUNK - 58)
price("P + 8 static hot replicas: chunk -58, decode straggler -0.46/-0.52/-0.3/-0.3",
      chunk=CHUNK - 58, step=dstep(c1=-0.46, c4=-0.52, c16=-0.3, c32=-0.3))
price("K small-M expert kernel to 85%: decode -0.75/-0.9 mid, warm -4 (range c1 -0.4..-1.1)",
      step=dstep(c1=-0.75, c4=-0.9, c16=-0.8, c32=-0.8), wf=WF - 4)
price("S small-M dense GEMM at 70% HBM: -1.3/-1.5; D1 to M=256: c16/c32 -1.5",
      step=dstep(c1=-1.3, c4=-1.5, c16=-1.5, c32=-1.5))
price("C combine-in-epilogue / glue beyond G4b-G7: -0.8 every rung", step=dstep(c1=-0.8, c4=-0.8, c16=-0.8, c32=-0.8))
price("N warm non-forward path 84 -> 15 ms (unmeasured; E1 minus measured GPU)", wf=WF - 69)
price("N half-way (-35)", wf=WF - 35)
price("tail 75 -> 35 (cold chunk -40)", chunk=CHUNK - 40)
price("decode hidden under prefill 30% at c16/c32 (not this week)", hidden=0.0)
for h in (0.3,):
    sc = {n: BASE[n] / wall(n, hidden=(h if n >= 16 else 0)) for n in R}
    s, _ = sm.score(res, sc)
    print(f"{'  (decode hidden 30% at c16/c32)':72s} {s:.3f} ({(s / S0 - 1) * 100:+5.1f}%)")

print("-- cumulative road (recommended order) --")
road = [
    ("scored stack (FGHIL + SR)", dict()),
    ("+ D1 + G5 (measured)", dict(step=dstep(c1=-1.01, c4=-1.81, c16=-0.35, c32=-0.35))),
    ("+ mfs12 (judge tonight)", dict(kb={1: 1, 4: 1, 16: 1.5, 32: 7.3})),
    ("+ G4b + G6 + G7 (built / next)", dict(dstep_add={"c1": -1.09, "c4": -1.26, "c16": -0.12, "c32": -0.12})),
    ("+ P placement + 8 hot replicas", dict(chunk_add=-58, dstep_add={"c1": -0.46, "c4": -0.52, "c16": -0.3, "c32": -0.3})),
    ("+ L1 overlap (-40)", dict(chunk_add=-40)),
    ("+ tail (T1 + C1 + bf16 layers, -40) + W(a) -6.5", dict(chunk_add=-40, wf_add=-6.5)),
    ("+ N warm non-forward -69 (if the measurement shows it)", dict(wf_add=-69)),
    ("+ S dense small-M + D1 to M=256", dict(dstep_add={"c1": -1.3, "c4": -1.5, "c16": -1.5, "c32": -1.5})),
    ("+ K expert kernel 85% (decode + warm)", dict(dstep_add={"c1": -0.75, "c4": -0.9, "c16": -0.8, "c32": -0.8}, wf_add=-4)),
    ("+ C combine/glue -0.8", dict(dstep_add={"c1": -0.8, "c4": -0.8, "c16": -0.8, "c32": -0.8})),
    ("+ L1 good case (-45 more)", dict(chunk_add=-45)),
]
cur = dict(chunk=CHUNK, wf=WF, step=dict(STEP), kb=dict(KB))
for label, ch in road:
    if "step" in ch:
        cur["step"] = ch["step"]
    if "kb" in ch:
        cur["kb"] = ch["kb"]
    for k, v in ch.get("dstep_add", {}).items():
        cur["step"][int(k[1:])] += v
    cur["chunk"] += ch.get("chunk_add", 0)
    cur["wf"] += ch.get("wf_add", 0)
    s = price(label, show=False, chunk=cur["chunk"], wf=cur["wf"], step=dict(cur["step"]), kb=dict(cur["kb"]))
    print(f"{label:58s} {s:.3f} ({(s / S0 - 1) * 100:+5.1f}%)  chunk {cur['chunk']:.0f} wf {cur['wf']:.0f} steps "
          + "/".join(f"{cur['step'][n]:.1f}" for n in R) + f" kb32 {cur['kb'][32]}")
