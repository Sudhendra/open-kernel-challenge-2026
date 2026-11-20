import sys, glob
sys.path.insert(0, sys.argv[1])
import score_model as sm, ceiling_physics as cp
res = sm.load(sorted(glob.glob(sys.argv[2]))[-1])
acc = {r["concurrency"]: r["acceptance"] for r in res["rungs"]}
CHUNK = sum(cp.CHUNK_NOW.values())          # 609
NOW_STEP = {1: 13.3, 4: 19.0, 16: 24.0, 32: 28.9}
KB = {1: 1, 4: 1, 16: 1.5, 32: 2.2}
def wall(n, chunk_ms=CHUNK, wf=127.0, step=None, kb=None, decode_hidden=0.0):
    step = step or NOW_STEP; kb = kb or KB
    cold = (5 * 0.166 + cp.FIRST * 0.028e-3) * chunk_ms / CHUNK
    warm = (wf + 0.028 * cp.NEW * kb[n]) / 1e3 / kb[n]
    d = cp.TURNS * cp.OUT / acc[n] * step[n] / 1e3
    return n * cold + n * 13 * warm + d * (1 - decode_hidden)
base = {n: wall(n) for n in (1, 4, 16, 32)}
s0, _ = sm.score(res)
def price(label, **kw):
    kws = {n: {k: (v[n] if isinstance(v, dict) and k == "decode_hidden" else v) for k, v in kw.items()} for n in (1, 4, 16, 32)}
    sc = {n: base[n] / wall(n, **kws[n]) for n in (1, 4, 16, 32)}
    s, r = sm.score(res, sc)
    print(f"{label:60s} {s:.3f} ({(s/s0-1)*100:+5.1f}%)  " + " ".join(f"c{k} x{v:.3f}" for k, v in sc.items()))
print(f"base model score {s0:.4f}")
price("1 ms off every decode step", step={k: v - 1 for k, v in NOW_STEP.items()})
price("10 ms off every 16k chunk", chunk_ms=CHUNK - 10)
price("10 ms off every warm forward fixed cost", wf=117)
price("R2 -55 ms/chunk (real counts, minus ~15 exposure -> -40)", chunk_ms=CHUNK - 40)
price("R2 best -55", chunk_ms=CHUNK - 55)
price("Expert-TP MoE: prefill -74 wait +15 kernel; decode -1.5/-2.0/-0.3/0", chunk_ms=CHUNK - 59,
      step={1: 11.8, 4: 17.0, 16: 23.7, 32: 28.9})
price("Expert-TP MoE hard (wait 0, no kernel penalty; decode straggler bytes 1.66/1.80)", chunk_ms=CHUNK - 74,
      step={1: 11.6, 4: 17.2, 16: 23.7, 32: 28.9})
price("G2a placement -0.35/-0.55 decode", step={1: 12.95, 4: 18.45, 16: 24, 32: 28.9})
price("L1 overlap -40 (range -85)", chunk_ms=CHUNK - 40)
price("L1 overlap best -85", chunk_ms=CHUNK - 85)
price("Collectives fully hidden (-128 + 25 residual = -103)", chunk_ms=CHUNK - 103)
price("P15 dispatch/combine (volume -35% of MoE AG/RS, unhidden) -20", chunk_ms=CHUNK - 20)
price("Prefill tail rewrite 75 -> 35 per forward (chunk -40, warm fixed -40)", chunk_ms=CHUNK - 40, wf=87)
price("Warm-forward fixed 127 -> 40 ms", wf=40)
price("Warm-forward fixed 127 -> 60 ms", wf=60)
price("Decode c1/c4 package (D1+G4+G5+G6+H1/H2): 13.3->10.5, 19->16", step={1: 10.5, 4: 16.0, 16: 23, 32: 28})
price("Decode practical floor all rungs", step={1: 7.1, 4: 14.1, 16: 21.5, 32: 24.5})
price("Decode hidden under prefill at c16/c32 50%", decode_hidden={1: 0, 4: 0, 16: 0.5, 32: 0.5})
price("Decode hidden under prefill c4/c16/c32 50%", decode_hidden={1: 0, 4: 0.5, 16: 0.5, 32: 0.5})
price("mfs12 alone (c32 kb 2.2->7)", kb={1: 1, 4: 1, 16: 1.5, 32: 7})
price("cold to practical 365", chunk_ms=365)
price("cold to hard 200", chunk_ms=200)
# a 3.38 stack composition
price("STACK A: R2+L1(-40)+tail(-40)+warm 60+decode pkg+mfs12", chunk_ms=CHUNK - 40 - 40 - 40, wf=60,
      step={1: 10.5, 4: 16.0, 16: 23, 32: 28}, kb={1: 1, 4: 1, 16: 1.5, 32: 7})
price("STACK B: A + expertTP instead of R2 + decode-hidden 30% c16/32", chunk_ms=CHUNK - 59 - 40 - 40, wf=60,
      step={1: 9.0, 4: 14.0, 16: 22.7, 32: 28}, kb={1: 1, 4: 1, 16: 1.5, 32: 7}, decode_hidden={1: 0, 4: 0, 16: 0.3, 32: 0.3})
