#!/usr/bin/env python3
"""Physics ceiling v2 (Nov 13 evening): floors re-derived from the measured STACK, not stock traces.

python3 -I harness/ceiling_physics_v2.py [out/judge/status_186157d12aeb4eba_*.json]

Report: research/ceiling_physics_v2_2026-11-13.md. v1 (constants, rung-wall model): harness/ceiling_physics.py.
New inputs (all cited in the report):
  - decode buckets, stack ctl/d1 at c1/c4: research/decode_breakdown_stack_{ctl,d1}_2026-11-13.txt   [M]
  - warm EXTEND forwards on the stack (571 tok ctl c1, 1022 tok d1 c1, least-skewed rank 7):
      out/modal_prof/vol/out/20261113T204844Z/variants/{ctl,d1}/prof/c1-*-TP-7-*                     [M]
  - distinct experts per rank per step from REAL per-layer popularity (Nov 11 counts):
      harness/ceiling_routing_v2.py (MC numbers copied below)                                         [E from M]
  - prefill straggler under placement policies: harness/ceiling_routing_v2.py                          [E from M]
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ceiling_physics as cp  # noqa: E402
import score_model  # noqa: E402

HBM, HBM_PR = cp.HBM, cp.HBM_PRACT          # 4.8 / 4.0 TB/s
EFF_EXPERT_PR = 0.85                        # practical streaming efficiency for the expert kernel [E]
FP4 = cp.FP4_BYTES; BF16E = cp.BF16_EXPERT_BYTES

# ---- distinct experts per rank per layer per step, real popularity (ceiling_routing_v2.py, 300-trial MC) [E from M]
# mean, E[max] contiguous EP8, E[max] static placement, E[max] static + 8 hot replicas (M<=32 only)
DISTINCT = {1: (5.40, 8.46, 8.05, 7.19), 4: (11.73, 15.65, 14.68, 14.06),
            16: (19.33, 23.16, 22.01, 22.01), 32: (21.83, 25.51, 24.32, 24.32)}
DISTINCT_UNIFORM = {1: 7.18, 4: 20.34, 16: 31.45, 32: 32.0}

# ---- measured decode step buckets on the stack, ms per step (rank median) [M]
# c1: d1 (ctl c1 window caught no decode); c4: d1 bs=4 and ctl bs=4.
MEAS = {
    "c1_d1": dict(period=13.41, experts=3.36, straggler=1.70, ar_transport=0.64, dense=2.62, fa3=1.16,
                  glue=1.31 + 1.14 + 0.37 + 0.29 + 0.42 + 0.11 + 0.08, bf16=0.35, draft=0.95,
                  bubbles_side_gaps=0.38 + 0.16 + 0.39, host_busy=8.45),
    "c4_d1": dict(period=18.31, experts=6.28, straggler=2.27, ar_transport=0.92, dense=2.43 + 0.40, fa3=1.59,
                  glue=1.27 + 1.19 + 0.51 + 0.38 + 0.30 + 0.09 + 0.08, bf16=0.72, draft=1.00,
                  bubbles_side_gaps=0.40 + 0.40 + 0.61, host_busy=8.58),
    "c4_ctl": dict(period=19.79, experts=6.50, straggler=2.41, ar_transport=0.76, dense=4.17 + 0.68,
                   fa3=1.65, glue=1.18 + 1.22 + 0.04 + 0.33 + 0.18 + 0.22 + 0.10, bf16=0.73, draft=0.99,
                   bubbles_side_gaps=0.38 + 0.68 + 0.59, host_busy=8.65),
}

# ---- measured warm EXTEND forwards on the stack (GPU span, least-skewed rank) [M]
WARM_MEAS = {
    571: dict(span=54.5, experts=15.16, bf16=1.52, fa3=8.95 + 0.34, ar=7.39, dense=4.16 + 3.34 + 2.29 + 1.22 + 0.29,
              norms=2.28 + 0.23 + 0.15, copies=0.70 + 0.69, glue_other=0.46 + 0.43 + 0.25 + 0.31 + 0.5),
    1022: dict(span=75.7, experts=18.42, bf16=1.55, fa3=12.74, ar=11.98, dense=12.85 + 6.99 + 3.07,
               norms=1.86 + 1.68, copies=1.12 + 1.08, glue_other=0.58 + 0.56 + 1.0),
}


def warm_gpu_now(new):
    """Linear in new tokens through the two measured stack forwards (different prefixes: ~70k, proxy corpus) [E from M]."""
    a, b = WARM_MEAS[571]["span"], WARM_MEAS[1022]["span"]
    slope = (b - a) / (1022 - 571)
    return a + slope * (new - 571), slope, a - slope * 571


def expert_floor_ms(n_rung, distinct, eff=1.0):
    return cp.L_FP4_MOE * distinct * FP4 / (HBM * eff) * 1e3


def decode_floors(n):
    """Return (hard, practical, parts) for the decode step at concurrency n (b = n, M = 8n). ms."""
    d_mean, d_max, d_static, d_rep = DISTINCT[n]
    by = cp.decode_bytes(n, distinct=d_mean, sharded=True)
    kv_ms = by["kv"] / HBM * 1e3
    exp_h = expert_floor_ms(n, d_mean)
    exp_p = expert_floor_ms(n, d_mean, EFF_EXPERT_PR)
    strag_ratio = d_rep / d_mean - 1                           # residual under placement + 8 hot replicas (2 GB cap)
    bf16_h = by["experts_bf16"] / HBM * 1e3
    dense_h = by["dense"] / HBM * 1e3                         # D1-style sharded pre/post (D1 gate M<=32 today)
    lm_draft_h = (by["lm_head"] + by["drafter"]) / HBM * 1e3
    ar_h = cp.ar_floor_s(n, True) * 1e3
    hard = dict(experts=exp_h, straggler=strag_ratio * exp_h, bf16=bf16_h, dense=dense_h, kv_attn=kv_ms,
                ar=ar_h, draft_lm=lm_draft_h, glue=0.0, bubbles=0.0)
    # practical: expert kernel at 85% of HBM on the real distinct count; residual straggler under placement + replicas;
    # dense at 70% (small-M bf16 GEMV class); FA3 kept at the measured / scaled kernel time (outside the patch) minus G7;
    # AR at the measured per-call floors (c1 0.64, c4 0.76) else v1's practical AR; glue fused to ~1/3 of today
    # (G4b + G5 + G6 + G7 + combine-in-epilogue); draft 0.6; bubbles 0.5.
    fa3 = {1: 1.16 - 0.12, 4: 1.59 - 0.12, 16: 2.6, 32: 4.0}[n]
    ar_p = {1: 0.64, 4: 0.76}.get(n, cp.ar_floor_s(n, False) * 1e3)
    glue_p = {1: 1.2, 4: 1.4, 16: 1.6, 32: 1.8}[n]
    pr = dict(experts=exp_p, straggler=strag_ratio * exp_p, bf16=bf16_h / EFF_EXPERT_PR, dense=dense_h / 0.70,
              kv_attn=fa3, ar=ar_p, draft_lm=0.6 + (0.0 if n <= 4 else 0.1), glue=glue_p, bubbles=0.5)
    return sum(hard.values()), sum(pr.values()), hard, pr


def chunk_tables():
    """16k cold chunk, ms. now = TOP5 §1.2 component table [E from M], total 609-625 [M]; straggler 73.7 [M, R2 §8]."""
    now = dict(experts=124, ep_wait=74, collectives=128, dense=92, fa3=71, tail=75, norms=40, gaps=5)
    # practical v2: straggler by static placement (+R2) = residual 11.5 out-of-sample (ceiling_routing_v2) -> 12;
    # collectives 80% hidden (L1 class) -> 25; dense at 95% of its 74 ms FLOP time; FA3 as is (outside the patch);
    # tail 35 (T1 + C1 + bf16-layer tuning); norms 20; experts at 85% of the bf16 roofline (103 ms) = 121.
    pract = dict(experts=121, ep_wait=12, collectives=25, dense=78, fa3=71, tail=35, norms=20, gaps=5)
    hard = dict(experts=103, ep_wait=0, collectives=0, dense=74, fa3=23, tail=0, norms=0, gaps=0)
    return now, pract, hard


def warm_floor(new=cp.NEW, kb=1):
    """GPU warm forward at 753 new tokens on ~80k context: hard (v1 function), practical [E]."""
    h = cp.warm_forward_hard(new=new, k=kb)
    h1 = cp.warm_forward_hard(new=new, k=1)
    lin = h1["linear"] * 1e3                                   # one extend's linear FLOP time
    # practical: experts streaming 11.3+1.3 ms at 85% -> 14.9 ms fixed per forward; linear rest at 80% of peak; FA3 at
    # the measured stack efficiency (571-tok forward: 9.3 ms on a ~70k prefix, ~18% of fp8 peak) scaled by tokens;
    # collectives under W(a) mostly hidden: 2 ms; norms/glue 2 ms.
    experts_fixed = (cp.L_FP4_MOE * cp.E_LOCAL * FP4 + cp.L_BF16_MOE * cp.E_LOCAL * BF16E) / (HBM * EFF_EXPERT_PR) * 1e3
    fa3_meas_per_tok = (WARM_MEAS[571]["fa3"]) / 571 * (cp.CTX_MEAN / 70000)
    pr = experts_fixed + kb * (lin * 0.45 / 0.8) + kb * new * fa3_meas_per_tok + 2.0 + 2.0
    return h["total"] * 1e3, pr


def main(path):
    res = score_model.load(path)
    rungs = {r["concurrency"]: r for r in res["rungs"]}
    acc = {c: rungs[c]["acceptance"] for c in rungs}
    st_pts = score_model.stock_points(res)

    print("== decode step floors v2 (ms per step; M = 8b) ==")
    floors = {}
    for n in (1, 4, 16, 32):
        h, p, hp, pp = decode_floors(n)
        floors[n] = (h, p)
        print(f"c{n:<2} hard {h:5.2f}  practical {p:5.2f}   hard parts " + " ".join(f"{k} {v:.2f}" for k, v in hp.items()))
        print(f"{'':4s}{'':28s}pract parts " + " ".join(f"{k} {v:.2f}" for k, v in pp.items()))
        d = DISTINCT[n]
        print(f"{'':4s}distinct mean {d[0]:.2f} (uniform {DISTINCT_UNIFORM[n]:.2f}); max EP8 {d[1]:.2f}, static {d[2]:.2f}, +8 replicas {d[3]:.2f};"
              f" expert floor at peak {expert_floor_ms(n, d[0]):.2f} ms (uniform {expert_floor_ms(n, DISTINCT_UNIFORM[n]):.2f})")
    print("\n== measured stack decode buckets vs v2 practical (ms) ==")
    for key, n in (("c1_d1", 1), ("c4_d1", 4), ("c4_ctl", 4)):
        m = MEAS[key]
        ks = [k for k in m if k not in ("period", "host_busy")]
        tot = sum(m[k] for k in ks)
        print(f"{key}: period {m['period']} = buckets {tot:.2f} - overlap {tot - m['period']:.2f}; host busy {m['host_busy']}; "
              f"expert kernel efficiency vs real-pop floor {expert_floor_ms(n, DISTINCT[n][0]) / m['experts']:.0%} "
              f"(vs uniform floor {expert_floor_ms(n, DISTINCT_UNIFORM[n]) / m['experts']:.0%}); "
              f"straggler / experts {m['straggler'] / m['experts']:.2f}")

    print("\n== warm forward (GPU, ms) ==")
    for t, w in WARM_MEAS.items():
        print(f"measured {t} tok: span {w['span']}  " + " ".join(f"{k} {v:.1f}" for k, v in w.items() if k != "span"))
    g753, slope, icpt = warm_gpu_now(cp.NEW)
    print(f"interpolated 753 tok: {g753:.1f} ms (= {icpt:.1f} + {slope*1e3:.1f} us/token)")
    for t in (571, 1022):
        rows_per_expert = t * 8 / 256
        floor_us = 32 * FP4 / HBM * 1e6
        print(f"  expert kernel at {t} tok: {WARM_MEAS[t]['experts']/60*1e3:.0f} us/layer, {rows_per_expert:.1f} rows/expert, "
              f"streaming floor {floor_us:.0f} us -> {floor_us/(WARM_MEAS[t]['experts']/60*1e3):.0%} of HBM roofline")
    for kb in (1, 2, 7):
        h, p = warm_floor(kb=kb)
        print(f"warm floor kb={kb}: hard {h:.1f}  practical GPU {p:.1f}")

    now_c, pr_c, hd_c = chunk_tables()
    print(f"\n== cold 16k chunk: now {sum(now_c.values())}  practical {sum(pr_c.values())}  hard {sum(hd_c.values())} "
          f"(v1 hard FLOP floor per chunk-equivalent {cp.cold_prompt_hard()['total']*16384/cp.FIRST*1e3:.0f}) ==")

    # ---------------- rung-wall model (v1 form) ----------------
    NOW_STEP = {1: 13.8, 4: 19.79, 16: 24.0, 32: 28.9}       # c1 judge decode p50 (4.2/304); c4 ctl bs=4 [M]; c16 [E]; c32 E1
    KB_NOW = {1: 1, 4: 1, 16: 1.5, 32: 2.2}
    cold_now = 5 * 0.166 + cp.FIRST * 0.028e-3                 # E1 main fit (2.91 s)
    WF_NOW = float(os.environ.get("V2_WF_NOW", 127.0))          # E1 per-forward fixed ms (sensitivity: 80)
    warm_now = lambda k: (WF_NOW + 0.028 * cp.NEW * k) / 1e3

    def wall(n, cold, warm_fn, step_ms, kb, hidden=0.0):
        steps = cp.TURNS * cp.OUT / acc[n]
        return n * cold + n * (cp.TURNS - 1) * warm_fn(kb) / kb + steps * step_ms / 1e3 * (1 - hidden)

    pr_chunk = sum(pr_c.values())
    cold_pr = pr_chunk * 1e-3 * cp.FIRST / 16384
    cold_hd = cp.cold_prompt_hard()["total"]
    NONFWD_PR = float(os.environ.get("V2_NONFWD_PR", 15.0))      # practical non-forward per warm forward [E]
    warm_pr = lambda k: (NONFWD_PR + warm_floor(kb=k)[1]) / 1e3
    warm_hd = lambda k: warm_floor(kb=k)[0] / 1e3
    KB_PR = {1: 1, 4: 1, 16: 2, 32: 7.3}
    KB_HD = {1: 1, 4: 1, 16: 4, 32: 7.3}
    meas = {c: cp.TURNS * cp.OUT / rungs[c]["tok_s_per_user_e2e"] for c in rungs}
    now = {n: wall(n, cold_now, warm_now, NOW_STEP[n], KB_NOW[n]) for n in (1, 4, 16, 32)}
    pr = {n: wall(n, cold_pr, warm_pr, floors[n][1], KB_PR[n]) for n in (1, 4, 16, 32)}
    hd = {n: wall(n, cold_hd, warm_hd, floors[n][0], KB_HD[n]) for n in (1, 4, 16, 32)}
    print(f"\n== rung walls (s per conversation); WF_NOW {WF_NOW}, practical non-forward {NONFWD_PR} ms ==")
    print(f"cold s now {cold_now:.2f} pract {cold_pr:.2f} hard {cold_hd:.2f}; warm per forward kb1 now {warm_now(1)*1e3:.0f} "
          f"pract {warm_pr(1)*1e3:.0f} hard {warm_hd(1)*1e3:.1f} ms; kb7 pract per extend {warm_pr(7)/7*1e3:.0f}")
    for n in (1, 4, 16, 32):
        parts = lambda cold, wf, st_, kb: (n * cold, n * 13 * wf(kb) / kb, cp.TURNS * cp.OUT / acc[n] * st_ / 1e3)
        pn = parts(cold_now, warm_now, NOW_STEP[n], KB_NOW[n])
        pp = parts(cold_pr, warm_pr, floors[n][1], KB_PR[n])
        ph = parts(cold_hd, warm_hd, floors[n][0], KB_HD[n])
        f = lambda t: "/".join(f"{x:.1f}" for x in t) + f"={sum(t):.1f}"
        print(f"c{n:<2} judge {meas[n]:6.1f}  model-now {f(pn)} ({now[n]/meas[n]-1:+.0%})  practical {f(pp)}  hard {f(ph)}")

    base, br = score_model.score(res)
    print(f"\n== scores (scale measured rung throughput by model_now / model_ceiling) ==\nbase {base:.4f} " +
          " ".join(f"c{k} {v:.3f}" for k, v in br.items()))
    out = {}
    for name, mdl in (("practical", pr), ("hard", hd)):
        scale = {n: now[n] / mdl[n] for n in mdl}
        s, r = score_model.score(res, scale)
        out[name] = (s, r, scale)
        print(f"{name:9s} {s:.3f}  " + " ".join(f"c{k} {v:.2f}" for k, v in r.items()) + "   scale " +
              " ".join(f"c{k} x{v:.2f}" for k, v in scale.items()))

    # current measured stack: FGHIL + SR + D1 + G5 (delayer pending): steps c1 13.8-0.7-0.31, c4 19.79-1.48-0.33,
    # c16/c32 G5 only (-0.35); D1 is gated off above M = 32.
    cur_step = {1: 13.8 - 0.7 - 0.31, 4: 19.79 - 1.48 - 0.33, 16: 24.0 - 0.35, 32: 28.9 - 0.35}
    for label, kb in (("FGHIL+SR+D1+G5", KB_NOW), ("FGHIL+SR+D1+G5+mfs12", {1: 1, 4: 1, 16: 1.5, 32: 7.3})):
        cur = {n: wall(n, cold_now, warm_now, cur_step[n], kb[n]) for n in (1, 4, 16, 32)}
        sc = {n: now[n] / cur[n] for n in cur}
        s, r = score_model.score(res, sc)
        sp, rp, _ = out["practical"]
        print(f"current {label:22s} {s:.3f} = {s/sp:.0%} of practical; per rung " +
              " ".join(f"c{k} {r[k]:.2f}/{rp[k]:.2f}={r[k]/rp[k]:.0%}" for k in r) + "  scale " +
              " ".join(f"c{k} x{v:.3f}" for k, v in sc.items()))

    print("\n== attribution: one bucket now -> practical (score), from the scored stack ==")
    for i, nm in enumerate(("cold", "warm", "decode")):
        sc = {}
        for n in (1, 4, 16, 32):
            pn = [n * cold_now, n * 13 * warm_now(KB_NOW[n]) / KB_NOW[n], cp.TURNS * cp.OUT / acc[n] * NOW_STEP[n] / 1e3]
            pp = [n * cold_pr, n * 13 * warm_pr(KB_PR[n]) / KB_PR[n], cp.TURNS * cp.OUT / acc[n] * floors[n][1] / 1e3]
            mix = list(pn); mix[i] = pp[i]
            sc[n] = sum(pn) / sum(mix)
        s_, _ = score_model.score(res, sc)
        print(f"  {nm:7s} {s_:.3f} ({s_/base-1:+.1%})  " + " ".join(f"c{k} x{v:.2f}" for k, v in sc.items()))


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "out/judge/status_186157d12aeb4eba_*.json")
    main(sorted(glob.glob(p))[-1] if "*" in p else p)
