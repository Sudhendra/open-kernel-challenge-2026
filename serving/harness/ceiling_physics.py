#!/usr/bin/env python3
"""Physics ceiling for the serving score (roofline + latency floors), Nov 13.

python3 -I harness/ceiling_physics.py [out/judge/status_186157d12aeb4eba_*.json]

Prints: model dims, decode-step floors per rung, prefill chunk / cold prompt / warm forward floors, the rung-wall model
(calibrated to the judge's own walls), and ceiling scores (hard, practical) through score_model.score. Report:
research/ceiling_physics_2026-11-13.md. Every constant below is labelled with its source.
"""
import glob
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_model  # noqa: E402

# ---------------- hardware (H200 SXM spec; NCCL bus bandwidth measured) ----------------
HBM = 4.8e12            # B/s peak per GPU [spec]
HBM_PRACT = 4.0e12      # B/s realistic for streaming GEMV-like kernels (E2 ~85%, dense nvjet ~80%) [E from deep dive G1]
BF16 = 989e12           # dense FLOP/s [spec]
FP8 = 1979e12           # dense FLOP/s [spec]
NVL = 450e9             # B/s per direction per GPU [spec]
NCCL_BUS = 361.8e9      # measured busbw, 64 MiB all-reduce (out/s5/.../nccl.jsonl via L1_FALLBACK_FEASIBILITY §2) [M]
TP = 8

# ---------------- model (out/20261107T055538Z/config_Muse-Spark-Open-816B-A42B-NVFP4.json) ----------------
H = 8192; NQ = 128; NKV = 16; HD = 64; L = 62; L_FULL = 16; L_SWA = 46; WIN = 2048
E = 256; TOPK = 8; MOE_H = 4096; MOE_I = 4096; VOCAB = 202048
L_BF16_MOE = 2                      # layers 0 and 61 excluded from NVFP4 (quantization_config.exclude_modules)
L_FP4_MOE = L - L_BF16_MOE
E_LOCAL = E // TP
EXPERT_PARAMS = MOE_H * 2 * MOE_I + MOE_I * MOE_H          # gate_up + down = 50,331,648
FP4_BYTES = EXPERT_PARAMS * (0.5 + 1 / 16)                 # e2m1 + fp8 scale per 16 (group_size 16) = 28.31 MB
BF16_EXPERT_BYTES = EXPERT_PARAMS * 2                      # 100.66 MB
QKVG = H * (NQ * HD + 2 * NKV * HD + NQ * HD)              # QKVOGateParallelLinear: q + k + v + gate(q-sized)
O = NQ * HD * H
PRE = H * MOE_H; POST = MOE_H * H
ROUTER_BYTES = H * E * 4                                   # fp32 router (router_in_fp32)
# drafter (config_Muse-Spark-Open-DFlash2.json): qwen3, 5 sliding layers, hidden 8192, 64 q x 128, 8 kv, MLP 28672
D_L = 5; D_I = 28672
D_LAYER = H * (64 * 128 + 2 * 8 * 128) + 64 * 128 * H + 3 * H * D_I
D_FC = 5 * H * H                                           # projection of 5 target hidden states (target_layer_ids)
KV_TOK_LAYER_GPU = 2 * (NKV // TP) * HD * 1                # fp8 K+V, 2 kv heads per rank = 256 B

# ---------------- workload (status JSON shape + E1) ----------------
FIRST = 74160; NEW = 753; OUT = 220; TURNS = 14             # shape: first_turn, new_per_turn, output_mean; 13 later + 1
CTX_MEAN = FIRST + NEW * 6.5 + OUT * 6.5                    # ~80.5k mean context over turns
TOKENS_PER_CONV = sum(FIRST + k * (NEW + OUT) for k in range(TURNS))   # what tok_s counts (incl. cache hits)


def distinct_per_rank(m):
    """Expected distinct local experts touched by m routed rows on one rank, uniform iid top-8 over 256."""
    return E_LOCAL * (1 - (1 - TOPK / E) ** m)


def max_rank_distinct(m, trials=4000, seed=1):
    """E[max over 8 ranks of distinct local experts] for m rows of uniform top-8 routing (Monte Carlo)."""
    rnd = random.Random(seed)
    acc = 0.0
    for _ in range(trials):
        seen = [set() for _ in range(TP)]
        for _ in range(m):
            for e in rnd.sample(range(E), TOPK):
                seen[e // E_LOCAL].add(e)
        acc += max(len(s) for s in seen)
    return acc / trials


def dense_bytes_per_layer(sharded=True):
    attn = (QKVG + O) / TP * 2
    proj = (PRE + POST) * 2 / (TP if sharded else 1)
    router = ROUTER_BYTES / (TP if sharded else 1)
    return attn + proj + router


def decode_bytes(b, distinct=None, sharded=True, ctx=CTX_MEAN):
    m = 8 * b
    d = distinct if distinct is not None else distinct_per_rank(m)
    experts = L_FP4_MOE * d * FP4_BYTES
    experts_bf16 = L_BF16_MOE * d * BF16_EXPERT_BYTES
    dense = L * dense_bytes_per_layer(sharded)
    lm_head = 2 * VOCAB * H * 2 / TP                       # verify + draft share it
    drafter = (D_L * D_LAYER + D_FC) * 2 / TP
    kv = b * (L_FULL * ctx + L_SWA * WIN) * KV_TOK_LAYER_GPU
    return dict(experts=experts, experts_bf16=experts_bf16, dense=dense, lm_head=lm_head, drafter=drafter, kv=kv)


def ar_floor_s(b, hard=True):
    """134 collectives per step (62 x 2 target + 10 draft); one-shot push of 8b rows x 8192 bf16 to 7 peers."""
    msg = 8 * b * H * 2
    lat = 1.0e-6 if hard else 2.5e-6
    eff = 1 if hard else 0.8
    tot = 0.0
    for width, calls in ((H, 62 + 10), (MOE_H, 62)):           # attn o_proj + draft ARs at 8192, MoE AR at 4096
        msg = 8 * b * width * 2
        if msg <= 512 * 1024:
            t = lat + 7 * msg / NVL / eff                         # one-shot: read 7 peers' buffers
        else:
            t = 2 * lat + 2 * (7 / 8) * msg / NVL / eff           # two-shot: RS + AG
        tot += calls * t
    return tot


def decode_step(b, mode, distinct=None):
    by = decode_bytes(b, distinct=distinct, sharded=True)
    tot = sum(by.values())
    if mode == "hard":
        return tot / HBM + ar_floor_s(b, True), by
    # practical: realistic HBM efficiency, measured AR transport (c1 0.52 ms, c4 0.89 ms -> table), FA3 small-kernel latency,
    # graph bubbles at ~600 nodes x 0.3 us, residual straggler 0.3 ms (placement/expert-TP leaves some), fused norms 0.4 ms.
    ar = {1: 0.52e-3, 4: 0.89e-3}.get(b, ar_floor_s(b, False))
    fa3 = 0.6e-3 + 0.02e-3 * b
    return tot / HBM_PRACT + ar + fa3 + 0.18e-3 + 0.3e-3 + 0.4e-3, by


def prefill_flops_per_token_linear():
    per_layer = 2 * (TOPK * EXPERT_PARAMS + PRE + POST + H * E + QKVG + O)
    return L * per_layer


def attn_flops_prompt(n):
    """Causal attention FLOPs over a prompt of n tokens (QK^T + PV, 4 * heads * hd per (q, k) pair)."""
    per_pair = 4 * NQ * HD
    full = L_FULL * per_pair * n * n / 2
    swa = L_SWA * per_pair * (WIN * (n - WIN) + WIN * WIN / 2) if n > WIN else L_SWA * per_pair * n * n / 2
    return full, swa


def cold_prompt_hard(n=FIRST):
    lin = prefill_flops_per_token_linear() * n / TP / BF16
    full, swa = attn_flops_prompt(n)
    attn = (full + swa) / TP / FP8
    # collectives under SP (lane H): AG attn in + RS o_proj (8192 wide) + AG expert in + RS MoE (4096 wide), 7/8 received
    comm = L * n * (2 * H * 2 + 2 * MOE_H * 2) * (7 / 8) / NVL      # per GPU received/sent, hideable under compute
    wbytes = (L_FP4_MOE * E_LOCAL * FP4_BYTES + L_BF16_MOE * E_LOCAL * BF16_EXPERT_BYTES) * math.ceil(n / 16384) / HBM
    return dict(linear=lin, attn=attn, comm=comm, weights=wbytes, total=max(lin + attn, comm, wbytes))


def warm_forward_hard(new=NEW, ctx=CTX_MEAN, k=1):
    lin = prefill_flops_per_token_linear() * new * k / TP / BF16
    attn = L_FULL * 4 * NQ * HD * ctx * new * k / TP / FP8 + L_SWA * 4 * NQ * HD * WIN * new * k / TP / FP8
    wbytes = (L_FP4_MOE * E_LOCAL * FP4_BYTES + L_BF16_MOE * E_LOCAL * BF16_EXPERT_BYTES + L * dense_bytes_per_layer()) / HBM
    kvread = k * (L_FULL * ctx + L_SWA * WIN) * KV_TOK_LAYER_GPU / HBM
    comm = L * new * k * (2 * H * 2 + 2 * MOE_H * 2) * (7 / 8) / NVL
    lat = 124 * 1.0e-6
    return dict(linear=lin, attn=attn, weights=wbytes, kv=kvread, comm=comm, lat=lat,
                total=max(lin + attn, wbytes + kvread, comm) + lat)


# Practical 16k chunk (component table from research/TOP5_LEVERS_2026-11-10.md §1.2 [E], targets per component [E])
CHUNK_NOW = dict(experts=124, ep_wait=74, collectives=128, dense=92, fa3=71, tail=75, norms=40, gaps=5)    # ~609 (+ host ~10)
CHUNK_PRACT = dict(experts=124, ep_wait=10, collectives=25, dense=75, fa3=71, tail=35, norms=20, gaps=5)


def rung_wall(n_users, cold, warm_fn, step_fn, accept, kb=1, parts=False):
    """Serial model (calibrated on the judge, see report §2.1): prefill forwards of all users serialize on the node,
    decode steps run with every decoding user in one batch. wall = N*(cold + 13*warm_fn(kb)/kb) + (3080/accept)*step(N)."""
    steps = TURNS * OUT / accept
    c = n_users * cold
    w = n_users * (TURNS - 1) * warm_fn(kb) / kb
    d = steps * step_fn(n_users)
    return (c, w, d) if parts else c + w + d


def main(path):
    res = score_model.load(path)
    rungs = {r["concurrency"]: r for r in res["rungs"]}
    st = score_model.stock_points(res)
    print(f"expert NVFP4 {FP4_BYTES/1e6:.2f} MB, bf16 expert {BF16_EXPERT_BYTES/1e6:.1f} MB, QKVG {QKVG/1e6:.1f}M, O {O/1e6:.1f}M,"
          f" drafter/GPU {(D_L*D_LAYER+D_FC)*2/TP/1e9:.2f} GB, tokens/conv {TOKENS_PER_CONV/1e6:.3f}M, ctx mean {CTX_MEAN:.0f}")
    print(f"linear prefill FLOP/token {prefill_flops_per_token_linear()/1e9:.1f} G; dense bytes/layer/GPU sharded "
          f"{dense_bytes_per_layer(True)/1e6:.1f} MB, replicated {dense_bytes_per_layer(False)/1e6:.1f} MB")

    print("\n== decode step floors (per GPU) ==")
    for b in (1, 4, 16, 32):
        m = 8 * b
        dmean = distinct_per_rank(m)
        dmax = max_rank_distinct(m, trials=600 if b >= 16 else 3000)
        th, by = decode_step(b, "hard")
        tp_, _ = decode_step(b, "practical")
        tot = sum(by.values())
        print(f"b={b:<2} M={m:<3} distinct mean {dmean:5.2f} max-rank {dmax:5.2f}  bytes {tot/1e9:5.1f} GB "
              + " ".join(f"{k} {v/1e9:.2f}" for k, v in by.items())
              + f"  hard {th*1e3:5.2f} ms (AR {ar_floor_s(b)*1e3:.2f})  practical {tp_*1e3:5.2f} ms"
              + f"  straggler bytes {(dmax-dmean)*(L_FP4_MOE*FP4_BYTES+L_BF16_MOE*BF16_EXPERT_BYTES)/HBM_PRACT*1e3:.2f} ms")

    print("\n== prefill ==")
    c = cold_prompt_hard()
    print("cold 74,160 prompt hard: " + " ".join(f"{k} {v*1e3:.0f} ms" for k, v in c.items()))
    per_chunk_hard = c["total"] * 16384 / FIRST
    print(f"  per 16k chunk-equivalent hard {per_chunk_hard*1e3:.0f} ms; now {sum(CHUNK_NOW.values())} ms; practical {sum(CHUNK_PRACT.values())} ms")
    for k in (1, 2, 7):
        w = warm_forward_hard(k=k)
        print(f"warm forward x{k}: " + " ".join(f"{a} {v*1e3:.1f}" for a, v in w.items()) + " ms")

    # ---- rung walls ----
    print("\n== rung walls (s per conversation) ==")
    meas = {c_: TURNS * OUT / rungs[c_]["tok_s_per_user_e2e"] for c_ in rungs}
    stock = {c_: TURNS * OUT / st[c_][0] for c_ in st}
    acc = {c_: rungs[c_]["acceptance"] for c_ in rungs}
    # our stack, calibrated params: E1 main fit (cold 5 forwards x cf 166 + 0.028 ms/token; lone warm 147 ms; step 12.9 + 0.516 (b-1) ms)
    our_cold = 5 * 0.166 + FIRST * 0.028e-3
    # now-steps: c1 from the judge decomposition (14.56 s - cold 2.91 - warm 1.91 over 733 steps) [M-derived];
    # c4 = deep-dive stack estimate 18.4-20.1 ms pure step [E]; c16 = stack bytes (replicated pre/post) / 4.0 TB/s + AR/FA3 [E];
    # c32 = E1 main fit b=32 [E fit]
    NOW_STEP = {1: 13.3e-3, 4: 19.0e-3, 16: 24.0e-3, 32: 28.9e-3}
    our_step = lambda n: NOW_STEP[n]
    our_kb = {1: 1, 4: 1, 16: 1.5, 32: 2.2}                     # E1: 2.2 extends per warm forward at c32 (§3 table)
    our_warm = lambda k: 0.127 + 0.028e-3 * NEW * k              # E1 main fit: wf 127 ms + cp per token
    model_now = {n: rung_wall(n, our_cold, our_warm, our_step, acc[n], our_kb[n]) for n in (1, 4, 16, 32)}

    hard_cold = c["total"]
    pr_cold = sum(CHUNK_PRACT.values()) * 1e-3 * FIRST / 16384
    hard_step = lambda n: decode_step(n, "hard")[0]
    pr_step = lambda n: max(decode_step(n, "practical")[0], 6e-3 if n == 1 else 0)   # host floor 6 ms at c1 [E]
    kb_hard = {1: 1, 4: 1, 16: 4, 32: 7}
    kb_pr = {1: 1, 4: 1, 16: 2, 32: 7}                           # c32: mfs12 batches ~7 (E1); c16: ~2
    hard_warm = lambda k: warm_forward_hard(k=k)["total"]
    pr_warm = lambda k: 0.025 + k * warm_forward_hard(k=1)["linear"] / 0.8 + k * warm_forward_hard(k=1)["attn"] / 0.3
    model_hard = {n: rung_wall(n, hard_cold, hard_warm, hard_step, acc[n], kb_hard[n]) for n in (1, 4, 16, 32)}
    model_pr = {n: rung_wall(n, pr_cold, pr_warm, pr_step, acc[n], kb_pr[n]) for n in (1, 4, 16, 32)}
    parts = {}
    for nm, cold_, wf, sf, kb in (("now", our_cold, our_warm, our_step, our_kb), ("practical", pr_cold, pr_warm, pr_step, kb_pr),
                                  ("hard", hard_cold, hard_warm, hard_step, kb_hard)):
        for n in (1, 4, 16, 32):
            parts[(nm, n)] = rung_wall(n, cold_, wf, sf, acc[n], kb[n], parts=True)
    print(f"cold s: now {our_cold:.2f} practical {pr_cold:.2f} hard {hard_cold:.2f}; warm per extend at k=1: now {our_warm(1)*1e3:.0f} "
          f"practical {pr_warm(1)*1e3:.0f} hard {hard_warm(1)*1e3:.1f} ms")
    for n in (1, 4, 16, 32):
        print(f"c{n:<2} parts cold/warm/decode s: " + "  ".join(f"{nm} " + "/".join(f"{x:.1f}" for x in parts[(nm, n)])
                                                            for nm in ("now", "practical", "hard"))
              + f"   step ms now {our_step(n)*1e3:.1f} pr {pr_step(n)*1e3:.1f} hard {hard_step(n)*1e3:.1f}")
    for n in (1, 4, 16, 32):
        print(f"c{n:<2} judge ours {meas[n]:6.1f}  model-now {model_now[n]:6.1f} ({model_now[n]/meas[n]-1:+.0%})  "
              f"stock {stock[n]:6.1f}  practical {model_pr[n]:6.1f}  hard {model_hard[n]:6.1f}")

    print("\n== attribution: replace ONE bucket now -> practical (score) ==")
    for i, nm in enumerate(("cold", "warm", "decode")):
        sc = {}
        for n in (1, 4, 16, 32):
            p_now = list(parts[("now", n)]); p_pr = parts[("practical", n)]
            p_mix = list(p_now); p_mix[i] = p_pr[i]
            sc[n] = sum(p_now) / sum(p_mix)
        s_, r_ = score_model.score(res, sc)
        print(f"  {nm:7s} score {s_:.3f}  " + " ".join(f"c{k} x{v:.2f}" for k, v in sc.items()))
    print("\n== ceiling scores (scale our measured rung throughput by model_now / model_ceiling) ==")
    base, br = score_model.score(res)
    print(f"base {base:.4f} " + " ".join(f"c{k} {v:.3f}" for k, v in br.items()))
    for name, mdl in (("practical", model_pr), ("hard", model_hard)):
        scale = {n: model_now[n] / mdl[n] for n in mdl}
        s, r = score_model.score(res, scale)
        print(f"{name:9s} score {s:.3f}  " + " ".join(f"c{k} {v:.2f}" for k, v in r.items())
              + "   scale " + " ".join(f"c{k} x{v:.2f}" for k, v in scale.items()))
    # what 3.38 needs
    for label, sc in (("uniform", None), ("c1/c4 flat", "flat14"), ("c32 flat", "flat32")):
        lo, hi = 1.0, 4.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if sc is None:
                scale = {n: mid for n in (1, 4, 16, 32)}
            elif sc == "flat14":
                scale = {1: 1, 4: 1, 16: mid, 32: mid}
            else:
                scale = {1: mid, 4: mid, 16: mid, 32: 1}
            s, r = score_model.score(res, scale)
            lo, hi = (mid, hi) if s < 3.38 else (lo, mid)
        print(f"3.38x {label:10s}: x{mid:.3f} -> " + " ".join(f"c{k} {v:.2f}" for k, v in r.items()))


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "out/judge/status_186157d12aeb4eba_*.json")
    main(sorted(glob.glob(p))[-1] if "*" in p else p)
