#!/usr/bin/env python3
"""EP straggler floors on REAL prefill routing counts (ceiling physics v2, Nov 13).

python3 -I harness/ceiling_routing_v2.py [out/s6/route_log_20261111/route_counts_rank0.pt]

Input: the Nov 11 recorder (lane rec-route-log): per (chunk, MoE layer) the global pair count of each of 256 experts
(rank 0 sees every token's top-k before dispatch, so these are global loads). 172 full chunks x 62 layers.

For each placement policy it prints the straggler excess of the grouped expert kernel, as a fraction of the mean rank
load (sum over layers of (max_r - mean_r) / mean_r, averaged per layer) and in ms per 16k chunk at the stack's measured
expert time (CHUNK_EXPERT_MS = 124 ms per chunk for all 60 NVFP4 layers + 2 bf16, research/TOP5_LEVERS_2026-11-10.md
section 1.2). Policies (all lossless: same experts, same kernel, only which rank computes which expert, or a replica of it):
  ep8       : contiguous EP8 (expert e on rank e // 32), today
  static    : one placement per layer, fitted on the first half of the chunks (mean loads, LPT bin-packing with exactly
              32 experts per rank), scored on the second half. Zero runtime cost; a load-time permutation of the experts.
  static_all: the same, fitted on all chunks (in-sample, optimistic)
  oracle    : per-chunk optimal-ish placement (LPT on this chunk's own loads): the floor of any re-placement policy
  r2        : today's built R2 (top-8 hot experts by the previous chunk, one replica, tokens split evenly) on contiguous EP8
  static+r2 : static placement plus R2 on top
  oracle_r2 : per-chunk LPT plus R2 with this chunk's loads (the floor of placement + replication)
  split     : perfect token-level balance (every rank gets exactly 1/8 of the pairs): excess 0 by construction (EP balance
              by splitting the hottest experts' tokens across ranks = expert-TP for hot experts only; costs replicas)
"""
import statistics as st
import sys

import torch

CHUNK_EXPERT_MS = 124.0
R = 8
E = 256
EL = E // R


def load(path):
    log = torch.load(path)
    chunks, cur, last = [], [], -1
    for layer_idx, n, counts in log:
        if layer_idx <= last and cur:
            chunks.append(cur)
            cur = []
        cur.append((layer_idx, int(n), [int(x) for x in counts.tolist()]))
        last = layer_idx
    if cur:
        chunks.append(cur)
    nl = max(len(c) for c in chunks)
    return [c for c in chunks if len(c) == nl], nl


def lpt(loads):
    """Longest-processing-time bin packing with exactly EL experts per rank. Returns rank of each expert."""
    order = sorted(range(E), key=lambda e: -loads[e])
    tot = [0.0] * R
    cnt = [0] * R
    owner = [0] * E
    for e in order:
        cands = [r for r in range(R) if cnt[r] < EL]
        r = min(cands, key=lambda r: tot[r])
        owner[e] = r
        tot[r] += loads[e]
        cnt[r] += 1
    return owner


def rank_loads(row, owner, rep=None):
    """rep: {expert: second_rank} -> tokens of that expert split evenly between its owner and the replica rank."""
    t = [0.0] * R
    for e, c in enumerate(row):
        if rep and e in rep:
            t[owner[e]] += c / 2
            t[rep[e]] += c / 2
        else:
            t[owner[e]] += c
    return t


def r2_choice(prev_row, owner, k=8):
    """Greedy as lane R2: hottest k experts (by prev loads) each get one replica on the currently least-loaded rank."""
    t = rank_loads(prev_row, owner)
    rep = {}
    for e in sorted(range(E), key=lambda e: -prev_row[e])[:k]:
        r = min((r for r in range(R) if r != owner[e]), key=lambda r: t[r])
        rep[e] = r
        t[owner[e]] -= prev_row[e] / 2
        t[r] += prev_row[e] / 2
    return rep


def excess(t):
    m = st.mean(t)
    return (max(t) - m) / m if m else 0.0


def main(path):
    chunks, nl = load(path)
    n = len(chunks)
    half = n // 2
    contiguous = [e // EL for e in range(E)]
    # static placements per layer
    def fit(rows):
        mean = [st.mean(r[e] for r in rows) for e in range(E)]
        return lpt(mean)
    static_half = [fit([c[l][2] for c in chunks[:half]]) for l in range(nl)]
    static_all = [fit([c[l][2] for c in chunks]) for l in range(nl)]
    pols = {k: [] for k in ("ep8", "static", "static_all", "oracle", "r2", "static+r2", "oracle_r2")}
    cv_rank = []
    max_expert_share = []
    test = range(half, n)
    for f in test:
        if f == 0:
            continue
        acc = {k: 0.0 for k in pols}
        for l in range(nl):
            row = chunks[f][l][2]
            prev = chunks[f - 1][l][2]
            t = rank_loads(row, contiguous)
            acc["ep8"] += excess(t)
            cv_rank.append(st.pstdev(t) / st.mean(t))
            max_expert_share.append(max(row) / (sum(row) / R))
            acc["static"] += excess(rank_loads(row, static_half[l]))
            acc["static_all"] += excess(rank_loads(row, static_all[l]))
            o = lpt(row)
            acc["oracle"] += excess(rank_loads(row, o))
            acc["r2"] += excess(rank_loads(row, contiguous, r2_choice(prev, contiguous)))
            acc["static+r2"] += excess(rank_loads(row, static_half[l], r2_choice(prev, static_half[l])))
            acc["oracle_r2"] += excess(rank_loads(row, o, r2_choice(row, o)))
        for k in pols:
            pols[k].append(acc[k] / nl)
    print(f"chunks {n} x {nl} MoE layers; scored on the second half ({len(pols['ep8'])} chunks); placement fitted on the first {half}")
    print(f"cross-rank CV of rank load (EP8) {st.mean(cv_rank):.3f}; hottest expert load / mean rank load: median "
          f"{st.median(max_expert_share):.2f}, p90 {sorted(max_expert_share)[int(0.9*len(max_expert_share))]:.2f}")
    base = st.mean(pols["ep8"])
    print(f"{'policy':11s} {'excess (max/mean-1)':>20s} {'ms per 16k chunk':>17s} {'removed vs ep8':>15s}")
    for k, v in pols.items():
        x = st.mean(v)
        print(f"{k:11s} {x:20.3f} {x*CHUNK_EXPERT_MS:17.1f} {(base-x)*CHUNK_EXPERT_MS:15.1f}")
    print("split      " + f"{0:20.3f} {0:17.1f} {base*CHUNK_EXPERT_MS:15.1f}  (perfect pair balance; needs every hot expert's tokens split)")
    decode_mc(chunks, nl, static_half)


def decode_mc(chunks, nl, static_half, trials=300, seed=7):
    """Decode: distinct experts per rank per step (bytes streamed), iid tokens drawn from each layer's REAL popularity
    (mean over all chunks), top-8 without replacement. Rows of one request are treated as independent (real rows of a
    request share experts, so mean distinct is an upper bound; the straggler (max - mean) is what placement changes)."""
    import random
    rnd = random.Random(seed)
    pop = []
    for l in range(nl):
        m = [st.mean(c[l][2][e] for c in chunks) + 1e-9 for e in range(E)]
        pop.append(m)
    contiguous = [e // EL for e in range(E)]
    print("\ndecode, distinct experts per rank per layer (real popularity, iid rows): mean / E[max over ranks]")
    for M in (8, 32, 128, 256):
        res = {"ep8": [0, 0], "static": [0, 0], "uniform": [0, 0]}
        for _ in range(trials):
            l = rnd.randrange(nl)
            w = pop[l]
            seen = set()
            for _ in range(M):
                cand = set()
                while len(cand) < 8:
                    cand.add(rnd.choices(range(E), weights=w)[0])
                seen |= cand
            useen = set()
            for _ in range(M):
                useen |= set(rnd.sample(range(E), 8))
            for name, owner, ss in (("ep8", contiguous, seen), ("static", static_half[l], seen), ("uniform", contiguous, useen)):
                per = [0] * R
                for e in ss:
                    per[owner[e]] += 1
                res[name][0] += st.mean(per)
                res[name][1] += max(per)
        print(f"  M={M:<3d} " + "  ".join(f"{k} {v[0]/trials:5.2f} / {v[1]/trials:5.2f}" for k, v in res.items()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out/s6/route_log_20261111/route_counts_rank0.pt")


def decode_replica_mc(path="out/s6/route_log_20261111/route_counts_rank0.pt", trials=300, seed=11):
    """Decode straggler with the k hottest experts of each layer also resident on a second rank (static placement first),
    each step assigning every replicated expert to whichever of its two ranks has fewer distinct experts so far (greedy,
    hottest first). Memory: k x 28.31 MB x 60 layers / 8 ranks per rank (k=8: 1.70 GB per rank, at the 2 GB cap)."""
    import random
    chunks, nl = load(path)
    half = len(chunks) // 2
    rnd = random.Random(seed)
    pops, owners = [], []
    for l in range(nl):
        pops.append([st.mean(c[l][2][e] for c in chunks) + 1e-9 for e in range(E)])
        owners.append(lpt([st.mean(c[l][2][e] for c in chunks[:half]) for e in range(E)]))
    print("\ndecode straggler with hot replicas (static placement + k replicas per layer), distinct experts mean / E[max]:")
    for M in (8, 32):
        for k in (0, 8, 16):
            mean_acc = max_acc = 0.0
            for _ in range(trials):
                l = rnd.randrange(nl)
                w, owner = pops[l], owners[l]
                hot = sorted(range(E), key=lambda e: -w[e])[:k]
                # replica rank: least-loaded rank (by mean load) other than the owner
                load_r = [0.0] * R
                for e in range(E):
                    load_r[owner[e]] += w[e]
                rep = {}
                for e in hot:
                    rep[e] = min((r for r in range(R) if r != owner[e]), key=lambda r: load_r[r])
                seen = set()
                for _ in range(M):
                    cand = set()
                    while len(cand) < 8:
                        cand.add(rnd.choices(range(E), weights=w)[0])
                    seen |= cand
                per = [0] * R
                for e in seen:
                    if e not in rep:
                        per[owner[e]] += 1
                for e in sorted((e for e in seen if e in rep), key=lambda e: -w[e]):
                    a, b = owner[e], rep[e]
                    per[a if per[a] <= per[b] else b] += 1
                mean_acc += st.mean(per)
                max_acc += max(per)
            print(f"  M={M:<3d} k={k:<2d} mean {mean_acc/trials:5.2f}  E[max] {max_acc/trials:5.2f}  max/mean {max_acc/mean_acc:4.2f}")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "replica":
    decode_replica_mc(sys.argv[1])
