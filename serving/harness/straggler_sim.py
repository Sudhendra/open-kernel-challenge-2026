#!/usr/bin/env python3
"""Decode MoE straggler simulator (CPU only, numpy).

Takes routed-expert ids for decode verify steps and, for a placement, computes per step and layer the
distinct experts and routed rows per rank, a predicted per-rank expert-path time from a linear model, and the
straggler wait sum_l (max_r T - mean_r T). Compares:
  base   trivial placement (expert e on rank e // 32)
  online the shipped online-placement objective (prefill marginals, alpha blend of load and hit probability,
         sticky LPT; port of muse_spark_online_placement_plan.py) fitted on PREFILL-like counts
  odec   same objective fitted on decode marginals of the training window (decode counts instead of prefill)
  coact  max-rank-load objective: pairwise-swap local search minimising mean_s max_r T on the training window
  speed  coact with per-rank speed factors in T (rank-speed weighting)
  oracle coact fitted on the evaluation window itself (best static placement this search finds; a bound)
  repK   coact placement + K hot-expert replicas per GPU per layer, rows of a replicated expert steered per step
         to the copy whose rank is currently less loaded (greedy, deterministic from global top-k ids)

Routes format (real capture): .npy int array [steps, layers, rows, topk] of LOGICAL expert ids, rows = the
verify rows of that step (bs x 8). Padding rows = -1. Optional --prefill-counts .npy [layers, 256] for the
'online' fit; without it 'online' uses a prefill-like sample from the synthetic generator.

Time model (us per layer per rank): T = a + b*D + c*P, D distinct local experts touched, P local (row, expert)
pairs. Defaults are calibrated by `calibrate` against the trace means (see research/moe_straggler_2026-11-18.md).

Usage:
  python3 -I harness/straggler_sim.py synth  [--rows 8] [--sigma-req 1.0] [--sigma-blk 0.5] ...
  python3 -I harness/straggler_sim.py routes --routes r.npy [--prefill-counts c.npy]
  python3 -I harness/straggler_sim.py grid   (sensitivity grid used in the report)
"""
import argparse, json, sys
import numpy as np

E, R, K = 256, 8, 8
PER = E // R
# Trace calibration targets: expert path (route+gather+persistent+combine) mean us per layer [measured]
TRACE_MEAN_US = {8: 73.5, 32: 121.6, 256: 203.8}
# Trace per-rank speed factors at 32 users (rank effect / mean) [measured; includes placement load, an upper bound]
TRACE_SPEED_C32 = np.array([-0.028, -0.042, -0.179, 0.255, -0.055, -0.058, 0.181, -0.074]) / 12.229 + 1.0


# ----------------------------------------------------------------------------- synthetic routes
class Synth:
    """Correlated routing: per layer popularity p ~ Dirichlet(alpha); per request persistent topic weights
    exp(sigma_req*N); per (request, step) block weights exp(sigma_blk*N) shared by the block's rows; each row takes
    the top-8 of log p + log topic + log block + Gumbel (sampling without replacement). Requests are replaced with
    probability `turnover` per step. Prefill popularity differs from decode by exp(sigma_pd*N)."""

    def __init__(self, layers, alpha, sigma_req, sigma_blk, sigma_pd, turnover, seed):
        self.rng = np.random.default_rng(seed)
        self.L, self.sr, self.sb, self.turn = layers, sigma_req, sigma_blk, turnover
        self.logp = np.log(self.rng.dirichlet(np.full(E, alpha), size=layers) + 1e-12)
        self.logp_pre = self.logp + sigma_pd * self.rng.standard_normal((layers, E))

    def _topic(self):
        return self.sr * self.rng.standard_normal((self.L, E))

    def routes(self, steps, users, block=8):
        topics = [self._topic() for _ in range(users)]
        out = np.empty((steps, self.L, users * block, K), dtype=np.int16)
        for s in range(steps):
            for u in range(users):
                if self.rng.random() < self.turn:
                    topics[u] = self._topic()
                blk = self.sb * self.rng.standard_normal((self.L, E))
                base = self.logp + topics[u] + blk                     # [L, E]
                g = self.rng.gumbel(size=(block, self.L, E))
                idx = np.argpartition(-(base[None] + g), K, axis=2)[:, :, :K]  # [block, L, K]
                out[s, :, u * block:(u + 1) * block, :] = idx.transpose(1, 0, 2)
        return out

    def prefill_counts(self, rows=60000):
        g = self.rng.gumbel(size=(rows // 64, self.L, E))   # subsample, scaled
        idx = np.argpartition(-(self.logp_pre[None] + g), K, axis=2)[:, :, :K]
        c = np.zeros((self.L, E))
        for l in range(self.L):
            c[l] = np.bincount(idx[:, l].ravel(), minlength=E)
        return c


def counts_per_step(routes):
    """routes [S, L, rows, K] -> M [L, S, E] row counts per expert per step."""
    S, L = routes.shape[:2]
    M = np.zeros((L, S, E), dtype=np.int32)
    for l in range(L):
        r = routes[:, l].reshape(S, -1).astype(np.int64)
        valid = r >= 0
        off = (np.arange(S)[:, None] * E + np.where(valid, r, 0))
        M[l] = np.bincount(off[valid], minlength=S * E).reshape(S, E)
    return M


# ----------------------------------------------------------------------------- time model
class Model:
    def __init__(self, a, b, c, speed=None):
        self.a, self.b, self.c = a, b, c
        self.speed = np.ones(R) if speed is None else np.asarray(speed, float)

    def rank_time(self, M, place):
        """M [S, E] counts, place [E] -> T [S, R] us."""
        oh = np.zeros((E, R)); oh[np.arange(E), place] = 1
        D = (M > 0).astype(float) @ oh
        P = M.astype(float) @ oh
        return (self.a + self.b * D + self.c * P) * self.speed

    def rank_time_rep(self, M, place, reps):
        """reps: dict expert -> replica rank. Rows of a replicated expert go to the less loaded copy (greedy)."""
        rep_e = np.array(sorted(reps), dtype=int)
        keep = M.copy(); keep[:, rep_e] = 0
        T = self.rank_time(keep, place)
        out = T.copy()
        for s in range(M.shape[0]):
            t = out[s]
            touched = [e for e in rep_e if M[s, e] > 0]
            touched.sort(key=lambda e: -M[s, e])
            for e in touched:
                cost = (self.b + self.c * M[s, e])
                r1, r2 = place[e], reps[e]
                r = r1 if t[r1] + cost * self.speed[r1] <= t[r2] + cost * self.speed[r2] else r2
                t[r] += cost * self.speed[r]
        return out


def straggler(T):
    """T [S, R] -> mean over steps of max - mean (us)."""
    return float(np.mean(T.max(1) - T.mean(1)))


# ----------------------------------------------------------------------------- placements
def trivial():
    return np.arange(E) // PER


def online_objective(counts, alpha=0.5, top_k=K):
    """Port of muse_spark_online_placement_plan.py weight + sticky LPT from the trivial layout."""
    pairs = counts.sum()
    load = counts / pairs
    pi = np.clip(counts * top_k / pairs, 0, 1)
    hit = np.zeros(E)
    for rows in (8, 16, 32, 64, 128, 256):
        h = 1.0 - (1.0 - pi) ** rows
        hit += h / h.sum()
    w = alpha * load + (1 - alpha) * hit / hit.sum()
    cur = trivial()
    order = np.argsort(-w, kind='stable')
    tot = np.zeros(R); cnt = np.zeros(R, int); new = np.empty(E, int)
    for e in order:
        cand = [r for r in range(R) if cnt[r] < PER]
        light = min(cand, key=lambda r: tot[r])
        r = cur[e] if (cnt[cur[e]] < PER and tot[cur[e]] - tot[light] <= 0.02 * w.sum() / R) else light
        new[e] = r; tot[r] += w[e]; cnt[r] += 1

    def excess(p):
        t = np.bincount(p, weights=w, minlength=R); return (t.max() - t.mean()) / t.mean()
    return new if excess(cur) - excess(new) >= 0.02 else cur


def local_search(M, model, start, iters=4000, seed=0):
    """Pairwise swaps minimising mean_s max_r T on counts M [S, E]."""
    rng = np.random.default_rng(seed)
    place = start.copy()
    Mb = (M > 0).astype(float); Mp = M.astype(float)
    oh = np.zeros((E, R)); oh[np.arange(E), place] = 1
    D = Mb @ oh; P = Mp @ oh
    T = (model.a + model.b * D + model.c * P) * model.speed
    best = np.mean(T.max(1))
    for _ in range(iters):
        e1, e2 = rng.integers(E, size=2)
        r1, r2 = place[e1], place[e2]
        if r1 == r2:
            continue
        dD = Mb[:, e2] - Mb[:, e1]; dP = Mp[:, e2] - Mp[:, e1]
        T2 = T.copy()
        T2[:, r1] += (model.b * dD + model.c * dP) * model.speed[r1]
        T2[:, r2] -= (model.b * dD + model.c * dP) * model.speed[r2]
        v = np.mean(T2.max(1))
        if v < best - 1e-9:
            best, T = v, T2
            place[e1], place[e2] = r2, r1
    return place


def choose_replicas(M, place, model, k):
    """k replica slots per rank: hottest experts by training load, each copied to the rank with the lowest
    mean training time among ranks with free replica slots (not its primary)."""
    load = M.sum(0)
    T = model.rank_time(M, place).mean(0)
    free = np.full(R, k); reps = {}
    for e in np.argsort(-load):
        if free.sum() == 0:
            break
        cand = [r for r in range(R) if free[r] > 0 and r != place[e]]
        if not cand:
            continue
        r = min(cand, key=lambda r: T[r])
        reps[int(e)] = r; free[r] -= 1
        share = (model.b + model.c * load[e] / M.shape[0]) / 2
        T[r] += share; T[place[e]] -= share
    return reps


# ----------------------------------------------------------------------------- evaluation
def evaluate(routes, model, prefill_counts=None, rep_ks=(4, 8, 16), iters=4000, train_frac=0.5, seed=0):
    M_all = counts_per_step(routes)
    L, S = M_all.shape[:2]
    ntr = max(1, int(S * train_frac))
    res = {k: [] for k in ['base', 'online', 'odec', 'coact', 'speed', 'oracle'] + [f'rep{k}' for k in rep_ks]}
    stats = {'distinct_per_step_mean': [], 'pairs_per_rank': [], 'rand_sd_us': [], 'sys_sd_us': []}
    plain = Model(model.a, model.b, model.c)  # placements other than 'speed' are fitted speed-blind
    for l in range(L):
        Mtr, Mev = M_all[l, :ntr], M_all[l, ntr:]
        stats['distinct_per_step_mean'].append(float((Mev > 0).sum(1).mean()))
        base = trivial()
        res['base'].append(straggler(model.rank_time(Mev, base)))
        pc = prefill_counts[l] if prefill_counts is not None else Mtr.sum(0)
        on = online_objective(pc)
        Ton = model.rank_time(Mev, on)
        res['online'].append(straggler(Ton))
        dev = Ton - Ton.mean(1, keepdims=True)
        sysc = dev.mean(0)
        stats['sys_sd_us'].append(float(sysc.std())); stats['rand_sd_us'].append(float((dev - sysc).std()))
        res['odec'].append(straggler(model.rank_time(Mev, online_objective(Mtr.sum(0)))))
        co = local_search(Mtr, plain, on, iters, seed + l)
        res['coact'].append(straggler(model.rank_time(Mev, co)))
        sp = local_search(Mtr, model, on, iters, seed + l)
        res['speed'].append(straggler(model.rank_time(Mev, sp)))
        orc = local_search(Mev, model, on, iters * 2, seed + l)
        res['oracle'].append(straggler(model.rank_time(Mev, orc)))
        for k in rep_ks:
            reps = choose_replicas(Mtr, co, plain, k)
            res[f'rep{k}'].append(straggler(model.rank_time_rep(Mev, co, reps)))
    out = {k: float(np.sum(v)) / 1000 * 60 / L for k, v in res.items()}   # ms per step, scaled to 60 layers
    out['_distinct_global_per_step'] = float(np.mean(stats['distinct_per_step_mean']))
    out['_sys_sd_us_online'] = float(np.mean(stats['sys_sd_us']))
    out['_rand_sd_us_online'] = float(np.mean(stats['rand_sd_us']))
    return out


def calibrate(gen_kwargs, block=8, b_us=None):
    """Fit a, b, c of T = a + b*D + c*P to the trace means at 1/4/32 users (8/32/256 rows) using the synthetic
    model's E[D], E[P] per rank under the trivial placement. 3 equations, 3 unknowns; if b_us is given, fit a, c
    by least squares with b fixed."""
    X, y = [], []
    for users, rows in ((1, 8), (4, 32), (32, 256)):
        g = Synth(seed=11, **gen_kwargs)
        M = counts_per_step(g.routes(16, users, block))
        oh = np.zeros((E, R)); oh[np.arange(E), trivial()] = 1
        D = np.mean([((M[l] > 0).astype(float) @ oh).mean() for l in range(M.shape[0])])
        P = rows * K / R
        X.append([1.0, D, P]); y.append(TRACE_MEAN_US[rows])
    X, y = np.array(X), np.array(y)
    if b_us is None:
        a, b, c = np.linalg.solve(X, y)
    else:
        (a, c), *_ = np.linalg.lstsq(X[:, [0, 2]], y - b_us * X[:, 1], rcond=None); b = b_us
    return dict(a=float(a), b=float(b), c=float(c), ED_per_rank=[float(v) for v in X[:, 1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['synth', 'routes', 'grid'])
    ap.add_argument('--routes'); ap.add_argument('--prefill-counts')
    ap.add_argument('--users', type=int, default=1); ap.add_argument('--block', type=int, default=8)
    ap.add_argument('--steps', type=int, default=160); ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--sigma-req', type=float, default=1.0); ap.add_argument('--sigma-blk', type=float, default=0.5)
    ap.add_argument('--sigma-pd', type=float, default=0.5); ap.add_argument('--turnover', type=float, default=0.0)
    ap.add_argument('--a', type=float); ap.add_argument('--b', type=float); ap.add_argument('--c', type=float)
    ap.add_argument('--b-fixed', type=float, default=None,
                    help='fix b (us per distinct expert); 6.0 = 28.3 MB NVFP4 expert at 4.7 TB/s')
    ap.add_argument('--speed', default='none', choices=['none', 'trace'])
    ap.add_argument('--iters', type=int, default=3000)
    args = ap.parse_args()
    gk = dict(layers=args.layers, alpha=args.alpha, sigma_req=args.sigma_req, sigma_blk=args.sigma_blk,
              sigma_pd=args.sigma_pd, turnover=args.turnover)
    if args.a is None:
        cal = calibrate(gk, args.block, args.b_fixed)
    else:
        cal = dict(a=args.a, b=args.b, c=args.c)
    speed = TRACE_SPEED_C32 if args.speed == 'trace' else None
    model = Model(cal['a'], cal['b'], cal['c'], speed)
    if args.mode == 'routes':
        routes = np.load(args.routes)
        pc = np.load(args.prefill_counts) if args.prefill_counts else None
        print(json.dumps(dict(model=cal, result=evaluate(routes, model, pc, iters=args.iters)), indent=1))
        return
    if args.mode == 'synth':
        g = Synth(seed=1, **gk)
        routes = g.routes(args.steps, args.users, args.block)
        print(json.dumps(dict(model=cal, gen=gk, users=args.users,
                              result=evaluate(routes, model, g.prefill_counts(), iters=args.iters)), indent=1))
        return
    # grid
    rows = []
    for sr, sb in ((0.0, 0.0), (0.5, 0.3), (1.0, 0.5), (1.5, 1.0)):
        for alpha in (0.3, 1.0):
            gk2 = dict(gk, sigma_req=sr, sigma_blk=sb, alpha=alpha)
            cal = calibrate(gk2, args.block, args.b_fixed)
            m = Model(cal['a'], cal['b'], cal['c'], speed)
            for users in (1, 4, 32):
                g = Synth(seed=1, **gk2)
                steps = args.steps if users < 32 else max(40, args.steps // 4)
                r = evaluate(g.routes(steps, users, args.block), m, g.prefill_counts(), iters=args.iters)
                row = dict(sigma_req=sr, sigma_blk=sb, alpha=alpha, users=users,
                           a=round(cal['a'], 1), b=round(cal['b'], 2), c=round(cal['c'], 3),
                           **{k: round(v, 3) for k, v in r.items()})
                rows.append(row); print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
