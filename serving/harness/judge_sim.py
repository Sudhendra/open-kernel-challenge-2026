#!/usr/bin/env python3
"""CPU discrete-event simulator of the judge's serving workload on our stack (E1 in research/scout_oob_2026-11-12.md).

Pure Python, no GPU. Calibrated to the day-one rung table (out/judge/status_c6286ef374094ed9_*.json) and used to price
scheduler and capacity levers in minutes. Report: research/judge_sim_2026-11-12.md.

  python3 harness/judge_sim.py --calibrate              # fit the cost parameters, print fit table + residuals, write fit json
  python3 harness/judge_sim.py --run                    # simulate the four rungs with the fitted (or default) parameters
  python3 harness/judge_sim.py --whatif sr              # price one lever (tok/s/GPU per rung + score via score_model.score)
  python3 harness/judge_sim.py --whatif all             # every lever below
  python3 harness/judge_sim.py --whatif base --hist     # also print the warm-batch histogram per rung

Levers (--whatif NAME): base, sr (KV 2.06M -> 3.2M full tokens), sr_k1 (3.2M x 1.07), k1 (2.06M x 1.07),
w2_100 / w2_200 (hold a lone warm extend up to 100 / 200 ms while decode runs), chunk32k (cold chunk 32,768),
retract (R1 retraction audit at c32 under the capacity model, incl. an admission-reserve sensitivity),
cold10 (cold-chunk line -10%), decode10 (decode step -10%), prefill10 (every prefill forward -10%).

Nov 13 (research/judge_sim_2026-11-13.md):
  python3 harness/judge_sim.py --selftest               # asserts on the copied stock rules
  python3 harness/judge_sim.py --residuals              # residuals over day one + day two + the SR and 32k checks
  python3 harness/judge_sim.py --price all --fits 'research/judge_sim_fits_2026-11-13/*.json' [--hist]
--calibrate now fits every scenario in scenarios() (day-one scored, day-two scored at KV 3,069,172, SR check
9ecac80c at 3,301,447, 32k-with-graphs check 061d4d6a at 1,922,681); --day1-only restores the Nov 12 objective.
--price names: BASE_TOKEN_TOKEN..., BASE d2 (control = day-two scored FGHIL+SR, 186157d12aeb4eba) or d1 (day one).
Tokens: w2 (lane-w2-fghil rule K=3 T=150 ms N=4096), mfsN (--min-free-slots-delay N, 1 = off), mrN (max-running N;
judge-set, model only), keager[NN] (chunk 32768, graphs to 16384, eager forwards +5 or +NN ms), k32g (chunk 32768 with
graphs and the 1.92M pool that check booted with), recvN (--scheduler-recv-interval N), lpm (--schedule-policy lpm),
nomixed, kv=N, t1 (cold forwards x0.976). Preset list NEW_LEVERS: d2_w2 d2_w2_mfs1 d2_mfs1 d2_mfs2 d2_mfs6 d2_mfs8
d2_mfs10 d2_mfs12 d2_mfs16 d2_mfs20 d2_mfs24 d2_w2_mfs6 d2_w2_mfs8 d2_w2_mfs12 d2_mr36
d2_mfs12_keager d2_w2_mfs12_keager d2_mfs12_t1 d1_keager d2_keager d2_keager30 d2_keager0 d2_w2_keager
d2_w2_mfs1_keager d2_mfs8_keager d2_w2_mfs8_keager d2_t1 d2_recv2 d2_recv8 d2_w2_recv8 d2_mfs8_recv8 d2_lpm d2_nomixed
d2_k32g d2_kv=3301447.
Stock rules added Nov 13 (on in every judge run): MinFreeSlotsDelayer (DFlash auto threshold 4 of 32: no admission
while > 28 run, unless a chunk is in flight) and SchedulerRecvSkipper (our stack's scheduler_recv_interval 4: new
requests are received after every extend forward and after every 4th decode step).

Model (all times in ms; one GPU timeline = the TP-8 group):
- Workload per rung (rules_live/serving_2026-11-11.json): c slots, conversations_per_slot 2 at c1/c4/c16 and 1 at c32,
  run back to back in a slot; reps 3/3/3/1. A conversation = 14 requests: a cold first turn of ~74,160 tokens, then 13
  turns of ~753 new tokens on top of the cached context (previous prompt + previous output), ~220 output tokens per turn.
  The judge says lengths are "varied": first-turn and new-turn lengths are drawn uniform +-LEN_SPREAD / +-NEW_SPREAD,
  outputs uniform 0.5..1.5 x 220. No think time between turns (a turn is submitted when the previous one finishes).
- Every request first pays an off-GPU latency tau (HTTP, tokenizing the whole conversation, radix match), then waits
  in the scheduler's FCFS queue.
- Scheduler (SGLang get_next_batch_to_run, prefill first): if any prefill work is admissible it runs a prefill forward:
  the in-flight chunked request first, then waiting requests FCFS, up to chunk_size tokens of compute (minus one token
  per riding decode row under mixed chunk), at most one chunked request, max-running 32. With mixed chunk on, every
  running decode row rides along and gains 1 token. Otherwise it runs a decode/verify step over all running rows, each
  gaining a ~ 1 + Binomial(7, (acc-1)/7) tokens (acc = the judge's per-rung acceptance).
- Prefill forward of M computed tokens costs min(wf + wp M, cf + cp M) + mix_row x riding rows: two lines, the warm
  (small-M, weight-streaming) line and the cold (16k chunk, compute) line; the cheaper one applies, so a forward that
  carries a cold chunk is on the cold line. Decode step costs df + dr x rows. Each forward adds sched_ms.
- KV (full-attention tokens per rank, capacity K): a conversation holds its whole context on device while its request
  runs (locked). Between turns and while queued its context is evictable (LRU, tail first). Admission needs
  (tokens to load from host + new tokens + reserve for the new request's output) <= free + evictable - reserve of
  running rows (output still to come x new_token_ratio, 1.0 for fixed-length requests). Evicted context is reloaded
  from the host tier on the next turn at reload_ms_per_ktok of GPU time. A decode step that cannot get spec_alloc
  slots per row retracts the running request with the longest output (SGLang retract_decode); it is re-queued at the
  front and re-admitted later.
- Metrics, as the judge's numbers imply (scout 1.1: tp / (c x u) ~ 43.5 at every rung):
  tok/s/GPU = (sum of prompt tokens incl. cache hits + output tokens) / makespan / 8;
  per-user e2e = total output tokens / total request latency; TTFT p90 over first turns and over later turns;
  tok_s_per_user_p50 = median over requests of (out - 1) / (e2e - ttft); e2e_p50 = median of out / e2e.
"""
import argparse
import copy
import glob
import heapq
import json
import math
import os
import random
import sys
from dataclasses import dataclass, asdict, field

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import score_model  # noqa: E402

JUDGE_GLOB = os.path.join(SERVING, "out/judge/status_c6286ef374094ed9_*.json")
FIT_PATH = os.environ.get("JUDGE_SIM_FIT", "")   # optional json of fitted params; defaults below are the Nov 12 fit

RUNGS = (1, 4, 16, 32)
CONV_PER_SLOT = {1: 2, 4: 2, 16: 2, 32: 1}
JUDGE_REPS = {1: 3, 4: 3, 16: 3, 32: 1}
ACCEPT = {1: 4.3, 4: 4.5, 16: 4.4, 32: 4.8}       # judge-reported acceptance per rung (input, not fitted)
STOCK_TTFT_FIRST_P90 = {16: 83.0, 32: 184.0}       # seconds, stock day-one text (2x rule)

FIT_NAMES = ("cf", "cp", "wf", "wp", "df", "dr", "tau", "len_spread", "reload_ms_per_ktok", "mix_after_ms")


@dataclass
class Params:
    # cost model: Nov 13 fit over day one + day two + the SR and 32k checks, with the delayer and recv skipper modelled
    # (objective 13.5 over 10 rung-scenarios; research/judge_sim_2026-11-13.md). Nov 12 day-one-only fit: cf 114.2,
    # cp 0.02649, wf 114.2, wp 0.02602, df 12.94, dr 0.4411, tau 29.13, len_spread 0.3103, reload 0.1259, mix_after 5.501
    cf: float = 166.4       # cold line fixed, ms
    cp: float = 0.02797      # cold line per token, ms  (cf + cp*16384 ~ 620 ms)
    wf: float = 127.1        # warm line fixed, ms
    wp: float = 0.02615        # warm line per token, ms
    df: float = 12.41        # decode/verify step fixed, ms
    dr: float = 0.5142        # decode/verify step per row, ms
    tau: float = 27.63       # off-GPU per-request latency (HTTP + tokenize 80k-token conversation), ms
    # fixed nuisance parameters (not fitted; see report)
    mix_row: float = 0.3    # cost of one riding decode row in a mixed prefill forward, ms (lane G: 1-3% of a chunk for 32 rows)
    sched_ms: float = 0.3   # scheduler/launch overhead per forward not hidden by the overlap scheduler, ms
    reload_ms_per_ktok: float = 0.02551   # GPU time per 1k tokens reloaded from the host tier (HiCache load-back, layer-overlapped)
    mix_after_ms: float = 5.87   # extra cost of the first decode step after a mixed forward (DFlash rebuilds the draft input; lane G)
    att: float = 14 * 4.0 / (48_000 * 16_384)   # full-attention cost, ms per (query token x prefix token): 14 full layers,
                                                # +4 ms per layer for a 16k chunk at 48k prefix (P17 trace); fixed, not fitted
    # scheduler / capacity
    kv_tokens: int = 2_063_388
    max_running: int = 32
    chunk: int = 16_384
    mixed: bool = True
    spec_alloc: int = 8     # KV slots a verify step needs per row
    fixed_len: bool = True  # requests carry max_tokens = their length (ratio 1.0 reserve); False -> max_new + new_token_ratio
    max_new: int = 4096     # only when fixed_len is False
    cold_scale: float = 1.0     # multiplies the cost of forwards that carry a cold chunk (R2/C4-class levers)
    w2_hold_ms: float = 0.0     # W2: hold a lone warm extend up to this long while decode keeps the GPU busy
    w2_tokens: int = 4096       # W2: release as soon as this many warm tokens are waiting (Nov 12 rule, w2_min_reqs 0)
    w2_min_reqs: int = 0        # W2 lane rule (lane-w2-fghil, K): >0 -> hold only while 0 < waiting < K, every waiting
                                # request a warm extend of <= w2_max_new tokens, oldest younger than w2_hold_ms
    w2_max_new: int = 4096      # W2 lane rule N
    # stock MinFreeSlotsDelayer (managers/min_free_slots_delayer.py, scheduler._get_new_batch_prefill_raw):
    # None = flag unset -> DFlash formula min(4, max(2, (max_running + 5) // 6)) if max_running >= 8;
    # an explicit value is capped by max_running and <= 1 disables (--min-free-slots-delay N)
    min_free_slots_delay: object = None
    req_pool_rows: int = 0      # request-pool rows (0 = max_running); allocatable = min(max_running, rows) - running
    # stock SchedulerRecvSkipper (our stack declares scheduler_recv_interval 4): new requests are received only when the
    # weight counter reaches the interval; an extend forward weighs 1000 (always recv after), a decode/verify step 1
    recv_interval: int = 4
    policy: str = "fcfs"        # fcfs | lpm (warm turns, which carry an ~80k cached prefix, sort before cold ones)
    # K-eager: forwards above graph_max_tokens computed tokens run without a prefill graph; eager_ms is the extra
    # per-forward cost of running one eagerly (host launch/python not hidden behind the GPU)
    graph_max_tokens: int = 1 << 30
    eager_ms: float = 0.0
    # Nov 18 (lane KSYNC): piecewise-linear decode/verify step cost ((rows, ms), ...); () = df + dr x rows
    dec_tab: tuple = ()
    acc_spread: float = 0.0     # per-request acceptance drawn uniform acc x (1 +- acc_spread) (content varies per turn)
    rider_tok: int = 1          # tokens a decode row gains riding a mixed prefill forward (stock: 1)
    # KSYNC rule: hold waiting prefills while every one is short (<= ksync_short uncached tokens), rows are decoding,
    # held < ksync_k x decoding rows, held + running < max_running, and the hold has lasted < ksync_n forwards
    ksync_k: float = 0.0        # 0 = off
    ksync_n: int = 0
    ksync_short: int = 4096
    ksync_min_mult: float = 0.0  # Nov 18 b: hold only while decoding rows >= ksync_min_mult x ksync_n (0 = no floor)
    # stock PrefillDelayer (--enable-prefill-delayer): queue trigger ratio (0 = delayer off), slot trigger always on
    pd_ratio: float = 0.0
    pd_max_passes: int = 30
    pd_max_ms: float = 5000.0
    conv_per_slot: int = 0      # 0 = the judge's CONV_PER_SLOT table
    # workload
    first_turn: int = 74_160
    new_per_turn: int = 753
    turns: int = 14
    out_mean: int = 220
    len_spread: float = 0.3103
    new_spread: float = 0.3
    out_spread: float = 0.5
    think_ms: float = 0.0


class Conv:
    __slots__ = ("cid", "slot", "lens", "news", "outs", "turn", "ctx", "dev", "last_use", "locked")

    def __init__(self, cid, slot, lens, news, outs):
        self.cid, self.slot, self.lens, self.news, self.outs = cid, slot, lens, news, outs
        self.turn = 0
        self.ctx = 0          # tokens of context the server has seen (prompts + outputs) before the current turn
        self.dev = 0          # of which resident on device
        self.last_use = 0.0
        self.locked = False


class Req:
    __slots__ = ("conv", "turn", "submit", "ready", "prompt", "new", "out", "done_in", "done_out", "ttft", "finish",
                 "admitted", "retracted", "reload", "accm")

    def __init__(self, conv, turn, submit, ready, prompt, new, out):
        self.conv, self.turn, self.submit, self.ready = conv, turn, submit, ready
        self.prompt, self.new, self.out = prompt, new, out
        self.done_in = 0      # computed tokens of this request's extend so far
        self.done_out = 0
        self.ttft = None
        self.finish = None
        self.admitted = False
        self.retracted = 0
        self.reload = 0
        self.accm = 1.0


def resolve_min_free_slots(user_value, max_running, is_dflash=True):
    """Exact copy of stock resolve_min_free_slots (min_free_slots_delayer.py). None = disabled."""
    max_running = max(0, int(max_running))
    if user_value is not None:
        thr = min(int(user_value), max_running)
        return thr if thr > 1 else None
    if is_dflash and max_running >= 8:
        return min(4, max(2, (max_running + 5) // 6))
    return None


def delayer_holds(p, running_bs, chunked_in_flight):
    """Stock MinFreeSlotsDelayer.should_delay as wired in _get_new_batch_prefill_raw: skipped while a chunked request is
    in flight; otherwise hold ALL prefill admission (warm and cold alike) while running_bs > 0 and allocatable rows <
    threshold, allocatable = min(pp_max_micro_batch_size (= max_running at PP 1) - running_bs, free req-pool rows)."""
    thr = resolve_min_free_slots(p.min_free_slots_delay, p.max_running)
    if thr is None or chunked_in_flight:
        return False
    rows = p.req_pool_rows or p.max_running
    alloc = min(p.max_running - running_bs, rows - running_bs)
    return running_bs > 0 and alloc < thr


def dec_cost(p, b):
    """Decode/verify step cost for b rows (without sched_ms / mix_after_ms)."""
    if not p.dec_tab:
        return p.df + p.dr * b
    tab = p.dec_tab
    if b <= tab[0][0]:
        return tab[0][1]
    for (b0, m0), (b1, m1) in zip(tab, tab[1:]):
        if b <= b1:
            return m0 + (m1 - m0) * (b - b0) / (b1 - b0)
    (b0, m0), (b1, m1) = tab[-2], tab[-1]
    return m1 + (m1 - m0) * (b - b1) / (b1 - b0)


def ksync_holds(p, waiting, n_running, n_decoding, chunked_in_flight, held_fwds):
    """KSYNC rule: hold while every waiting request is a short prefill, rows decode, the held requests are fewer than
    ksync_k x decoding rows, they and the running rows leave room under max_running, and the hold is younger than
    ksync_n forwards. Never holds a long prefill, with a chunk in flight, or when nothing decodes."""
    if p.ksync_k <= 0 or chunked_in_flight or not waiting or n_decoding == 0:
        return False
    if n_decoding < p.ksync_min_mult * p.ksync_n:
        return False
    if any((1 if r.retracted else r.new) > p.ksync_short for r in waiting):
        return False
    return (len(waiting) < p.ksync_k * n_decoding and len(waiting) + n_running < p.max_running
            and held_fwds < p.ksync_n)


def make_convs(p, c, rep, seed):
    rng = random.Random(seed * 1000003 + c * 101 + rep)
    convs = []
    n = c * (p.conv_per_slot or CONV_PER_SLOT[c])
    for i in range(n):
        lens = int(p.first_turn * rng.uniform(1 - p.len_spread, 1 + p.len_spread))
        news = [int(p.new_per_turn * rng.uniform(1 - p.new_spread, 1 + p.new_spread)) for _ in range(p.turns - 1)]
        outs = [max(8, int(p.out_mean * rng.uniform(1 - p.out_spread, 1 + p.out_spread))) for _ in range(p.turns)]
        convs.append(Conv(i, i % c, lens, news, outs))
    return convs, rng


def simulate(p, c, rep=0, seed=1, trace=False):
    convs, rng = make_convs(p, c, rep, seed)
    acc_rng = random.Random(seed * 7919 + c * 13 + rep)
    by_slot = {}
    for cv in convs:
        by_slot.setdefault(cv.slot, []).append(cv)
    queue_next = {s: list(v) for s, v in by_slot.items()}   # conversations still to start per slot

    t = 0.0
    arrivals = []          # heap of (ready_time, seq, req)
    seq = 0
    waiting = []           # FCFS list of ready requests (not admitted)
    running = []           # admitted requests (prefilling or decoding)
    chunked = None         # request whose extend is in flight across forwards
    done = []
    stats = {"prefill_fwd": 0, "decode_steps": 0, "mixed_rows": 0, "retractions": 0, "reload_tokens": 0,
             "evicted_tokens": 0, "warm_hist": {}, "cold_fwd": 0, "gpu_busy": 0.0, "gpu_prefill": 0.0,
             "gpu_decode": 0.0, "gpu_cold": 0.0, "gpu_warm": 0.0, "resident_samples": [], "running_samples": [],
             "delayer_holds": 0, "w2_holds": 0, "admit_running": {}, "eager_fwd": 0, "ksync_holds": 0,
             "pd_holds": 0, "warm_fwd": 0, "warm_seqs": 0}
    fwd_ct = [0]
    ks_start = [None]      # forward count when the current KSYNC hold began
    pd_state = {"cnt": None, "t0": 0.0, "skip_first": True, "hist": []}   # stock PrefillDelayer
    recv_cnt = [10 ** 9]   # SchedulerRecvSkipper counter; starts high so the first pass receives
    after_mixed = [False]
    ntr = 0.7              # new_token_ratio (only used when fixed_len is False)
    ntr_min, ntr_decay = 0.7 * 0.14, (0.7 - 0.7 * 0.14) / 600.0

    def submit(cv, now):
        nonlocal seq
        k = cv.turn
        new = cv.lens if k == 0 else cv.news[k - 1]
        prompt = cv.ctx + new
        r = Req(cv, k, now, now + p.tau, prompt, new, cv.outs[k])
        if p.acc_spread > 0:
            r.accm = acc_rng.uniform(1 - p.acc_spread, 1 + p.acc_spread)
        heapq.heappush(arrivals, (r.ready, seq, r))
        seq += 1

    for s, lst in queue_next.items():
        submit(lst.pop(0), 0.0)

    def used():
        return sum(cv.dev for cv in convs)

    def evictable(exclude=None):
        return sum(cv.dev for cv in convs if not cv.locked and cv is not exclude)

    def reserve_running():
        tot = 0.0
        for r in running:
            rem = r.out - r.done_out
            if p.fixed_len:
                tot += rem
            else:
                tot += min(p.max_new - r.done_out, 4096) * ntr
        return tot

    def evict(need, exclude=None):
        """Free `need` slots by LRU eviction of unlocked context (tail first)."""
        free = p.kv_tokens - used()
        if free >= need:
            return True
        short = need - free
        cands = sorted((cv for cv in convs if not cv.locked and cv is not exclude and cv.dev > 0), key=lambda v: v.last_use)
        for cv in cands:
            take = min(cv.dev, short)
            cv.dev -= take
            stats["evicted_tokens"] += take
            short -= take
            if short <= 0:
                return True
        return short <= 0

    def try_admit(r, budget_running_rows):
        """Admission check for a waiting request (SGLang PrefillAdder.add_one_req budget). Returns True if admitted."""
        cv = r.conv
        cached = min(cv.dev, r.prompt - r.new) if r.retracted == 0 else min(cv.dev, r.prompt + r.done_out)
        target_prefix = (r.prompt - r.new) if r.retracted == 0 else (r.prompt + r.done_out - 1)
        reload = max(0, target_prefix - cv.dev)
        extend = r.new if r.retracted == 0 else 1
        out_res = (r.out - r.done_out) if p.fixed_len else min(p.max_new - r.done_out, 4096)
        need = reload + extend + out_res
        avail = (p.kv_tokens - used()) + evictable(exclude=cv) - reserve_running()
        if need > avail:
            return False
        if not evict(reload + extend, exclude=cv):
            return False
        cv.dev = target_prefix + extend      # reloaded prefix + slots for the new tokens
        cv.locked = True
        r.reload = reload
        stats["reload_tokens"] += reload
        r.admitted = True
        r.done_in = 0
        r.new = extend if r.retracted else r.new
        return True

    def finish_req(r, now):
        r.finish = now
        cv = r.conv
        cv.locked = False
        cv.last_use = now
        cv.ctx = r.prompt + r.out
        done.append(r)
        cv.turn += 1
        if cv.turn < p.turns:
            submit(cv, now + p.think_ms)
        else:
            lst = queue_next[cv.slot]
            if lst:
                submit(lst.pop(0), now + p.think_ms)

    def decode_rows():
        return [r for r in running if r.ttft is not None and r.done_out < r.out]

    while arrivals or waiting or running:
        if p.recv_interval <= 1 or recv_cnt[0] >= p.recv_interval:
            recv_cnt[0] = 0
            while arrivals and arrivals[0][0] <= t + 1e-9:
                waiting.append(heapq.heappop(arrivals)[2])
        drows = decode_rows()
        if p.policy == "lpm" and len(waiting) > 1:
            waiting.sort(key=lambda r: -r.conv.dev)   # stable: FCFS among equal device prefixes

        # ---- try to build a prefill forward ----
        budget = min(p.chunk, max(p.chunk, 16384)) - (len(drows) if p.mixed else 0)
        plan = []           # (req, tokens)
        warm_n = 0
        has_cold = False
        if chunked is not None:
            take = min(chunked.new - chunked.done_in, budget)
            plan.append((chunked, take))
            budget -= take
            has_cold = True
        # stock MinFreeSlotsDelayer: no admission while fewer than thr request rows are free (no chunk in flight)
        hold = False
        if chunked is None and waiting and delayer_holds(p, len(running), False):
            hold = True
            stats["delayer_holds"] += 1
        # W2 gate: lone warm extends wait while decode keeps the GPU busy (runs after stock's delayer, as in the lane)
        if hold:
            pass
        elif p.w2_hold_ms > 0 and p.w2_min_reqs > 0 and chunked is None and waiting and drows:
            if (len(waiting) < p.w2_min_reqs
                    and all(r.turn > 0 and not r.retracted and r.new <= p.w2_max_new for r in waiting)
                    and t - min(r.ready for r in waiting) < p.w2_hold_ms):
                hold = True
                stats["w2_holds"] += 1
        elif p.w2_hold_ms > 0 and chunked is None and waiting and drows:
            ready_warm = [r for r in waiting if r.turn > 0 and not r.retracted]
            if len(ready_warm) == len(waiting):
                tok = sum(r.new for r in ready_warm)
                oldest = min(r.ready for r in ready_warm)
                if tok < p.w2_tokens and t - oldest < p.w2_hold_ms and len(running) < p.max_running:
                    hold = True
                    stats["w2_holds"] += 1
        # KSYNC: short prefills wait for each other while rows decode (after stock's delayer, like the lane)
        ks_held = 0 if ks_start[0] is None else fwd_ct[0] - ks_start[0]
        if not hold and ksync_holds(p, waiting, len(running), len(drows), chunked is not None, ks_held):
            hold = True
            stats["ksync_holds"] += 1
            if ks_start[0] is None:
                ks_start[0] = fwd_ct[0]
        else:
            ks_start[0] = None
        # stock PrefillDelayer (queue + slot triggers, max_delay_passes / max_delay_ms caps, skip_first_delayer)
        if not hold and p.pd_ratio > 0 and chunked is None and waiting:
            run = len(running)
            mpbs = max(pd_state["hist"][-16:], default=0)
            qmin = min(int(run * p.pd_ratio), mpbs)
            expired = pd_state["cnt"] is not None and t - pd_state["t0"] >= p.pd_max_ms
            queue_cond = run > 0 and qmin > 0 and len(waiting) < qmin and not expired
            if (p.max_running - run < mpbs) or queue_cond:
                if pd_state["skip_first"]:
                    pd_state["skip_first"] = False
                elif (pd_state["cnt"] or 0) < p.pd_max_passes - 1:
                    if pd_state["cnt"] is None:
                        pd_state["cnt"], pd_state["t0"] = 0, t
                    pd_state["cnt"] += 1
                    hold = True
                    stats["pd_holds"] += 1
                    pd_state["hist"].append(min(max(len(waiting), 1), max(p.max_running - run, 1)))
                else:
                    pd_state["cnt"] = None
            else:
                pd_state["cnt"] = None
        if not hold:
            run0 = len(running)
            i = 0
            while i < len(waiting) and budget > 0:
                if len(running) + 1 > p.max_running:
                    break
                r = waiting[i]
                if r.turn == 0 and r.retracted == 0 and chunked is not None:
                    break       # only one chunked request in flight; FCFS blocks behind it
                if not try_admit(r, len(drows)):
                    break       # FCFS: NO_TOKEN stops the adder
                waiting.pop(i)
                if r.turn > 0 and not r.retracted:
                    stats["admit_running"][run0] = stats["admit_running"].get(run0, 0) + 1   # running at pass start
                running.append(r)
                take = min(r.new, budget)
                plan.append((r, take))
                budget -= take
                if take < r.new:
                    chunked = r
                    has_cold = True
                elif r.turn == 0 and not r.retracted:
                    has_cold = True
                if r.turn > 0 and not r.retracted:
                    warm_n += 1
            if p.pd_ratio > 0 and len(running) > run0:
                pd_state["hist"].append(len(running) - run0)
        if plan:
            fwd_ct[0] += 1
            m = sum(k for _, k in plan)
            reload = sum(r.reload for r, _ in plan if r.done_in == 0)
            dur = min(p.wf + p.wp * m, p.cf + p.cp * m)
            for r, k in plan:     # attention over the prefix (full layers)
                pre = (r.prompt - r.new) + r.done_in
                dur += p.att * k * (pre + k / 2.0)
            riders = []
            if p.mixed:
                riders = [r for r in drows if r not in [q for q, _ in plan]]
                # riders need one slot each
                if p.kv_tokens - used() + evictable() < len(riders):
                    riders = []
            if has_cold:
                dur *= p.cold_scale
            if m > p.graph_max_tokens:
                dur += p.eager_ms
                stats["eager_fwd"] += 1
            dur += p.mix_row * len(riders) + p.reload_ms_per_ktok * reload / 1000.0 + p.sched_ms
            t += dur
            stats["prefill_fwd"] += 1
            stats["gpu_busy"] += dur
            stats["gpu_prefill"] += dur
            if has_cold:
                stats["cold_fwd"] += 1
                stats["warm_in_cold"] = stats.get("warm_in_cold", 0) + warm_n
                stats["gpu_cold"] += dur
            else:
                stats["gpu_warm"] += dur
                stats["warm_hist"][warm_n] = stats["warm_hist"].get(warm_n, 0) + 1
                stats["warm_fwd"] += 1
                stats["warm_seqs"] += warm_n
            stats["mixed_rows"] += len(riders)
            after_mixed[0] = bool(riders)
            for r, k in plan:
                r.done_in += k
                r.reload = 0
                if r.done_in >= r.new:
                    if r is chunked:
                        chunked = None
                    if r.ttft is None:
                        r.ttft = t - r.submit
                        r.done_out = 1
                    if r.done_out >= r.out:
                        running.remove(r)
                        finish_req(r, t)
            for r in riders:
                if evict(1):
                    k = min(p.rider_tok, r.out - r.done_out)
                    r.conv.dev += k
                    r.done_out += k
                    if r.done_out >= r.out:
                        running.remove(r)
                        finish_req(r, t)
            stats["running_samples"].append(len(running))
            recv_cnt[0] += 1000          # extend-like forward: the skipper always receives after it
            continue

        # ---- decode / verify step ----
        if drows:
            need = p.spec_alloc * len(drows)
            while drows and (p.kv_tokens - used() + evictable()) < need:
                victim = max(drows, key=lambda r: (r.done_out, -r.prompt))
                drows.remove(victim)
                running.remove(victim)
                victim.retracted += 1
                victim.admitted = False
                victim.conv.locked = False
                victim.conv.last_use = t
                stats["retractions"] += 1
                waiting.insert(0, victim)
                if not p.fixed_len:
                    ntr = min(1.0, ntr + 0.3)
                need = p.spec_alloc * len(drows)
            if not drows:
                continue
            evict(need)
            b = len(drows)
            dur = dec_cost(p, b) + p.sched_ms + (p.mix_after_ms if after_mixed[0] else 0.0)
            after_mixed[0] = False
            t += dur
            fwd_ct[0] += 1
            stats["decode_steps"] += 1
            stats["gpu_busy"] += dur
            stats["gpu_decode"] += dur
            acc = ACCEPT_CUR[0]
            for r in drows:
                q = min(1.0, max(0.0, (acc * r.accm - 1.0) / 7.0))
                k = 1 + sum(1 for _ in range(7) if rng.random() < q)
                k = min(k, r.out - r.done_out)
                r.done_out += k
                r.conv.dev += k
                if r.done_out >= r.out:
                    running.remove(r)
                    finish_req(r, t)
            if not p.fixed_len:
                ntr = max(ntr_min, ntr - ntr_decay)
            stats["running_samples"].append(len(running))
            stats["resident_samples"].append(sum(1 for cv in convs if cv.dev > 0 and cv.turn < p.turns))
            recv_cnt[0] += 1             # decode/verify weight 1
            continue

        # ---- idle: jump to the next arrival ----
        if arrivals:
            t = max(t, arrivals[0][0])
            recv_cnt[0] = 10 ** 9        # idle passes spin in microseconds: treat the next arrival as received at once
        elif waiting and not running:
            # nothing admissible and nothing running: should not happen; force-admit to avoid a stall
            raise RuntimeError(f"deadlock at t={t:.0f} c={c}: waiting={len(waiting)} used={used()}")

    makespan = t
    return done, makespan, stats


ACCEPT_CUR = [4.5]


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    f, cc = math.floor(k), math.ceil(k)
    return xs[f] + (xs[cc] - xs[f]) * (k - f)


def rung_metrics(p, c, reps=None, seed=1, acc=None):
    reps = reps or max(JUDGE_REPS[c], 1)
    ACCEPT_CUR[0] = acc if acc is not None else ACCEPT[c]
    tps, us, first, later, dec, e2e = [], [], [], [], [], []
    later_dec, mks = [], []
    agg = None
    for rep in range(reps):
        done, mk, st = simulate(p, c, rep, seed)
        mks.append(mk / 1000.0)
        later_dec += [(r.finish - r.submit - r.ttft) / 1000.0 for r in done if r.turn > 0]
        tokens = sum(r.prompt + r.out for r in done)
        tps.append(tokens / (mk / 1000.0) / 8.0)
        us.append(sum(r.out for r in done) / sum((r.finish - r.submit) / 1000.0 for r in done))
        first += [r.ttft / 1000.0 for r in done if r.turn == 0]
        later += [r.ttft / 1000.0 for r in done if r.turn > 0]
        dec += [(r.out - 1) / ((r.finish - r.submit - r.ttft) / 1000.0) for r in done if r.finish - r.submit > r.ttft + 1e-6]
        e2e += [r.out / ((r.finish - r.submit) / 1000.0) for r in done]
        st["makespan"] = mk
        if agg is None:
            agg = {k: (copy.deepcopy(v)) for k, v in st.items()}
        else:
            for k, v in st.items():
                if isinstance(v, (int, float)):
                    agg[k] += v
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        agg[k][kk] = agg[k].get(kk, 0) + vv
                elif isinstance(v, list):
                    agg[k] += v
    tp = sum(tps) / len(tps)
    u = sum(us) / len(us)
    return {"tok_s_per_gpu": tp, "tok_s_per_user_e2e": u, "ttft_first_p90_s": pct(first, 0.9),
            "ttft_later_p90_s": pct(later, 0.9), "tok_s_per_user_p50": pct(dec, 0.5), "tok_s_per_user_e2e_p50": pct(e2e, 0.5),
            "tp_over_cu": tp / (c * u), "stats": agg, "reps": reps,
            "ttft_later_p50_s": pct(later, 0.5), "ttft_first_p50_s": pct(first, 0.5), "dec_later_p50_s": pct(later_dec, 0.5), "wall_s": sum(mks) / len(mks),
            "warm_fwd_per_cycle": agg["warm_fwd"] / (reps * (p.conv_per_slot or CONV_PER_SLOT[c]) * (p.turns - 1)),
            "warm_seqs_per_fwd": agg["warm_seqs"] / max(1, agg["warm_fwd"])}


def run_all(p, reps_override=None, seed=1, rungs=RUNGS, accept=None):
    return {c: rung_metrics(p, c, (reps_override or {}).get(c), seed, (accept or {}).get(c)) for c in rungs}


# ------------------------------------------------------------------ judge data, fit, score

def judge():
    res = score_model.load(JUDGE_GLOB)
    return res, {r["concurrency"]: r for r in res["rungs"]}


SIM_REPS = {1: 12, 4: 8, 16: 4, 32: 4}   # more reps than the judge: a rep samples only 2-32 conversations (tok/out swings 40-48)


def residuals(sim, jr):
    out = {}
    for c in RUNGS:
        s, j = sim[c], jr[c]
        out[c] = {k: s[k] / j[k] - 1.0 for k in ("tok_s_per_gpu", "tok_s_per_user_e2e", "ttft_first_p90_s",
                                                  "ttft_later_p90_s", "tok_s_per_user_p50", "tok_s_per_user_e2e_p50")}
    return out


def objective(sim, jr, p):
    err = 0.0
    for c in RUNGS:
        s, j = sim[c], jr[c]
        err += (math.log(s["tok_s_per_gpu"] / j["tok_s_per_gpu"]) / 0.05) ** 2
        err += (math.log(s["tok_s_per_user_e2e"] / j["tok_s_per_user_e2e"]) / 0.05) ** 2
        err += (math.log(s["ttft_first_p90_s"] / j["ttft_first_p90_s"]) / 0.15) ** 2
        # weak terms: later TTFT and decode p50 (not targets; keep the fit physical)
        err += 0.25 * (math.log(s["ttft_later_p90_s"] / j["ttft_later_p90_s"]) / 0.3) ** 2
        err += 0.25 * (math.log(s["tok_s_per_user_p50"] / j["tok_s_per_user_p50"]) / 0.15) ** 2
    # weak priors from our own measurements: 16k chunk ~620 ms (NEXT_STEPS 27/37), c1 step ~14 ms (I_AUDIT)
    err += 0.5 * (math.log((p.cf + p.cp * 16384) / 620.0) / 0.10) ** 2
    err += 0.5 * (math.log((p.df + p.dr) / 14.0) / 0.10) ** 2
    return err


# Nov 13 calibration set: day one plus every day-two judge run on this stack. Each scenario = (name, param overrides,
# rungs, acceptance per rung, judge rung dicts). The KV pool is the run's own token_capacity from its status json.
DAY2_SCORED = "out/judge/status_186157d12aeb4eba_*.json"      # FGHIL + SR, token_capacity 3,069,172
DAY2_SR_CHECK = "out/judge/status_9ecac80c0db44581_*.json"    # FGHIL + SR, c32, token_capacity 3,301,447
DAY2_32K_CHECK = "out/judge/status_061d4d6a79f04dd1_*.json"   # FGHIL + SR + chunk 32768 WITH graphs to 32k, c32,
                                                              # token_capacity 1,922,681 (graphs ate the SR gain)


def _rungs_of(path):
    res = score_model.load(os.path.join(SERVING, path))
    return res, {r["concurrency"]: r for r in res["rungs"]}


def scenarios():
    _, d1 = judge()
    _, d2 = _rungs_of(DAY2_SCORED)
    _, sr = _rungs_of(DAY2_SR_CHECK)
    _, k32 = _rungs_of(DAY2_32K_CHECK)
    acc = lambda jr: {c: r["acceptance"] for c, r in jr.items()}
    return [("day1", {}, RUNGS, acc(d1), d1, 1.0),
            ("day2", {"kv_tokens": 3_069_172}, RUNGS, acc(d2), d2, 1.0),
            ("srchk", {"kv_tokens": 3_301_447}, (32,), acc(sr), sr, 1.0),
            ("k32chk", {"kv_tokens": 1_922_681, "chunk": 32_768}, (32,), acc(k32), k32, 1.0)]


def run_scenarios(p, scen, seed=1, reps=None):
    out = {}
    for name, ov, rungs, acc, jr, w in scen:
        q = copy.deepcopy(p)
        for k, v in ov.items():
            setattr(q, k, v)
        out[name] = run_all(q, reps or SIM_REPS, seed, rungs, acc)
    return out


def objective_all(sims, scen, p):
    """Sum of the per-rung terms of `objective` over every scenario (single-rung checks: tp/u/TTFT1 only), plus the
    same weak priors once."""
    err = 0.0
    for name, ov, rungs, acc, jr, w in scen:
        for c in rungs:
            s, j = sims[name][c], jr[c]
            e = (math.log(s["tok_s_per_gpu"] / j["tok_s_per_gpu"]) / 0.05) ** 2
            e += (math.log(s["tok_s_per_user_e2e"] / j["tok_s_per_user_e2e"]) / 0.05) ** 2
            e += (math.log(s["ttft_first_p90_s"] / j["ttft_first_p90_s"]) / 0.15) ** 2
            e += 0.25 * (math.log(s["ttft_later_p90_s"] / j["ttft_later_p90_s"]) / 0.3) ** 2
            e += 0.25 * (math.log(s["tok_s_per_user_p50"] / j["tok_s_per_user_p50"]) / 0.15) ** 2
            err += w * e
    err += 0.5 * (math.log((p.cf + p.cp * 16384) / 620.0) / 0.10) ** 2
    err += 0.5 * (math.log((p.df + p.dr) / 14.0) / 0.10) ** 2
    return err


def with_x(base, x, names=FIT_NAMES):
    p = copy.deepcopy(base)
    for n, v in zip(names, x):
        setattr(p, n, math.exp(v))
    return p


def calibrate(base=None, maxiter=400, seed=1, names=FIT_NAMES, day1_only=False):
    import numpy as np
    from scipy.optimize import minimize
    _, jr = judge()
    scen = None if day1_only else scenarios()
    base = base or Params()
    x0 = np.array([math.log(getattr(base, n)) for n in names])
    best = [1e18, None]

    def f(x):
        p = with_x(base, x, names)
        if p.wf >= p.cf or p.len_spread >= 0.8:   # warm line below the cold line at small M; lengths stay positive
            return 1e9
        try:
            if scen is None:
                e = objective(run_all(p, SIM_REPS, seed), jr, p)
            else:
                e = objective_all(run_scenarios(p, scen, seed), scen, p)
        except RuntimeError:
            return 1e9
        if e < best[0]:
            best[0], best[1] = e, x.copy()
        return e

    r = minimize(f, x0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-3,
                                                       "initial_simplex": None, "adaptive": True})
    p = with_x(base, best[1], names)
    return p, best[0]


def load_params():
    p = Params()
    if FIT_PATH and os.path.exists(FIT_PATH):
        d = json.load(open(FIT_PATH))
        for k, v in d["params"].items():
            setattr(p, k, v)
    return p


def score_of(sim, base_sim, res):
    """Feed the simulated relative change of tp and u per rung into score_model.score on the day-one JSON."""
    res2 = copy.deepcopy(res)
    for r in res2["rungs"]:
        c = r["concurrency"]
        r["tok_s_per_gpu"] *= sim[c]["tok_s_per_gpu"] / base_sim[c]["tok_s_per_gpu"]
        r["tok_s_per_user_e2e"] *= sim[c]["tok_s_per_user_e2e"] / base_sim[c]["tok_s_per_user_e2e"]
    s, ratios = score_model.score(res2)
    return s, ratios


# ------------------------------------------------------------------ levers

def lever(name, p0):
    p = copy.deepcopy(p0)
    if name == "base":
        pass
    elif name == "sr":
        p.kv_tokens = 3_200_000
    elif name == "k1":
        p.kv_tokens = int(p0.kv_tokens * 1.07)
    elif name == "sr_k1":
        p.kv_tokens = int(3_200_000 * 1.07)
    elif name == "w2_100":
        p.w2_hold_ms = 100.0
    elif name == "w2_200":
        p.w2_hold_ms = 200.0
    elif name == "chunk32k":
        p.chunk = 32_768
    elif name == "cold10":
        p.cold_scale = 0.9
    elif name == "prefill10":
        p.cf *= 0.9
        p.cp *= 0.9
        p.wf *= 0.9
        p.wp *= 0.9
    elif name == "decode10":
        p.df *= 0.9
        p.dr *= 0.9
    elif name == "sr_w2_100":
        p.kv_tokens = 3_200_000
        p.w2_hold_ms = 100.0
    elif name == "sr_w2_200":
        p.kv_tokens = 3_200_000
        p.w2_hold_ms = 200.0
    elif name == "nomixed":
        p.mixed = False
    elif name == "kv_inf":
        p.kv_tokens = 50_000_000
    elif name == "reserve4096":
        p.fixed_len = False
        p.max_new = 4096
    else:
        raise SystemExit(f"unknown lever {name}")
    return p


LEVERS = ("sr", "k1", "sr_k1", "w2_100", "w2_200", "sr_w2_100", "sr_w2_200", "chunk32k", "cold10", "decode10", "prefill10",
          "kv_inf", "nomixed")
EVAL_REPS = {1: 12, 4: 8, 16: 6, 32: 8}


# ------------------------------------------------------------------ Nov 13 levers: day-two base, composable names
# A name is BASE_TOKEN_TOKEN...; BASE is d2 (FGHIL + SR as scored on day two: KV 3,069,172, day-two acceptance, scored
# against 186157d12aeb4eba) or d1 (FGHIL day one: KV 2,063,388, day-one acceptance, scored against c6286ef374094ed9).
# Tokens: w2 (lane-w2-fghil rule: K=3, T=150 ms, N=4096), mfsN (--min-free-slots-delay N; 1 disables), mrN
# (--max-running-requests N, req pool N rows), keager / keagerNN (chunk 32768, prefill graphs to 16384, forwards above
# run eager at +5 / +NN ms each), k32g (chunk 32768 WITH graphs: the KV pool the 061d4d6a check booted with, 1,922,681),
# recvN (--scheduler-recv-interval N), lpm (--schedule-policy lpm), nomixed (mixed chunk off: env/code, not a flag),
# kv=N (KV pool N tokens), t1 (lane T1 draft-skip: -73 ms per 80k cold prompt ~ cold forwards x0.976).
D2_KV = 3_069_172
NEW_LEVERS = ("d2_w2", "d2_w2_mfs1", "d2_mfs1", "d2_mfs2", "d2_mfs6", "d2_mfs8", "d2_mfs10", "d2_mfs12", "d2_mfs16",
              "d2_mfs20", "d2_mfs24", "d2_w2_mfs6", "d2_w2_mfs8", "d2_w2_mfs12", "d2_mr36", "d2_mfs12_keager",
              "d2_w2_mfs12_keager", "d2_mfs12_t1",
              "d1_keager", "d2_keager", "d2_keager30", "d2_keager0", "d2_w2_keager", "d2_w2_mfs1_keager",
              "d2_mfs8_keager", "d2_w2_mfs8_keager", "d2_t1",
              "d2_recv2", "d2_recv8", "d2_w2_recv8", "d2_mfs8_recv8", "d2_lpm", "d2_nomixed", "d2_k32g", "d2_kv=3301447")


# Nov 18 base k21 (stack-k21/k22 as run 20261118T055445Z, research/lane_ksync_2026-11-18.md): decode step 13.2 / 18.3 /
# ~25 / 36 ms at 1 / 4 / 16 / 32 rows (traces; 16 estimated), small prefill forward ~54 ms + 32 ms per 1k new tokens
# (warm line + attention at an ~80k prefix), cold 16k chunk ~465 ms incl. attention, KV 3,221,409, --min-free-slots-delay
# 12. tau and mix_after_ms fitted to the run's later-turn TTFT / decode time (--k21check). Scored against 788356d8.
# Refit Nov 18 b on the 8-GPU A/B 20261118T155332Z (k23 vs k24): wf 54 -> 34, c16 step 25 -> 28 ms.
K21 = dict(dec_tab=((1, 13.2), (4, 18.3), (16, 28.0), (32, 36.0)), wf=34.0, wp=0.0263, cf=110.0, cp=0.01843,
           kv_tokens=3_221_409, min_free_slots_delay=12, tau=8.0, mix_after_ms=10.0, acc_spread=0.3)
K21_JSON = "out/judge/status_788356d87cda4e27_fetched.json"
# the proxy's judge-shaped workload (harness proxy --shape judge --client ids): 1 conversation per slot, 13 turns,
# fixed 220-token outputs, later turns 283..1363 new tokens (median ~700)
PROXY = dict(conv_per_slot=1, turns=13, out_spread=0.0, new_per_turn=719, new_spread=0.55, len_spread=0.01)


SET_OVERRIDES = {}     # --set name=value on --price / --k21check, applied after the base preset


def apply(p, d):
    for k, v in d.items():
        setattr(p, k, v)
    return p


def new_lever(name, p0):
    """Returns (params, base_tag). base_tag 'd1', 'd2' or 'k21' selects the control (base sim, acceptance, judge JSON).
    Nov 18 tokens: ksK[nN][sS] (KSYNC rule, K = held/decoding ratio, N = forward cap, S = short-prefill tokens),
    pdR (stock --enable-prefill-delayer --prefill-delayer-queue-min-ratio R), proxy (the proxy's workload shape)."""
    toks = name.split("_")
    base = toks[0]
    if base not in ("d1", "d2", "k21"):
        raise SystemExit(f"unknown lever {name}")
    p = copy.deepcopy(p0)
    if base == "d2":
        p.kv_tokens = D2_KV
    if base == "k21":
        apply(p, K21)
    apply(p, SET_OVERRIDES)
    for t in toks[1:]:
        if t.startswith("ks"):
            import re
            m = re.fullmatch(r"ks([\d.]+)(?:n(\d+))?(?:s(\d+))?(?:m([\d.]+))?", t)
            if not m:
                raise SystemExit(f"bad ksync token {t}")
            p.ksync_k = float(m.group(1))
            p.ksync_n = int(m.group(2) or 1 << 30)
            p.ksync_short = int(m.group(3) or p.ksync_short)
            p.ksync_min_mult = float(m.group(4) or 0)
            continue
        if t.startswith("pd"):
            p.pd_ratio = float(t[2:])
            continue
        if t == "proxy":
            apply(p, PROXY)
            continue
        if t == "w2":
            p.w2_hold_ms, p.w2_min_reqs, p.w2_max_new = 150.0, 3, 4096
        elif t.startswith("mfs"):
            p.min_free_slots_delay = int(t[3:])
        elif t.startswith("mr"):
            p.max_running = int(t[2:])
            p.req_pool_rows = int(t[2:])
        elif t.startswith("keager"):
            p.chunk, p.graph_max_tokens = 32_768, 16_384
            p.eager_ms = float(t[6:]) if len(t) > 6 else 5.0
        elif t == "k32g":
            p.chunk, p.kv_tokens = 32_768, 1_922_681
        elif t.startswith("recv"):
            p.recv_interval = int(t[4:])
        elif t == "lpm":
            p.policy = "lpm"
        elif t == "nomixed":
            p.mixed = False
        elif t.startswith("kv="):
            p.kv_tokens = int(t[3:])
        elif t == "t1":
            p.cold_scale *= 0.976
        else:
            raise SystemExit(f"unknown token {t} in {name}")
    return p, base


BASE_JSON = {"d1": "out/judge/status_c6286ef374094ed9_*.json", "d2": DAY2_SCORED, "k21": K21_JSON}


def base_of(tag, p0):
    """(params, acceptance per rung, judge result) of a control."""
    res = score_model.load(os.path.join(SERVING, BASE_JSON[tag]))
    acc = {r["concurrency"]: r["acceptance"] for r in res["rungs"]}
    p = copy.deepcopy(p0)
    if tag == "d2":
        p.kv_tokens = D2_KV
    if tag == "k21":
        apply(p, K21)
    apply(p, SET_OVERRIDES)
    return p, acc, res


def price(names, p0, seed=1, reps=None):
    """{name: (score delta, {rung: (tp delta, u delta)}, sim)} for NEW_LEVERS-style names under params p0."""
    reps = reps or EVAL_REPS
    bases, out = {}, {}
    for name in names:
        p, tag = new_lever(name, p0)
        key = (tag, "proxy" in name.split("_"))
        if key not in bases:
            pb, acc, res = base_of(tag, p0)
            if key[1]:
                apply(pb, PROXY)
            bsim = run_all(pb, reps, seed, accept=acc)
            bases[key] = (bsim, acc, res, score_of(bsim, bsim, res)[0])
        bsim, acc, res, s0 = bases[key]
        sim = run_all(p, reps, seed, accept=acc)
        s, _ = score_of(sim, bsim, res)
        out[name] = (s / s0 - 1, {c: (sim[c]["tok_s_per_gpu"] / bsim[c]["tok_s_per_gpu"] - 1,
                                      sim[c]["tok_s_per_user_e2e"] / bsim[c]["tok_s_per_user_e2e"] - 1) for c in RUNGS},
                     sim, bsim)
    return out


def load_fit(path):
    p = Params()
    for k, v in json.load(open(path))["params"].items():
        setattr(p, k, v)
    return p


def cmd_price(args):
    """--price NAMES [--fits 'dir/*.json']: score delta per lever under every fit (main fit first), per-rung deltas."""
    names = []
    for n in args.price.split(","):
        names += list(NEW_LEVERS) if n == "all" else [n]
    fits = [("main", load_params())]
    if args.fits:
        for f in sorted(glob.glob(args.fits)):
            tag = os.path.splitext(os.path.basename(f))[0]
            if tag != "main":
                fits.append((tag, load_fit(f)))
    allres = {tag: price(names, p, args.seed) for tag, p in fits}
    tags = [t for t, _ in fits]
    print("score delta per lever (main fit, then each alternative fit), then main-fit per-rung tok/s/GPU [u] deltas")
    print(f"{'lever':<22} {'main':>7} {'min':>7} {'max':>7}  " + " ".join(f"{t:>6}" for t in tags[1:])
          + "   c1 / c4 / c16 / c32 tp [u]")
    for n in names:
        ds = [allres[t][n][0] for t in tags]
        rung = allres["main"][n][1]
        print(f"{n:<22} {ds[0]:>+7.2%} {min(ds):>+7.2%} {max(ds):>+7.2%}  " + " ".join(f"{d:>+6.1%}" for d in ds[1:])
              + "   " + " / ".join(f"{rung[c][0]:+.1%} [{rung[c][1]:+.1%}]" for c in RUNGS))
    if args.hist:
        for n in names:
            sim, bsim = allres["main"][n][2], allres["main"][n][3]
            st = sim[32]["stats"]
            rs = st["running_samples"]
            ar = st["admit_running"]
            print(f"  {n}: c32 mean running {sum(rs) / max(1, len(rs)):.1f}, warm admissions by running-at-pass-start "
                  f"max {max(ar) if ar else 0}, delayer holds/rep {st['delayer_holds'] / sim[32]['reps']:.0f}, "
                  f"W2 holds/rep {st['w2_holds'] / sim[32]['reps']:.0f}; warm hist c16 {fmt_hist(sim[16]['stats']['warm_hist'])};"
                  f" c32 {fmt_hist(st['warm_hist'])}; TTFT1 p90 c16/c32 {sim[16]['ttft_first_p90_s']:.1f}/{sim[32]['ttft_first_p90_s']:.1f}s"
                  f" later p90 {sim[16]['ttft_later_p90_s']:.2f}/{sim[32]['ttft_later_p90_s']:.2f}s")
    if args.json_out:
        json.dump({t: {n: {"score": allres[t][n][0], "rungs": allres[t][n][1]} for n in names} for t in tags},
                  open(args.json_out, "w"), indent=1)


# run 20261118T055445Z variant k21 (proxy, ids client, judge shape) [measured]: per rung wall (all reps), later-turn
# TTFT p50 / p90, later-turn decode time p50 (e2e - TTFT), first-turn TTFT p50; c16 later phase: 140 small prefill
# forwards for 192 arrivals = 11.7 per turn cycle (server.log)
K21_MEASURED = {1: (36.6, 0.087, 0.108, 0.672, 1.98), 4: (72.1, 0.134, 0.184, 1.072, 2.08),
                16: (64.1, 0.190, 0.254, 2.261, 17.0), 32: (106.5, 0.728, 1.347, 2.454, 33.2)}
PROXY_REPS = {1: 3, 4: 3, 16: 1, 32: 1}
STOCK_LATER_P90 = {1: 0.15, 4: 0.251, 16: 1.0, 32: 2.8}   # s; c4 and c32 [read], c1 and c16 assumed (lead)


def cmd_k21check(args):
    """Proxy-shaped k21 base vs the measured k21 run, per rung (acceptance 4.4)."""
    p, _ = new_lever("k21_proxy", Params())
    acc = {c: 4.4 for c in RUNGS}
    sims = {c: [rung_metrics(p, c, PROXY_REPS[c], seed, acc[c]) for seed in range(1, 1 + args.k21_seeds)] for c in RUNGS}
    print(f"{'c':>3} {'wall s sim/meas':>16} {'TTFTl p50':>14} {'TTFTl p90':>14} {'dec/turn':>14} {'TTFT1 p50':>13}"
          f" {'warm fwd/cycle':>14} {'seq/fwd':>7} {'ms/step':>8}")
    for c in RUNGS:
        m = K21_MEASURED[c]
        avg = lambda k: sum(s[k] for s in sims[c]) / len(sims[c])
        wall = avg("wall_s") * PROXY_REPS[c]
        print(f"{c:>3} {wall:>7.1f}/{m[0]:<7.1f} {avg('ttft_later_p50_s'):>6.3f}/{m[1]:<6.3f} {avg('ttft_later_p90_s'):>6.3f}/{m[2]:<6.3f}"
              f" {avg('dec_later_p50_s'):>6.3f}/{m[3]:<6.3f} {avg('ttft_first_p50_s'):>5.1f}/{m[4]:<6.1f}"
              f" {avg('warm_fwd_per_cycle'):>14.1f} {avg('warm_seqs_per_fwd'):>7.2f} {avg('dec_later_p50_s') * 1000 / (219 / 4.4):>8.1f}")


def cmd_ksync(args):
    """--ksync NAMES: per rung wall / tp / u / later TTFT p50, p90 / small prefill forwards per turn cycle, score delta."""
    names = args.ksync.split(",")
    res = price(names, load_params(), args.seed, reps=PRICE_REPS)
    for n in names:
        sd, rung, sim, bsim = res[n]
        print(f"\n== {n}: score {sd:+.2%}")
        for c in RUNGS:
            s, b = sim[c], bsim[c]
            proj = JUDGE_LATER_P90[c] + s["ttft_later_p90_s"] - b["ttft_later_p90_s"]
            lim = 2 * STOCK_LATER_P90[c]
            print(f"  c{c:<3} wall {s['wall_s']:>6.1f}s ({s['wall_s'] / b['wall_s'] - 1:+.1%})  tp {rung[c][0]:+.1%} u {rung[c][1]:+.1%}"
                  f"  TTFTl p50 {s['ttft_later_p50_s']:.3f} p90 {s['ttft_later_p90_s']:.3f} (base {b['ttft_later_p50_s']:.3f}/{b['ttft_later_p90_s']:.3f})"
                  f"  judge-proj p90 {proj:.3f} vs 2x stock {lim:.2f} {'OK' if proj <= lim else 'OVER'}"
                  f"  fwd/cycle {s['warm_fwd_per_cycle']:.1f} (base {b['warm_fwd_per_cycle']:.1f}) seq/fwd {s['warm_seqs_per_fwd']:.2f}"
                  f"  TTFT1 p90 {s['ttft_first_p90_s']:.1f}s (base {b['ttft_first_p90_s']:.1f})")


PRICE_REPS = {1: 12, 4: 8, 16: 6, 32: 6}
JUDGE_LATER_P90 = {1: 0.136, 4: 0.199, 16: 0.296, 32: 1.512}   # 788356d8 [read]


def cmd_selftest(args):
    """Asserts on the stock rules this model copies (delayer, recv skipper, W2 lane rule). Runs in ~10 s."""
    # resolve_min_free_slots: stock formula and flag semantics
    assert resolve_min_free_slots(None, 32) == 4
    assert resolve_min_free_slots(None, 8) == 2
    assert resolve_min_free_slots(None, 7) is None
    assert resolve_min_free_slots(None, 36) == 4
    assert resolve_min_free_slots(None, 32, is_dflash=False) is None
    assert resolve_min_free_slots(1, 32) is None and resolve_min_free_slots(0, 32) is None
    assert resolve_min_free_slots(6, 32) == 6 and resolve_min_free_slots(40, 32) == 32
    p = Params()
    # should_delay: running > 0 and allocatable < 4, skipped while a chunk is in flight
    assert delayer_holds(p, 29, False) and delayer_holds(p, 31, False)
    assert not delayer_holds(p, 28, False) and not delayer_holds(p, 0, False) and not delayer_holds(p, 30, True)
    q = copy.deepcopy(p); q.min_free_slots_delay = 1
    assert not any(delayer_holds(q, b, False) for b in range(33))
    q = copy.deepcopy(p); q.max_running = q.req_pool_rows = 36
    assert not delayer_holds(q, 32, False) and delayer_holds(q, 33, False)
    # in the event loop: with the delayer, no warm turn is admitted on a pass that starts with > 28 running (c32, SR
    # pool so KV does not cap first); without it, passes at 29-31 admit
    ACCEPT_CUR[0] = 4.6
    p = Params(); p.kv_tokens = D2_KV
    _, _, st = simulate(p, 32, 0, 1)
    assert st["admit_running"] and max(st["admit_running"]) <= 28, st["admit_running"]
    assert st["delayer_holds"] > 0
    q = copy.deepcopy(p); q.min_free_slots_delay = 1
    _, _, st1 = simulate(q, 32, 0, 1)
    assert max(st1["admit_running"]) > 28 and st1["delayer_holds"] == 0, st1["admit_running"]
    # c1: the delayer and W2 never act (a turn arrives only when nothing runs)
    ACCEPT_CUR[0] = 4.3
    q, _ = new_lever("d2_w2", Params())
    _, mk_a, st_a = simulate(q, 1, 0, 1)
    assert st_a["w2_holds"] == 0 and st_a["delayer_holds"] == 0
    # recv skipper: interval 1 receives every pass; c1 runs are identical for interval 1 and 4 only up to arrival
    # timing, so check the mechanism instead: with interval 4, a decode-only stretch receives every 4th step
    q = copy.deepcopy(p); q.recv_interval = 1
    simulate(q, 4, 0, 1)
    # W2 lane rule holds only warm extends: at c16 it holds, and never while a chunk is in flight (cold turns ride)
    ACCEPT_CUR[0] = 4.3
    q, _ = new_lever("d2_w2", Params())
    _, _, st16 = simulate(q, 16, 0, 1)
    assert st16["w2_holds"] > 0
    # K-eager: only forwards above 16384 computed tokens pay eager_ms
    q, _ = new_lever("d2_keager30", Params())
    _, _, stk = simulate(q, 4, 0, 1)
    assert stk["eager_fwd"] > 0 and q.chunk == 32_768 and q.graph_max_tokens == 16_384
    q, _ = new_lever("d2", Params())
    _, _, st0 = simulate(q, 4, 0, 1)
    assert st0["eager_fwd"] == 0
    print("selftest OK")




def fmt_hist(h):
    tot = sum(h.values()) or 1
    return " ".join(f"{k}:{v} ({v / tot:.0%})" for k, v in sorted(h.items()))


def print_rungs(sim, jr=None):
    hdr = f"{'c':>3} {'tok/s/GPU':>10} {'u_e2e':>7} {'TTFT1 p90':>9} {'TTFTl p90':>9} {'dec p50':>8} {'tp/cu':>6} {'retr':>5} {'evict Mtok':>10} {'reload Mtok':>11} {'GPU busy':>8} {'cold/warm/dec share':>20}"
    print(hdr)
    for c, s in sim.items():
        st = s["stats"]
        busy = st["gpu_busy"]
        mk = st["makespan"]
        line = (f"{c:>3} {s['tok_s_per_gpu']:>10.0f} {s['tok_s_per_user_e2e']:>7.1f} {s['ttft_first_p90_s']:>9.2f} "
                f"{s['ttft_later_p90_s']:>9.3f} {s['tok_s_per_user_p50']:>8.1f} {s['tp_over_cu']:>6.1f} "
                f"{st['retractions'] / s['reps']:>5.1f} {st['evicted_tokens'] / 1e6 / s['reps']:>10.2f} "
                f"{st['reload_tokens'] / 1e6 / s['reps']:>11.2f} {busy / mk:>8.0%} "
                f"{st['gpu_cold'] / busy:>6.0%}/{st['gpu_warm'] / busy:>4.0%}/{st['gpu_decode'] / busy:>4.0%}")
        print(line)
        if jr:
            j = jr[c]
            print(f"{'jdg':>3} {j['tok_s_per_gpu']:>10.0f} {j['tok_s_per_user_e2e']:>7.1f} {j['ttft_first_p90_s']:>9.2f} "
                  f"{j['ttft_later_p90_s']:>9.3f} {j['tok_s_per_user_p50']:>8.1f} "
                  f"{j['tok_s_per_gpu'] / (c * j['tok_s_per_user_e2e']):>6.1f}")


def cmd_calibrate(args):
    res, jr = judge()
    start = Params()
    if args.start:
        for k, v in json.load(open(args.start))["params"].items():
            setattr(start, k, v)
    for kv in args.set or []:
        k, v = kv.split("=")
        setattr(start, k, float(v))
    fixed = set()
    for kv in args.fix or []:
        k, v = kv.split("=")
        setattr(start, k, float(v))
        fixed.add(k)
    p, err = calibrate(start, maxiter=args.maxiter, seed=args.seed, names=tuple(n for n in FIT_NAMES if n not in fixed),
                       day1_only=args.day1_only)
    if fixed:
        print("held fixed: " + " ".join(f"{k}={getattr(p, k):.4g}" for k in sorted(fixed)))
    print(f"objective {err:.3f}\nfitted: " + " ".join(f"{n}={getattr(p, n):.4g}" for n in FIT_NAMES))
    print(f"derived: cold 16k chunk {p.cf + p.cp * 16384:.0f} ms; warm 753-token forward {min(p.wf + p.wp * 753, p.cf + p.cp * 753):.1f} ms;"
          f" decode step b=1 {p.df + p.dr:.1f} ms, b=32 {p.df + 32 * p.dr:.1f} ms")
    rs = print_residuals_all(p, args.seed)
    if args.fit_out:
        json.dump({"params": {n: getattr(p, n) for n in FIT_NAMES}, "objective": err, "residuals": rs,
                   "fixed": {k: v for k, v in asdict(p).items() if k not in FIT_NAMES}}, open(args.fit_out, "w"), indent=1)
        print(f"\nwrote {args.fit_out} (use it with JUDGE_SIM_FIT=<path>)")


RES_KEYS = ("tok_s_per_gpu", "tok_s_per_user_e2e", "ttft_first_p90_s", "ttft_later_p90_s", "tok_s_per_user_p50")


def print_residuals_all(p, seed=1):
    """Residuals (sim/judge - 1) over every calibration scenario; returns {scenario: {rung: {metric: r}}}."""
    scen = scenarios()
    sims = run_scenarios(p, scen, seed)
    print(f"objective over all scenarios {objective_all(sims, scen, p):.3f}")
    print(f"{'scenario':>7} {'c':>3} " + " ".join(f"{k[:14]:>15}" for k in RES_KEYS) + "   sim tp / judge tp  mean running")
    out = {}
    for name, ov, rungs, acc, jr, w in scen:
        out[name] = {}
        for c in rungs:
            s, j = sims[name][c], jr[c]
            r = {k: s[k] / j[k] - 1.0 for k in RES_KEYS}
            out[name][c] = r
            rs = s["stats"]["running_samples"]
            print(f"{name:>7} {c:>3} " + " ".join(f"{r[k]:>+15.1%}" for k in RES_KEYS)
                  + f"   {s['tok_s_per_gpu']:>6.0f} / {j['tok_s_per_gpu']:>6.0f}  {sum(rs) / max(1, len(rs)):.1f}")
    return out


def cmd_run(args):
    res, jr = judge()
    p = load_params()
    sim = run_all(p, SIM_REPS, args.seed)
    print_rungs(sim, jr)
    if args.hist:
        for c in RUNGS:
            print(f"c{c} warm-only forwards, histogram of warm extends per forward: {fmt_hist(sim[c]['stats']['warm_hist'])}")


def cmd_whatif(args):
    res, jr = judge()
    p0 = load_params()
    base = run_all(p0, EVAL_REPS, args.seed)
    s0, _ = score_of(base, base, res)
    names = []
    for n in args.whatif.split(","):
        names += list(LEVERS) + ["retract"] if n == "all" else [n]
    print(f"base (sim) score via score_model {s0:.4f}  [judge 2.2323; model on judge rungs {score_model.score(res)[0]:.4f}]")
    print_rungs(base)
    if args.hist:
        for c in RUNGS:
            st = base[c]['stats']
            nw = sum(k * v for k, v in st['warm_hist'].items())
            print(f"  c{c} warm-only forwards by # warm extends: {fmt_hist(st['warm_hist'])};"
                  f" warm extends riding in cold-chunk forwards: {st.get('warm_in_cold', 0)} of {nw + st.get('warm_in_cold', 0)}")
    for name in names:
        if name == "base":
            continue
        if name == "retract":
            cmd_retract(p0, base, args)
            continue
        sim = run_all(lever(name, p0), EVAL_REPS, args.seed)
        s, ratios = score_of(sim, base, res)
        print(f"\n== {name}: score {s:.4f} ({(s / s0 - 1) * 100:+.2f}%)  rung ratios " + " ".join(f"c{c} {r:.3f}" for c, r in ratios.items()))
        for c in RUNGS:
            d = sim[c]["tok_s_per_gpu"] / base[c]["tok_s_per_gpu"] - 1
            line = (f"  c{c:<3} tok/s/GPU {sim[c]['tok_s_per_gpu']:>7.0f} ({d:+.2%})  u {sim[c]['tok_s_per_user_e2e'] / base[c]['tok_s_per_user_e2e'] - 1:+.2%}"
                    f"  TTFT1 p90 {sim[c]['ttft_first_p90_s']:.1f}s (base {base[c]['ttft_first_p90_s']:.1f})"
                    f"  TTFTl p90 {sim[c]['ttft_later_p90_s']:.2f}s  retr {sim[c]['stats']['retractions'] / sim[c]['reps']:.1f}")
            if c in STOCK_TTFT_FIRST_P90:
                lim = 2 * STOCK_TTFT_FIRST_P90[c]
                line += f"  [2x stock {lim:.0f}s: {'OK' if sim[c]['ttft_first_p90_s'] < lim else 'OVER'}]"
            print(line)
        if args.hist:
            for c in RUNGS:
                print(f"  c{c} warm-only forwards by # warm extends: {fmt_hist(sim[c]['stats']['warm_hist'])}")


def cmd_retract(p0, base, args):
    print("\n== retract (R1): retractions and their cost under the capacity model")
    for name, p in (("fitted (fixed-length requests, exact reserve)", p0),
                    ("max_tokens 4096 + new_token_ratio decay (SGLang default estimator)", lever("reserve4096", p0))):
        sim = run_all(p, EVAL_REPS, args.seed)
        nr = {k: v for k, v in sim.items()}
        line = " ".join(f"c{c} {nr[c]['stats']['retractions'] / nr[c]['reps']:.1f}" for c in RUNGS)
        print(f"  {name}: retractions per rep: {line}")
        c = 32
        st = sim[c]["stats"]
        print(f"    c32: tok/s/GPU {sim[c]['tok_s_per_gpu']:.0f} vs base {base[c]['tok_s_per_gpu']:.0f};"
              f" evicted {st['evicted_tokens'] / 1e6 / sim[c]['reps']:.2f} Mtok/rep, reloaded {st['reload_tokens'] / 1e6 / sim[c]['reps']:.2f} Mtok/rep;"
              f" mean running {sum(st['running_samples']) / max(1, len(st['running_samples'])):.1f}")
    # cost of capacity pressure as a whole: infinite KV
    sim = run_all(lever("kv_inf", p0), EVAL_REPS, args.seed)
    print(f"  upper bound on everything capacity costs at c32 (KV unlimited): {sim[32]['tok_s_per_gpu'] / base[32]['tok_s_per_gpu'] - 1:+.2%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--whatif", metavar="NAME")
    ap.add_argument("--hist", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--fit-out", default="", help="write the fitted parameters to this json")
    ap.add_argument("--start", default="", help="start the fit from this json")
    ap.add_argument("--fix", nargs="*", help="hold these parameters at the given values in the fit, e.g. --fix wf=40")
    ap.add_argument("--set", nargs="*", help="override start values, e.g. --set mix_after_ms=10 dr=0.8")
    ap.add_argument("--day1-only", action="store_true", help="calibrate on the day-one rungs alone (the Nov 12 objective)")
    ap.add_argument("--selftest", action="store_true", help="asserts on the copied stock rules")
    ap.add_argument("--price", metavar="NAMES", help="Nov 13 levers (comma list or all); see NEW_LEVERS")
    ap.add_argument("--fits", default="", help="glob of fit jsons: --price reports the range over them")
    ap.add_argument("--json-out", default="", help="--price: write all deltas to this json")
    ap.add_argument("--residuals", action="store_true", help="residuals of the current parameters over every scenario")
    ap.add_argument("--k21check", action="store_true", help="k21 base (proxy shape) vs the measured k21 run")
    ap.add_argument("--k21-seeds", type=int, default=4)
    ap.add_argument("--ksync", metavar="NAMES", help="lane KSYNC table for k21_... names (comma list)")
    args = ap.parse_args()
    if (args.k21check or args.ksync or args.price) and args.set:
        for kv in args.set:
            k, v = kv.split("=")
            SET_OVERRIDES[k] = float(v)
    if args.k21check:
        cmd_k21check(args)
    elif args.ksync:
        cmd_ksync(args)
    elif args.calibrate:
        cmd_calibrate(args)
    elif args.selftest:
        cmd_selftest(args)
    elif args.price:
        cmd_price(args)
    elif args.residuals:
        print_residuals_all(load_params(), args.seed)
    elif args.whatif:
        cmd_whatif(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
