#!/usr/bin/env python3
"""Per-bucket anatomy of the DFlash decode step (draft graph + target-verify graph + eager tail + host loop)
from torch-profiler traces, per rank and aggregated. CPU only; reads the format of out/<ts>/prof/.

  python3 -I harness/decode_breakdown.py <prof_dir> [--tag c1] [--ranks 0,1] [--bs 4] [--json out.json] [--top 25]

<prof_dir> holds files named <tag>-<time>-TP-<r>-EP-<r>-DECODE.trace.json.gz (one per rank, written by
/start_profile with profile_by_stage). Every tag (c1, c4, ...) is reported separately.

Method (lane I's, made mechanical; see research/decode_deep_dive_2026-11-13.md section 1):
- Graphs are labelled by the main-stream `step[DRAFT bs=N]` / `step[VERIFY bs=N]` gpu_user_annotation their
  kernels fall in (majority vote per graph id); main stream = the stream with the most step[VERIFY annotations.
  The stack captures one draft and one verify graph per batch size (c4 runs mix bs=3 and bs=4), so this is per bs.
- A step starts at the first draft-graph kernel on the main stream after a verify-graph kernel and ends at the
  next one; it is kept when it holds exactly draft->verify of one bs (no MIXED/EXTEND graph) and its period is
  < 1.5 x the median for that bs. Every bs with >= 5 such steps on the first rank is reported (--bs to force).
  A trace window that holds only prefill (EXTEND/MIXED) reports "no decode steps".
- Buckets come from kernel names (first match wins, RULES below); kernels in the verify graph that run on a
  stream other than the main stream are reported as side-stream (the router GEMM in Muse).
- Bubbles: inside a graph span, time when no kernel runs on any stream. Side-only time: some side stream
  busy while the main stream idles (the main stream waits on an event from the side stream).
- All-reduce wait vs transport: the same AR call (step matched by start time across ranks, which share the host
  clock; same index in the verify graph) on every rank; PDL pre-start overlap with the preceding kernel is cut;
  the shortest duration over ranks is the floor (transport + the last arriver's own sync); wait = dur - floor.
  MoE vs attention ARs: an AR whose preceding main-stream kernels hit an expert kernel before an attention
  kernel is a MoE AR. Straggler: per MoE AR, expert compute since the previous AR, max over ranks - mean.
- Host: the scheduler thread (the one that records `step[DRAFT` on the CPU), per steady host step: CUDA
  runtime calls by name (graph launches split draft/verify by graph label), record_function annotations, and,
  when the trace has Python stacks, the named Python functions in HOST_PY (inclusive times; rows nest).
Numbers are profiled (with_stack adds ~10% host time; GPU time is unaffected).
"""
import bisect
import collections
import glob
import gzip
import json
import os
import re
import statistics
import sys

RULES = [
    ("moe_experts_nvfp4", ("marlin_moe", "Marlin<", "moe_fused_persistent", "moe_gather_kernel")),
    ("moe_experts_bf16", ("fused_moe_kernel",)),
    ("moe_glue", ("moe_align", "count_and_sort", "act_and_mul", "post_expert_norm", "moe_route_pairs",
                  "moe_sum", "finalize", "moe_combine")),
    ("router", ("routing_kernel", "s1688gemm")),
    ("allreduce", ("all_reduce", "allreduce", "AllReduce", "cross_device_reduce")),
    ("allgather", ("all_gather", "AllGather", "multimem")),
    ("attention_fa", ("flash::", "FlashAttn", "prepare_varlen", "flash_fwd", "fmha")),
    ("kv_write", ("store_kvcache", "set_kv_buffer", "kv_buffer", "hicache")),
    ("dense_gemm", ("nvjet", "gemm", "Gemm", "splitKreduce", "sm90_xmma")),
    ("norm", ("norm", "Norm", "rsqrt", "MeanOps", "pow_tensor")),
    ("convert_copy", ("ConvertTo", "bfloat16_copy", "direct_copy", "copy_kernel", "memcpy_kernel", "memcpy32")),
    ("spec_sample", ("argmax", "ArgMax", "topk", "TopK", "sampling", "dflash", "accept", "sbtopk", "bitonic",
                     "selector", "softcap", "embedding")),
    ("memcpy_memset", ("Memcpy", "Memset")),
    ("elementwise_other", ("",)),
]
MOE_BUCKETS = ("moe_experts_nvfp4", "moe_experts_bf16", "moe_glue")
HOST_PY = [  # (label, regex on the python_function name); inclusive times, rows can nest
    ("graph replay: DecodeCudaGraphRunner.execute", r"decode_cuda_graph_runner\.py\(\d+\): execute$"),
    ("graph replay: load_batch (static-buffer copy-in)", r"cuda_graph_runner\.py\(\d+\): load_batch$"),
    ("attn metadata: init_forward_metadata*", r"attention/.*\.py\(\d+\): (init_forward_metadata\w*|_init_full_cg\w*)$"),
    ("ForwardBatch.init_new", r"forward_batch_info\.py\(\d+\): init_new$"),
    ("dflash prepare_for_verify", r"dflash_info\.py\(\d+\): prepare_for_verify$"),
    ("dflash prepare_for_decode", r"dflash_info_v2\.py\(\d+\): prepare_for_decode$"),
    ("noise embedding (eager)", r"_NormalizedEmbedding|noise_embedding"),
    ("deferred_argmax (gather + all-gather)", r"logits_processor\.py\(\d+\): deferred_argmax$"),
    ("_accept_block", r"dflash_worker\w*\.py\(\d+\): _accept_block$"),
    ("draft-KV append (_append_target_hidden*)", r"dflash_worker\w*\.py\(\d+\): _append_target_hidden_to_draft_kv\w*$"),
    ("future-map publish", r"overlap_utils\.py\(\d+\): publish$"),
    ("process_batch_result_decode", r"batch_result_processor\.py\(\d+\): process_batch_result_decode$"),
    ("HiCache ready-count sync", r"unified_radix_cache\.py\(\d+\): _sync_hicache_ready_counts$"),
    ("HiCache flush_pending_backups", r"unified_radix_cache\.py\(\d+\): flush_pending_backups$"),
    ("request broadcast across ranks", r"request_receiver\.py\(\d+\): _broadcast_reqs_across_ranks$"),
    ("update_running_batch", r"scheduler\.py\(\d+\): update_running_batch$"),
]
HOST_ANN = ("scheduler.run_batch", "scheduler.get_next_batch_to_run", "scheduler.process_batch_result",
            "scheduler.recv_requests", "scheduler.process_input_requests", "gloo:all_reduce", "gloo:broadcast",
            "nccl:_all_gather_base", "nccl:broadcast", "nccl:all_reduce", "copy_result_to_cpu", "step[DRAFT]",
            "step[VERIFY]")


def bucket(name):
    for b, keys in RULES:
        if any(k in name for k in keys):
            return b
    return "elementwise_other"


def union_len(iv):
    """Total length of the union of [s, e) intervals."""
    tot, cs, ce = 0.0, None, None
    for s, e in sorted(iv):
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + (ce - cs if ce is not None else 0.0)


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def norm_ann(name):
    return re.sub(r"^(step\[\w+).*", r"\1]", name)


_CACHE = {}


def load_rank(path):
    if path in _CACHE:
        return _CACHE[path]
    with gzip.open(path, "rt") as f:
        ev = json.load(f)["traceEvents"]
    kern, rt, gua, ua, py, cpu = [], [], [], [], [], []
    for e in ev:
        c = e.get("cat")
        if e.get("ph") != "X" or "dur" not in e:
            continue
        if c in ("kernel", "gpu_memcpy", "gpu_memset"):
            kern.append(e)
        elif c in ("cuda_runtime", "cuda_driver"):
            rt.append(e)
        elif c == "gpu_user_annotation":
            gua.append(e)
        elif c == "user_annotation":
            ua.append(e)
        elif c == "python_function":
            py.append(e)
        elif c == "cpu_op":
            cpu.append(e)
    kern.sort(key=lambda e: e["ts"])
    _CACHE[path] = (kern, rt, gua, ua, py, cpu)
    return _CACHE[path]


STEP_RE = re.compile(r"^step\[(\w+) bs=(\d+)")


def classify_graphs(kern, gua):
    """Graph id -> (kind, bs) from the main-stream step[...] gpu annotations its kernels fall in (majority vote).
    Main stream = the stream carrying the most step[VERIFY gpu annotations (side streams get partial mirrors)."""
    vt = collections.Counter(e["tid"] for e in gua if e["name"].startswith("step[VERIFY"))
    if not vt:
        vt = collections.Counter(e["tid"] for e in gua if e["name"].startswith("step["))
    main = vt.most_common(1)[0][0] if vt else collections.Counter(e["tid"] for e in kern).most_common(1)[0][0]
    anns = sorted([e for e in gua if e["tid"] == main and STEP_RE.match(e["name"])], key=lambda e: e["ts"])
    ats = [a["ts"] for a in anns]
    votes = collections.defaultdict(collections.Counter)
    for e in kern:
        g = e.get("args", {}).get("graph id")
        if not g:
            continue
        i = bisect.bisect_right(ats, e["ts"]) - 1
        if i >= 0 and e["ts"] < anns[i]["ts"] + anns[i]["dur"]:
            m = STEP_RE.match(anns[i]["name"])
            votes[g][(m.group(1), int(m.group(2)))] += 1
    gkind = {}
    for g, c in votes.items():
        (kind, bs), n = c.most_common(1)[0]
        gkind[g] = ({"VERIFY": "verify", "DRAFT": "draft"}.get(kind, kind.lower()), bs)
    return main, gkind, anns


def analyse_rank(path, top, want_bs=None):
    kern, rt, gua, ua, py, cpu = load_rank(path)
    main, gkind, anns = classify_graphs(kern, gua)
    gid_of_corr = {}
    by_corr = collections.Counter()
    for e in kern:
        c = e["args"].get("correlation")
        by_corr[c] += 1
        if e["args"].get("graph id"):
            gid_of_corr[c] = e["args"]["graph id"]
    kind_of_corr = {c: gkind.get(g, ("graph_other", 0))[0] for c, g in gid_of_corr.items()}
    gnodes = collections.defaultdict(list)
    for e in rt:
        if e["name"] == "cudaGraphLaunch":
            c = e["args"]["correlation"]
            if c in gid_of_corr:
                gnodes[gid_of_corr[c]].append(by_corr[c])
    nodes = {g: max(v) for g, v in gnodes.items()}
    # step starts: the first draft-graph kernel on the main stream after a verify-graph kernel (a step's
    # draft phase can span two step[DRAFT] annotations and launches); bs = the draft graph's bs
    seq = []  # (ts, kind, bs, graph id) for graph kernels on the main stream, phase changes only
    for e in kern:
        if e["tid"] != main:
            continue
        g = e["args"].get("graph id")
        if g in gkind and gkind[g][0] in ("draft", "verify"):
            k, bs = gkind[g]
            if not seq or seq[-1][1] != k:
                seq.append((e["ts"], k, bs, g))
    starts_all = [(ts, bs) for ts, k, bs, g in seq if k == "draft"]
    bss = collections.Counter(bs for _, bs in starts_all)
    if want_bs is None:
        want_bs = bss.most_common(1)[0][0] if bss else None
    vgr = [g for g, (k, b) in gkind.items() if k == "verify" and b == want_bs]
    dgr = [g for g, (k, b) in gkind.items() if k == "draft" and b == want_bs]
    verify_n = max((nodes.get(g, 0) for g in vgr), default=None)
    draft_n = max((nodes.get(g, 0) for g in dgr), default=None)
    # candidate steps: consecutive draft starts; keep only windows whose draft and verify both have bs=want_bs
    # and that hold no prefill/mixed graph kernels
    cand = []
    for (a, ba), (b, bb) in zip(starts_all, starts_all[1:]):
        if ba != want_bs:
            continue
        ph = [x for x in seq if a <= x[0] < b]
        if [x[1] for x in ph] != ["draft", "verify"] or ph[1][2] != want_bs:
            continue
        cand.append((a, b))
    pm = med([b - a for a, b in cand])
    steps = [(a, a, b) for a, b in cand if b - a < 1.5 * pm]  # (key = start ts, used for cross-rank matching)
    starts = [a for a, _ in starts_all]
    bs_label = f"step bs={want_bs} (bs mix of draft starts: {dict(bss)})"
    kts = [e["ts"] for e in kern]

    per = collections.defaultdict(list)  # metric -> per-step values (us)
    bseg = collections.defaultdict(lambda: collections.defaultdict(float))  # bucket -> seg -> total us
    bcnt = collections.defaultdict(lambda: collections.defaultdict(int))
    names = collections.defaultdict(lambda: [0, 0.0])
    eager_names = collections.defaultdict(lambda: [0, 0.0])
    ar_calls = {}  # step index -> [(dur, class, moe_compute_before)]
    d2h = [0, 0.0]
    pdl = [0.0]
    for si, a, b in steps:
        ks = kern[bisect.bisect_left(kts, a):bisect.bisect_left(kts, b)]
        comp = [e for e in ks if not (e["tid"] != main and e.get("cat") == "gpu_memcpy")]
        side_copies = [e for e in ks if e["tid"] != main and e.get("cat") == "gpu_memcpy"]
        d2h[0] += len(side_copies)
        d2h[1] += sum(e["dur"] for e in side_copies)
        per["period"].append(b - a)
        per["gpu_busy"].append(union_len([(e["ts"], e["ts"] + e["dur"]) for e in comp]))
        segk = collections.defaultdict(list)
        for e in comp:
            kind = kind_of_corr.get(e["args"].get("correlation"), "eager")
            seg = kind if kind in ("verify", "draft") else "eager"
            if seg == "verify" and e["tid"] != main:
                seg = "verify_side"
            segk[seg].append(e)
            bk = bucket(e["name"])
            bseg[bk][seg] += e["dur"]
            bcnt[bk][seg] += 1
            nk = (seg, e["name"][:90])
            names[nk][0] += 1
            names[nk][1] += e["dur"]
        for g in ("verify", "draft"):
            ksg = segk[g] + (segk["verify_side"] if g == "verify" else [])
            if not ksg:
                continue
            s0 = min(e["ts"] for e in ksg)
            s1 = max(e["ts"] + e["dur"] for e in ksg)
            allu = union_len([(e["ts"], e["ts"] + e["dur"]) for e in ksg])
            mainu = union_len([(e["ts"], e["ts"] + e["dur"]) for e in ksg if e["tid"] == main])
            per[f"{g}_span"].append(s1 - s0)
            per[f"{g}_busy"].append(allu)
            per[f"{g}_bubbles"].append(s1 - s0 - allu)
            per[f"{g}_side_only"].append(allu - mainu)
            per[f"{g}_kernels"].append(len(ksg))
        per["eager_kernels"].append(len(segk["eager"]))
        per["eager_kernel_us"].append(sum(e["dur"] for e in segk["eager"]))
        for e in segk["eager"]:
            eager_names[e["name"][:90]][0] += 1
            eager_names[e["name"][:90]][1] += e["dur"]
        # AR sequence in the verify graph (main stream), with class and expert compute since the previous AR
        seq = sorted(segk["verify"], key=lambda e: e["ts"])
        calls, moe_acc, last, prev_end = [], 0.0, [], None
        for e in seq:
            bk = bucket(e["name"])
            # PDL: an AR launched with programmatic dependent launch starts before its predecessor ends; that
            # pre-start overlap is not AR time (it is counted in the predecessor), so it is cut from the AR dur
            ov = max(0.0, min(e["dur"], prev_end - e["ts"])) if prev_end is not None else 0.0
            prev_end = max(prev_end or 0.0, e["ts"] + e["dur"])
            if bk == "allreduce":
                pdl[0] += ov
                cls = "other"
                for p in reversed(last[-40:]):
                    if p in MOE_BUCKETS:
                        cls = "moe"
                        break
                    if p == "attention_fa":
                        cls = "attn"
                        break
                calls.append((e["dur"] - ov, cls, moe_acc))
                moe_acc, last = 0.0, []
            else:
                if bk in MOE_BUCKETS:
                    moe_acc += e["dur"]
                last.append(bk)
        ar_calls[si] = calls

    # host side
    host = collections.defaultdict(list)
    # host steps: on the scheduler thread, a step[DRAFT bs=want_bs] that follows a non-DRAFT step annotation, to the
    # next such; kept when the window holds exactly DRAFT..VERIFY of that bs (no MIXED/EXTEND) and is steady
    cst = sorted([e for e in ua if STEP_RE.match(e["name"])], key=lambda e: e["ts"])
    tc = collections.Counter(e["tid"] for e in cst if e["name"].startswith("step[DRAFT"))
    sched_tid = tc.most_common(1)[0][0] if tc else None
    cst = [(e["ts"], STEP_RE.match(e["name"]).group(1), int(STEP_RE.match(e["name"]).group(2))) for e in cst
           if e["tid"] == sched_tid]
    hs = [i for i, x in enumerate(cst) if x[1] == "DRAFT" and (i == 0 or cst[i - 1][1] != "DRAFT")]
    hc = []
    for i, j in zip(hs, hs[1:]):
        kinds = {(k, b) for _, k, b in cst[i:j]}
        if kinds == {("DRAFT", want_bs), ("VERIFY", want_bs)}:
            hc.append((cst[i][0], cst[j][0]))
    cpm = med([b - a for a, b in hc])
    hsteps = [(a, b) for a, b in hc if b - a < 1.5 * cpm]
    pyre = [(lab, re.compile(rx)) for lab, rx in HOST_PY]
    py_lab = []
    for e in py:
        for lab, rx in pyre:
            if rx.search(e["name"]):
                py_lab.append((e["ts"], e["dur"], lab, e["tid"]))
                break
    py_lab.sort()
    pts = [x[0] for x in py_lab]
    rt_s = sorted(rt, key=lambda e: e["ts"])
    rts = [e["ts"] for e in rt_s]
    ua_s = sorted(ua, key=lambda e: e["ts"])
    uts = [e["ts"] for e in ua_s]
    for a, b in hsteps:
        acc = collections.defaultdict(float)
        cnt = collections.defaultdict(int)
        for e in rt_s[bisect.bisect_left(rts, a):bisect.bisect_left(rts, b)]:
            who = "sched" if e["tid"] == sched_tid else "other_thr"
            nm = e["name"]
            if nm == "cudaGraphLaunch":
                nm += "[" + kind_of_corr.get(e["args"].get("correlation"), "?") + "]"
            acc[f"rt {who} {nm}"] += e["dur"]
            cnt[f"rt {who} {nm}"] += 1
        for e in ua_s[bisect.bisect_left(uts, a):bisect.bisect_left(uts, b)]:
            if e["tid"] == sched_tid:
                acc["ann " + norm_ann(e["name"])] += e["dur"]
                cnt["ann " + norm_ann(e["name"])] += 1
        for ts, dur, lab, tid in py_lab[bisect.bisect_left(pts, a):bisect.bisect_left(pts, b)]:
            if tid == sched_tid:
                acc["py " + lab] += dur
                cnt["py " + lab] += 1
        host["__period"].append(b - a)
        for k in set(acc) | set(host):
            if k != "__period":
                host[k].append(acc.get(k, 0.0))
                host["#" + k].append(cnt.get(k, 0))
    # pad rows that appeared late
    nh = len(host["__period"])
    for k in list(host):
        if len(host[k]) < nh:
            host[k] = [0.0] * (nh - len(host[k])) + host[k]
    n = max(1, len(steps))
    return {
        "path": os.path.basename(path), "bs_label": bs_label, "bs": want_bs, "bs_mix": dict(bss),
        "step_candidates": len(cand), "main_stream": main, "verify_nodes": verify_n,
        "draft_nodes": draft_n, "steps_all": len(starts), "steps_steady": len(steps), "step_index": [s[0] for s in steps],
        "per": {k: v for k, v in per.items()},
        "buckets": {bk: {sg: us / n for sg, us in segs.items()} for bk, segs in bseg.items()},
        "bucket_counts": {bk: {sg: c / n for sg, c in segs.items()} for bk, segs in bcnt.items()},
        "top_kernels": sorted(((k[0], k[1], c / n, us / n) for k, (c, us) in names.items()), key=lambda r: -r[3])[:top],
        "eager_top": sorted(((k, c / n, us / n) for k, (c, us) in eager_names.items()), key=lambda r: -r[2])[:12],
        "side_copies_per_step": (d2h[0] / n, d2h[1] / n), "ar_pdl_overlap_per_step": pdl[0] / n,
        "ar_calls": ar_calls, "host": dict(host), "host_steps": nh, "has_python": bool(py),
    }


def ar_wait(ranks):
    """Cross-rank AR floor/wait per call. Steps are matched by start time (ranks share the host clock): each
    rank's step nearest to the reference rank's step start, within 25% of the period."""
    ref = min(ranks, key=lambda rk: rk)
    out = {rk: collections.defaultdict(list) for rk in ranks}
    floors = collections.defaultdict(list)
    strag = []
    used = 0
    keys = {rk: sorted(r["ar_calls"]) for rk, r in ranks.items()}
    for k0 in keys[ref]:
        tol = 0.25 * med(ranks[ref]["per"]["period"])
        seqs = {}
        for rk, ks in keys.items():
            i = bisect.bisect_left(ks, k0)
            best = min((ks[j] for j in (i - 1, i) if 0 <= j < len(ks)), key=lambda x: abs(x - k0), default=None)
            if best is None or abs(best - k0) > tol:
                break
            seqs[rk] = ranks[rk]["ar_calls"][best]
        if len(seqs) != len(ranks):
            continue
        lens = {len(s) for s in seqs.values()}
        if len(lens) != 1:
            continue
        used += 1
        L = lens.pop()
        step_tot = {rk: collections.defaultdict(float) for rk in ranks}
        step_strag = 0.0
        for j in range(L):
            durs = {rk: s[j][0] for rk, s in seqs.items()}
            cls = collections.Counter(s[j][1] for s in seqs.values()).most_common(1)[0][0]
            fl = min(durs.values())
            floors[cls].append(fl)
            for rk, d in durs.items():
                step_tot[rk][cls + "_dur"] += d
                step_tot[rk][cls + "_wait"] += d - fl
                step_tot[rk][cls + "_n"] += 1
            if cls == "moe":
                comp = [s[j][2] for s in seqs.values()]
                step_strag += max(comp) - sum(comp) / len(comp)
        for rk in ranks:
            for k, v in step_tot[rk].items():
                out[rk][k].append(v)
        strag.append(step_strag)
    return out, floors, strag, used


def fmt_ms(us):
    return f"{us / 1e3:7.3f}"


def report(tag, ranks, top, as_json):
    print(f"\n######## {tag}: {len(ranks)} rank(s), {next(iter(ranks.values()))['bs_label']}")
    rk0 = sorted(ranks)[0]
    r0 = ranks[rk0]
    if not any(r["steps_steady"] for r in ranks.values()):
        print("  no decode steps (no step[DRAFT]..step[VERIFY] pair of one bs) in this trace window: "
              "it holds prefill/mixed steps only; nothing to report")
        return
    print(f"main stream {r0['main_stream']}, verify graph {r0['verify_nodes']} nodes, draft graph {r0['draft_nodes']} nodes, "
          f"steps {r0['steps_all']} ({r0['steps_steady']} steady); python stacks in trace: {r0['has_python']}")

    def agg(fn):
        vals = [fn(r) for r in ranks.values()]
        vals = [v for v in vals if v == v]
        return (med(vals), min(vals), max(vals)) if vals else (float("nan"),) * 3

    print("\n## 1. Step timing (ms per steady step; median over steps, then median [min..max] over ranks)")
    rows = [("period (draft start to next draft start)", "period"), ("GPU busy, all streams (excl. side-stream D2H copies)", "gpu_busy"),
            ("draft graph span", "draft_span"), ("  draft graph busy", "draft_busy"), ("  draft graph bubbles", "draft_bubbles"),
            ("verify graph span", "verify_span"), ("  verify graph busy (union of streams)", "verify_busy"),
            ("  verify bubbles (no stream busy)", "verify_bubbles"), ("  verify side-stream-only time (main waits)", "verify_side_only"),
            ("eager kernels: summed kernel time", "eager_kernel_us")]
    for lab, key in rows:
        m, lo, hi = agg(lambda r: med(r["per"].get(key, [])))
        print(f"| {lab:<52} | {fmt_ms(m)} | [{fmt_ms(lo)} .. {fmt_ms(hi)}] |")
    m, lo, hi = agg(lambda r: med(r["per"]["period"]) - med(r["per"]["gpu_busy"]))
    print(f"| {'GPU idle inside the step (period - busy)':<52} | {fmt_ms(m)} | [{fmt_ms(lo)} .. {fmt_ms(hi)}] |")
    for key, lab in (("verify_kernels", "verify kernels/step"), ("draft_kernels", "draft kernels/step"), ("eager_kernels", "eager kernels/step")):
        m, lo, hi = agg(lambda r: med(r["per"].get(key, [])))
        print(f"| {lab:<52} | {m:7.0f} | [{lo:7.0f} .. {hi:7.0f}] |")
    m, lo, hi = agg(lambda r: r["side_copies_per_step"][0])
    m2, _, _ = agg(lambda r: r["side_copies_per_step"][1])
    print(f"| {'side-stream memcpys/step (HiCache D2H), count / ms':<52} | {m:7.0f} / {m2 / 1e3:.3f} ms |")

    print("\n## 2. GPU time by bucket (ms per step, summed kernel durations; rank-median; side = verify kernels off the main stream)")
    segs = ("verify", "verify_side", "draft", "eager")
    allb = sorted({b for r in ranks.values() for b in r["buckets"]},
                  key=lambda b: -sum(agg(lambda r: r["buckets"].get(b, {}).get(s, 0.0))[0] for s in segs))
    print(f"| {'bucket':<20} | " + " | ".join(f"{s:>11}" for s in segs) + " |   total | n/step (verify) | max-rank total |")
    tots = collections.defaultdict(float)
    for b in allb:
        vals = [agg(lambda r: r["buckets"].get(b, {}).get(s, 0.0))[0] for s in segs]
        hi = agg(lambda r: sum(r["buckets"].get(b, {}).values()))[2]
        cnt = agg(lambda r: r["bucket_counts"].get(b, {}).get("verify", 0) + r["bucket_counts"].get(b, {}).get("verify_side", 0))[0]
        for s, v in zip(segs, vals):
            tots[s] += v
        print(f"| {b:<20} | " + " | ".join(f"{fmt_ms(v):>11}" for v in vals) + f" | {fmt_ms(sum(vals))} | {cnt:15.0f} | {fmt_ms(hi)} |")
    print(f"| {'SUM':<20} | " + " | ".join(f"{fmt_ms(tots[s]):>11}" for s in segs) + f" | {fmt_ms(sum(tots.values()))} | | |")
    m, lo, hi = agg(lambda r: r.get("ar_pdl_overlap_per_step", 0.0))
    print(f"  note: allreduce above includes {fmt_ms(m).strip()} ms/step [{fmt_ms(lo).strip()}..{fmt_ms(hi).strip()}] of PDL pre-start "
          f"overlap with the preceding kernel (double-counted with that kernel; section 4 removes it)")

    print(f"\n## 3. Top kernels, rank {rk0} (segment, calls/step, ms/step, us/call)")
    for seg, nm, c, us in r0["top_kernels"][:top]:
        print(f"  {seg:<11} {c:7.1f} {fmt_ms(us)} {us / max(c, 1e-9):7.1f}  {nm}")
    print(f"  eager kernels, rank {rk0}:")
    for nm, c, us in r0["eager_top"]:
        print(f"  eager       {c:7.1f} {fmt_ms(us)} {us / max(c, 1e-9):7.1f}  {nm}")

    print("\n## 4. All-reduce in the verify graph: floor (min over ranks per call) vs wait (dur - floor), ms per step;"
          " dur excludes PDL pre-start overlap")
    if len(ranks) < 2:
        print("  needs >= 2 ranks")
    else:
        out, floors, strag, used = ar_wait(ranks)
        print(f"  steps matched across ranks: {used}")
        print("| rank | " + " | ".join(f"{c}: n, dur, wait" for c in ("moe", "attn", "other")) + " |")
        for rk in sorted(out):
            cells = []
            for c in ("moe", "attn", "other"):
                d = out[rk]
                cells.append(f"{med(d[c + '_n']):4.0f} {fmt_ms(med(d[c + '_dur']))} {fmt_ms(med(d[c + '_wait']))}")
            print(f"| {rk} | " + " | ".join(cells) + " |")
        for c, fl in floors.items():
            print(f"  floor per {c} AR call: median {med(fl):.1f} us, p10 {sorted(fl)[len(fl) // 10]:.1f}, p90 {sorted(fl)[9 * len(fl) // 10]:.1f}")
        print(f"  MoE straggler (sum over MoE ARs of max-rank minus mean-rank expert compute since the previous AR): "
              f"median {med(strag) / 1e3:.3f} ms/step")

    print("\n## 5. Host, scheduler thread (ms per steady host step; inclusive; rank-median [min..max]; '#'=calls/step)")
    m, lo, hi = agg(lambda r: med(r["host"].get("__period", [])))
    print(f"| {'host step period':<62} | {fmt_ms(m)} | [{fmt_ms(lo)} .. {fmt_ms(hi)}] | |")
    keys = sorted({k for r in ranks.values() for k in r["host"] if not k.startswith("#") and k != "__period"},
                  key=lambda k: -agg(lambda r: med(r["host"].get(k, [0.0])))[0])
    for k in keys:
        m, lo, hi = agg(lambda r: med(r["host"].get(k, [0.0])))
        c = agg(lambda r: med(r["host"].get("#" + k, [0])))[0]
        if m < 5 and hi < 20:
            continue
        print(f"| {k[:62]:<62} | {fmt_ms(m)} | [{fmt_ms(lo)} .. {fmt_ms(hi)}] | #{c:.0f} |")
    if as_json is not None:
        as_json[tag] = {rk: {k: v for k, v in r.items() if k not in ("ar_calls",)} for rk, r in ranks.items()}


def main():
    a = sys.argv[1:]
    if not a or a[0].startswith("-"):
        print(__doc__)
        sys.exit(1)
    d = a[0]
    opt = lambda k, dflt=None: a[a.index(k) + 1] if k in a else dflt
    tag_f, ranks_f, top = opt("--tag"), opt("--ranks"), int(opt("--top", "25"))
    files = sorted(glob.glob(os.path.join(d, "*DECODE*.trace.json.gz")))
    groups = collections.defaultdict(dict)
    for f in files:
        b = os.path.basename(f)
        tag = b.split("-")[0]
        m = re.search(r"-TP-(\d+)", b)
        rk = int(m.group(1)) if m else 0
        if tag_f and tag != tag_f:
            continue
        if ranks_f and rk not in {int(x) for x in ranks_f.split(",")}:
            continue
        groups[tag][rk] = f
    if not groups:
        print(f"no *DECODE*.trace.json.gz under {d}")
        sys.exit(1)
    js = {} if opt("--json") else None
    for tag in sorted(groups):
        rk0 = min(groups[tag])
        print(f"  loading {tag} rank {rk0} ...", file=sys.stderr, flush=True)
        first = analyse_rank(groups[tag][rk0], top)
        if opt("--bs"):
            bss = [int(x) for x in opt("--bs").split(",")]
        else:  # every bs with >= 5 clean draft->verify steps on the first rank, most common first
            bss = [b for b, n in sorted(first["bs_mix"].items(), key=lambda x: -x[1]) if n >= 5] or [first["bs"]]
        for bs in bss:
            ranks = {}
            for rk, f in sorted(groups[tag].items()):
                print(f"  analysing {tag} bs={bs} rank {rk} ...", file=sys.stderr, flush=True)
                ranks[rk] = analyse_rank(f, top, bs)
            report(f"{tag} bs={bs}", ranks, top, js)
        _CACHE.clear()
    if js is not None:
        with open(opt("--json"), "w") as f:
            json.dump(js, f, indent=1, default=str)


if __name__ == "__main__":
    main()
