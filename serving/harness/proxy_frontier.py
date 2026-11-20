#!/usr/bin/env python3
"""Read a proxy A/B the way the judge scores: per-rung raw ratios AND frontier ratios ("throughput at stock's per-user
speed", score_model.frontier_at), from ladder.py JSONs. Needs the judge's four concurrencies in both runs (the frontier
at stock's c16 speed reads OUR c32 point), e.g. --rungs 32x1x1,16x1x1,4x1x2,1x1x2.

  python3 harness/proxy_frontier.py <stock proxy.json> <cand proxy.json> [--vs <ctl proxy.json>] [--judge out/judge/status_<scored>.json] [--host-ms auto]

Per run and concurrency, like the judge's rung fields:
  tok_s_per_gpu      = (prompt + completion tokens) / wall / 8   (the judge counts cached prompt tokens too: its c32
                       stock 12,645 tok/s/GPU = 32 x 1.04M tokens / ~330 s / 8)
  tok_s_per_user_e2e = mean over conversations of output tokens / (last turn end - first turn start)
  hit                = cached / prompt tokens (the judge's cache_hit; 0.92 for the judge shape when nothing is evicted)
--judge prints the scored run's per-rung ratios next to ours and flags |proxy/judge - 1| > 5% (the validation criterion).
--vs prints cand/ctl per rung (raw and frontier-score delta), e.g. P vs FGHIL+D1; with --judge also the judge-anchored
  prediction (scored per-rung ratio x cand/ctl), the number to quote for a candidate when ctl is the scored build.

--host-ms: Modal host-overhead correction for the latency-bound rungs (proxy_rebase_2026-11-15.md section 6).
  Modal's API process adds a fixed per-turn host cost (chat template + full-context re-tokenize, serial, ~120-150 ms,
  lane N) that the judge's node does not pay (judge later-turn TTFT p90 140 ms at c1 vs our 295 ms, same build). Both
  stock and candidate pay it, so at c1/c4 it dilutes OUR ratio (1.41 vs judge 1.61 raw at c1). The option subtracts
  H seconds per turn from every conversation's critical path (span and rep wall) on BOTH runs, at --host-rungs only
  (default 1,4: at c16/c32 the host work overlaps GPU-bound batches and the proxy already matches the judge).
    --host-ms 155           one H for every corrected rung
    --host-ms c1:155,c4:209 per rung
    --host-ms auto          per rung, H = cand's later-turn TTFT p90 - the --judge run's ttft_later_p90 (use it with
                            cand = the scored build, i.e. ctl; then reuse the printed H for other A/Bs on that node)
  This corrects only the per-turn (TTFT-side) host cost. Modal also decodes slower per step at c1 (ctl 296 tok/s on
  later turns vs judge p50 327 at a HIGHER acceptance, 4.9 vs 4.4), a per-step host cost the option does not model.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_model import frontier_at  # noqa: E402

RUNGS = (1, 4, 16, 32)


def parse_host(spec, rungs):
    """--host-ms spec -> {conc: seconds}. 'auto' is resolved later from the judge JSON."""
    if not spec or spec == "auto":
        return {}
    if ":" not in spec:
        return {c: float(spec) / 1e3 for c in rungs}
    out = {}
    for part in spec.split(","):
        k, v = part.split(":")
        out[int(k.lstrip("c"))] = float(v) / 1e3
    return out


def p90(xs):
    xs = sorted(xs)
    return xs[int(0.9 * (len(xs) - 1) + 0.5)] if xs else 0.0


def later_ttft_p90(path):
    by = collections.defaultdict(list)
    for r in json.load(open(path))["records"]:
        if r.get("ttft") is not None and r.get("completion_tokens") and not r["key"].endswith("/t0"):
            by[int(r["key"].split("/")[0][1:])].append(r["ttft"])
    return {c: p90(v) for c, v in by.items()}


def rung_stats(path, host=None):
    """host: {conc: seconds per turn} removed from each conversation's critical path (see --host-ms)."""
    host = host or {}
    recs = json.load(open(path))["records"]
    by_rep = collections.defaultdict(list)
    for r in recs:
        if r.get("ttft") is None or not r.get("completion_tokens"):
            continue
        conc, rep = r["key"].split("/")[:2]
        by_rep[(int(conc[1:]), rep)].append(r)
    out = {}
    for c in sorted({k[0] for k in by_rep}):
        toks = wall = prompt = cached = 0.0
        users = []
        h = host.get(c, 0.0)
        for (cc, _), rs in by_rep.items():
            if cc != c:
                continue
            t0 = min(r["t_start"] for r in rs)
            conv = collections.defaultdict(list)
            for r in rs:
                conv[r["key"].rsplit("/t", 1)[0]].append(r)
            # rep wall = latest user-session end (a session s<k> runs its conversations back to back), each session
            # shortened by h per turn it ran
            sess = collections.defaultdict(list)
            for r in rs:
                sess[r["key"].split("/")[2]].append(r)
            wall += max(max(r["t_start"] + r["latency"] for r in sr) - t0 - h * len(sr) for sr in sess.values())
            toks += sum((r.get("prompt_tokens") or 0) + r["completion_tokens"] for r in rs)
            prompt += sum(r.get("prompt_tokens") or 0 for r in rs)
            cached += sum(r.get("cached_tokens") or 0 for r in rs)
            for cr in conv.values():
                span = max(r["t_start"] + r["latency"] for r in cr) - min(r["t_start"] for r in cr) - h * len(cr)
                if span > 0:
                    users.append(sum(r["completion_tokens"] for r in cr) / span)
        out[c] = {"tok_s_per_gpu": toks / wall / 8 if wall else 0.0, "u": sum(users) / len(users) if users else 0.0,
                  "hit": cached / prompt if prompt else 0.0}
    return out


def frontier(stock, cand):
    pts = sorted(((v["u"], v["tok_s_per_gpu"], c) for c, v in cand.items()), key=lambda p: -p[0])
    return {c: frontier_at(pts, stock[c]["u"]) / stock[c]["tok_s_per_gpu"] for c in RUNGS if c in stock and c in cand}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stock")
    ap.add_argument("cand")
    ap.add_argument("--vs", help="a control run (same stock) to read cand against")
    ap.add_argument("--judge", help="a scored judge JSON whose frontier/rung ratios the proxy should reproduce")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--host-ms", help="per-turn Modal host-overhead correction: N | c1:N,c4:M | auto (needs --judge)")
    ap.add_argument("--host-rungs", default="1,4", help="rungs the --host-ms correction applies to (default 1,4)")
    a = ap.parse_args()
    hr = [int(x) for x in a.host_rungs.split(",") if x]
    host = {k: v for k, v in parse_host(a.host_ms, hr).items() if k in hr}
    if a.host_ms == "auto":
        if not a.judge:
            sys.exit("--host-ms auto needs --judge")
        jres = json.load(open(a.judge))
        jres = jres.get("result", jres)
        jl = {r["concurrency"]: r["ttft_later_p90_s"] for r in jres["rungs"]}
        cl = later_ttft_p90(a.cand)
        host = {k: max(0.0, cl[k] - jl[k]) for k in hr if k in cl and k in jl}
    if host:
        print("host correction (s/turn, both runs): " + "  ".join(f"c{k} {v * 1e3:.0f} ms" for k, v in sorted(host.items())))
    s, c = rung_stats(a.stock, host), rung_stats(a.cand, host)
    fr = frontier(s, c)
    raw = {k: c[k]["tok_s_per_gpu"] / s[k]["tok_s_per_gpu"] for k in RUNGS if k in s and k in c}
    # cache hit = cached / prompt tokens per rung; the judge reads 0.92 at every rung (13 turns, nothing evicted). A stock
    # hit well below the candidate's at c32 means stock evicted contexts (host cache too small) and re-prefilled them cold.
    print("conc  stock tok/s/GPU  stock u  hit    cand tok/s/GPU  cand u  hit     raw x   frontier x")
    for k in RUNGS:
        if k in raw:
            print(f"{k:>4}  {s[k]['tok_s_per_gpu']:>15.0f}  {s[k]['u']:>7.1f}  {s[k]['hit']:.3f}  {c[k]['tok_s_per_gpu']:>14.0f}  {c[k]['u']:>6.1f}  {c[k]['hit']:.3f}  {raw[k]:>6.3f}  {fr.get(k, 0):>10.3f}")
    if len(fr) == 4:
        print(f"proxy score (mean frontier) {sum(fr.values()) / 4:.4f}")
    else:
        print(f"WARNING: rungs {sorted(fr)} only; the frontier needs c1/c4/c16/c32 in both runs, so frontier x is not judge-comparable")
    if a.judge:
        d = json.load(open(a.judge))
        res = d.get("result", d)
        jfr = {p["concurrency"]: p["ratio"] for p in res["frontier"]["points"]}
        st = {p["concurrency"]: p["stock_tok_s_per_gpu"] for p in res["frontier"]["points"]}
        jraw = {r["concurrency"]: r["tok_s_per_gpu"] / st[r["concurrency"]] for r in res["rungs"]}
        print(f"judge {res['frontier']['score']}: tol +-{a.tol:.0%}")
        for k in RUNGS:
            if k in fr:
                ef, er = fr[k] / jfr[k] - 1, raw[k] / jraw[k] - 1
                ok = "ok" if abs(ef) <= a.tol and abs(er) <= a.tol else "OFF"
                print(f"  c{k}: frontier proxy {fr[k]:.3f} judge {jfr[k]:.3f} ({ef:+.1%}); raw proxy {raw[k]:.3f} judge {jraw[k]:.3f} ({er:+.1%})  {ok}")
    if a.vs:
        v = rung_stats(a.vs, host)
        vfr = frontier(s, v)
        print("cand vs ctl:  " + "  ".join(f"c{k} raw {c[k]['tok_s_per_gpu'] / v[k]['tok_s_per_gpu'] - 1:+.1%}" for k in RUNGS if k in c and k in v))
        if len(fr) == 4 and len(vfr) == 4:
            print(f"score delta (frontier mean) {sum(fr.values()) / sum(vfr.values()) - 1:+.2%}")
            if a.judge:
                # judge-anchored prediction: the scored run's per-rung frontier ratio times cand/ctl on the proxy. Valid
                # when ctl is the scored build; it cancels every proxy-vs-judge offset that is common to ctl and cand.
                pred = {k: jfr[k] * fr[k] / vfr[k] for k in RUNGS}
                print("judge-anchored: " + "  ".join(f"c{k} {pred[k]:.3f}" for k in RUNGS)
                      + f"  -> predicted score {sum(pred.values()) / 4:.4f} (scored {res['frontier']['score']})")


if __name__ == "__main__":
    main()
