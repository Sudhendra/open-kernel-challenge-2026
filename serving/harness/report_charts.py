#!/usr/bin/env python3
"""Charts from one profile_session.sh output directory -> PNGs in <out>/charts/ + summary.json.

  python report_charts.py out/<ts>

Reads stock.json / stock2.json (ladder.py), gpu_util.csv (nvidia-smi 1 Hz), nccl.jsonl,
timeline.txt, startup.txt. Trace-derived numbers come from SGLang's own analyzer separately.
"""
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_json(p):
    with open(p) as f:
        return json.load(f)


def ladder_chart(out, run, charts):
    rungs = run["rungs"]
    conc = [r["concurrency"] for r in rungs]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].bar([str(c) for c in conc], [r["throughput_tok_s"] for r in rungs], color="#4C78A8")
    ax[0].set_title("Output throughput (tok/s, node)")
    ax[1].bar([str(c) for c in conc], [r["interactivity_tok_s_median"] or 0 for r in rungs], color="#F58518")
    ax[1].set_title("Per-stream rate after first token (tok/s)")
    ax[2].bar([str(c) for c in conc], [r["ttft_s_median"] or 0 for r in rungs], color="#54A24B")
    ax[2].bar([str(c) for c in conc], [r["ttft_s_p90"] or 0 for r in rungs], color="#54A24B", alpha=0.35)
    ax[2].set_title("TTFT median / p90 (s)")
    for a in ax:
        a.set_xlabel("concurrency")
    fig.tight_layout()
    p = os.path.join(charts, "ladder.png")
    fig.savefig(p, dpi=130)
    return {"rungs": rungs}


def per_request_chart(out, run, charts):
    """TTFT and latency per turn index: shows prefix-cache effect turn 1 vs later."""
    by_turn = defaultdict(list)
    for r in run["records"]:
        if r.get("ttft") is None:
            continue
        t = int(r["key"].rsplit("/t", 1)[1])
        by_turn[t].append(r)
    turns = sorted(by_turn)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(turns, [statistics.median(x["ttft"] for x in by_turn[t]) for t in turns], marker="o", label="TTFT median (s)")
    ax.plot(turns, [statistics.median(x["latency"] for x in by_turn[t]) for t in turns], marker="s", label="request latency median (s)")
    ax.set_xlabel("turn index in conversation")
    ax.legend()
    ax.set_title("Cold first turn vs cached later turns")
    fig.tight_layout()
    fig.savefig(os.path.join(charts, "turns.png"), dpi=130)
    return {t: {"n": len(by_turn[t]), "ttft_med": statistics.median(x["ttft"] for x in by_turn[t]),
                "prompt_tokens_med": statistics.median(x["prompt_tokens"] or 0 for x in by_turn[t]),
                "cached_tokens_med": statistics.median(x["cached_tokens"] or 0 for x in by_turn[t])} for t in turns}


def gpu_util_chart(out, charts):
    p = os.path.join(out, "gpu_util.csv")
    if not os.path.exists(p):
        return None
    rows = defaultdict(list)
    with open(p) as f:
        for row in csv.reader(f):
            if len(row) < 5 or row[0].startswith("timestamp"):
                continue
            try:
                ts = datetime.strptime(row[0].strip()[:19], "%Y/%m/%d %H:%M:%S")
                rows[int(row[1])].append((ts, float(row[2].strip().split()[0]), float(row[3].strip().split()[0])))
            except (ValueError, IndexError):
                continue
    if not rows:
        return None
    t0 = min(v[0][0] for v in rows.values())
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    for g, v in sorted(rows.items()):
        x = [(t - t0).total_seconds() / 60 for t, _, _ in v]
        ax[0].plot(x, [u for _, u, _ in v], lw=0.6, label=f"gpu{g}")
        ax[1].plot(x, [w for _, _, w in v], lw=0.6)
    ax[0].set_ylabel("SM util %")
    ax[1].set_ylabel("power W")
    ax[1].set_xlabel("minutes since session start")
    marks = {}
    tl = os.path.join(out, "timeline.txt")
    if os.path.exists(tl):
        for line in open(tl):
            name, hms = line.split()
            h, m, s = map(int, hms.split(":"))
            mins = (h * 3600 + m * 60 + s - (t0.hour * 3600 + t0.minute * 60 + t0.second)) / 60
            if mins < 0:
                mins += 24 * 60
            marks[name] = mins
            for a in ax:
                a.axvline(mins, color="k", ls=":", lw=0.8)
            ax[0].text(mins, 100, name, rotation=90, fontsize=6, va="top")
    ax[0].legend(fontsize=6, ncol=8)
    fig.tight_layout()
    fig.savefig(os.path.join(charts, "gpu_util.png"), dpi=130)
    util0 = [u for _, u, _ in rows[0]]
    return {"gpu0_util_mean": statistics.mean(util0), "gpu0_util_median": statistics.median(util0), "marks_min": marks}


def nccl_chart(out, charts):
    p = os.path.join(out, "nccl.jsonl")
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p) if l.strip().startswith("{")]
    ar = [r for r in rows if "allreduce" in r]
    peer = next((r["peer_access"] for r in rows if "peer_access" in r), None)
    if ar:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        labels = [f'{r["allreduce"]} M={r["M"]}' for r in ar]
        ax.bar(range(len(ar)), [r["us"] for r in ar], color="#B279A2")
        ax.set_xticks(range(len(ar)))
        ax.set_xticklabels(labels, rotation=60, fontsize=7)
        ax.set_ylabel("all-reduce latency (us)")
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(charts, "nccl.png"), dpi=130)
    return {"allreduce": ar, "peer_access": peer}


def main():
    out = sys.argv[1]
    charts = os.path.join(out, "charts")
    os.makedirs(charts, exist_ok=True)
    summary = {}
    sp = os.path.join(out, "stock.json")
    if os.path.exists(sp):
        run = load_json(sp)
        summary["ladder"] = ladder_chart(out, run, charts)
        summary["turns"] = per_request_chart(out, run, charts)
    summary["gpu_util"] = gpu_util_chart(out, charts)
    summary["nccl"] = nccl_chart(out, charts)
    st = os.path.join(out, "startup.txt")
    if os.path.exists(st):
        summary["startup"] = open(st).read().strip()
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps({k: (v if k != "ladder" else "…") for k, v in summary.items()}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
