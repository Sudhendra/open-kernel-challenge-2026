#!/usr/bin/env python3
"""Read a judge check or scored run against the incumbent, the way a decision should be made.

  python3 harness/read_check.py <job-id> [--base <job-id>] [--touches c1,c4] [--noise 0.02]

Pulls out/judge/status_<job>_*.json (and the saved text for a check's stock numbers; fetches with okc when absent),
then prints per rung: our tok/s, stock tok/s, ratio, and the delta vs the base run's ratio at that rung, with:
  - the noise band (default 3%: the Nov 12 noop check read 0.983x, judge nodes differ ~4%; a scored run's c1/c4 spreads are 6-9%),
  - the untouched-rung test: rungs the lever cannot touch (everything not in --touches) are the control; if they moved as
    much as the touched rungs, the difference is the node, not the change,
  - pass/fail of the lossless gates and memory, TTFT ratio where the text has it,
  - a verdict: KEEP-CANDIDATE (touched rungs up beyond the band, controls inside it), NODE (controls moved too),
    NO-GAIN (inside the band), SLOWER, or FAILED (gates).
Base defaults to the day-one scored run c6286ef374094ed9 (FGHIL 2.232x).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
JUDGE = os.path.join(SERVING, "out", "judge")
DEFAULT_BASE = "c6286ef374094ed9"


def latest(pattern):
    fs = sorted(glob.glob(pattern))
    return fs[-1] if fs else None


def fetch(job):
    """status JSON and text for a job, from disk or okc."""
    js = latest(f"{JUDGE}/status_{job}_*.json")
    # the status text (not the --log startup dump): check_<job>_*.txt, scored_<job>_*.txt or text_<job>_*.txt
    tx = latest(f"{JUDGE}/check_{job}_*.txt") or latest(f"{JUDGE}/scored_{job}_*.txt") or latest(f"{JUDGE}/text_{job}_*.txt")
    env = dict(os.environ, PATH=os.path.expanduser("~/.okc/bin") + ":" + os.environ.get("PATH", ""))
    def unfinished(path):  # a cached status saved while the job was queued/running is stale: never trust it (Nov 16 miss)
        try:
            return not json.load(open(path)).get("result")
        except Exception:
            return True
    if not js or unfinished(js):
        out = subprocess.run(["okc", "serve", "status", job, "--json"], capture_output=True, text=True, env=env).stdout
        js = f"{JUDGE}/status_{job}_fetched.json"; open(js, "w").write(out); tx = None
    if not tx:
        out = subprocess.run(["okc", "serve", "status", job], capture_output=True, text=True, env=env).stdout
        tx = f"{JUDGE}/text_{job}_fetched.txt"; open(tx, "w").write(out)
    d = json.load(open(js))
    return d, open(tx).read()


def rungs_of(d, text):
    """{concurrency: {"ours", "stock", "ratio", ...}} for a run (frontier) or a check (text)."""
    res = d.get("result") or {}
    out = {}
    fr = res.get("frontier")
    if fr:
        # a scored run: the frontier ratio is what the score uses, but a CHECK at concurrency c reports the RAW ratio
        # (our tok/s at c over stock's at c), so the comparable base number is the raw one; keep both.
        raw = {r["concurrency"]: r["tok_s_per_gpu"] for r in res.get("rungs", [])}
        for p in fr["points"]:
            c = p["concurrency"]
            out[c] = {"ours": raw.get(c), "stock": p["stock_tok_s_per_gpu"], "frontier_ratio": p["ratio"],
                      "ratio": (raw[c] / p["stock_tok_s_per_gpu"]) if raw.get(c) else p["ratio"]}
    else:
        # check: "vs stock    c16  2.103×" and "stock 12,419 tok/s/GPU"
        m = re.search(r"vs stock\s+c(\d+)\s+([\d.]+)×", text)
        s = re.search(r"stock ([\d,]+) tok/s/GPU", text)
        if m:
            c = int(m.group(1)); ratio = float(m.group(2))
            r = next((x for x in res.get("rungs", []) if x["concurrency"] == c), None)
            ours = r["tok_s_per_gpu"] if r else None
            out[c] = {"ours": ours, "stock": float(s.group(1).replace(",", "")) if s else None, "ratio": ratio}
    for r in res.get("rungs", []):
        c = r["concurrency"]
        out.setdefault(c, {})
        out[c].update(accept=r.get("acceptance"), spread=r.get("spread"), ttft_first=r.get("ttft_first_p90_s"),
                      e2e=r.get("tok_s_per_user_e2e"), p50=r.get("tok_s_per_user_p50"))
    return out


def gates(d, text):
    res = d.get("result") or {}
    c = res.get("correctness") or {}; lt = res.get("later_turns") or {}; mem = res.get("gpu_memory") or {}
    ttft = re.search(r"TTFT p90 first ([\d.]+)× · later ([\d.]+)×", text)
    return {"passed": bool(res.get("passed")), "prefill_kl": c.get("prefill_kl_mean"), "decode_kl": c.get("decode_kl_mean"),
            "agree_first": lt.get("first_agreement"), "agree_later": lt.get("later_agreement"),
            "mem_over_gb": mem.get("over_stock_gb"), "ttft_first_x": float(ttft.group(1)) if ttft else None,
            "ttft_later_x": float(ttft.group(2)) if ttft else None, "reasons": (c.get("reasons") or []) + (lt.get("reasons") or [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job"); ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--touches", default="", help="rungs the lever can move, e.g. c1,c4; the rest are controls")
    ap.add_argument("--noise", type=float, default=0.03)
    a = ap.parse_args()
    d, text = fetch(a.job); bd, btext = fetch(a.base)
    cur, base = rungs_of(d, text), rungs_of(bd, btext)
    g = gates(d, text)
    mode = d.get("mode"); print(f"job {a.job} ({mode}) vs base {a.base} ({bd.get('mode')}); noise band ±{a.noise:.0%}")
    print(f"gates: passed={g['passed']} prefill_kl={g['prefill_kl']} (≤0.11) decode_kl={g['decode_kl']} (≤0.035) "
          f"agree {g['agree_first']}/{g['agree_later']} mem {('%+.1f' % g['mem_over_gb']) if g['mem_over_gb'] is not None else 'n/a'} GB TTFT first {g['ttft_first_x']}x later {g['ttft_later_x']}x"
          + (f" REASONS {g['reasons']}" if g["reasons"] else ""))
    touched = {int(x.strip().lstrip("c")) for x in a.touches.split(",") if x.strip()}
    if any("frontier_ratio" in v for v in base.values()):
        print("base raw ratios (what a check at that concurrency compares to): " + " ".join(f"c{c} {v['ratio']:.3f}" for c, v in sorted(base.items()))
              + " | frontier (score) " + " ".join(f"c{c} {v.get('frontier_ratio', float('nan')):.3f}" for c, v in sorted(base.items())))
    print(f"{'rung':<5}{'ours':>9}{'stock':>9}{'ratio':>8}{'base':>8}{'delta':>8}{'accept':>8}{'spread':>8}  role")
    moves = {}
    for c in sorted(cur):
        r, b = cur[c], base.get(c, {})
        if r.get("ratio") is None:
            continue
        delta = r["ratio"] / b["ratio"] - 1 if b.get("ratio") else float("nan")
        moves[c] = delta
        role = "touched" if (not touched or c in touched) else "control"
        print(f"c{c:<4}{r.get('ours') or 0:>9.0f}{r.get('stock') or 0:>9.0f}{r['ratio']:>8.3f}{b.get('ratio', float('nan')):>8.3f}"
              f"{delta:>+8.1%}{r.get('accept') or 0:>8.1f}{(r.get('spread') or 0):>8.1%}  {role}")
    if not g["passed"]:
        print("VERDICT: FAILED gates"); return
    t = [moves[c] for c in moves if (not touched or c in touched)]
    k = [moves[c] for c in moves if touched and c not in touched]
    if k and max(abs(x) for x in k) > a.noise:
        print(f"VERDICT: NODE — control rungs moved {max(k, key=abs):+.1%} (> band); the judge node, not the change, decides this read")
    elif t and min(t) > a.noise:
        print(f"VERDICT: KEEP-CANDIDATE — touched rungs up {min(t):+.1%}..{max(t):+.1%} beyond the band; confirm in a stack before scoring")
    elif t and max(t) < -a.noise:
        print(f"VERDICT: SLOWER — touched rungs down {min(t):+.1%}")
    else:
        print("VERDICT: NO-GAIN — inside the noise band; a single check cannot resolve this lever (see score_model.py runs_needed)")


if __name__ == "__main__":
    main()
