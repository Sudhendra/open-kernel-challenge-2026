#!/usr/bin/env python3
"""Compare ladder.py runs: speed vs stock, the 2x TTFT gate, 220-token gate, output agreement.

  python agree.py stock.json cand.json                  # speed + agreement
  python agree.py stock.json cand.json --self stock2.json   # also stock-vs-stock noise floor

The rule is that outputs must agree with stock "nearly as well as it agrees with itself",
so the candidate's agreement only means something next to a second stock run (--self).
The judge's own agreement metric is not published; these are text proxies, not the metric.
The candidate should have been run with --replay stock.json so prompts are identical.
"""
import argparse
import json
import statistics
from collections import defaultdict


def load(path):
    with open(path) as f:
        return json.load(f)


def prefix_frac(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / max(len(a), len(b), 1)


def agreement(ref, other):
    a = {r["key"]: r["text"] for r in ref["records"]}
    b = {r["key"]: r["text"] for r in other["records"]}
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    fr = [prefix_frac(a[k], b[k]) for k in keys]
    return {
        "n": len(keys),
        "exact": sum(1 for k in keys if a[k] == b[k]) / len(keys),
        "prefix_mean": statistics.mean(fr),
        "prefix_median": statistics.median(fr),
    }


def by_conc(run):
    out = defaultdict(list)
    for r in run["rungs"]:
        out[r["concurrency"]].append(r)
    return out


def med(rows, field):
    vals = [r[field] for r in rows if r.get(field) is not None]
    return statistics.median(vals) if vals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stock")
    ap.add_argument("cand")
    ap.add_argument("--self", dest="self_run", help="a second stock run, for the noise floor")
    args = ap.parse_args()
    stock, cand = load(args.stock), load(args.cand)
    s, c = by_conc(stock), by_conc(cand)

    print(f"{'conc':>4} {'thr stock':>10} {'thr cand':>10} {'x':>6} {'inter x':>8} {'ttft x':>7}  gates")
    for conc in sorted(set(s) & set(c)):
        ts, tc = med(s[conc], "throughput_tok_s"), med(c[conc], "throughput_tok_s")
        is_, ic = med(s[conc], "interactivity_tok_s_median"), med(c[conc], "interactivity_tok_s_median")
        fs, fc = med(s[conc], "ttft_s_median"), med(c[conc], "ttft_s_median")
        gates = []
        if fs and fc and fc > 2 * fs:
            gates.append("TTFT>2x")
        if sum(r["wrong_length"] for r in c[conc]):
            gates.append("LEN!=220")
        if sum(r["failed"] for r in c[conc]):
            gates.append("FAILED")
        if not ts or not tc:
            print(f"{conc:>4} {ts or 0:>10.1f} {tc or 0:>10.1f}   n/a  (no throughput recorded)")
            continue
        print(
            f"{conc:>4} {ts:>10.1f} {tc:>10.1f} {tc / ts:>6.3f} "
            f"{(ic / is_) if is_ and ic else float('nan'):>8.3f} {(fc / fs) if fs and fc else float('nan'):>7.2f}  "
            f"{' '.join(gates) or 'ok'}"
        )

    ag = agreement(stock, cand)
    print("\nagreement cand vs stock :", json.dumps(ag))
    if args.self_run:
        print("agreement stock vs stock:", json.dumps(agreement(stock, load(args.self_run))))
    else:
        print("no --self run: there is no noise floor to judge the number above against")


if __name__ == "__main__":
    main()
