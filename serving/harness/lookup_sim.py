#!/usr/bin/env python3
"""Offline prompt-lookup drafting simulation on the ladder's recorded conversations.

Reconstructs each prompt exactly as ladder.py does (same corpus dir, seed, keys), tokenises
(Qwen3-8B tokenizer from the local HF cache as a stand-in for Muse; falls back to tiktoken
cl100k_base, then to a regex tokenizer), then walks each recorded 220-token reply as greedy
decoding would, proposing k tokens from the most recent earlier occurrence of the last n
generated tokens (n=4,3,2, longest first) in context + reply so far.

  HF_HUB_OFFLINE=1 python lookup_sim.py --records ../out/<run>/stock.json --out report.md
"""
import argparse, collections, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ladder

DFLASH_TPS = 3.6   # tokens emitted per DFlash step (convention: baseline steps = 220/3.6)


def get_tok(name):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        return (lambda s: t.encode(s, add_special_tokens=False)), "Qwen/Qwen3-8B (HF cache)"
    except Exception:
        pass
    try:
        import tiktoken
        e = tiktoken.get_encoding("cl100k_base")
        return (lambda s: e.encode(s, disallowed_special=())), "tiktoken cl100k_base"
    except Exception:
        rx = re.compile(r"\w+|[^\w\s]|\s+")
        return (lambda s: rx.findall(s)), "regex word/punct"


def build_index(seq, ns, idx=None, start=0):
    idx = idx if idx is not None else {n: {} for n in ns}
    for n in ns:
        d = idx[n]
        for e in range(max(start, n - 1), len(seq)):
            d[tuple(seq[e - n + 1:e + 1])] = e      # end position; later overwrites -> most recent
    return idx


def simulate(ctx, reply, k, ns=(4, 3, 2), min_n=2, dflash=DFLASH_TPS):
    """Stepped walk. Returns dict of stats for one turn."""
    ns = [n for n in ns if n >= min_n]
    seq = list(ctx)
    idx = build_index(seq, ns)
    L0 = len(ctx)
    T = len(reply)
    # (a) per-token walk: does a match exist at each position, and how long would accepted be
    pos_match = pos_first = 0
    acc_when_match = []
    full = list(ctx)
    idx1 = build_index(full, ns)
    for p in range(T):
        found = None
        for n in ns:
            if len(full) >= n:
                e = idx1[n].get(tuple(full[-n:]))
                if e is not None:
                    found = (n, e); break
        if found:
            n, e = found
            d = full[e + 1:e + 1 + k]
            a = 0
            for j, tok in enumerate(d):
                if p + j < T and reply[p + j] == tok: a += 1
                else: break
            pos_match += 1; acc_when_match.append(a)
            if a >= 1: pos_first += 1
        # append token, index n-grams ending at the previous end (lookup precedes insertion)
        full.append(reply[p])
        for n in ns:
            if len(full) > n:
                idx1[n][tuple(full[-n - 1:-1])] = len(full) - 2   # ngram ending before newest token
        # note: ngram ending at the newest token is inserted next iteration after its lookup
    # (b) stepped walk with fallback to DFlash
    full = list(ctx); idx2 = build_index(full, ns)
    p = 0.0; steps = lk_steps = df_steps = 0
    lk_acc = []; lk_tokens = 0
    while int(p) < T:
        ip = int(p)
        cur = full[:L0 + ip]
        found = None
        for n in ns:
            if len(cur) >= n:
                e = idx2[n].get(tuple(cur[-n:]))
                if e is not None and e < len(cur) - 1:
                    found = (n, e); break
        if found:
            n, e = found
            d = cur[e + 1:e + 1 + k]
            a = 0
            for j, tok in enumerate(d):
                if ip + j < T and reply[ip + j] == tok: a += 1
                else: break
            adv = a + 1
            lk_steps += 1; lk_acc.append(a); lk_tokens += min(adv, T - ip)
        else:
            adv = dflash; df_steps += 1
        steps += 1
        new_ip = min(T, int(p + adv)) if not found else min(T, ip + adv)
        p = p + adv
        # extend sequence with true tokens up to new position and index them
        upto = min(T, int(p))
        old = len(full)
        full = list(ctx) + list(reply[:upto])
        for n in ns:
            d_ = idx2[n]
            for e in range(max(old - 1, n - 1), len(full) - 1):   # exclude ngram ending at newest token
                d_[tuple(full[e - n + 1:e + 1])] = e
    return dict(T=T, pos_match=pos_match, pos_first=pos_first, acc_when_match=acc_when_match,
                steps=steps, lk_steps=lk_steps, df_steps=df_steps, lk_acc=lk_acc, lk_tokens=lk_tokens)


def copy_spans(ctx, reply, min_len=3):
    """Greedy segmentation of reply into longest exact substrings found earlier (ctx + reply so far)."""
    idx = {}
    seq = list(ctx)
    for e in range(2, len(seq)):
        idx.setdefault(tuple(seq[e - 2:e + 1]), []).append(e - 2)   # start positions
    T = len(reply); p = 0; spans = []
    full = list(ctx)
    while p < T:
        best = 0
        for s in idx.get(tuple(reply[p:p + 3]), [])[-3000:]:
            l = 0
            while p + l < T and s + l < len(full) and full[s + l] == reply[p + l]: l += 1
            if l > best: best = l
        if best >= min_len:
            spans.append(best); adv = best
        else: adv = 1
        for q in range(p, min(T, p + adv)):
            full.append(reply[q])
            e = len(full) - 1
            if e >= 2: idx.setdefault(tuple(full[e - 2:e + 1]), []).append(e - 2)
        p += adv
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--corpus", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sglang", "python", "sglang", "srt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-turns", type=int, default=0)
    a = ap.parse_args()
    d = json.load(open(a.records))
    args = argparse.Namespace(**{**d["args"], "trace_convs": None})
    files = ladder.load_corpus(a.corpus)
    tok, tokname = get_tok(None)
    recs = d["records"]
    byconv = collections.defaultdict(dict)
    for r in recs:
        conv, t = r["key"].rsplit("/t", 1); byconv[conv][int(t)] = r
    rows = []
    for conv, turns in sorted(byconv.items()):
        rung = int(conv.split("/")[0][1:])
        built = ladder.build_conversation(conv, files, args, 0)
        ctx = []
        for t in range(len(built)):
            for m in built[t]:
                ctx += tok(f"<{m['role']}>\n" + m["content"] + "\n")
            if t not in turns: break
            r = turns[t]
            reply = tok(r["text"])[:220]
            ctx_now = list(ctx)
            row = dict(key=r["key"], rung=rung, turn=t, ctx_tokens=len(ctx_now), prompt_tokens=r["prompt_tokens"], T=len(reply), ks={})
            for k in (4, 8, 16):
                row["ks"][k] = {mn: simulate(ctx_now, reply, k, min_n=mn) for mn in (2, 3)}
            row["spans"] = copy_spans(ctx_now, reply)
            rows.append(row)
            print(row["key"], row["ctx_tokens"], row["prompt_tokens"], file=sys.stderr, flush=True)
            ctx += tok("<assistant>\n" + r["text"] + "\n")
            if a.max_turns and len(rows) >= a.max_turns: break
        if a.max_turns and len(rows) >= a.max_turns: break
    write_report(rows, tokname, a.out)


def agg(rs, k, mn):
    S = [r["ks"][k][mn] for r in rs]
    T = sum(s["T"] for s in S)
    am = [x for s in S for x in s["acc_when_match"]]
    la = [x for s in S for x in s["lk_acc"]]
    steps = sum(s["steps"] for s in S); look = sum(s["lk_steps"] for s in S)
    pos = sum(len(s["acc_when_match"]) for s in S)
    return dict(
        turns=len(S), T=T,
        match_pos=pos / T,
        first_ok=sum(s["pos_first"] for s in S) / T,
        acc_match=sum(am) / max(1, len(am)),
        nomatch_steps=1 - look / max(1, steps),
        lk_cov=sum(s["lk_tokens"] for s in S) / T,
        lk_acc_mean=sum(la) / max(1, len(la)),
        lk_ge6=sum(1 for x in la if x >= 6) / max(1, steps),
        lk_ge1=sum(1 for x in la if x >= 1) / max(1, steps),
        steps_turn=steps / len(S), base_turn=(T / len(S)) / DFLASH_TPS,
    )


def write_report(rows, tokname, out):
    L = []
    P = L.append
    P(f"# Prompt-lookup simulation\n\nTokenizer: {tokname} (approximation of Muse tokenizer; replies re-tokenised from saved text, truncated to 220).")
    P(f"Turns analysed: {len(rows)}. Mean reconstructed ctx tokens vs logged prompt_tokens: "
      f"{sum(r['ctx_tokens'] for r in rows)/len(rows):.0f} vs {sum(r['prompt_tokens'] or 0 for r in rows)/len(rows):.0f}.\n")
    rungs = sorted({r["rung"] for r in rows})
    groups = [(f"c{c}", [r for r in rows if r["rung"] == c]) for c in rungs] + [("ALL", rows)]
    for mn in (2, 3):
        P(f"\n## Policy: lookup when longest-match n>={mn} (n tried 4,3,2), else DFlash ({DFLASH_TPS} tokens/step)\n")
        P("match_pos = fraction of positions (token-by-token walk) with any n-gram match; acc|match = mean accepted drafts at those positions; "
          "nomatch = fraction of steps with no match (stepped walk); lk_cov = fraction of reply tokens emitted by lookup steps; "
          "lk_acc = mean accepted drafts per lookup step; >=6 = fraction of ALL steps with 6+ accepted; steps = expected steps/turn vs DFlash-only.\n")
        P("| k | rung | turns | match_pos | acc\\|match | nomatch | lk_cov | lk_acc | lookup steps >=6 | steps/turn | DFlash steps/turn | saving |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k in (4, 8, 16):
            for name, rs in groups:
                g = agg(rs, k, mn)
                P(f"| {k} | {name} | {g['turns']} | {g['match_pos']:.1%} | {g['acc_match']:.2f} | {g['nomatch_steps']:.1%} | {g['lk_cov']:.1%} | "
                  f"{g['lk_acc_mean']:.2f} | {g['lk_ge6']:.1%} | {g['steps_turn']:.1f} | {g['base_turn']:.1f} | {1-g['steps_turn']/g['base_turn']:+.1%} |")
    P("\n## Copy spans (longest exact shared substrings, greedy, min 3 tokens)\n")
    P("| rung | tokens | in spans>=3 | >=4 | >=8 | >=16 | span count/turn | mean span len |")
    P("|---|---|---|---|---|---|---|---|")
    for name, rs in groups:
        T = sum(r["T"] for r in rs); sp = [x for r in rs for x in r["spans"]]
        P(f"| {name} | {T} | {sum(sp)/T:.1%} | {sum(x for x in sp if x>=4)/T:.1%} | {sum(x for x in sp if x>=8)/T:.1%} | "
          f"{sum(x for x in sp if x>=16)/T:.1%} | {len(sp)/len(rs):.1f} | {sum(sp)/max(1,len(sp)):.1f} |")
    P("\n## Per turn (k=8, n>=2): match_pos, lk_cov, lookup steps, steps/turn\n")
    P("| key | ctx tok | match_pos | lk_cov | lookup steps | steps | span>=8 tok |")
    P("|---|---|---|---|---|---|---|")
    for r in rows:
        s = r["ks"][8][2]
        P(f"| {r['key']} | {r['ctx_tokens']} | {len(s['acc_when_match'])/s['T']:.0%} | {s['lk_tokens']/s['T']:.0%} | {s['lk_steps']} | {s['steps']} | {sum(x for x in r['spans'] if x>=8)} |")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
