#!/usr/bin/env python3
"""Local replica of the OKC serving ladder.

Drives a running SGLang server (OpenAI-compatible endpoint) with synthetic multi-turn
agentic-coding conversations at the judge's rungs and records, per request: time to
first token, end time, completion token count and the output text.

The judge's real conversations are hidden, so absolute numbers will not match the
official score. What this gives is a like-for-like A/B: stock vs patched on the same
node with the same prompts.

  # stock, recording the transcripts
  python ladder.py --out stock.json
  # patched, replaying stock's assistant turns so every prompt is byte-identical
  python ladder.py --out cand.json --replay stock.json
  # proxy preset = the judge's conversation shape (Nov 15): 13 turns, ~74k-token first turn,
  # ~753 new tokens per later turn, 220 output tokens; rungs 16x1x2 unless --rungs
  python ladder.py --proxy --out proxy.json
  # the pre-Nov-15 proxy (4 turns, ~64k first turn, ~690 new tokens): reproduces old runs byte for byte
  python ladder.py --proxy-short --out proxy.json

Workload shape (synthetic corpus): the first turn is ~--base-chars of source text, later turns
add a slab drawn uniformly from [--turn-chars/2, 2*--turn-chars) plus the previous 220-token
answer. The sizes are in characters. Measured on Modal (out/q09/.../20261114T161657Z, 42
conversations, the Muse tokenizer): 280_000 chars -> 63,918 first-prompt tokens (4.38 chars/token)
and 2_150 -> 690 new tokens per later turn (+220 for the previous answer, which is never a cache
hit). The judge's JSON `shape` block says first_turn 74,160, new_per_turn 753, output_mean 220,
turns 13, so the judge shape scales both: 325_000 chars (~74.2k) and 2_350 chars (~754).
Check prompt_tokens in the output records to confirm on a new corpus or tokenizer.

Shapes (--shape, or set by --proxy / --proxy-short; explicit --turns/--base-chars/--turn-chars win):
  judge  13 turns, 325_000 / 2_350 chars  (cache hit 0.92 when nothing is evicted, as the judge reads)
  short   4 turns, 280_000 / 2_150 chars  (cache hit ~0.75: the old, prefill-heavy proxy)
  legacy  6 turns, 280_000 / 2_150 chars  (the default without a preset; profile_session's LADDER passes --turns)

Only needs aiohttp, which the SGLang image has.

Client modes (--client; profile_session.sh PROXY_CLIENT):
  chat (default)  POST /v1/chat/completions with the messages as text: the server renders the chat template and tokenizes
                  the whole ~80k-token conversation on every turn (our pre-Nov-17 proxy; output unchanged byte for byte).
  ids             the judge's client (organizer, Nov 17): POST /generate with `input_ids`, greedy, streamed. Every
                  conversation's ids are prepared before its rung's clock starts (the judge has its ids ready before it
                  sends): turn 1 = the model's chat template rendered and encoded once; each later turn = the previous
                  turn's input_ids + the server's returned output ids for it + the template glue that closes the assistant
                  turn and opens the next user turn + the new user text + the generation prompt, the glue taken from the
                  template itself (IdsTemplate), the new text encoded alone (a few hundred tokens). So the prefix is
                  exactly what the server cached. With --replay, the replayed reply's ids are the record's output_ids if it
                  has them, else this run's own output ids when its text equals the replayed text, else the replayed text
                  encoded. Records add output_ids, input_len and ttft_ids (first chunk with an output id); ttft stays the
                  first chunk with text, as in chat mode. Needs transformers and the model's tokenizer (--tokenizer, the
                  server's --model-path, read from the HF cache); --tokenizer fake is a stand-in for the laptop dry run.
"""
import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import time

import aiohttp

try:
    import orjson  # serialises an 80k-int input_ids list in ~1 ms (json: ~15 ms on the event loop)
except ImportError:
    orjson = None

TOKENIZER = "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"   # the server's --model-path (profile_session.sh start_server)
CHANNEL_HEADER = re.compile(r"^ ?to=(self|user|commentary)")   # Muse recipient header (sglang reasoning_parser MuseGlimmerDetector)

# GET /api/v1/serving, 2026-11-04: (concurrency, reps, conversations_per_slot)
SCORED_RUNGS = [(1, 3, 2), (4, 3, 2), (16, 3, 2), (32, 1, 1)]
CHECK_RUNGS = [(16, 1, 2)]
# Conversation shapes. judge = the judge JSON's `shape` block (first_turn 74160, new_per_turn 753, output_mean 220,
# turns 13), with chars scaled by the measured 4.38 (first turn) / 3.89 (later slab, mean draw 1.25x) chars per token. short/legacy = the old defaults.
SHAPES = {
    "judge": {"turns": 13, "base_chars": 325_000, "turn_chars": 2_350},
    "short": {"turns": 4, "base_chars": 280_000, "turn_chars": 2_150},
    "legacy": {"turns": 6, "base_chars": 280_000, "turn_chars": 2_150},
}

SYSTEM = (
    "You are a coding agent working in a repository. You are given tool results and "
    "must reason about the code and state the next concrete step."
)
ASKS = [
    "Explain what this code does and identify the hot path.",
    "Find a bug or a risky assumption in this file and propose a fix.",
    "Write the next edit you would make here, as a unified diff.",
    "Summarise how this module interacts with its callers.",
    "List the tests you would add for this code and sketch the first one.",
]


def load_corpus(root):
    files = []
    for d, _, names in os.walk(root):
        for n in sorted(names):
            if n.endswith((".py", ".md", ".cu", ".cuh", ".h", ".cpp")):
                p = os.path.join(d, n)
                try:
                    if 2000 < os.path.getsize(p) < 400_000:
                        files.append(p)
                except OSError:
                    pass
    files.sort()
    if not files:
        raise SystemExit(f"no corpus files under {root}")
    return files


def chunk(rng, files, n_chars):
    """Deterministic slab of source text of about n_chars, tagged like a tool result."""
    out, left = [], n_chars
    while left > 0:
        p = rng.choice(files)
        with open(p, errors="replace") as f:
            text = f.read()
        start = rng.randrange(0, max(1, len(text) - 1000))
        piece = text[start : start + min(left, rng.randrange(1500, 12000))]
        out.append(f"<tool_result path={os.path.basename(p)!r}>\n{piece}\n</tool_result>")
        left -= len(piece)
    return "\n".join(out)


def load_trace(path):
    """SGLang agentic-trace JSON (benchmark/datasets/agentic_trace.py): the format the
    track's 220-token multi-turn workload matches. Each turn holds its new messages."""
    with open(path) as f:
        convs = json.load(f)["conversations"]
    return [[t["messages"] for t in c if t.get("messages")] for c in convs]


def build_conversation(key, files, args, index):
    """Per-turn message lists; assistant turns come from the model or a replay."""
    if args.trace_convs:
        return args.trace_convs[index % len(args.trace_convs)][: args.turns]
    rng = random.Random(f"{args.seed}:{key}")
    turns = []
    for t in range(args.turns):
        n = args.base_chars if t == 0 else rng.randrange(args.turn_chars // 2, args.turn_chars * 2)
        msgs = [{"role": "system", "content": SYSTEM}] if t == 0 else []
        msgs.append({"role": "user", "content": f"{chunk(rng, files, n)}\n\n{rng.choice(ASKS)}"})
        turns.append(msgs)
    return turns


async def one_request(session, args, messages):
    body = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.ignore_eos:
        body["ignore_eos"] = True
    t0 = time.perf_counter()
    t_first, text, usage, n_chunks, error, status, raw_head = None, [], None, 0, None, None, []
    async with session.post(f"{args.url}/v1/chat/completions", json=body) as r:
        status = r.status
        async for raw in r.content:
            line = raw.decode(errors="replace").strip()
            if len(raw_head) < 3 and line:
                raw_head.append(line[:300])  # first lines verbatim, for debugging a bad run
            if not line.startswith("data:"):
                if line and status != 200:
                    error = (error or "") + line[:500]
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except json.JSONDecodeError:
                error = f"bad json: {data[:200]}"
                continue
            if ev.get("error"):
                error = json.dumps(ev["error"])[:500]
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                # Reasoning models stream thoughts as reasoning_content; both count as output.
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    n_chunks += 1
                    text.append(piece)
    t1 = time.perf_counter()
    usage = usage or {}
    return {
        "status": status,
        "error": error,
        "raw_head": raw_head,
        "t_start": t0,
        "ttft": (t_first - t0) if t_first else None,
        "latency": t1 - t0,
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "chunks": n_chunks,
        "text": "".join(text),
    }


class FakeTokenizer:
    """--tokenizer fake: ChatML-like template, one id per 4 characters. Dry runs only (no model files on a laptop)."""
    chat_template = "fake"
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        s = "".join(f"<|start|>{m['role']}\n{m['content']}<|end|>\n" for m in messages) + ("<|start|>assistant\n" if add_generation_prompt else "")
        return self.encode(s) if tokenize else s
    def encode(self, text, add_special_tokens=True):
        return [1000 + (sum(map(ord, text[i:i + 4])) % 50000) for i in range(0, len(text), 4)]
    def __call__(self, texts, add_special_tokens=True):
        return {"input_ids": [self.encode(t) for t in texts]}


def load_tokenizer(name):
    if name == "fake":
        return FakeTokenizer()
    from transformers import AutoTokenizer  # the server loads it the same way (no trust_remote_code in the judge's launch)
    tok = AutoTokenizer.from_pretrained(name)
    # The server defaults --chat-template to the snapshot's chat_template.jinja for this model (server.log: "Defaulting
    # --chat-template to .../chat_template.jinja") and loads it with strip("\n"): use exactly that text.
    try:
        if os.path.isdir(name):
            path = os.path.join(name, "chat_template.jinja")
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(name, "chat_template.jinja", local_files_only=True)
        if os.path.isfile(path):
            with open(path) as f:
                tok.chat_template = f.read().strip("\n")
    except Exception as e:  # no such file: the tokenizer's own template, as the server would use
        print(json.dumps({"chat_template_jinja": f"not used ({type(e).__name__})"}), file=sys.stderr, flush=True)
    return tok


class IdsTemplate:
    """Client-side ids for --client ids. The glue between turns comes from the model's own chat template: render
    [system S, user U1] and [system S, user U1, assistant A1, user U2] (generation prompt on) with sentinel contents; the
    first must be a prefix of the second (else the template rewrites history and no ids client can keep a stable prefix:
    raised). history_header = what the template puts between the generation prompt and an assistant message's content
    (Muse: " to=user<|message|>"; the model emits its own header, " to=self<|message|>" for reasoning, inside the
    generated ids, so the ids client does not add it). mid = text between A1 and U2, end = text after U2 (the generation
    prompt)."""
    S, U1, A1, U2 = "QXSYSQX", "QXUSERONEQX", "QXASSISTANTQX", "QXUSERTWOQX"

    def __init__(self, tok):
        self.tok = tok
        t1 = tok.apply_chat_template([{"role": "system", "content": self.S}, {"role": "user", "content": self.U1}], tokenize=False, add_generation_prompt=True)
        t2 = tok.apply_chat_template([{"role": "system", "content": self.S}, {"role": "user", "content": self.U1},
                                      {"role": "assistant", "content": self.A1}, {"role": "user", "content": self.U2}], tokenize=False, add_generation_prompt=True)
        i = t2.find(self.A1); j = t2.find(self.U2, i)
        if i < 0 or j < 0 or not t2[:i].startswith(t1):
            raise SystemExit(f"chat template is not prefix-stable across turns: turn-1 prompt {t1!r} vs turn-2 history {t2[:max(i, 0)]!r}")
        self.history_header = t2[len(t1):i]
        self.mid, self.end = t2[i + len(self.A1):j], t2[j + len(self.U2):]

    def encode(self, texts):
        # add_special_tokens=False: the rendered template carries its own specials (the server encodes the same way).
        return self.tok(texts, add_special_tokens=False)["input_ids"]

    def plan(self, turns):
        """turns: per-turn message lists of one conversation -> (turn-1 ids, [delta ids of turns 2..]): every encode
        done here, before the rung starts. Later turns must be one user message each (the synthetic corpus is)."""
        first = self.tok.apply_chat_template(turns[0], tokenize=False, add_generation_prompt=True)
        later = []
        for msgs in turns[1:]:
            if len(msgs) != 1 or msgs[0]["role"] != "user":
                raise SystemExit("--client ids: later turns must be a single user message")
            later.append(self.mid + msgs[0]["content"] + self.end)
        ids = self.encode([first] + later)
        return ids[0], ids[1:]


def _dumps(body):
    return orjson.dumps(body) if orjson is not None else json.dumps(body).encode()


async def one_request_ids(session, args, input_ids):
    """POST /generate with input_ids, streamed; the same record fields as one_request (+ output_ids, input_len, ttft_ids).
    Handles both streaming styles: --incremental-streaming-output (the judge's launch: each chunk a delta) and cumulative."""
    body = {"input_ids": input_ids, "stream": True,
            "sampling_params": {"max_new_tokens": args.max_tokens, "temperature": args.temperature, "ignore_eos": bool(args.ignore_eos)}}
    t0 = time.perf_counter()
    t_first, t_first_id, text, out_ids, meta, n_chunks, error, status, raw_head = None, None, "", [], {}, 0, None, None, []
    async with session.post(f"{args.url}/generate", data=_dumps(body), headers={"Content-Type": "application/json"}) as r:
        status = r.status
        async for raw in r.content:
            line = raw.decode(errors="replace").strip()
            if len(raw_head) < 3 and line:
                raw_head.append(line[:300])
            if not line.startswith("data:"):
                if line and status != 200:
                    error = (error or "") + line[:500]
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except json.JSONDecodeError:
                error = f"bad json: {data[:200]}"
                continue
            if ev.get("error"):
                error = json.dumps(ev["error"])[:500]
            meta = ev.get("meta_info") or meta
            ids = ev.get("output_ids") or []
            n_total = meta.get("completion_tokens")
            cumulative = n_total is not None and len(ids) == n_total and len(ids) > len(out_ids) and out_ids == ids[: len(out_ids)]
            piece = ev.get("text") or ""
            if cumulative and len(out_ids) > 0:   # cumulative stream: text and ids are the whole output so far
                piece = piece[len(text):] if piece.startswith(text) else piece
                new_ids = ids[len(out_ids):]
            else:
                new_ids = ids
            if new_ids:
                if t_first_id is None:
                    t_first_id = time.perf_counter()
                out_ids.extend(new_ids)
            if piece:
                if t_first is None:
                    t_first = time.perf_counter()
                n_chunks += 1
                text += piece
    t1 = time.perf_counter()
    # /generate has no reasoning parser: its text starts with the model's channel header (" to=self", <|message|> is
    # skipped as special). The chat path's reasoning parser strips it; strip it here too so agree.py compares like text.
    m = CHANNEL_HEADER.match(text)
    return {
        "status": status,
        "error": error,
        "raw_head": raw_head,
        "t_start": t0,
        "ttft": (t_first - t0) if t_first else None,
        "latency": t1 - t0,
        "completion_tokens": meta.get("completion_tokens"),
        "prompt_tokens": meta.get("prompt_tokens"),
        "cached_tokens": meta.get("cached_tokens"),
        "chunks": n_chunks,
        "text": text[m.end():] if m else text,
        "output_ids": out_ids,
        "input_len": len(input_ids),
        "ttft_ids": (t_first_id - t0) if t_first_id else None,
    }


async def run_slot_ids(session, args, plans, replay, rung_key, slot, convs, records):
    for c in range(convs):
        key = f"{rung_key}/s{slot}/c{c}"
        first, deltas = plans[key]
        ids = list(first)
        for t in range(len(deltas) + 1):
            rec = await one_request_ids(session, args, ids)
            rec["key"] = f"{key}/t{t}"
            records.append(rec)
            if t == len(deltas):
                break
            prior = replay.get(rec["key"]) if replay else None
            # an ids-mode replay carries the generated ids: prompts stay identical across runs. A chat-mode replay has only
            # text, which never was the generated sequence (the chat path strips the model's " to=self<|message|>" header and
            # re-renders the reply as " to=user" content), so this run's own generated ids are used, as the judge's client would.
            reply = prior["output_ids"] if prior and prior.get("output_ids") else rec["output_ids"]
            ids = ids + list(reply) + deltas[t]


async def run_slot(session, args, files, replay, rung_key, slot, convs, records, base):
    for c in range(convs):
        key = f"{rung_key}/s{slot}/c{c}"
        messages = []
        for t, new in enumerate(build_conversation(key, files, args, base + slot * convs + c)):
            messages.extend(new)
            rec = await one_request(session, args, messages)
            rec["key"] = f"{key}/t{t}"
            records.append(rec)
            # Replay keeps prompts identical across runs even when outputs differ.
            prior = replay.get(rec["key"]) if replay else None
            messages.append({"role": "assistant", "content": prior if prior is not None else rec["text"]})


def summarise(conc, recs, wall):
    ok = [r for r in recs if r["ttft"] is not None and r["completion_tokens"]]
    toks = sum(r["completion_tokens"] for r in ok)
    # Interactivity: tokens/s a single stream sees once it has started.
    inter = [(r["completion_tokens"] - 1) / (r["latency"] - r["ttft"]) for r in ok if r["latency"] > r["ttft"]]
    ttft = sorted(r["ttft"] for r in ok)
    return {
        "concurrency": conc,
        "requests": len(recs),
        "failed": len(recs) - len(ok),
        "wall_s": wall,
        "throughput_tok_s": toks / wall if wall else 0.0,
        "interactivity_tok_s_median": statistics.median(inter) if inter else None,
        "ttft_s_median": statistics.median(ttft) if ttft else None,
        "ttft_s_p90": ttft[int(0.9 * (len(ttft) - 1))] if ttft else None,
        "wrong_length": sum(1 for r in ok if r["completion_tokens"] != 220),
    }


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default="default")
    ap.add_argument("--out", required=True)
    ap.add_argument("--replay", help="earlier run whose assistant turns are reused as history")
    ap.add_argument("--plan", choices=["scored", "check"], default="scored")
    ap.add_argument("--rungs", help="override, e.g. 1x1x1,16x1x2 (concurrency x reps x convs)")
    ap.add_argument("--corpus", default=os.path.join(os.path.dirname(__file__), "..", "sglang", "python", "sglang", "srt"))
    ap.add_argument("--trace", help="agentic-trace JSON to replay instead of the synthetic corpus")
    ap.add_argument("--turns", type=int, default=None, help="turns per conversation (cap, with --trace); default from --shape")
    ap.add_argument("--shape", choices=sorted(SHAPES), default=None,
                    help="conversation shape (see the docstring); default judge with --proxy, short with --proxy-short, else legacy")
    ap.add_argument("--proxy", action="store_true",
                    help="proxy preset, judge shape: rungs 16x1x2, 13 turns; explicit --rungs/--turns/--shape win")
    ap.add_argument("--proxy-short", action="store_true",
                    help="the pre-Nov-15 proxy preset: rungs 16x1x2, 4 turns, 280k/2150 chars (old runs stay reproducible)")
    ap.add_argument("--base-chars", type=int, default=None, help="first-turn context size, chars (default from --shape)")
    ap.add_argument("--turn-chars", type=int, default=None, help="later-turn size, chars; draws from [x/2, 2x) (default from --shape)")
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0, help="first --trace conversation to use")
    ap.add_argument("--client", choices=["chat", "ids"], default="chat", help="chat: /v1/chat/completions text (default); ids: /generate input_ids (see the docstring)")
    ap.add_argument("--tokenizer", default=TOKENIZER, help="--client ids: tokenizer path or HF id (the server's --model-path); fake = dry-run stand-in")
    args = ap.parse_args()
    if args.proxy and args.proxy_short:
        ap.error("--proxy and --proxy-short are exclusive")
    if args.proxy or args.proxy_short:
        args.rungs = args.rungs or "16x1x2"
    if args.shape is None:
        args.shape = "judge" if args.proxy else "short" if args.proxy_short else "legacy"
    sh = SHAPES[args.shape]
    if args.turns is None:
        args.turns = sh["turns"]
    if args.base_chars is None:
        args.base_chars = sh["base_chars"]
    if args.turn_chars is None:
        args.turn_chars = sh["turn_chars"]

    rungs = CHECK_RUNGS if args.plan == "check" else SCORED_RUNGS
    if args.rungs:
        rungs = [tuple(int(x) for x in r.split("x")) for r in args.rungs.split(",")]
    args.trace_convs = load_trace(args.trace) if args.trace else None
    files = None if args.trace else load_corpus(args.corpus)
    replay = None
    if args.replay:
        with open(args.replay) as f:
            if args.client == "ids":
                replay = {r["key"]: {"text": r["text"], "output_ids": r.get("output_ids")} for r in json.load(f)["records"]}
            else:
                replay = {r["key"]: r["text"] for r in json.load(f)["records"]}
    ids_mode = args.client == "ids"
    if ids_mode:
        t_tok = time.perf_counter()
        args.ids_template = IdsTemplate(load_tokenizer(args.tokenizer))
        print(json.dumps({"client": "ids", "tokenizer": args.tokenizer, "history_header": args.ids_template.history_header,
                          "glue_mid": args.ids_template.mid, "glue_end": args.ids_template.end,
                          "load_s": round(time.perf_counter() - t_tok, 2)}), file=sys.stderr, flush=True)

    # chat mode keeps the pre-Nov-17 args block byte for byte (no client/tokenizer keys)
    hidden = {"trace_convs", "ids_template"} | (set() if ids_mode else {"client", "tokenizer"})
    result = {"args": {k: v for k, v in vars(args).items() if k not in hidden}, "rungs": [], "records": []}
    if ids_mode:
        result["client"] = "ids"
    base = args.offset  # every rung and rep gets fresh conversations, so nothing is pre-cached
    timeout = aiohttp.ClientTimeout(total=None, sock_read=1800)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        for conc, reps, convs in rungs:
            for rep in range(reps):
                recs = []
                if ids_mode:   # every encode of this rung happens here, before its clock starts
                    tp = time.perf_counter(); rk = f"c{conc}/r{rep}"; plans = {}
                    for s in range(conc):
                        for c in range(convs):
                            key = f"{rk}/s{s}/c{c}"
                            plans[key] = args.ids_template.plan(build_conversation(key, files, args, base + s * convs + c))
                    print(json.dumps({"ids_plan": rk, "conversations": len(plans), "s": round(time.perf_counter() - tp, 2),
                                      "turn1_len_median": statistics.median(len(p[0]) for p in plans.values())}), file=sys.stderr, flush=True)
                t0 = time.perf_counter()
                if ids_mode:
                    await asyncio.gather(*(run_slot_ids(session, args, plans, replay, f"c{conc}/r{rep}", s, convs, recs) for s in range(conc)))
                else:
                    await asyncio.gather(
                        *(run_slot(session, args, files, replay, f"c{conc}/r{rep}", s, convs, recs, base) for s in range(conc))
                    )
                base += conc * convs
                summ = summarise(conc, recs, time.perf_counter() - t0)
                summ["rep"] = rep
                print(json.dumps(summ), flush=True)
                result["rungs"].append(summ)
                result["records"].extend(recs)
                with open(args.out, "w") as f:  # rewrite each rung so a crash keeps what ran
                    json.dump(result, f)


if __name__ == "__main__":
    asyncio.run(main())
