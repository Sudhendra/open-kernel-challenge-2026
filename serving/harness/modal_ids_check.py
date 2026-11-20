"""CPU-only check of ladder.py --client ids against the real Muse tokenizer on the okc-muse-weights volume (no GPU).

  modal run harness/modal_ids_check.py

Builds a judge-shape conversation with ladder.py's synthetic corpus, then compares the ids ladder's IdsTemplate builds
(turn 1; turns 2-3 with each reply given as its re-tokenised text) against the ids the server's chat path makes from the
same messages at the 297e54fd pin (serving_chat._render_and_encode_chat_template: render the Jinja template to text, encode
with add_special_tokens=False when encode("") adds specials, else plain; apply_chat_template(tokenize=True) when that round
trip is lossy). Prints the first difference if any. Reads only the volume's HF cache; prints no credentials.
"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIGEST = os.environ.get("OKC_IMAGE", "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f")
app = modal.App("okc-ids-check")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
         .env({"HF_HOME": "/vol/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_file(os.path.join(HERE, "ladder.py"), "/root/harness/ladder.py"))


@app.function(image=image, volumes={"/vol": vol}, timeout=1200, cpu=8, memory=16 * 1024)
def check():
    import importlib.util, json, sys, time
    sys.path.insert(0, "/root/harness")
    import ladder
    t = time.time(); tok = ladder.load_tokenizer(ladder.TOKENIZER); print(f"tokenizer {type(tok).__name__} loaded in {time.time() - t:.1f}s", flush=True)
    corpus = os.path.join(importlib.util.find_spec("sglang").submodule_search_locations[0], "srt")
    # the server's encode decision (serving_chat: _tokenizer_auto_adds_specials, _prompt_text_round_trip_is_lossy)
    auto_specials = len(tok.encode("")) > 0
    kw = {"add_special_tokens": False} if auto_specials else {}
    probe = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    lossy = list(tok.encode(tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True), **kw)) != list(
        tok.apply_chat_template(probe, tokenize=True, add_generation_prompt=True, return_dict=False))
    print(json.dumps({"auto_adds_specials": auto_specials, "round_trip_lossy": lossy}), flush=True)

    def server_ids(msgs):
        if lossy:
            return list(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False))
        return list(tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), **kw))

    tmpl = ladder.IdsTemplate(tok); print(json.dumps({"history_header": tmpl.history_header, "glue_mid": tmpl.mid, "glue_end": tmpl.end}), flush=True)
    H = tmpl.history_header   # emulate a model that generated exactly the template's history header before its reply
    args = type("A", (), dict(trace_convs=None, seed=4242, turns=3, base_chars=ladder.SHAPES["judge"]["base_chars"], turn_chars=ladder.SHAPES["judge"]["turn_chars"]))()
    files = ladder.load_corpus(corpus)
    replies = ["Here is the next step: the hot path is the loop in `run_batch`, which calls `_prepare` for every request.\n\n"
               "```diff\n- x = foo(a)\n+ x = foo(a, cache=True)\n```\nThis keeps the cached prefix valid.",
               "The second answer.  It ends with spaces and a newline?  No: it ends here."]
    ok = True
    for conv in range(2):
        turns = ladder.build_conversation(f"c32/r0/s{conv}/c0", files, args, 4000 + conv)
        t = time.time(); first, deltas = tmpl.plan(turns); plan_s = time.time() - t
        msgs = list(turns[0]); built = list(first); ref = server_ids(msgs)
        res = {"conv": conv, "plan_s": round(plan_s, 2), "turn1_len": len(built), "turn1_equal": built == ref}
        for k, d in enumerate(deltas):
            r = replies[k % len(replies)]
            msgs = msgs + [{"role": "assistant", "content": r}] + turns[k + 1]
            built = built + list(tok.encode(H + r, add_special_tokens=False)) + d
            ref = server_ids(msgs)
            res[f"turn{k + 2}_len"] = len(built); res[f"turn{k + 2}_new"] = len(d); res[f"turn{k + 2}_equal"] = built == ref
            if built != ref:
                ok = False; i = next((i for i, (x, y) in enumerate(zip(built, ref)) if x != y), min(len(built), len(ref)))
                res[f"turn{k + 2}_first_diff"] = {"index": i, "built": tok.decode(built[max(0, i - 8):i + 8]), "server": tok.decode(ref[max(0, i - 8):i + 8]),
                                                  "len_built": len(built), "len_server": len(ref)}
        ok = ok and res["turn1_equal"]
        print(json.dumps(res), flush=True)
    # what the chat path re-sends vs what the model generated, for a reasoning reply: generated " to=self<|message|>" + text
    gen = tok.encode(" to=self<|message|>" + replies[0], add_special_tokens=False)
    hist = tok.encode(H + replies[0], add_special_tokens=False)
    k = next((i for i, (x, y) in enumerate(zip(gen, hist)) if x != y), min(len(gen), len(hist)))
    print(json.dumps({"reasoning_reply_generated_vs_chat_history": {"common_prefix_tokens": k, "generated": len(gen), "chat_history": len(hist),
                      "generated_head": tok.convert_ids_to_tokens(gen[:6]), "history_head": tok.convert_ids_to_tokens(hist[:6])}}), flush=True)
    print("RESULT", "ALL EQUAL" if ok else "DIFFERENCES (see above)", flush=True)


@app.local_entrypoint()
def main():
    check.remote()
