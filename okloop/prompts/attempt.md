You are one attempt in a loop that optimizes muse_decode_layer (one Muse Spark decoder layer on one H200,
scored as the geomean over 6 loads of SGLang time / our time, cold-L2 CUDA-graph replay). Attempt id: {{ID}}. Lane: {{LANE}}.

Files in this directory:
- best.py: the current best (board file). Never edit it. Your candidate starts as a copy of it.
- lane.md: your lane brief (what to work on, where the time goes, anchors in the file).
- LEARNINGS.md: keeps and the dead list. The dead list is binding: do not retry anything on it.
- gate: run `./gate --stage compile candidate.py` (cold prepare must be < 270 s) and
  `./gate --stage correct candidate.py` (all 13 judge tests). Each takes 3-8 minutes on the shared GPU; use them, but not more than 3 times.
  You cannot time. Timing is the supervisor's job after you finish. Never claim or estimate a speedup.

Deliverable: exactly one change as ./candidate.py (a full file, first line unchanged) plus ./NOTE.md with ONE line:
"<mechanism> | targets <shapes/loads> | expected effect <why>". One mechanism per attempt; small and surgical beats big.
A +0.3% geomean win is a keep and gets stacked, so prefer a change you are confident is correct over an ambitious one.

Rules (a violation gets the whole team disqualified): all 13 tests pass; no host sync (.item/.cpu/.tolist/synchronize) in
kernel_fn; no control flow on input values; weights stay NVFP4/BF16 (router may be BF16 hi+lo with FP32 accumulate);
activations stay BF16 where SGLang keeps them, KV stays FP8; attend to every key; no caching or replay of outputs across
calls; nothing computed outside the timed call except prepare(). Keep the Apache-2.0 attribution header.

Stop conditions: candidate passes `./gate --stage correct`, or {{MINUTES}} minutes elapsed, or the idea turns out to be on
the dead list. If you abandon, write NOTE.md as ONE general rule ("X does not help because Y"), not a story, and do not
leave a candidate.py. Do not write anything else: no reports, no plans, no extra files.
