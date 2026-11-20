# METHOD — serving track (Nov 12, 2026 onward)

How we pick, build, measure and ship levers for the rest of the track. Adapted from the kernel-track method of
gauravjain14 (kernel-autoresearch, 7th on the kernel board; review in `research/kar_plugin_review_2026-11-12.md`) and
from what the judge's own JSON taught us on day one. Every rule here replaced a mistake we made at least once.

## 0. The loop

1. Model the score in milliseconds (section 1). Know what one ms per chunk and one ms per decode step is worth at each rung.
2. Bound every lever before building it: the cheapest measurement or arithmetic that would show it cannot pay, with the
   decision rule written down first ("build only if the bound saves >= X ms").
3. Build in a lane: one lever, one branch, CPU tests in the judge image, a patch file frozen under `patches/`.
4. Measure where the signal is: paired Modal A/B for anything under ~6%, a free judge check only for boot + agreement
   and for effects a check can resolve (section 3).
5. Keep every verified win, stack it, and audit the stack before it becomes the scored base (section 4).
6. Score only stacks. The scored run is the scarce resource; a check is cheap.
7. Re-score the value table after every measurement; kill directions on a measured bound, not on a first miss.

## 1. The score, as the judge computes it (confirmed on c6286ef374094ed9)

score = mean over c1/c4/c16/c32 of ratio_r, where ratio_r = OUR throughput at STOCK's per-user speed at rung r, read
off our own frontier (per-user speed -> tok/s/GPU over our measured concurrencies, log-log between points, capped at
our highest concurrency), divided by stock's throughput at r. `harness/score_model.py reproduce` rebuilds it to 0.4%.

Consequences (day-one numbers):
- Our per-user speed is >2x stock everywhere, so the judge reads our frontier at HIGHER concurrency than the rung: the
  c16 and c32 rungs both read our c32 point (28,250 tok/s/GPU). **c32 throughput is half the score**: +1% at c32 =
  +0.50% score; +1% at c4 = +0.39%; at c16 = +0.22%; at c1 = +0.18%.
- Raw ratios are c1 1.52, c4 1.81, c16 2.16, c32 2.23; the frontier turns c1 into 2.13. The "c1 rung" is really our
  throughput between the c1 and c4 points at 141.6 tok/s/user: c4 throughput and c1 per-user speed both move it.
- 10 ms off every 16k prefill chunk = +1.1% score. 1 ms off every decode step = +3.5% score (c4 +1.5, c1 +0.9, c32 +0.6,
  c16 +0.5). Decode-step ms are worth 3x prefill-chunk ms per unit, because the decode step is ~25x shorter.
- TTFT is 0.40x stock with a 2x allowance: 5x headroom for trades that spend TTFT to buy throughput.
- The c32 point is capped by the judge's max-running-requests 32: per-user speed above stock's at c32 buys nothing;
  only c32 throughput does. KV capacity (31-32 resident), acceptance (4.8) and chunk time at c32 are the levers there.

Run `python3 harness/score_model.py value <scored run json>` after every scored run; the table above is replaced by it.

## 2. Reading a judge result

`python3 harness/read_check.py <job> --touches c4,c16` prints per-rung raw ratios against the incumbent's RAW ratios
(a check at c16 reports raw c16, not the frontier number: 2.103 vs 2.293 on day one was not a loss), the lossless
gates, memory and TTFT, and a verdict:
- the noise band is ±2% (noop check 0.983x); c1/c4 scored-run spreads are 6-9% across reps;
- the **untouched-rung test**: rungs the lever cannot move are controls; if they moved as much as the touched rungs, the
  node moved (identical source scored 3.4% apart on the kernel boards; judge nodes differ ~4% in HBM speed);
- NO-GAIN inside the band is not "no gain": it is "this lever cannot be resolved by one check" (section 3).

## 3. What a check can resolve (runs_needed, sd(log) ≈ 0.02 at c16, ≈ 0.05 at c1/c4)

| lever size | runs per arm at c16 | at c1/c4 |
|---|---|---|
| +1% | 64 | 390 |
| +2% | 16 | 100 |
| +3% | 8 | 44 |
| +5% | 3 | 17 |
| +10% | 1 | 5 |

So: a check proves boot, agreement, memory and TTFT for any patch, and a throughput effect only when it is >= ~6% at
c16/c32 or >= ~15% at c1/c4. Levers under that are accepted on construction (provably skipped work with identical
outputs, e.g. T1, C1) or measured on a paired Modal replay (same boot image, stock-self control, 2+ reps), never on a
single check. Never spend the scored run on a single small piece.

## 4. Lanes, stacks, audits

- A lane = one lever, one worktree cut from the current best branch, one `README.md` with: the mechanism, why it is
  lossless under the rules, the bound (ms it can remove, from a trace or replay), the exact rows/paths it changes, the
  tests run in the judge image, the patch files and their lint, and "what only a GPU run can show". Template:
  `lanes/LANE_TEMPLATE.md`. A lane never launches GPU work, Slack or outer-repo commits.
- Before combining lanes, diff every pair for same-line touches (muse_spark.py is the hot file: H, C1, R2, D1 all
  touch it). Stack frozen patch files, never live worktrees.
- **Audit before scoring**: every stack that may become the scored base gets one read-only audit session hunting what
  tests cannot reach (state between calls, graph-capture vs eager divergence, shapes only the proxy takes, reduction
  order), and must prove a hazard with a test before raising it. Audits done: none yet (C1 eager break, R2 symm-mem
  init are the first two).
- Dead ends go to `research/dead_ends_considered.md` with the numbers; a dead end is reopened only on a new bound
  (K-eager was reopened on the judge's TTFT slack).

## 4b. Pre-GPU gates (added Nov 12 after R2 session A burned $19 on a capture crash)

What went wrong: session A ran on code whose unit tests covered the math but not the contract the code runs under. The
prefill path replays CUDA graphs with no Python, and the lane had three capture-contract bugs (a CPU->CUDA copy at capture,
a static-buffer row read as if live, padding rows counted). A one-hour read found all three for cents. The harness change
that carried the kernel test had also never been proven to reach the container. Rules, enforced before any Modal serving
session on a lane:

1. **Contract review before GPU.** Any lane that runs inside a captured region (prefill breakable graph, decode/verify
   graphs) or on a side stream gets a read-only capture-safety review (the `CAPTURE_REVIEW.md` state table: every state
   item classed as live-tensor / eager-break / frozen-Python, with file:line) before the first GPU session. The review
   is a separate session on Opus, never the author.
2. **Replay-simulation test.** The lane's CPU tests must include one that calls the per-chunk path twice with changed
   inputs and asserts the second call reflects them (what replay does), and one with padded rows past the live count.
3. **No Python scalars into device tensors on a captured path** (`t[i] = int`, `.item()`, `.tolist()`, `numel()` gates,
   `if tensor:`); grep for them in the lane diff before the review (`harness/capture_lint.sh`).
4. **Harness changes are proven end to end on the free path first**: a new session flag or env must show up in the
   container's log banner (`SRC_STOCK=...` block) in a CPU dry run or the first minute of the session, else stop the session.
5. **Stop at the first failed gate**, and read the failure in the same hour: a $19 session that crashes at boot still
   pays for itself only if the fix lands before the next one.
6. **Session line goes through a checklist in the lane README section 6** (gates in order, what each proves, what to
   look for in the log) and the session is launched only against that list.

## 5. GPU spend

- Judge checks are the free H200s: 3 per day plus 1 scored. Plan the day's checks the evening before; the bell script
  (`harness/open_bell.sh`) queues them at the 00:00 UTC reset and can gate the scored run on the checks.
- Modal (8xH200, ~$36/h): one job at a time, state the worst case before launch, paired A/B with a stock-self control,
  stop at the first failed gate. Microbenchmarks that gate a multi-day build (overlap, NCCL) cost cents to a few
  dollars and come before the build, never after.
- Calibration proxy -> judge (log-linear, refit per check with a proxy twin): c1 1.33x, c4 1.13x, c16 1.05x, c32 n/a.

## 6. Record

- `VALUE_TABLE.md`: prediction written before measuring, never edited; "measured" line added under it.
- `NEXT_STEPS_2026-11-08.md`: one dated subsection per session or decision, with job ids and file paths.
- Memory: one fact per file; MEMORY.md index line.

## Pre-Modal gates for the HARNESS (added Nov 13 evening, after three containers were lost to harness bugs)

A lane's code may fail on first boot; the harness must not. Before any 8xH200 launch:
1. `bash harness/modal_launch.sh --variants "<exact line>" [--reuse-out <ts>] [--replace <app>]` is the only launcher. It runs
   `bash -n` + `ast.parse`, `harness/test_modal_queue.py` (queue loop with a fake runner), `harness/dryrun_session.sh` in the
   exact mode with the exact VARIANTS (the real profile_session.sh against a mock server/ladder/GPU; catches undefined
   functions, wrong branches, bad variant lines in ~1 min), checks every SRC=<name> is a mounted tree, and refuses to stop an
   app that has a running task (a stale "0 tasks" reading killed lane W's run on Nov 13).
2. A harness change that touches control flow (new mode, new key, new loop) gets the three dry runs (proxy, reuse, full) and
   the unit test green BEFORE the launch, not after a container dies.
3. Nothing a running container needs is edited under it: trees and harness are snapshots at launch; a fix means a new launch.
4. Each launch states: expected wall time per item, worst-case cost, and which file proves each result.
