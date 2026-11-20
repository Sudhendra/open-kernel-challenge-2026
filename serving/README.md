# Serving track: 3.073x stock SGLang on 8xH200

The serving track of the Open Kernel Challenge (Nov 12 to Nov 19, 2026) asked for a patch to SGLang at commit `297e54fd` (v0.5.23) that serves Muse Spark Open 816B (NVFP4 weights, DFlash speculative decoding, TP = EP = 8) faster on one node of eight H200s. The judge replays multi-turn agentic coding conversations at 1, 4, 16 and 32 concurrent users and scores throughput at stock SGLang's interactivity, as a geometric mean over the four loads. The patch must be lossless: nothing below stock's precision, no approximations, no skipped work.

**Final scored run: 3.073x**, job `71184012d50f4135`, all correctness, latency and memory gates passed. Standings are provisional until the organizers finish their code review and reruns.

## Result

Throughput at stock's interactivity (the scored quantity), from [evidence/judge_result_71184012d50f4135.json](evidence/judge_result_71184012d50f4135.json):

| Users | Stock tok/s per GPU | Patched tok/s per GPU | Ratio |
|---:|---:|---:|---:|
| 1 | 6,419 | 19,953 | 3.108x |
| 4 | 9,926 | 30,745 | 3.097x |
| 16 | 12,244 | 37,937 | 3.098x |
| 32 | 12,695 | 37,937 | 2.988x |

The measured ladder of the patched server:

| Users | tok/s per GPU | tok/s per user (end to end) | TTFT p90, first turn | TTFT p90, later turns | Accepted tokens per step |
|---:|---:|---:|---:|---:|---:|
| 1 | 12,311 | 267.8 | 2.95 s | 0.14 s | 4.3 |
| 4 | 21,730 | 122.7 | 6.61 s | 0.19 s | 4.4 |
| 16 | 33,056 | 47.3 | 25.93 s | 0.43 s | 4.4 |
| 32 | 37,937 | 27.6 | 57.14 s | 1.39 s | 4.4 |

Gates:

| Gate | Limit | This run |
|---|---|---|
| Prefill KL vs stock (mean) | at most 0.11 | 0.054 |
| Decode KL vs stock (mean) | at most 0.035 | 0.011 |
| Token agreement, first turn | at least 0.85 | 0.95 |
| Token agreement, later turns | at least 0.92 | 0.97 |
| GPU memory vs stock | at most +2 GB per GPU | 3.4 GB under |
| TTFT vs stock | at most 2x at each load | passed |

Scored runs over the week (times as the CLI prints them, US Central):

| Submitted | Job | Score | Note |
|---|---|---:|---|
| Nov 12 11:00 | `c6286ef374094ed9` | 2.232x | first scored file |
| Nov 12 20:52 | `186157d12aeb4eba` | 2.277x | |
| Nov 14 09:15 | `73da895e1a564297` | timed out | |
| Nov 14 12:54 | `82580c00e7154993` | 2.370x | |
| Nov 16 01:19 | `9f1b15a8d7804399` | 2.416x | online expert placement |
| Nov 16 20:11 | `faa832057bb147ba` | 2.681x | |
| Nov 17 20:50 | `788356d87cda4e27` | 2.859x | |
| Nov 18 20:16 | `c37d7cec730e4707` | 2.833x | prompt lookup removed; the score went down |
| Nov 19 23:55 | `71184012d50f4135` | 3.073x | the file in this repo |

## What the patch changes

Everything computes with stock's operand precisions and FP32 accumulation. What changes is where work runs, the order of FP32 sums, how many kernel launches a step takes, and when requests are admitted. [notes/NOTES_kfinal6ct.md](notes/NOTES_kfinal6ct.md) is the reviewer document attached to the scored job; it has a one-table map of the patch, a per-feature account of what differs from stock, accuracy tables, the source of every constant, and a list of what was not verified. In short:

- **Routed experts.** One persistent grouped NVFP4 kernel runs both expert GEMMs, in place of Marlin.
- **Decode-step glue.** A small-M GEMM with programmatic dependent launch, a one-launch router, a fused FP8 KV store, and all-gather fused with the norm.
- **Prefill.** An FP8 attention kernel with FlashAttention-3's arithmetic and key order, token-sharded prefill with FP32 folds at the owner rank, and a last layer that computes only the rows that are read.
- **Speculative decoding.** Prompt-lookup drafts from the request's own text, and an in-row branch verify that checks a second draft branch in the same forward. Committed tokens stay the target model's greedy tokens.
- **Verify attention.** A prefix kernel for the tree-masked verify on all 62 layers, with FA3 kept as the fallback.
- **Scheduling.** Online expert placement from the run's own routing counts, mixed prefill and decode chunks, a short-prefill delayer, and two admission flags.
- **Cache.** HiCache staging and write-back kernels and a KV pool split.

Two CUDA kernels (the grouped expert kernel and the prefill attention kernel) are derived from public Apache-2.0 kernel-track submissions; see [Attribution](#attribution).

## Apply it

```
git clone https://github.com/sgl-project/sglang && cd sglang
git checkout 297e54fd105e974cdf401e5b4d20253a22c16f90
git apply /path/to/serving/patches/kfinal6ct/okc_patch_kfinal6ct_u1.diff
```

The patch is 409,271 bytes across 67 files under `python/sglang/`, sha256 `7c77c0c81cf2f9bec2319973bb980c57744a5b22e1f34707ddfe3b9f96624771`. It is the reduced-context form of `git diff 297e54fd -- python/sglang` (one context line per hunk), produced by `harness/min_diff.py` to fit the track's 409,600-byte limit. Most explanatory comments were trimmed for the same reason, so read the patch with the notes beside it.

Run it in the image `lmsysorg/sglang:v0.5.23` with the judge's launch flags (the full line is in `harness/profile_session.sh`) plus the two flags the scored run used:

```
--swa-full-tokens-ratio 0.13 --min-free-slots-delay 12
```

It targets Hopper (`sm_90a`) and this model. The new paths sit behind compatibility guards with stock's code as the fallback (notes, section 5); the guards and fallbacks were covered by CPU tests only, as section 11 of the notes says.

## What is here

| Path | What |
|---|---|
| `patches/kfinal6ct/okc_patch_kfinal6ct_u1.diff` | the scored patch |
| `notes/NOTES_kfinal6ct.md` | reviewer notes attached to the scored job |
| `evidence/` | the judge's result for the scored run, and the kernel accuracy, timing and test logs the notes cite |
| `research/lane_kdatt_2026-11-19.md` | the verify-attention prefix kernel: design, accuracy against FA3 and FP64, timings |
| `research/lane_kdiet_2026-11-19.md` | how 12 KB were removed to fit the byte limit |
| `research/dead_ends_considered.md` | ideas that were measured and dropped, with the numbers |
| `METHOD.md` | how levers were priced and A/B tested |
| `harness/` | Modal 8xH200 session runner, load generator, judge simulator, result readers, patch tools, and the single-GPU microbenchmarks |

Notes on reading these:

- The notes cite logs as `out/<name>`; they are here as `evidence/<name>`. In those copies the local home directory and the Modal workspace name are replaced with placeholders. Nothing else is changed. The judge result is a trimmed copy; its `_note` field says what was left out.
- The harness and research files are working files from the competition, not a packaged tool. They refer to SGLang worktrees named `lanes/<name>` and to a Modal volume holding the weights; neither is in this repository. To recreate a tree, apply a patch to a clean checkout as above.
- The day-by-day ledger, the other 130 or so candidate patches and the raw 8-GPU run outputs are not published.

## What I would do differently

- **Price every hot kernel against its physical floor in the first week.** The best idea of the last week came from one such table on the last evening: stock's verify attention moved bytes at well under half the memory bandwidth. Found a week earlier it could have been finished and checked on the judge.
- **Trust the official check over the local proxy.** The local load test and the judge disagreed more than once. Expert placement tables read +8 to 14% locally and nothing on the judge. Prompt lookup showed no gain locally, was removed, and the next scored run went down.
- **Hold one official check for the true final file.** The file that was scored had not been checked on the judge itself; it went in on clean 8-GPU boots and two independent reviews. It passed, but that was a risk taken, not a plan.
- **Gate late changes hard.** Exactness against stock and an FP64 reference on adversarial inputs, CUDA-graph replay against eager, the CPU suite in the judge image, an independent review, and a boot of the exact tree. Reviews caught a stale cached schedule inside a captured graph that no test had reached.
- **Write reviewer notes that match the code line for line**, with a stated reason above every bare constant.

## Attribution

- SGLang (`sgl-project/sglang`, Apache-2.0) is the base this patch modifies.
- `kernels/jit/csrc/moe/muse_spark_fp4_grouped.cuh` and `kernels/jit/csrc/muse_spark/prefill_attention.cuh` are derived from public kernel-track submissions of this challenge: vjeux's `okc-2026` repository (Apache-2.0), the expert kernel through alygg77's submission of it. Their attribution headers are kept in the patch.
