# Muse Spark serving on 8 SM90 GPUs (TP = EP = 8, DFlash): kfinal6ct (branch verify v2 + gate kernel + FP32 prefill folds + prefill and verify attention on one FP8 kernel family in FA3's key order)

Base `297e54fd`. The file is the reduced-context form of `git diff 297e54fd -- python/sglang` (one context line per hunk, no `index` lines, no function name after `@@`); `git apply` and `patch -p1` both rebuild the tree exactly. 67 files, 409,271 bytes (`patches/kfinal6ct/okc_patch_kfinal6ct_u1.diff`, sha256 `7c77c0c81cf2f9bec2319973bb980c57744a5b22e1f34707ddfe3b9f96624771`; tree lanes/kfinal6ct `d396422efc`). Paths are under `python/sglang/`; tests live in the tree under `test/registered/`, not in the file.

**We removed most explanatory comments from this file to fit the 409,600-byte limit, after the organizer said they need not stay. Comments stating a numerical property and a one-line reason above each bare scheduling constant are kept; these notes carry the rest (section 7).**

## 1. What the patch does (map)

Everything computes with stock's operand precisions and FP32 accumulation. What changes is where work runs, in what order FP32 sums are taken, how many launches a step takes, and when requests are admitted. Names like stack-k27 or kpatt3t below are our own earlier submissions.

| Part | Files (under `python/sglang/`) | Class | Section |
|---|---|---|---|
| Grouped NVFP4 expert kernel (both expert GEMMs in one persistent kernel) | `kernels/jit/csrc/moe/muse_spark_fp4_grouped.cuh`, `kernels/ops/moe/muse_spark_fp4_grouped.py` | order; one rounding fewer | 2 |
| Decode-step GEMMs, router, fused KV store, all-gather + norm, all-reduce launch | `kernels/.../small_m_pdl_gemm*`, `router_mma.cuh`, `srt/layers/muse_spark/muse_spark_decode_shard.py` | order / exact | 2, 3 |
| FP8 prefill attention kernel (FA3's FP8 arithmetic, FA3's key order) | `kernels/jit/csrc/muse_spark/prefill_attention.cuh` | order | 6 |
| Token-sharded prefill with FP32 owner folds, expert pair push | `srt/layers/muse_spark/muse_spark_sp_prefill.py`, `muse_spark_sp_overlap.py` | order; rounding | 2, 3, 6 |
| Last layer computes only rows that are read | `srt/models/muse_spark.py` | nothing read is skipped | 2 |
| Online expert placement from this run's routing counts | `srt/layers/muse_spark/muse_spark_online_placement*.py` | order | 2 |
| Mixed prefill + decode chunks, short-prefill delayer, admission flags | `srt/managers/scheduler.py`, `min_free_slots_delayer.py`, `srt/arg_groups/muse_spark_hook.py` | scheduling only | 3, 10 |
| Prompt-lookup drafts (verify and accept stay stock's) | `srt/speculative/dflash_prompt_lookup.py` | committed tokens = target greedy | 3 |
| In-row branch verify: gate, layout, accept kernels; tree-masked attention | `srt/speculative/dflash_branch.py`, `dflash_worker_v2.py`, `srt/layers/attention/fa3_verify_tree.py` | committed tokens = target greedy; attention order | 4, 5 |
| Verify attention prefix kernel (all 62 layers) + correction / merge kernel | `kernels/jit/csrc/muse_spark/verify_tree_attention.cuh`, `kernels/ops/attention/muse_spark_verify_tree.py` | order | 4, 6 |
| HiCache staging and write-back, KV pool split | `kernels/jit/csrc/kvcacheio/staged_write_back.cuh`, `srt/mem_cache/*` | exact | 2 |

**Provenance.** `kernels/jit/csrc/moe/muse_spark_fp4_grouped.cuh` and `kernels/jit/csrc/muse_spark/prefill_attention.cuh` are derived from public kernel-track submissions of this challenge (vjeux's `okc-2026` repository, Apache 2.0; the expert kernel through alygg77's submission of it). Their Apache attribution headers are retained at the top of both files. No other file in the patch carries a third-party notice. The deterministic sampling renormalization at TP > 1 follows upstream SGLang #48111 (section 6, "For other models").

**What is new in this file over the ones we had judge-checked today** (kpick3mt at 32 and 16 users, kpatt3t at 32 users):
- (h) over kpick3mt: the FP8 prefill attention kernel walks its key tiles last to first, as FA3 walks its key blocks (15 lines). Same per-element arithmetic; its error now equals stock's (section 6). This is in kpatt3t.
- (i) over kpatt3t: the tree verify's prefix call runs on a Muse FP8 prefix kernel instead of FA3, on the 16 full-attention layers and (k) the 46 sliding-window layers (FA3's per-element arithmetic, order only; section 6).
- (j) byte reduction: 26 environment switches and tunables that this patch had added are removed, each replaced by its default; CPU-only reference arms moved into the tests. The default path is unchanged (section 6).
- A host-side guard (the kernel is not used for a sliding window of 8 keys or fewer) and comment wording.
- FA3's tree split schedules are no longer built before every decode step: since the prefix kernel serves every tree layer of this model, they are built only when a layer actually calls FA3 (at each such call, so a fallback captured in a CUDA graph rebuilds its schedule at every replay). 12 lines in `fa3_verify_tree.py`; no numerical effect; the eager per-step tree setup drops from 0.205 to 0.104 ms at every batch size on one H200 (`out/kdatt2_meta_old.log`, `out/kfinal4_meta2.log`).

**Earlier steps of the branch verify, for readers of our earlier notes:** (a) the gate is an expected-gain test on the drafter's lattice; (b) layout and accept are two Triton kernels instead of roughly 70-90 small torch ops per step; (c) compatibility guards (including `gamma == 7` for the pick) and a stock fallback; (d) the token-sharded prefill starts at 2049 rows, not 1024; (e) the Laguna drafter keeps its own projection under TP; (f) the gate is one Triton kernel (`_pick_kernel`); (g) the token-sharded prefill's folds sum in FP32 and round to bf16 once.

Tree tables are built eagerly before each replay (an in-graph build saved ~0.005-0.01 ms per step and was left out, section 9); FA3's tree schedules only when a layer calls FA3.

## 2. Base (stack-k27), unchanged except the prefill threshold

Labels: **exact** = bit-identical; **scheduling only** = same per-token math, batch composition changes; **order** = same products at the same precision, FP32 sums in another order; **rounding** = same precision, bf16 rounding at other points (stated).

- **Grouped NVFP4 experts**: FP4/E4M3 decoded exactly to bf16, bf16 wgmma, FP32 accumulation and combine, one rounding. Order; rounding: gate/up not rounded before SiLU x up (stock rounds); each expert GEMM output is rounded to bf16 once, after the FP32 global scale (stock's Marlin rounds the accumulator to bf16, then multiplies by the bf16 global scale). Activations stay bf16 (W4A16, as stock).
- **Decode boundary**: 1/8-column projections + all-gather at <= 512 rows; router: stock's products (TF32 at <= 8 rows, bf16 hi + lo above) in one launch for 2-64 rows, FP32 order; small-M PDL GEMM, FP32 split-K, one launch shape per (weight, row range) (order). Fused FP8 KV store, AG + norm (exact).
- **Attention**: FA3 verify schedule (exact); split counts (order); FP8 prefill attention with FA3's FP8 numerics (order).
- **Last layer**: middle chunks write K/V only (output unread); prompt-ending batches run the rest on read rows (order); logprobs / hidden states keep the full layer.
- 2shot_push all-reduce, FP32 sum (order). **Token-sharded prefill >= 2049 rows (section 6)**: o_proj replica (one FP32 GEMM); the owner sums the ranks' partials (or, with the pair push, the token's routed rows in k order) in FP32 and rounds to bf16 once (rounding: stock's NCCL ring rounds at every hop there, so this is the same precision class as stock's own custom all-reduce and above the ring). Online placement (order). Mixed chunks (scheduling only). HiCache staging (exact).
- **Load and numerics.** Precision never depends on load. As in stock (FA3's split heuristic, cuBLAS's choice by M), tile, split and kernel choices key on the forward's own row, request and route counts, so the order of a request's FP32 sums, and which keys its FP8 P is quantized against, can depend on what it is batched with. Queue state (waiting / decoding counts) is read by two admission rules only (section 3) and changes when a request is admitted, never what is computed.

## 3. Unchanged from our earlier files (described here in full)

- **Prompt-lookup drafts** (`srt/speculative/dflash_prompt_lookup.py`, `dflash_worker_v2.py`). On the host, each request's n-gram index of its own prompt and output finds the newest earlier occurrence of its last 8 tokens (exact compare). On the GPU, without a sync, if the realized bonus token lies on that row with >= 7 tokens after it, they replace the drafter's 7. Verify stays stock's 8-row block and prefix accept: every committed token is the target's greedy token; no rows are added. A later turn whose prompt starts with one of the 64 most recent indexed prompts reuses that prompt's key chunks (never output keys) and hashes only the new tail; the rows equal a fresh build (tested per step over 5- and 20-turn conversations). It reads only the request's own tokens; rows are identical alone and in any batch (tested). On by default only for this model's architecture (`MuseSparkForConditionalGeneration`); off: `SGLANG_ENABLE_DFLASH_PROMPT_LOOKUP=0`.
- **Short-prefill delayer** (`srt/managers/min_free_slots_delayer.py`, `scheduler.py`). Holds a request with <= 4096 uncached tokens for at most 4 scheduler passes while at least 4 requests decode, fewer wait than decode, and waiting + decoding < max running; FCFS. Admission only.
- **Small-M decode paths** (`kernels/ops/gemm/small_m_pdl_gemm.py`, `muse_spark_bf16_moe.py`): the small-M PDL GEMM (already used at <= 16 rows) also serves qkv + gate, o_proj and the pre-expert block at 17-32 rows where cuBLAS ran before (bf16 products, FP32 accumulation, fixed-order reduction, one bf16 rounding; error vs FP64 equal to cuBLAS's at every M); the two BF16-expert layers call stock's Triton finalize at every row count (stock uses it from 32 tokens): same FP32 norm / weight / sum, one rounding. Order.
- **Expert pair push** (`muse_spark_fp4_grouped.cuh`, `muse_spark_sp_overlap.py`, token-sharded prefill only): the expert kernel stores each routed output row into the token owner's slot instead of a local buffer; the owner sums the token's routed rows in k order in FP32 and rounds to bf16 once (section 6). Every routed row is stored once; nothing is skipped. The older routed-row fold remains as the path for pair rows that do not fit the slots; at this model's shapes they always fit, so it does not run.
- **Scheduler poll interval**: for this model the scheduler polls for new requests every 4th decode step (`scheduler_recv_interval`, stock 1; extend steps poll every step). Scheduling only.

## 4. In-row branch verify

Files as stack-k37 (`srt/speculative/dflash_branch.py`, `dflash_worker_v2.py`, `srt/layers/attention/fa3_verify_tree.py`, `flashattention_backend.py`, `kernels/.../verify_tree*`).

**Gate (`pick_branch` = one launch of `_pick_kernel`, in the captured draft graph).** It reads only the drafter's own lattice: `candidate_ids` [bs, 7, 16] and `scores` [bs, 7, 16, 16]. One program per request; the row softmax is FP32 (`max`, `sum(exp(x - max))`); ties go to the lowest index, and a later slot wins only with a strictly larger gain. No RNG, no atomics; every rank reads the same replicated inputs.
- Along the drafter's greedy chain, p_j is the top row-softmax probability, and surv_k = p_1 ... p_k.
- For slot e in 1-3: gain_e = surv_(e-1) x q2_e x (1 + c1_e), where q2 is the rank-2 probability and c1 is the top probability after it.
- loss = surv_6 + surv_7 (expected drafts 6-7 given up). The request branches at e = argmax gain iff gain - loss > `SGLANG_DFLASH_BRANCH_MIN_GAIN` (-0.3) and it is greedy.

How the constant was picked: offline, on recorded drafter lattices from our load generator's synthetic conversations (prompts built from SGLang's own source files), not the benchmark's data (20,520 anchors, 341 requests; research/branch_shape_opt_2026-11-19.md). The slot range 1-3 and the chain cut at 5 are stack-k37's, chosen in the same study; e >= 4 needs a per-request chain cut, and at e = 5 the swap would overlap itself.
- With -0.3 fixed, the replay estimate is +1.10% tokens per step over the 0.49 margin rule [+0.92, +1.26]. Cross-fit picks -0.15 to -0.3 (+1.03 to +1.09%).
- The offset corrects the raw softmax's miscalibration (a binned calibration with offset 0 gives the same tokens). Gated steps 70% vs 62%.
- An estimate from recorded lattices, not measured in serving. One value at every load.
- The torch form of the rule equals the closed form on 768 recorded lattices: slot 100%, branch tokens 100% on gated rows, gated 71.7% (`out/krec32/pick_parity.txt`). The shipped kernel is checked against that torch form (now the test's reference): under the Triton interpreter on 128 random cases plus six exact-tie cases, and compiled for sm_90a on an H200: 1,696 rows, 6 within 1e-4 of a tie excluded, 0 mismatches (`out/kwide2/kpick_bench_run1.log`).

**Rows, layout kernel (`branch_layout`).** The block stays 8 rows: anchor, drafts 1-5, then rows 6-7 = the rank-2 candidate at e and the lattice argmax after it, at positions p+e, p+e+1, KV written to the slots reserved for p+6, p+7. A request whose drafts the lookup changed keeps the chain. The whole step stays on the chain when the verify does not replay a captured graph, uses DP attention, or has a non-greedy, logprob, grammar, penalty, logit-bias or custom-processor request.

**Accept kernel (`branch_accept`).** When gated, the chain accepts at most 5 drafts.
- The branch wins only if the chain stopped exactly at e and row 6 equals the target's argmax at row e-1; row 7 then counts if it equals the target's argmax at row 6; the bonus is the target token at row 6 or 7.
- On a win, the kernel swaps the `req_to_token` entries of p+e, p+e+1 and p+6, p+7 (the same slot set, now in sequence order).
- It writes committed tokens, lengths and their rows/slots/positions, plus counter rows indexed by batch slot (no atomics; the log line that read them is removed in this file, the tests read them).
- Every committed token is the target's greedy token over exactly its committed context.
- Both kernels are integer selects and gathers (no floating-point arithmetic).

**Attention: lossless argument.** It runs on every row of every captured verify step, gated or not.
- **Query and K/V**: the same e4m3 query stock builds with its one cast; the stored e4m3 K/V bytes, made FP32 exactly; no descale (none in stock here).
- **Prefix**: one call per layer over the committed prefix (non-causal; sliding-window layers use window (w-8, 7)): the Muse FP8 prefix kernel (section 6), or FA3 forced to >= 2 splits if that kernel cannot serve the layer. Either returns FP32 per-split partials (O, LSE). The merge reads those partials, never a bf16 output.
- **Correction**: one kernel per layer attends each row over its <= 8 remaining keys with FA3's FP8 convention (P = exp2(s*c - (m*c - 8)) in FP32, FP32 row sum of the unrounded P, e4m3 P for P V, FP32 accumulation), merges with the partials in FP32 as FA3's split combine does, and rounds once to bf16. Built with `--use_fast_math`, as stock builds FA3.
- **What changes**: only the key partition (FP32 summation order). Rows 6-7 never see chain rows e..5, and chain rows never see rows 6-7.
- **Tables**: `_tables_kernel` (Triton, one launch per page table) builds them before each replay; this is the code stack-k37 runs on GPU.

**Reads:** the drafter's scores, the sampling mode, whether the lookup changed the drafts, whether the step replays a graph. None changes a committed token.

## 5. Compatibility guards and stock fallback

The tree verify is on only if all hold (else stock's chain verify): `SGLANG_DFLASH_BRANCH`; DFLASH target runner; FA3 with its verify split policy; page size 1; e4m3 KV; (heads, kv heads, head dim) = (16, 2, 64); no softcap; full attention or SWA with w >= 8 and its own pool; 8 draft tokens.

The draft-side pick is built only with gamma == 7 (the 8-token block its lattice walk is written for) and selector top_k >= 2 (`topk(2)`). The worker decides once (`_branch_ready`): selector sampler, gain constant, FA3 backend, `tree_verify`.

**Fallback.** `_tree_verify_attention` returns None, and the layer takes the stock FA3 call, for any of:
kwargs beyond k/v descale or descale without float scales; local, cascade or cross attention; a logit cap; a window without a schedule or table.

It then sets `tree_verify = False`. It never raises. The other layers keep the tree call with every request on the chain (the same keys; FP32 summation order only), and a layer that fell back runs FA3's own split choice, as stock does.

Every layer meets this check during the boot-time graph capture, before the worker first reads the flag, so a configuration that falls back builds no branch row (read from code; tested on a fake layer only). Muse Spark takes none of these paths.

## 6. Defaults that change, and why (same precision)

**Token-sharded prefill: from 2049 rows, FP32 folds** (`muse_spark_sp_prefill.py` `min_tokens`; `muse_spark_sp_overlap.py` `_ring_fold_kernel`, `_routed_ring_fold_kernel`, `_pair_fold_kernel`).
- Stock all-reduces bf16 rank partials. Its custom all-reduce handles tensors up to and including 16 MiB (2048 rows of the 8 KB expert output) and sums in FP32 with one rounding; above that it uses NCCL, whose ring rounds to bf16 at every hop.
- In the region, the owner's fold now accumulates in FP32 across all source ranks and rounds once. With the pair push it sums the token's routed rows directly in k order (no per-rank bf16 partial), one rounding. Never below stock: equal to the custom all-reduce's class, above the ring's.
- Two of the 62 MoE layers (the BF16-expert layers, which do not use the expert push) and the attention scatter when its slots do not fit use NCCL's reduce-scatter (bf16 per hop). That equals stock only above 16 MiB, which is why the region still starts at 2049 rows; below it stock's path runs unchanged.
- Our earlier files rounded per hop inside these folds to reproduce NCCL's ring bit for bit; this file does not depend on which algorithm NCCL picks on a given node.
- Error vs an FP64 sum on an H200 (`out/kpick2/gpu_fold_bench.log`): ring and routed folds 3.1-4.6 -> 1.0-1.9 (max, bf16 ulp scale of the test data); pair fold 0.077-0.111 -> 0.031.

**FP8 prefill attention: FA3's key order** (`kernels/jit/csrc/muse_spark/prefill_attention.cuh`).
- The kernel's per-element arithmetic is FA3's FP8 path: P relative to the running row max with the 2^8 offset, `ex2.approx`, E4M3 round-to-nearest (saturating), FP32 row sum of the unrounded P, FP32 accumulation of P.V, the FP32 rescale of the sums by exp2((m_old - m_new) * c) when the max grows (as FA3), one bf16 rounding of the output. What differed from FA3 was only the ORDER in which keys are visited: our earlier files walked key tiles first to last, FA3 walks its key blocks last to first. With a large early score (for example an attention sink at key 0) forward order quantizes every later key's P against that maximum, so a long tail lands on E4M3's subnormal grid; FA3's order quantizes the tail against its own maximum first.
- This file walks key tiles last to first. Max abs error vs an FP64 reference over every query row, head and dimension, 19 input classes on an H200 (`out/kpatt3/bench_run4.log`; table in research/lane_kpatt3_2026-11-19.md): identical to FA3's to four digits on 16 classes (random, sink, peaked, flat, large, growing; full and sliding-window); 1.014x, 1.018x and 1.022x FA3's on two long-tail classes and one flat sliding-window class, which is the remaining difference in block size (our 128-key tiles vs FA3's 160-key blocks: order of FP32 summation and of the running max, not arithmetic). Mean error within 1% of FA3's except +8% on 1k-token rows with no history and +12% on one growing sliding-window class.
- Our earlier files (forward order) read above FA3 on some classes: long tail 0.0070 vs 0.0039, growing 0.0512 vs 0.0418, random 0.0330 vs 0.0304; below FA3 on others (peaked sliding-window 0.065 vs 0.085). Same arithmetic, different order.
- Speed is unchanged: 70.68 ms per mean cold chunk for the 62 layers vs FA3's 79.10 ms on one H200 (forward order 71.38).

**Full-layer verify attention: Muse FP8 prefix kernel** (`kernels/jit/csrc/muse_spark/verify_tree_attention.cuh` `prefix_kernel`, op `muse_spark_verify_tree.prefix_verify`, call site `fa3_verify_tree.tree_verify_attention`).
- On the 16 full-attention layers the tree verify used to call FA3 non-causally over each request's committed prefix (p keys) and merge FA3's FP32 split partials with each row's <= 8 new keys in our correction kernel. That FA3 call is replaced by our kernel; the correction / merge kernel is unchanged.
- The kernel computes what FA3 computes for that call: E4M3 Q and K/V read from the paged cache as stored (no scales other than FA3's); S = Q K^T by FP8 wgmma with FP32 accumulators; the running row maximum with P = ex2.approx(S * c - (m * c - 8)) (2^8 at the running max, as FA3's FP8 path); the FP32 row sum of the unrounded P; P rounded to E4M3 for an FP8 P V wgmma with FP32 accumulators; the FP32 rescale of O and l by exp2((m_old - m_new) * c) whenever the maximum grows, as FA3; every key attended; per split a normalized FP32 O and natural-log LSE in FA3's partial layout, merged in FP32 by LSE. These per-element steps are the prefill kernel's (`prefill_attention.cuh` `softmax` / `issue_qk` / `issue_pv`, reused unchanged).
- What differs from FA3 is order only: 128-key tiles (FA3: its own block size); each split's tiles walked last to first (as FA3); a different split partition (max(2, min(128, num_SMs // batch)) contiguous ranges per request, capped at the request's tile count: 1 split for a short prefix, none for an empty one); the merge order.
- The speed comes from structure: one CTA per (request, split) covers both kv heads, so each cache row is one 128-byte gather; a producer warpgroup keeps 3 tiles of K/V in flight into a 6-stage ring and transposes V in shared memory.
- Non-FP8 caches, KV shapes other than 2 heads x 64, page size != 1, and a sliding window of 8 keys or fewer (tree window <= 0) keep FA3. Every check that can raise runs on the host before the first launch and depends only on dtypes and shapes, so all 16 layers of a graph take the kernel or all take FA3. No persistent memory (partials come from the graph pool, ~4.3 MB per call at most, replacing FA3's split buffers).
- Two changes to `prefill_attention.cuh` serve it and are bit-identical for prefill: `Ctx.nwg` / `Ctx.stages` (defaults = the prefill values) and a shared V-transpose helper (same loads and stores). The prefill kernel's 11 GPU tests pass unchanged.
- Time, 1 x H200, judge image, CUDA graphs of 16 full + 46 sliding layers, own KV per layer, prefixes 75k-86k, page size 1 (`out/kdatt_gpu12.log`), ms over stock's causal FA3 for the 62 layers: FA3 tree path +0.262 / +0.275 / +0.329 / +0.360, with the prefix kernel +0.222 / +0.096 / -0.662 / -1.834 at 1 / 4 / 16 / 32 requests = 0.04 / 0.18 / 0.99 / 2.19 ms per decode step saved. Prefix call alone per full layer 12.7 / 34.1 / 111.9 / 226.7 us vs FA3 20.7 / 54.0 / 184.6 / 360.6 us (`out/kdatt_diag7.log`, forward order; last-to-first costs ~0.1 ms per step at 32 requests).
- Accuracy vs an FP64 softmax on the dequantized FP8 inputs:
  - 64-configuration verify gate (B 1/4/16/32; prefixes 0 to 100k; full and sliding; chain and branch masks): worst ratio of our error to stock FA3's 1.0008, identical to the FA3 tree path it replaces. Full layers, max / mean error: p 70k-100k 0.00501 / 0.00319 vs stock FA3 0.00655 / 0.00407; p ~ 2k 0.00448 / 0.00377 vs 0.00591 / 0.00455.
  - Seeded adversarial classes at the verify shape (`out/kdatt_adv_seeds.log`; sink at key 0, flat tails 15.8 / 17.8 / 19.8 binades below the max, growing, peaked, flat, random; 5-20 seeds each, B = 1 and 32). Ratio of our max error to stock FA3's, min / mean / max per class: random 0.39-0.67 / 0.85-0.93 / 1.00; sink 0.67-0.98 / 0.85-1.00 / 1.04; peaked 0.62-0.69 / 0.88-0.94 / 1.00; flat 0.86-1.00 / 0.97-1.02 / 1.09; growing 0.91-1.00 / 0.99-1.00 / 1.13; tails 0.96-1.00 / 1.00 / 1.08. For scale, stock FA3 against itself at another split count on the same inputs: mean up to 2.67, max up to 5.15 (1 split), max 2.06 (16 splits). The per-class mean is 0.85-1.04; no class sits systematically above FA3.
  - With an empty prefix the kernel does no work and the output is the already-shipped correction kernel's (ratio 1.00-1.22 on that class, identical to our earlier files); an empty prefix at a full layer does not occur in decode.
  - Deterministic; CUDA-graph replay equals eager at every batch size 1-40, 48, 64 (`out/kdatt_sweep1.log`).

**Sliding-window verify attention on the same kernel** (same files; the kernel is a template on the window, so the full-layer build is unchanged).
- The 46 sliding-window layers (w = 2048: FA3 window (2047, 0)) use the same tree verify: one call over each request's committed prefix (p old keys), then the unchanged correction kernel, which adds each row's <= 8 new keys and, for branch rows, the window-edge keys from the tree's table. That prefix call was FA3 non-causal with window (2039, 7), so verify row r (0..7) attends exactly the old keys [p + r - 2047, p) (positions below 0 dropped).
- The kernel reproduces that per-row range: it reads key tiles from row 0's first key to p - 1, and the prefill softmax's existing per-row lower bound masks each row's keys below p + r - 2047 to -inf before the row max, as for any masked key. Per-element arithmetic is the full-layer kernel's; only order differs (128-key tiles, last to first; the same split rule max(2, min(128, num_SMs // batch)), which for a 2,048-key window means one 128-key tile per split up to 7 requests and 4 tiles per split at 32).
- A row can have every key of a split masked: that split holds O = 0 and LSE = -inf for the row (FA3's empty-split convention) and the merge gives it weight 0.
- Time, same-container A/B on one H200 (`out/kdatt_ab2.log`, `out/kdatt_ab3.log`): a further 0.18 / 0.18 / 0.13 / 0.16 / 0.17 / 0.20 / 0.19 ms per decode step at 1 / 4 / 8 / 12 / 16 / 24 / 32 requests. Both kernel steps vs FA3's tree path: 0.22 / 0.36 / 1.18 / 2.40 ms per step at 1 / 4 / 16 / 32 (`out/kdatt2_gpu4.log`).
- Accuracy vs an FP64 softmax on the dequantized inputs:
  - 64-configuration gate (incl. window boundaries p = 2040 / 2048 / 2100; chain and branch masks): worst ratio of our error to stock FA3's on sliding layers 1.0031. The FA3 tree path it replaces (in our earlier files) reads 1.114.
  - Seeded adversarial classes at w = 2048 (`out/kdatt2_adv_3path.log`; ratio of max error to stock causal FA3's, mean over 5-20 seeds; (1) this kernel, (2) the FA3 tree path of our earlier files, (3) stock FA3 at 1 split):
    - 32 requests: (1) at or below (2) in every class (random 0.757 vs 0.795; growing 0.997 vs 1.158; flat window 0.968 vs 1.011; edge p = 2048 0.945 vs 1.022).
    - 12 requests: likewise (flat window 0.985 vs 0.999).
    - 1 request: (1) at or below (2) in every class except the flat-window class (a run of near-equal logits filling the window): **mean 1.062, against 1.002 for (2) and 1.123 for (3)**; worst seed 1.312 (FA3 at 1 split reads 1.381 on that seed). Fewer splits made it worse (4 splits: mean 1.107), so one 128-key tile per split at 1 request is kept. This is the one class where the mean is above stock's default; it lies inside stock's own spread across split counts.
  - Deterministic; CUDA-graph replay equals eager at batch sizes 1-40, 48, 64 for both window kinds (`out/kdatt2_sweep2.log`).
- GPU tests on an H200 (`out/kdatt2_gputests3.log`): 21 passed = the prefill kernel's 11 + 10 prefix-kernel tests (full and w = 2047, incl. p 2040 / 2048 / 2100).

**Removed switches (bytes)**. To fit the kernel above under the size limit, this file deletes environment variables that our own patch had added, keeping the arm that ran when they were unset: the off switches for the all-reduce PDL trigger and workspace input/output, the DFLASH context GEMM warm-up and TP context projection, the FA3 verify split policy, the mixed-chunk final-chunk rule, the all-gather + norm row kernel, the decode row split, the expert pair push and sparse push, the grouped FP4 row split, the last-layer read rows, the prefill key split, the short-prefill delayer and the small-M decode paths; the tunables for the HiCache quiet sync interval (32), the delayer (4096 tokens, 4 passes), the lookup match length (8), the online placement (60,000 rows, 0.5, 0.02, 0.02, 4 layers per forward); and the branch counter log line. Kept: `SGLANG_DFLASH_BRANCH`, `SGLANG_DFLASH_BRANCH_MIN_GAIN`, `SGLANG_ENABLE_DFLASH_PROMPT_LOOKUP`. CPU-only reference arms of the verify schedule fill and the sharded-prefill folds moved into their tests (production tensors are always CUDA there; a CPU tensor now raises). An independent read of every hunk found the default path identical; no shape, dtype or model guard and no stock fallback was removed.

**Laguna drafter** (`srt/models/dflash.py`): `DFlashLagunaForCausalLM.project_target_hidden_tp` returns its own replicated projection (per-slice norms before `fc`), so the column-split TP projection applies only to drafters without that override. No effect on Muse Spark.

**For other models:** `SGLANG_SPEC_TP_SYNC` all -> init with deterministic renorm at TP > 1 (upstream #48111); FA3 verify split policy for any FA3 non-MLA verify; 2shot_push up to 2.5 MB on SM90 x8; mixed-chunk final-chunk rule and the delayer; boot `torch.mm` warm-up; the branch verify for any DFLASH target meeting section 5 (`SGLANG_DFLASH_BRANCH=0`).

## 7. Comment trim

`harness/comment_trim.py` (not in the patch): whole-line comments removed, docstrings cut to their first sentence; licence headers, pragmas and comments stating a numerical property kept. lanes/kfinal6c `3f0433ce0a` -> lanes/kfinal6ct `d396422efc`; `--verify-against` the untrimmed head: VERIFY OK (Python AST and token streams equal; C++/CUDA token streams equal).

## 8. Memory

Measured on 8 GPUs for the predecessor kpickt (`20261119T190315Z`, section 9; this file adds no buffer): -4.53 GB per GPU vs stock (kbranch2wt -4.50, stack-k35c3 -4.51 in `20261119T165108Z`). Over stack-k37 the only persistent buffer change is the branch counter: [max_bs, 3] int64 instead of [3].

## 9. Tests

**CPU suite, judge image**: 748 passed, 0 failed, 24 skipped on lanes/kfinal4b `131000e020` (`out/kfinal4b_cpu_suite.log`), of which this file's untrimmed tree `3f0433ce0a` is a comment-only edit (token streams equal) = 753 on our file kpatt3t - 7 tests of removed off switches + 2 new routing / split-policy tests. The trimmed tree is token-identical. GPU tests on an H200: 23 passed (`out/kfinal4b_gputests1.log`) = the 21 above + a forced fallback to FA3 captured in a CUDA graph and replayed at other prefix lengths, for both window kinds (max difference from stock FA3 0.023 / 0.039, bound 0.125). No test exercises the pick with another gamma. The branch test file covers:
the closed-form gate on random lattices; gate and guard conditions evaluated from source; the fallback on a fake layer; layout/accept/swap vs torch references; greedy replay through a position-keyed KV model with the tree mask (committed tokens == target greedy, each slot holds its token); junk branch tokens; mutation checks; the table reference; the Triton kernels vs the torch references under the Triton interpreter (120 random cases).

**1-GPU microbench, H200** (`out/kwide2/kbranch2_bench_2026-11-19_run2.log`, lanes/kbranch2 `fc3fdfb7c0`; its layout/accept/table kernels equal this tree's apart from comments the trim removed; its gate is the torch body this tree replaces):
- Layout, accept and table kernels compiled for sm_90a vs the torch references, 120 random cases: `BAD 0 []`.
- Worker branch bookkeeping per step, host ms, v1 torch ops -> v2 kernels: 1.04-1.05 -> 0.19-0.20 at bs 1/4/16/32.
- The in-graph fill measured 1.0119 -> 1.0072 ms (bs 1) and 1.6261 -> 1.6163 ms (bs 4) on a 62-layer tree-attention graph, with bits equal. That gain is too small to keep, so the fill is not in this tree.
- Isolated costs, not end-to-end. On 8 GPUs v1 measured +0.55-0.60 ms per step at 1 user and +0.75 at 12-16 (research/audit_2026-11-19.md). How much of that v2 removes is not measured.

**Folds, 1 GPU, H200** (`out/kpick2/gpu_fold_bench.log`): ring and routed folds equal their CPU reference paths (0 mismatches); `_pair_fold_kernel` in a CUDA graph 8.1 -> 5.5 / 9.8 -> 6.3 / 72.5 -> 46.5 us per call at 128 / 256 / 2048 rows.

**Gate kernel, 1-GPU graph replay, H200** (`out/kwide2/kpick_bench_run1.log`, 500 replays): the torch body 0.145 / 0.155 / 0.167 / 0.187 ms per replay at bs 1/4/16/32, the kernel 0.0037-0.0040 ms; slot and tokens equal.

**8 GPUs** (full ladder, the judge's ids client, control = our judge-checked file kpatt3t in the same container; newest first; the predecessors kfinalt / kfinal2t differ from this file's kernel code by one host-side guard that Muse Spark never takes):
- Run `20261120T050033Z` (this exact file, booted first; control kpatt3t second): clean boot; 0 failed requests, 0 wrong-length outputs, 0 tracebacks; peak memory -4.42 GB per GPU vs stock.
  - Throughput at 1/4/16/32 users: 244.6 / 496.4 / 773.8 / 902.6 tok/s vs the control's 248.1 / 494.8 / 772.3 / 886.2 (-1.4 / +0.3 / +0.2 / +1.8%); median per-user rate -3.4 / +3.7 / +1.4 / +3.9%.
  - Median decode step period at 1 running request: 10.90 ms vs the control's 11.15 (kfinal3t in the previous container: 11.00 / 10.95).
- Run `20261120T040824Z` (soak of this file's code: kfinal3t, whose code tokens equal kfinal3ct's), order kfinal3t, kpatt3t, kfinal3t on one node: clean boots; 0 failed requests, 0 wrong-length outputs, 0 tracebacks; peak memory -4.42 GB per GPU vs stock (control -4.50).
  - Throughput at 1/4/16/32 users: control 241.1 / 496.3 / 762.8 / 899.2 tok/s; first boot 241.4 / 495.4 / 778.1 / 908.0 (+0.1 / -0.2 / +2.0 / +1.0%); second boot 241.1 / 490.3 / 782.7 / 913.8 (+0.0 / -1.2 / +2.6 / +1.6%). Median per-user rate +4.6 / +4.8% at 16 users and +3.2 / +3.2% at 32.
  - Median decode step period at 1 running request: 11.00 / 11.15 / 10.95 ms (first boot / control / second boot).
- Run `20261120T022113Z`, order kpatt3t, kfinalt (full-attention layers on the kernel), kfinal2t (all 62 layers): clean boots on 8 ranks (each JIT-compiles the kernel); 0 failed requests, 0 wrong-length outputs, 0 tracebacks; peak memory -4.50 GB per GPU vs stock in all three arms.
  - Throughput at 1/4/16/32 users: control 244.3 / 494.3 / 776.4 / 907.3 tok/s; kfinalt 235.7 / 489.9 / 778.2 / 917.0 (-3.5 / -0.9 / +0.2 / +1.1%); kfinal2t 243.7 / 494.5 / 779.6 / 916.6 (-0.2 / +0.0 / +0.4 / +1.0%). Median per-user rate, kfinal2t vs control: +1.2 / -0.3 / +3.0 / +2.4%.
  - Median decode step period from adjacent `Decode batch` lines, 1 running request: control 11.15 ms, kfinalt 11.08, kfinal2t 10.95 (the 1-GPU tables predict -0.04 and -0.22).
- Run `20261120T013319Z`, order kfinalt, kpatt3t (a slower node): clean; kfinalt vs control +0.0 / -1.6 / +0.9 / +0.9% throughput, +1.1 / -0.9 / +7.1 / +4.1% per-user rate.
- This test's per-rung noise is 1-2% (two equal-speed files differed by up to 2.2% at a rung in `20261119T234922Z` / `20261120T004154Z`, and the arm booted first read ~1% low), so the throughput gain is not resolved by it; the step-time and 1-GPU kernel measurements are the evidence for speed, these runs are the evidence for health.

**8 GPUs, our previous file kpickt (this file minus section 1 (g))** (lanes/kpickt `c305f3b1dd`; run `20261119T190315Z`, one arm at 1/4/32 users, the judge's ids client; the planned control arm did not run, so this is a boot and health run, not an A/B):
- Clean boot; 0 failed requests, 0 wrong-length outputs, 0 tracebacks; peak memory -4.53 GB per GPU vs stock.
- Throughput 239.1 / 489.3 / 894.9 tok/s at 1/4/32 users (kbranch2wt on another node that morning: 235.5 / 485.2 / 883.5). Mean accept length by batch size 1 / 2-4 / 17-32: 3.552 / 3.920 / 4.025.
- Median decode step period from adjacent `Decode batch` lines: 11.10 ms at 1 user and 16.10 ms at 4 users (kbranch2wt 11.35 / 16.55, stack-k35c3 11.00 / 16.25, both on the morning's node). Cross-node figures; indicative only.
- The 16-user rung was not run for this file; batch sizes 5-16 occurred inside the 32-user rung.

**8 GPUs, the same tree with the torch gate** (kbranch2wt `4a7b9c6964`; run `20261119T165108Z`, one container, control = stack-k35c3 `33c0a47ca6` booted first, full ladder, the judge's ids client):
- Clean boot; 0 failed requests, 0 wrong-length outputs, 0 tracebacks at 1/4/16/32 users; peak memory -4.50 GB per GPU vs stock (control -4.51).
- Throughput 235.5 / 485.2 / 758.8 / 883.5 tok/s vs the control's 226.4 / 471.0 / 739.0 / 876.4 = +4.0 / +3.0 / +2.7 / +0.8%; median per-user rate +3.3 / +5.8 / +5.4 / +5.5%.
- Mean accept length from the `Decode batch` lines by batch size 1 / 2-4 / 5-16 / 17-32: 3.549 / 3.945 / 3.889 / 3.925 vs 3.295 / 3.682 / 3.629 / 3.755 (+7.7 / +7.1 / +7.2 / +4.5%).
- Both arms ran with the harness's `--hicache-size 42`. Later-turn TTFT p90 at 16 users: 0.397 s vs the control's 0.382 s (all-turn p90 0.586 vs 0.437 s; median +4%).
- One container, one repetition of the comparison; the test node is not the judge.

**Inherited from v1, labelled as v1** (lanes/kbranch, 0.49 margin gate, torch layout/accept):
- 1-GPU exactness gate (`out/kmask_run3_kbranch.log`): 64 configurations; error vs FP64 <= 1.0008x stock's (full attention) and <= 1.018x (sliding window).
- 8 GPUs, one container (`20261119T083430Z`) vs stack-k35: +0.9 / +0.6 / +1.1 / +1.0% tok/s at 1/4/16/32 users; accept length +6.9 / +3.3 / +3.0 / +4.0%. stack-k37's own boot (`20261119T132644Z`): clean, 0 failed.

## 10. Launch flags, scheduling constants, and how each was chosen

None of these changes a numerical result, and none reads the content of a request. We state plainly where a value came from our own load tests or from a judge check.

- `--swa-full-tokens-ratio 0.13 --min-free-slots-delay 12`, passed as flags and equal to the in-code defaults `muse_spark_hook.py` declares for this model when unset (an explicit flag wins). 0.13: the sliding-window layers hold one 2,048-token window per request plus chunks in flight, and the rest of the KV bytes go to the full-attention pool. 12: DFlash admissions wait for 12 free slots (stock's automatic value is at most 4), so more new requests share each prefill chunk. Both were picked on our own load generator (synthetic multi-turn conversations built from SGLang's source files) at 16-32 concurrent conversations and kept after judge checks.
- Scheduler poll interval 4 (section 3): a poll is a TP-wide broadcast on the scheduler's critical path; 4 bounds the added admission latency to three decode steps.
- Short-prefill delayer: 4,096 tokens (about a quarter of one prefill chunk) and 4 passes. We first shipped 8 passes; a judge check read later-turn TTFT p90 at 1.67x stock (the limit is 2x), so we halved the hold to stay well inside the latency limit. Today's judge check at 16 users read 1.26x.
- Online placement (`muse_spark_online_placement*.py`): fits once 60,000 prefill rows are counted (60k rows x top-8 / 256 experts is about 1,875 routes per expert, a load-share error near 0.4%); weight 0.5 between prefill load balance and decode hit chance; `_DECODE_ROWS = (8, ..., 256)` is 8 verify rows per request times the power-of-two decode batch sizes of the CUDA-graph capture range, each weighted equally. Counts come from this run's own routing; nothing is loaded from a file.
- Prefill attention kernel thresholds: from 1,024 query rows (below that the kernel leaves SMs idle and FA3 is faster) and, in a mixed batch, above 33 blocks of 24 rows. From 1-GPU kernel timings against FA3.
- Token-sharded prefill from 2,049 rows: one row past the 16 MiB cap of stock's custom all-reduce (section 6).
- Last-layer read rows up to 256 rows; HiCache quiet re-sync at least every 32 steps: bounds on eager work, not tuned.
- Branch gate constant -0.3: section 4 (recorded drafter lattices from our load generator).

## 11. Not verified / known limitations

- The 8-GPU throughput comparisons (section 9) do not resolve the gain; the kernel's speed and accuracy tables are 1-GPU measurements. The Laguna override is not on Muse Spark's path and was not run; the guards and the stock fallback did not run on 8 GPUs (CPU tests only). **The judge checks we ran today were of our earlier files kpick3mt (32 and 16 users) and kpatt3t (32 users), which do not contain the prefix kernel; this file has had no judge check.**
- Requests that ask for logprobs, grammar or non-greedy sampling take stock's chain verify for that step (`_can_skip_verify_logits`); our load client sends none of those, so that hand-over was not exercised under load.
- The gate's gain is a replay estimate on our recordings; gated steps cut the chain to 5 drafts.
- **Prefix lengths 0-5** (as stack-k37): largest gap vs stock 0.094 abs vs 0.066 for stock's own re-split envelope; error vs FP64 unchanged (FP32 accumulation of e4m3 QK products in another order).
- Byte-identical text vs stock on our own load generator is under 1% of prompts (greedy decoding diverges after the first differing token; no stock-vs-stock floor was run). The judge's own agreement measure read 0.95-0.96 (first turns) and 0.97-0.98 (later turns) on our earlier files today.
- The accept kernel still writes its counter rows (the log line that read them is removed; tests read them).
- The token-sharded prefill's 2,049-row boundary assumes the default cap of stock's custom all-reduce (`SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB` = 16384). If that cap were raised, stock would sum those rows in FP32 while the two BF16-expert layers (and the attention scatter when its slots do not fit) would still take NCCL's reduce-scatter from 2,049 rows, i.e. below stock there. We did not derive the boundary from the variable.
- The tree call's fallback is a bare `except Exception` around the prefix kernel: every check that can raise runs on the host before the first launch, so a layer is served whole by one path, but a fallback is silent (no log line).
- The decode router's one-launch path accepts a 1-row batch (TF32 `torch.mm`, where stock uses its 1-row GEMV); under DFlash the target only sees 8-row verify blocks, so that case does not occur in this configuration.
- Non-greedy sampling at TP > 1 takes a deterministic renormalization (upstream #48111; section 6 "For other models"). Our load is greedy, so it did not run; we did not compare its precision with stock's.
- Identifiers in the three Triton kernels of the branch verify (`_pick_kernel`, `_accept_kernel`, `_tables_kernel`) are short. A legend comment sits above `_pick_kernel` (b request; e slot; m, z row max and sum(exp); ...); the legends of the other two were trimmed for bytes and are: `_accept_kernel`: b request; j row; e branch slot; chain = accepted drafts; won, both = rows 6 / 6-7 accepted; rows, rows1 = rows of tokens j, j + 1. `_tables_kernel`: t verify row (b request, r row); p prefix length; e branch slot; d row offset; [lo, lo + edge) old keys outside the window. Longer names did not fit the byte limit.
