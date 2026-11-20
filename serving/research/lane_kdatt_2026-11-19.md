# Lane KDATT: decode/verify attention vs its floors, and what can ship tonight (Thu Nov 19 2026, from 18:20 CST)

Labels: [measured file] = a GPU number read from that file; [read file:line]; [computed] = arithmetic on those;
[estimate] = judgement. No GPU run was needed for Phase 1: existing 1-GPU microbenchmarks (KPAGE, KATT, KMASK) already
time the exact verify call per layer type and request count.

## Phase 1 (written 18:25-18:45 CST)

### 1.1 Sources

- **FA3 verify call, per layer, page size 1, our split policy (`pol`)**: research/lane_kpage_2026-11-17.md 1a/1b
  (raw out/kpage/run{1,2}_stdout.log). The call is the serving call: sgl_kernel FA3 `flash_attn_with_kvcache`, FP8
  q/k/v, 8 rows per request, 16 q / 2 kv heads, d 64, page-table width 131,072, KATT shared schedule + split policy;
  contexts 73k-77k; CUDA graph of 8 x [128 MB L2 scrub, call], scrub subtracted, 200 reps, median. The time is the
  whole call (FA3 main + FA3 combine). B = 32 reproduces the c32 server trace's 0.322 ms per full layer
  [read research/trace_koproj_c32_2026-11-17.md:269].
- Page size on the server is 1: `--page-size 64` was A/B'd on 8 GPUs and closed flat-to-negative
  [read NEXT_STEPS_2026-11-08.md:2079,2083].
- **Tree path overhead** (non-causal prefix call with forced >= 2 splits + our FP32 correction/merge kernel), 62 layers,
  p ~ U[95k,105k]: research/lane_kmask_2026-11-19.md:183-195 (out/kmask_run2.log). Merge kernel 3.8-5.6 us/layer;
  FA3's own combine 1.8-4 us/layer still runs.
- **Existence proof of a faster kernel on the same inputs**: kernel-track "vjeux" kernel (FP32-accumulating FP8 wgmma
  QK and PV, FP8 P, ex2.approx as FA3, but fixed-reference softmax + FP32-atomic split merge), research/lane_katt_2026-11-16.md 1c
  (out/katt/20261116T204902Z/kt.json). Its times include fused QK-norm/RoPE/KV-store/output-norm.
- Split-count sweep of the same FA3 call: lane_katt 1a (out/katt/20261116T204902Z/sweep.json).

### 1.2 Floors [computed]

- Full layer: 73k-77k keys x 2 kv heads x 64 d x 2 tensors x 1 B = 19.2 MB per request (KPAGE: 19 / 77 / 308 / 615 MB
  at B 1/4/16/32 [read lane_kpage:65-70]). At 4.6 TB/s: 4.2 / 16.7 / 67.0 / 133.7 us.
- Sliding layer: 2,048 keys x 256 B = 0.52 MB per request: 0.11 / 0.46 / 1.8 / 3.6 us.
- Tensor-core work: QK + PV = 2 x 2 x (8 rows x 16 heads) x 64 x keys = 32.8 kFLOP per key per request
  = 128 FLOP per KV byte. At the 4.6 TB/s byte floor that needs 590 TFLOP/s of FP8 (30% of the 1,979 dense peak):
  below the byte floor, but not by much. The exp count is 128 per key per request = 0.5 exp per byte; at 4.6 TB/s
  that is 2.3e12 exp/s against ~4.2e12/s of MUFU (16/SM/clk x 132 x ~1.98 GHz) [computed]: a full-speed kernel
  must keep MUFU ~55% busy, which is why the organizer's bit-trick exp2 ruling matters.
- Launch floor: one kernel ~10 us behind a PDL neighbour (brief: 8-12 us). Floor per layer = max(bytes, ~10 us) for
  one kernel; today's verify layer is 3 kernels (FA3 main, FA3 combine, our correction merge).

### 1.3 Table: measured us per layer vs floor, and ms/step recoverable at 1.3x floor

| layer type | requests (rows) | measured us/layer (FA3 main+combine) | floor us | 1.3x floor | recoverable us/layer | x layers | **ms/step recoverable** | kernel-track kernel us (fused, for scale) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full (16) | 1 (8) | 25.6 [measured lane_kpage 1a] | 10 (launch; bytes 4.2) | 13.0 | 12.6 | 16 | **0.20** | 15.6 |
| full | 4 (32) | 53.7 | 16.7 | 21.7 | 32.0 | 16 | **0.51** | 32.8 |
| full | 16 (128) | 168.3 | 67.0 | 87.1 | 81.2 | 16 | **1.30** | 95.0 |
| full | 32 (256) | 323.3 (1.90 TB/s) | 133.7 | 173.8 | 149.5 | 16 | **2.39** | 169.4 |
| sliding (46) | 1 | 13.6 (rand/p1, today's recycled slots) | 10 (launch) | 13.0 | 0.6 | 46 | 0.03 | 7.7 |
| sliding | 4 | 14.9 | 10 | 13.0 | 1.9 | 46 | 0.09 | 8.9 |
| sliding | 16 | 19.4 | 10 | 13.0 | 6.4 | 46 | 0.29 | 12.2 |
| sliding | 32 | 25.6 | 10 | 13.0 | 12.6 | 46 | 0.58 | 16.0 |
| tree path add-on (prefix call forced >= 2 splits + merge kernel), all 62 layers | 1 / 4 / 16 / 32 | +0.299 / +0.287 / +0.362 / +0.383 ms/step [measured lane_kmask:185-188] | 0 if merged into the attention kernel's own split combine | | | | up to 0.25-0.35 (merge kernel 3.8-5.6 us x 62) [computed] | |

All [computed] from the [measured] columns; "recoverable" = measured - 1.3 x floor. The FA3 and correction kernels
cannot be separated in the KPAGE numbers (no tree path then); KMASK times the tree add-on separately (row above).

Split-count headroom (schedule-only) at uniform long contexts [read lane_katt 1a]: best forced split vs stock is
-0.7% at full B=16 (f4 178.5 vs 179.8) and -2.2% at full B=32 (f8 329.4 vs 336.9); sliding B=16/32 -1.8% / -0.5%.
The policy we ship already removed the wave cliffs (B 18-28: -30..-36%). So option (a) is worth <= ~0.13 ms/step at
c32 and ~0 at c16 [computed: (336.9-329.4) x 16].

### 1.4 Verdict

**GATE PASSED, from the 16 full-attention layers at 16 and 32 users: ~1.3 ms/step at c16 and ~2.4 ms/step at c32 sit
between FA3 (1.9 TB/s at c32, ~2x the byte floor) and 1.3x the byte floor.** The kernel-track kernel on the same inputs
(page size 1) already runs within 1.27x (B=32) / 1.42x (B=16) of the floor, so the gap is not physics. At 4 users the
full layers leave 0.51 ms (>= 0.3 line); at 1 user 0.20 ms (below). Sliding layers leave 0.3-0.6 ms at c16/c32, mostly
launch/latency, not bytes.

Why FA3 is slow here [estimate, consistent with the data]: each FA3 tile covers one kv head, so with page size 1 it
gathers 64-byte rows per token through the non-TMA paged loader (KPAGE: page 64 alone recovers 25%; more splits do not
help: f8 at B=32 is within 2%), and its pipeline per SM is not deep enough to cover HBM latency at that granularity.

Phase 2 plan: (a) schedule-only is worth <= 0.13 ms (above); (b) the merge kernel is worth <= 0.35 ms; (a)+(b) leave
>= 1 ms at c16/c32, so (c) applies if it can be measured by 22:30.

## Phase 2 (c): the full-layer prefix kernel (from 18:28 CST)

### 2.1 Structure vs arithmetic (lead's 18:30 question, answered on paper before the GPU run)

The organizer's 18:30 reversal leaves only order changes. The kernel built here takes **no** arithmetic from the
kernel-track file: every per-element operation is `prefill_attention.cuh`'s, which is FA3's FP8 class (scores FP32 from
FP8 wgmma, running row max, P = ex2.approx(s*c - (m*c - 8)), P rounded to E4M3 for an FP8 P.V wgmma with FP32
accumulators, FP32 row sum of the unrounded P, exact rescale when the max grows, FP32 split combine via LSE). It reuses
that file's `softmax` / `pipe_step` / `issue_qk` / `issue_pv` unchanged. What differs from FA3 is order only: 128-key
tiles, a different split partition, and the merge (the existing correction kernel).
The structural differences that should carry the speed [estimate, to be measured]: one CTA per (request, split)
covering BOTH kv heads, so each cache row is one 128-byte gather (FA3: two 64-byte gathers by two CTAs); a producer
warpgroup with cp.async 16-byte gathers 4 tiles deep (64 KB of K/V in flight per SM); splits sized to one wave
(132 // B CTAs per request). Whether the kernel-track's fixed reference contributed is not separable from its file;
this kernel answers it directly: if it is fast, the speed is structure.

### 2.2 What was built (lanes/kdatt, branch lane-kdatt, on kpick3m e42dcbaee9; uncommitted at 18:35)

- `verify_tree_attention.cuh`: `prefix_kernel` + `prefix<kUsePDL>` host op (includes `prefill_attention.cuh`).
  Grid (B, S), 384 threads: warpgroup 0 = producer (key `tid` of each 128-key tile: K by cp.async into the SW64 K-major
  slot of its head; V through a 2-deep row buffer, then stage_kernel's 4x4 byte transposes into the SW128 V^T slot);
  warpgroups 1-2 = consumers, one per kv head (64 rows = 8 verify rows x 8 q heads). Writes FA3's partial layout:
  O normalized (x v_scale) [S, 16, 8B, 64], LSE natural log [S, 16, 8B]; an empty split writes O = 0, LSE = -inf;
  writes S into a per-request split-count vector the correction kernel reads (order_off = -1).
- `fa3_verify_tree.tree_verify_attention`: full layers (FP8 q and KV) call `prefix` then the unchanged `correct`;
  S = max(2, min(64, num_sms // bs)) (static per captured bs). Any exception falls back to the FA3 call. Sliding
  layers and everything else stay on FA3. Kill switch `SGLANG_DFLASH_TREE_PREFIX_KERNEL=0`.
- CPU compile (Modal CPU, judge image, sm_90a): prefix_kernel 168 registers, 0 spills, no wgmma serialization
  warning [measured out/kdatt_compile1.log].
- Bytes: +12.2 KB of added lines (raw diff 15.0 KB) against 842 B of room + ~4-5 KB from byte_trim [measured
  `git diff | wc -c`]. **The judge file does not fit without ~7 KB more cuts.**

### 2.3 GPU run 1-2 (18:34-18:37 CST, 1 x H200, judge image; harness/modal_kmask_ubench.py + lanes/kmask_tools/kdatt_adapter.py)

- Run 1 (out/kdatt_gpu1.log): bench/API mismatch only (kmask_ubench passed `lagged=`, the adapter `num_splits=`; both
  removed in kpick2 item D). Fixed in the tools (kmask_ubench now passes `lagged` only if the tree accepts it).
- Run 2 (out/kdatt_gpu2.log), first kernel version (producer 1 tile ahead):
  - **Exactness: PASS, same class as FA3.** ext (tree path with the prefix kernel on full layers) vs stock, all 64
    configs: full max 1160 ulp / 0.094 abs (= fixp, the FA3 tree path); chain6 0.078, branch2 0.0625, ungated 0.094;
    **error vs FP64 / stock's error vs FP64, worst: 1.049 on full layers** (fixp 1.0008; stock's own re-split
    envelope 1.34-3.72); SWA 1.018 (FA3 there, unchanged). Graph replay == eager: yes [measured EXACT_WORST line].
  - **Timing: SLOWER than FA3.** Added over stock, 62-layer graph: ext +0.419 / +0.445 / +1.008 / +1.643 ms vs fixp
    +0.257 / +0.274 / +0.328 / +0.357 at B 1/4/16/32 (p 75k-86k). Prefix kernel ~440 us per full layer at B=32
    [computed: 113.6 us/layer-avg x 62 / 16]. Cause [estimate]: the producer kept one tile (32 KB) in flight and
    loaded each tile's page-table entries on the critical path: ~13 GB/s per SM.
- Fix (18:38): producer 3 tiles ahead (4 row buffers, 223 KB smem), page-table entry prefetched one tile ahead;
  splits per request = max(2, min(128, num_sms // bs)). Run 3 launched 18:38.

### 2.4 Runs 3-8 and diagnostics (18:38-19:07 CST) -- the kernel now beats FA3 at every request count

Diagnostics (lanes/kmask_tools/kdatt_diag.py via `harness/modal_kmask_ubench.py::diag`; out/kdatt_diag{1..7}.log):
the prefix kernel alone, modes m0 = full, m1 = producer only (consumers just release), m2 = consumers only.
- The producer was the bottleneck (m1 ~460 us at B=32 while m2 = 152-166 us): the per-row integer division by the
  runtime page size cost more than everything else. Page-size-1 fast path (host RuntimeCheck page == 1, else FA3):
  m1 373 -> 176 us at B=32 (~3.7 TB/s).
- Then producer and consumers serialized (m0 ~ m1 + m2): the producer finished tile t only after issuing tile t+4,
  which waited for the consumer's previous step. Finishing before issuing, 3 tiles ahead, 6-stage ring per head
  (prefill_attention.cuh: `Ctx.nwg` / `Ctx.stages` with defaults NWG / STAGES, bit-identical for prefill), V lands
  row-major in its V^T slot and is transposed in place (no scratch).
- Other fixes on the way: the prefix launch is not PDL (its one-CTA-per-SM grid, resident early, starved the
  predecessor's tail: the correction kernel went from "118 us" to 4-6 us); coalesced 16-byte gathers.

Prefix kernel alone, us per full-layer call, p ~ U[75k, 86k], page 1, 64-token-scattered slots, CUDA graph of 10 calls
[measured out/kdatt_diag7.log, m10 = this version]:

| B (rows) | kdatt prefix (one kernel) | FA3 prefix call (main + combine) at the tree's split count | consumers-only bound (m2) |
|---|---:|---:|---:|
| 1 (8) | 12.7 | 20.7 (64 splits) | 9.5 |
| 4 (32) | 34.1 | 54.0 (16 splits) | 23.7 |
| 16 (128) | 111.9 | 184.6 (4 splits) | 81.3 |
| 32 (256) | 226.7 (2.9 TB/s) | 360.6 (2 splits; 1.8 TB/s) | 154.0 |

**Run 8, the gate: the served tree path end to end** (out/kdatt_gpu8.log; JSON out/kdatt/gpu8_kmask.json;
`fa3_verify_tree.tree_verify_attention` as served via lanes/kmask_tools/kdatt_adapter.py; 62 layers = 16 full + 46 SWA,
own KV per layer, p ~ U[75k, 86k], CUDA graphs, 200 interleaved replays, median of paired differences vs stock):

| requests (rows) | stock causal FA3, ms (62 layers) | FA3 tree path (fixp), added ms | **kdatt tree path (ext), added ms** | **ms per step saved vs today's tree path** |
|---|---:|---:|---:|---:|
| 1 (8) | 0.860 | +0.265 | +0.230 | **0.035** |
| 4 (32) | 1.413 | +0.279 | +0.084 | **0.195** |
| 16 (128) | 3.654 | +0.331 | -0.714 | **1.045** |
| 32 (256) | 6.708 | +0.362 | -1.932 | **2.294** |

(graph medians, p10-p90 spread <= 0.02 ms; the eager schedule build is unchanged for SWA and not counted.)

**Exactness (run 8, all 64 configurations of the gate: B 1/4/16/32 x p 0/5/2040/2048/2100/70k/100k/mixed x full/SWA,
chain and branch masks, ungated rows):** PASS.
- vs stock FA3: full max 1160 ulp / 0.094 abs (identical to the FA3 tree path; it occurs at p = 0/5 where only the
  correction keys exist); chain6 0.078, branch2 0.0625; graph replay == eager; finite; deterministic.
- Error vs FP64 (relative L2, first 3 requests), full layers, max / mean per class [measured, out/kdatt/exact_table.py]:

| class | stock FA3 | FA3 tree path | kdatt tree path |
|---|---|---|---|
| p = 0 / 5 | 0.00725 / 0.00561 | 0.00551 / 0.00482 | 0.00551 / 0.00482 |
| p ~ 2k | 0.00591 / 0.00455 | 0.00587 / 0.00450 | 0.00448 / 0.00377 |
| p 70k-100k | 0.00655 / 0.00407 | 0.00655 / 0.00407 | 0.00501 / 0.00319 |
| mixed | 0.00639 / 0.00514 | 0.00520 / 0.00448 | 0.00519 / 0.00437 |
| worst ratio to stock, any config | 1 | 1.0008 | 1.0008 |

  kdatt's error is at or below FA3's in every class (more, shorter splits; same per-element arithmetic).
- SWA layers are untouched (FA3): worst ratio 1.018 as before.

## Phase 2 status at 19:30 CST: lane-kdatt head 53222bea1f (on lane-kpatt3 01e47dbe25)

Commits (branch lane-kdatt, lanes/kdatt): 525b953df5 kernel; ecc1da7b10 shared `vt_block` V^T helper (stage_kernel uses
it too: same loads/stores) + compaction; 81b72d7957 key tiles last to first (as FA3 / 01e47dbe25); 53222bea1f merge
folded into the op (`prefix_verify`), splits with keys first (no empty-split writes), env switch removed, FP8 + 2x64 KV
guard, tests. Merged by the lead into lanes/kfinal 89b2236036.

Measured on 53222bea1f's tree (1 x H200, judge image):
- 64-config gate (out/kdatt_gpu12.log): PASS, worst error-vs-FP64 ratio to stock 1.0008 (= FA3 tree path), graph
  replay == eager. Time saved vs today's FA3 tree path: **2.19 / 0.99 / 0.18 / 0.04 ms per step at 32 / 16 / 4 / 1
  requests** (ext -1.834 / -0.662 / +0.096 / +0.222 vs fixp +0.360 / +0.329 / +0.275 / +0.262 ms over stock).
  (Forward key order measured 2.29 / 1.05 / 0.20 / 0.04: the backward order costs ~0.1 ms at c32.)
- GPU tests (out/kdatt_gputests4.log): 16 passed = prefill kernel 11 (unchanged after the shared helper) + 5 new
  `test_muse_spark_verify_prefix_kernel.py`.
- Batch-size sweep bs 1..40, 48, 64, mixed prefixes incl. 0/3/127/128/129 (out/kdatt_sweep1.log): CUDA-graph replay ==
  eager at all 42, finite, max |kdatt - FA3 tree path| 0.043 (stock's own re-split envelope 0.066).

### Robustness (lead's item 4)

| case | behaviour | evidence |
|---|---|---|
| page size != 1 | host `RuntimeCheck(page == 1)` raises before any launch -> FA3 tree call for that layer | code; CPU test (raising kernel -> FA3) |
| non-FP8 q/KV, KV not 2 heads x 64 | Python guard `q.dtype == key_cache.dtype == e4m3 and key_cache.shape[-2:] == (2, 64)` -> FA3 | CPU test (bf16 KV -> FA3) |
| softcap, sinks, cross-attention | never reach it: `_tree_verify_attention` already declines those layers | existing test_dflash_branch |
| head counts / TP rank != 0 | kernel is fixed at 16 q / 2 kv heads x 64 (the per-rank shape of every TP8 rank); q matcher {T,16,64} and the 128-byte KV row enforce it; any other shape raises -> FA3 | code |
| short / zero prefix | nt = 0 -> no split has keys, merge uses the <= 8 table keys only; p = 0/1/3/5/127/128/129 | gate, adversarial, sweep, GPU test |
| sms // bs < 2 | splits = max(2, min(128, sms // bs)); bs > 66 -> 2 splits, grid B x 2 in several waves | sweep bs 48 / 64; CPU test of the policy |
| CUDA graphs | allocations (partials 4.3 MB at most for bs <= 66, plus lse / counts) come from the graph pool; capture at bs 1..40/48/64 replays == eager | sweep |
| capture-time failure | every check that can raise runs on the host before the first launch, and depends only on dtypes / shapes, which are the same for all 16 full layers of a graph: all layers of a graph take the kernel or all take FA3. A device fault cannot be caught (as for any kernel); none seen in 64 + 42 + 40 + 16 runs | code + runs |
| 8 ranks JIT-compiling at once | the kernel lives in the existing `muse_spark_verify_tree` JIT module (same `load_jit` + cache as today's shipped correct kernel), so no new module or cache key path; compile +~0.4 s ptxas. Not exercised on 8 GPUs by me | code (unverified on 8 GPUs) |
| persistent memory | none outside the graph pool; per call ~4.3 MB of FP32 partials, replacing FA3's own split buffers of similar size; net ~0 | computed |
| PDL | the prefix launch is not PDL (one CTA per SM resident early would starve the predecessor); the correct kernel keeps PDL | measured (correct 118 us -> 4-6 us) |

### Seeded adversarial distribution (lead's ask 1; out/kdatt_adv_seeds.log, lanes/kmask_tools/kdatt_adv.py, 1 x H200, 53222bea1f tree)

Per case: max |out - FP64| over the first min(B,4) requests' 8 rows x 16 heads x 64, divided by stock causal FA3's (default splits).
kdatt = the served tree path with the prefix kernel. For scale, stock FA3 itself at num_splits 1 and 16 (its own re-split envelope).
Tail classes 20 seeds, others 5; all inputs seeded.

| B | class | seeds | kdatt / stock: min / mean / max | FA3 1 split / stock: min / mean / max | FA3 16 splits / stock: min / mean / max |
|---|---|---:|---|---|---|
| 1 | rand_80k | 5 | 0.6743 / 0.9349 / **1.0000** | 1.6511 / 2.6678 / 4.2260 | 1.0000 / 1.6542 / 2.0569 |
| 1 | sink_30k | 5 | 0.9764 / 1.0038 / **1.0428** | 1.6412 / 2.1168 / 2.4444 | 1.0000 / 1.0624 / 1.2571 |
| 1 | peaked_8k | 5 | 0.6877 / 0.9375 / **1.0000** | 1.0988 / 2.4580 / 5.1496 | 1.0000 / 1.1979 / 1.8350 |
| 1 | flat_8k | 5 | 0.8563 / 0.9695 / **1.0658** | 0.9950 / 1.1014 / 1.1847 | 0.8360 / 1.0023 / 1.1056 |
| 1 | growing_16k | 5 | 0.9108 / 0.9895 / **1.1294** | 1.1399 / 1.4315 / 2.1052 | 1.0000 / 1.0893 / 1.2341 |
| 1 | tail_g15.8_64k | 20 | 0.9845 / 1.0033 / **1.0801** | 0.9667 / 1.0071 / 1.0848 | 0.9567 / 1.0015 / 1.1012 |
| 1 | tail_g17.8_64k | 20 | 0.9771 / 0.9976 / **1.0000** | 0.9891 / 1.0004 / 1.0103 | 0.9872 / 1.0002 / 1.0271 |
| 1 | tail_g19.8_64k | 20 | 1.0000 / 1.0000 / **1.0000** | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| 1 | short_5 | 5 | 0.9212 / 0.9593 / **1.0000** | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| 1 | zero | 5 | 1.0000 / 1.0440 / **1.2202** | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| 32 | rand_80k | 5 | 0.3948 / 0.8534 / **1.0000** | 1.0000 / 1.1364 / 1.4195 | 1.0000 / 1.0000 / 1.0000 |
| 32 | sink_30k | 5 | 0.6738 / 0.8450 / **1.0000** | 1.0000 / 1.2462 / 2.0505 | 1.0000 / 1.0000 / 1.0000 |
| 32 | peaked_8k | 5 | 0.6193 / 0.8752 / **1.0000** | 1.0000 / 1.0323 / 1.1028 | 1.0000 / 1.0000 / 1.0000 |
| 32 | flat_8k | 5 | 1.0000 / 1.0225 / **1.0887** | 0.9393 / 1.0304 / 1.1538 | 1.0000 / 1.0000 / 1.0000 |
| 32 | growing_16k | 5 | 1.0000 / 1.0000 / **1.0000** | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| 32 | tail_g15.8_64k | 20 | 0.9609 / 0.9981 / **1.0078** | 0.9607 / 0.9975 / 1.0347 | 1.0000 / 1.0000 / 1.0000 |
| 32 | tail_g17.8_64k | 20 | 0.9616 / 1.0013 / **1.0762** | 0.9693 / 0.9980 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| 32 | tail_g19.8_64k | 20 | 0.9989 / 0.9999 / **1.0000** | 0.9989 / 1.0009 / 1.0188 | 1.0000 / 1.0000 / 1.0000 |
| 32 | short_5 | 5 | 0.7832 / 0.9337 / **1.0000** | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| 32 | zero | 5 | 1.0000 / 1.0440 / **1.2202** | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |

Reading: kdatt's mean ratio is 0.85-1.04 in every class and its max ratio stays inside FA3's own re-split envelope in
every class with a long prefix (worst 1.129 growing B=1, FA3 1-split 2.105 there; tails <= 1.080 vs FA3 re-split up to
1.101). The `zero` class (p = 0: no prefix keys, the kernel does no work, the row is the existing correction kernel's
<= 8 keys in FP32 QK) shows 1.22 max: that is the already-shipped tree path's behaviour (identical output to the FA3
tree path there), not this kernel; p = 0 at a full layer does not occur at decode (the prefix holds the prompt).


### Reviewer paragraph (for the reviewer notes)

**Full-layer verify attention: Muse FP8 prefix kernel (python/sglang/kernels/jit/csrc/muse_spark/verify_tree_attention.cuh
`prefix_kernel`, op `muse_spark_verify_tree.prefix_verify`, call site `fa3_verify_tree.tree_verify_attention`).**
On the 16 full-attention layers, the DFLASH tree verify used to call FA3 non-causally over each request's committed
prefix (p keys) and merge FA3's FP32 split partials with each row's <= 8 new keys in our correction kernel. That FA3
call is replaced by our kernel; the correction / merge kernel is unchanged. The kernel computes exactly what FA3
computes for that call: E4M3 Q and K/V read from the paged cache as stored (no scales other than FA3's), S = Q K^T by
FP8 wgmma with FP32 accumulators, the running row maximum with P = ex2.approx(S * c - (m * c - 8)) (2^8 at the
running max, as FA3 FP8), the FP32 row sum of the unrounded P, P rounded to E4M3 for an FP8 P V wgmma with FP32
accumulators, an exact FP32 rescale of O and l whenever the maximum grows, every key attended, and per split a
normalized FP32 O and natural-log LSE in FA3's partial layout, merged in FP32 by LSE. These per-element steps are the
prefill kernel's (prefill_attention.cuh `softmax` / `issue_qk` / `issue_pv`, reused unchanged). What differs from FA3 is
order only: 128-key tiles (FA3: its own block size), each split's tiles walked last to first (as FA3), a different
split partition (max(2, min(128, num_SMs // batch)) contiguous ranges per request), and the merge order. The speed comes
from structure: one CTA per (request, split) covers both kv heads, so each cache row is one 128-byte gather; a producer
warpgroup keeps 3 tiles of K/V in flight into a 6-stage ring and transposes V in shared memory. Sliding-window layers,
non-FP8 caches, page size != 1 and any other shape keep FA3 (the call falls back on any host-side error).

| requests (verify rows) | today's FA3 tree path, 62 layers, ms over stock | with the prefix kernel | ms per decode step saved |
|---|---:|---:|---:|
| 1 (8) | +0.262 | +0.222 | 0.04 |
| 4 (32) | +0.275 | +0.096 | 0.18 |
| 16 (128) | +0.329 | -0.662 | 0.99 |
| 32 (256) | +0.360 | -1.834 | 2.19 |

(1 x H200, judge image, CUDA graphs of 16 full + 46 sliding layers, own KV per layer, prefixes 75k-86k, page size 1;
out/kdatt_gpu12.log.) Prefix call alone per full layer: 12.7 / 34.1 / 111.9 / 226.7 us vs FA3 20.7 / 54.0 / 184.6 /
360.6 us at 1 / 4 / 16 / 32 requests (out/kdatt_diag7.log, forward order).

Accuracy (all vs an FP64 softmax on the dequantized FP8 inputs): on the 64-configuration verify gate (B 1/4/16/32,
prefixes 0 to 100k, full and sliding, chain and branch masks) the worst ratio of our error to stock FA3's is 1.0008,
identical to the FA3 tree path it replaces (stock FA3's own error moves by up to 3.7x between split counts). Mean error
per class is at or below FA3's (full layers, max / mean: p 70-100k 0.00501 / 0.00319 vs FA3 0.00655 / 0.00407). On lane
kpatt3's adversarial classes at the verify shape (sink at key 0, long flat tails 15.8 / 17.8 / 19.8 binades, growing,
peaked, flat; 5-20 seeds each, B = 1 and 32) the max-error ratio to FA3 has mean 0.85-1.04 per class and stays inside
FA3's own re-split envelope in every class (table above). Outputs are deterministic and CUDA-graph replay equals eager at
every batch size 1-40, 48, 64.

### CPU suite and bytes (19:37 CST)

- Judge-image CPU suite on 53222bea1f (stack_k26 list + prompt_lookup + branch): **755 passed, 0 failed, 24 skipped,
  3,347 subtests** (753 + the 2 new routing / split-policy cases) [measured out/kdatt_cpu_suite1.log]. 1 of 3 runs used.
- Bytes: min_diff U1 judge delta of lane-kdatt over lane-kpatt3 = 11,834 B (449,623 vs 437,789; ~0.9 KB of it comments);
  `git diff 01e47dbe25 -- python/sglang | wc -c` = 18,000. The judge file itself is built by the lead in lanes/kfinal
  (89b2236036, byte diet + this head); I did not build or lint it.
- Not done by me: 8-GPU boot / A/B (lead's nov19l), JIT compile race on 8 ranks, the judge file, trimmed-tree suite.
- Speed headroom left (not committed, freeze): at B=32 the kernel runs 227 us vs a 154 us consumers-only bound;
  the producer's per-tile barriers + in-place transposes overlap poorly with the consumers. A deeper decoupling could
  be worth ~0.5-1 ms/step at c32 [estimate], medium risk (new synchronization); not attempted after 20:00.

## Phase 3 (19:39-20:12 CST): sliding-window layers on the prefix kernel -- lane-kdatt2 0c144c0e09 (on 53222bea1f)

Choice (A) over (B) (producer decoupling): the tree's SWA prefix call is FA3 non-causal over p old keys with window
(w-8, 7), so row r reads exactly [p+r-w, p) [read fa3_verify_tree.tree_window; FA3 window rule]; the correction kernel
already supplies the new keys and branch rows' window-edge keys from swa_table. The kernel reproduces those per-row
ranges through the prefill softmax's existing per-row floor (kminA/kminB) and loads tiles from row 0's first tile. No
change to the correction kernel. A row with no key in a split writes O = 0 / LSE = -inf (FA3's empty-split convention).

Findings on the way:
- Prefill timing check (lead's interrupt; kpatt3 bench, same container, B/A/B/A) [measured out/kdatt_pfab1.log]: mean
  cold chunk 70.88 / 70.70 ms on 01e47dbe25 vs 70.87 / 70.72 ms on 53222bea1f; per shape within 0.3%. No regression.
- A first version shared one kernel build for both windows: ptxas spilled 28 B (runtime per-row floor), and full layers
  got slower (same-container A/B: +0.25 ms at B=16, +0.43 ms at B=32) [measured out/kdatt_ab1.log,
  out/kdatt2_compile.log]. Templating the kernel on the window restores 0 spills in both builds [out/kdatt2_compile2.log].
- With the spill gone, sliding-window layers on the kernel pay at every batch size, so no batch threshold.

Timing, served tree path (62 layers), same-container A/B, ms per step saved over 53222bea1f [measured
out/kdatt_ab2.log, out/kdatt_ab3.log; two interleaved reps each, agree within 0.01]:

| requests | 1 | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| saved vs 53222bea1f | 0.18 | 0.18 | 0.13 | 0.16 | 0.17 | 0.20 | 0.19 |

Total vs today's FA3 tree path (run out/kdatt2_gpu4.log, ext vs fixp): 0.22 / 0.36 / 1.18 / 2.40 ms per step at
1 / 4 / 16 / 32 requests.

Gates on 0c144c0e09's tree:
- 64-config exactness PASS; SWA worst error-vs-FP64 ratio to stock 1.0031 (FA3 tree path 1.114), full 1.0008
  [out/kdatt2_gpu4.log].
- GPU tests 21 passed: prefill 11 (bit-identical, unchanged file) + 10 prefix-kernel accuracy tests (full and w = 2047,
  incl. p = 2040 / 2048 / 2100; tail data seeded; tolerance = FA3's own re-split movement) [out/kdatt2_gputests3.log].
- CUDA-graph sweep bs 1-40 / 48 / 64 x full / SWA: 84 / 84 replay == eager, finite, max |diff| vs FA3 tree path 0.055
  [out/kdatt2_sweep2.log].
- Seeded adversarial at w = 2048 (B = 1 / 12 / 32; tails 20 seeds, others 5; ratio = kdatt max err / stock FA3 max err)
  [out/kdatt2_adv_swa.log (B 1, 12; spilled build, same arithmetic), out/kdatt2_adv_swa2.log (B 1, 32)]: mean ratio
  0.75-1.06 per class; worst 1.31 (tail, B=1), 1.19 (tail, B=32), 1.18 (edge p=2048, B=32), 1.15 (sink, B=1);
  stock FA3 at 1 split vs default on the same classes: 1.68 / 1.21 / 1.54 / 3.24. Inside FA3's re-split envelope everywhere.
- Bytes: min_diff U1 450,328 vs 449,623 for 53222bea1f = **+705 B**.

### SWA accuracy, three paths on the same seeds (lead's last ask; out/kdatt2_adv_3path.log, lanes/kmask_tools/kdatt_adv.py 2047)

w = 2048 (FA3 window (2047, 0)); ratio = path's max |out - FP64| / stock causal FA3's (default splits), per seed; min / mean / max over seeds.
(1) kdatt2 = prefix kernel on SWA (0c144c0e09); (2) FA3 tree path = 53222bea1f's SWA path (FA3 non-causal (w-8, 7) + correction),
run in the same process by forcing the fallback; (3) stock FA3 at 1 split. The tail classes are one class at w = 2048 (the
sink key 0 is outside every window at p = 64k; g15.8 / 17.8 / 19.8 give identical rows), shown once.

| B | class | seeds | (1) kdatt2 min / mean / max | (2) FA3 tree path | (3) FA3 1 split |
|---|---|---:|---|---|---|
| 1 | rand_80k | 5 | 0.7192 / **0.8781** / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.1000 / 2.0375 / 3.0031 |
| 1 | sink_30k | 5 | 0.7030 / **0.9171** / 1.1543 | 1.0000 / 1.0019 / 1.0094 | 1.1154 / 1.9369 / 3.2438 |
| 1 | peaked_8k | 5 | 0.9839 / **0.9968** / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.4088 / 1.8207 |
| 1 | flat_8k | 5 | 0.9145 / **0.9812** / 1.0519 | 0.9747 / 0.9983 / 1.0167 | 0.8858 / 1.0875 / 1.2783 |
| 1 | growing_16k | 5 | 0.8718 / **0.9988** / 1.1145 | 1.0000 / 1.0000 / 1.0000 | 1.1479 / 1.4963 / 2.1938 |
| 1 | tail (flat 2048-key window) | 20 | 0.8478 / **1.0622** / 1.3116 | 0.9843 / 1.0020 / 1.0359 | 0.7730 / 1.1234 / 1.6793 |
| 1 | short_5 | 5 | 0.9212 / **0.9593** / 1.0000 | 0.9212 / 0.9593 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| 1 | zero | 5 | 1.0000 / **1.0440** / 1.2202 | 1.0000 / 1.0440 / 1.2202 | 1.0000 / 1.0000 / 1.0000 |
| 1 | swa_edge_2040 | 5 | 0.5018 / **0.7486** / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0877 / 1.2513 / 1.3730 |
| 1 | swa_edge_2048 | 5 | 0.5106 / **0.7103** / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.1410 / 1.2272 / 1.3986 |
| 1 | swa_edge_2100 | 5 | 0.4422 / **0.7114** / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0212 / 1.2464 / 1.3919 |
| 12 | rand_80k | 5 | 0.6205 / **0.8046** / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.1747 / 1.5084 / 1.6733 |
| 12 | sink_30k | 5 | 0.7162 / **0.8225** / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.2414 / 1.4566 |
| 12 | peaked_8k | 5 | 0.6306 / **0.9157** / 1.0160 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.2917 / 1.6314 |
| 12 | flat_8k | 5 | 0.8474 / **0.9550** / 1.0213 | 0.9963 / 0.9993 / 1.0000 | 0.8596 / 0.9591 / 1.0249 |
| 12 | growing_16k | 5 | 0.7227 / **0.9083** / 1.0480 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0010 / 1.0051 |
| 12 | tail (flat 2048-key window) | 20 | 0.8103 / **0.9851** / 1.2904 | 0.9747 / 0.9990 / 1.0193 | 0.7884 / 1.0427 / 1.3255 |
| 12 | short_5 | 5 | 0.7832 / **0.9337** / 1.0000 | 0.7832 / 0.9337 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| 12 | zero | 5 | 1.0000 / **1.0440** / 1.2202 | 1.0000 / 1.0440 / 1.2202 | 1.0000 / 1.0000 / 1.0000 |
| 12 | swa_edge_2040 | 5 | 0.7803 / **0.8823** / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.1382 / 1.4475 |
| 12 | swa_edge_2048 | 5 | 0.7090 / **0.8488** / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.1205 / 1.4076 |
| 12 | swa_edge_2100 | 5 | 0.6252 / **0.7918** / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0341 / 1.1703 |
| 32 | rand_80k | 5 | 0.6523 / **0.7573** / 0.8651 | 0.6523 / 0.7945 / 1.0000 | 1.0000 / 1.0359 / 1.0953 |
| 32 | sink_30k | 5 | 0.5129 / **0.7776** / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0420 / 1.2099 |
| 32 | peaked_8k | 5 | 0.5392 / **0.8937** / 1.0000 | 1.0000 / 1.0156 / 1.0779 | 1.0000 / 1.0181 / 1.0779 |
| 32 | flat_8k | 5 | 0.8898 / **0.9997** / 1.0613 | 0.9152 / 0.9923 / 1.0683 | 0.9268 / 0.9932 / 1.0472 |
| 32 | growing_16k | 5 | 0.9825 / **0.9972** / 1.0036 | 1.0000 / 1.1578 / 1.7855 | 1.0000 / 1.2245 / 1.7855 |
| 32 | tail (flat 2048-key window) | 20 | 0.7198 / **0.9683** / 1.1949 | 0.7604 / 1.0108 / 1.3215 | 0.8079 / 1.0355 / 1.2103 |
| 32 | short_5 | 5 | 0.7832 / **0.9337** / 1.0000 | 0.7832 / 0.9337 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| 32 | zero | 5 | 1.0000 / **1.0440** / 1.2202 | 1.0000 / 1.0440 / 1.2202 | 1.0000 / 1.0000 / 1.0000 |
| 32 | swa_edge_2040 | 5 | 0.6740 / **0.9119** / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.1483 / 1.5006 |
| 32 | swa_edge_2048 | 5 | 0.7236 / **0.9454** / 1.1806 | 0.9282 / 1.0218 / 1.1806 | 1.0000 / 1.1875 / 1.5413 |
| 32 | swa_edge_2100 | 5 | 0.5415 / **0.7463** / 1.0000 | 0.6372 / 0.9000 / 1.0000 | 1.0000 / 1.0014 / 1.0072 |

Worst cases, same seeds [out/kdatt2_adv_3path.log per-seed lines]:
- tail B=1 seed 5: kdatt2 1.312, FA3 tree path 1.000, FA3 1-split 1.381; seed 18: 1.282 / 1.036 / 1.311; seed 8: 1.212 / 1.000 / 1.157.
- tail B=32 seed 1: kdatt2 1.195, FA3 tree path 0.994, FA3 1-split 1.113 (seed 13: 1.130 / 1.322 / 1.210: the FA3 tree path's own worst).
- edge p=2048 B=32 seed 0: kdatt2 1.181, FA3 tree path 1.181, FA3 1-split 1.541 (same value: both are the correction kernel's row there).

Verdict: kdatt2's mean ratio is at or below the FA3 tree path's in every class and batch except flat B=32 (0.9997 vs 0.9923,
within seed noise) and ONE real case: tail (flat window) at B=1,
mean 1.062 vs 1.002 (stock 1.000; FA3 1-split 1.123), +0.06 over 20 seeds (seed spread 0.85-1.31; ~2.4 standard errors, so
probably real but small). At B=12 the same class reads 0.985 vs 0.999 and at B=32 0.968 vs 1.011.

Fewer splits does not help (out/kdatt2_adv_split4.log, prefix splits forced to 4; B=32 already uses 4): B=1 gets WORSE --
tail mean 1.107 (max 1.645), growing 1.312, sink 1.343, flat 1.093; B=12 tail 1.005. Longer per-split accumulation hurts;
B=1 already runs one 128-key tile per split (17 splits), so the default is the most split it can be. Cost of 4 splits
in time was not measured (it can only be slower at B=1: 4 CTAs instead of 17).


### CPU suite on 0c144c0e09

Judge-image CPU suite (stack_k26 list + prompt_lookup + branch): **755 passed, 0 failed, 24 skipped, 3,347 subtests**
[measured out/kdatt2_cpu_suite1.log].

### Reviewer paragraph: sliding-window layers (lane-kdatt2 0c144c0e09)

**Sliding-window verify attention on the same Muse FP8 prefix kernel.** The 46 sliding-window layers (window w = 2048:
FA3 window (2047, 0)) go through the same DFLASH tree verify as the full layers: one call over each request's committed
prefix (p old keys), then the unchanged correction kernel, which adds each row's <= 8 new keys and, for branch rows, the
window-edge keys from the tree's swa_table. That prefix call was FA3 non-causal with window (2039, 7), so verify row
r (r = 0..7) attends exactly the old keys [p + r - 2047, p) (positions below 0 dropped). The prefix kernel now
reproduces that per-row range: it reads key tiles from row 0's first key to p - 1, and the prefill softmax's existing
per-row lower bound masks each row's keys below p + r - 2047 to -inf before the row max, exactly as for any masked key.
The per-element arithmetic is the full-layer kernel's (FA3's FP8 class: E4M3 Q/K/V, FP32 scores, running-max
ex2.approx softmax with P at 2^8, P rounded to E4M3 for the FP8 P V wgmma with FP32 accumulation, FP32 row sum of the
unrounded P, exact rescale on a new max, FP32 LSE merge). Only the order differs (128-key tiles, last to first; split
partition). A row can have every key of a split masked (its window starts after the split's keys): that split then
holds O = 0 and LSE = -inf for the row, FA3's empty-split convention, and the merge gives it weight 0. The kernel is a
template on the window, so the full-layer build is unchanged (0 spills in both builds).

| requests | 1 | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ms per decode step saved by this step (62-layer verify, same-container A/B) | 0.18 | 0.18 | 0.13 | 0.16 | 0.17 | 0.20 | 0.19 |

(1 x H200, judge image, CUDA graphs, prefixes 75k-86k; out/kdatt_ab2.log, out/kdatt_ab3.log.) Both kernel steps
together vs FA3's tree path: 0.22 / 0.36 / 1.18 / 2.40 ms per step at 1 / 4 / 16 / 32 requests (out/kdatt2_gpu4.log).

Accuracy vs an FP64 softmax on the dequantized inputs:
- **64-configuration gate** (incl. window boundaries p = 2040 / 2048 / 2100, chain and branch masks): worst ratio of
  error to stock FA3's on sliding layers 1.0031. The FA3 tree path it replaces reads 1.114.
- **Seeded adversarial classes at w = 2048** (max-error ratio to stock causal FA3, mean over 5-20 seeds; path (1) this
  kernel, (2) the FA3 tree path, (3) FA3 at 1 split, its own re-split envelope; full table above):
  - B = 32: (1) is at or below (2) in every class (e.g. rand 0.757 vs 0.795, growing 0.997 vs 1.158, flat-window tail
    0.968 vs 1.011, edge p = 2048 0.945 vs 1.022).
  - B = 12: likewise (tail 0.985 vs 0.999).
  - B = 1: (1) is at or below (2) in every class except the flat-window tail class (a long run of near-equal small
    logits filling the window). There **its mean is 1.062, against 1.002 for the FA3 tree path and 1.123 for FA3 at
    1 split**; worst seed 1.312 (FA3 1-split 1.381 on the same seed).
  - Forcing fewer splits makes it worse (4 splits: 1.107 mean), so the default (one 128-key tile per split at B = 1)
    is kept.
- Outputs are deterministic, and CUDA-graph replay equals eager at batch sizes 1-40, 48 and 64 for both window kinds.

## Phase 4 (21:27-22:16 CST): per-step eager work -- lane-kfinal4 (from c367f36b48)

| commit | what | eager tree metadata per step, host ms at bs 1 / 4 / 16 / 32 | U1 bytes |
|---|---|---|---:|
| c367f36b48 (base) | FA3 tree schedules built every step | 0.204 / 0.206 / 0.205 / 0.217 | |
| 37e71aac26 (C) | FA3 tree schedules built lazily, only for a layer that falls back to FA3 (`VerifyTree.build`) | 0.104 / 0.103 / 0.102 / 0.102 | +312 |
| 26443069f7 | table kernel reads seq_lens / branch slot / locations directly and writes prefix lengths (no eager copies) | 0.064 / 0.062 / 0.065 / 0.064 | +81 |

[measured out/kdatt2_meta_old.log, out/kfinal4_meta2.log, out/kfinal4_meta3.log; lanes/kmask_tools/kdatt_meta.py, eager,
median of 200; GPU span tracks host time, i.e. the section is launch/host-bound.]

Gates (both commits): 64-config full 1.0008 / SWA 1.0031 (unchanged); GPU tests 21 passed incl. the forced FA3
fallback with a lazily built schedule checked against stock FA3; CUDA-graph sweep 84/84; verify-graph timing unchanged
(ext -2.019 / -0.841 / -0.101 / +0.040 ms vs stock at 32/16/4/1); CPU suite 748 passed / 0 failed on each
[out/kfinal4_gpu1.log, gpu2, gputests1/2, sweep1/2, cpu_suite_C.log, cpu_suite_T.log].

Dropped: PDL launch of the prefix kernel (same-container A/B: 0.008 ms per step, out/kdatt_ab_pdl.log).

### lane-kfinal4b 131000e020 (from 37e71aac26): FA3 fallback schedule rebuilt at every fallback call

Review finding on 37e71aac26: the graph runner's warm-up runs share the capture's VerifyTree, so a cached schedule leaves
the build out of the graph. Fix: `schedule = tree.build(window)` at every FA3 fallback call (no cache), so the build is
captured and re-runs at each replay. The table fusion (26443069f7) was dropped by the lead: runtime Triton
re-specialisation risk.
- New GPU test `test_fa3_fallback_captured_in_a_graph_follows_new_lengths` (full and w = 2047): two warm-ups + capture
  with a raising kernel at prefixes [3000, 5000, 777], replays at [6000, 100, 2500] and [1, 4999, 3000]; max |out -
  stock FA3| 0.023-0.039 (bound 0.125) [measured out/kfinal4b_gputests1.log]. GPU tests: 23 passed.
- The same test on 37e71aac26 (cached) also passes, with the same numbers [measured out/kfinal4_captured_old.log]. A stale
  schedule only fixes FA3's split counts (>= 2) and batch order, and FA3 takes each request's key range from the live
  lengths. So the stale state changed the partition, not the result, in this test. The fix makes the fallback rebuild
  every step anyway; the test guards output correctness but does not show the old code failing.
- 64-config gate unchanged (full 1.0008, SWA 1.0031) [out/kfinal4b_gpu1.log]; CPU suite 748 passed / 0 failed
  [out/kfinal4b_cpu_suite.log]; U1 439,171 vs 439,231 for 37e71aac26 (-60 B).
