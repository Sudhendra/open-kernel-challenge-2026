# Dead ends considered for the top-5 plan (2026-11-10)

Companion to `TOP5_LEVERS_2026-11-10.md`. Each row: the idea, the number that killed it, and the source. "Dead" means not worth a lane
before Nov 19 on this evidence; "parked" means a measurement could revive it. Rows marked (lane) were established by the named lane
and are restated here so the list is complete; rows without a mark are rejections made while writing the plan.

## Prefill: attention

| Idea | Killing number | Source / note |
|---|---|---|
| FA3 tuning for head_dim 64 fp8 (tiles, pipelining) | FA3 is 68-75 ms of a 620 ms chunk (11%); the kernel lives in `sgl-kernel`, outside `python/sglang/`, so the patch cannot change it; a JIT replacement attention kernel for 80k prefill is weeks | P17 rows 7/13; RULES.md patch rules |
| SWA layers attending only the window | already true: 45 SWA launches at 0.30-0.32 ms regardless of prefix; full layers 1.7-6.7 ms | P17 section 2 (d) |
| BF16 KV on the SWA layers (vLLM note: FP8 KV gives little on bounded-window layers) | rules: "the FP8 KV cache stays FP8" | RULES.md; web research (d) |
| FP8 query path changes | stock already casts the query to the KV dtype in FA3; rules say it stays as it is | RULES.md |
| Context/sequence-parallel attention (ring or Ulysses) for one 80k prompt at c1 | attention is 11% of the chunk; the chunk already runs on all 8 GPUs (TP8 heads + EP8 experts); CP would add KV exchange for a kernel we cannot modify | P17 (d); lane H dead_ends row 5 |
| Prefill chunk pipelining across GPUs (PP-like) | layer l of chunk k+1 needs chunk k's layer-l KV; each chunk is one breakable graph replay; nothing independent exists until the next forward batch (lane J (a)) | J_DESIGN section 3 (a) |
| Fewer, larger full-attention launches (merge the 14 full layers' FA3 calls) | separate layers, separate K/V; 0 |  |

## Prefill: MoE and the expert boundary

| Idea | Killing number | Source / note |
|---|---|---|
| Faster expert GEMM beyond the grouped kernel (FlashInfer SM90 CUTLASS W4A16 #24816, NVFP4 MegaMoE #45613, Humming) | grouped kernel at M = 16k is 124 ms vs 100 ms bf16-peak ideal (~80%); MegaMoE/Humming SM90 paths are W4A8 (FP8 activations), forbidden; #45613 needs a FlashInfer newer than the image | VALUE_TABLE P13 microbench 3.16x; RESOURCES section 4 |
| Dequant-once + BF16 grouped GEMM | bf16 copy of 50.6 GiB NVFP4 experts = 202 GiB per rank vs 141 GiB HBM; per-chunk dequant streams 50 GiB (10 ms) and loses the W4 bandwidth advantage at decode | MOE_RUNNER_FEASIBILITY section 2 |
| Second weight copy for a Marlin fallback at some sizes | 50.6 GiB extra vs 44.6 GiB KV pool | MOE_RUNNER_FEASIBILITY section 1 |
| Expert TP inside EP (EP4 x TP2) to cut load variance | variance of a 64-term sum drops ~1/sqrt(2): -20 ms/chunk; grouped kernel hard-codes 32 experts and 4096/4096; NVFP4 loader has no TP slicing and `model_loader/` is off-limits | P15_dead_ends (lane) |
| MoE-TP instead of EP at prefill (balanced by construction) | weights are loaded EP-sharded by `model_loader/` (off-limits); narrower N = 1024 per rank hurts the grouped kernel | P17 lever 7 |
| Token-routed dispatch/combine (FlashInfer `MoeAlltoAll`) instead of all-reduce/reduce-scatter | combine moves 56 KiB/token vs 7 KiB for reduce-scatter; needs MNNVL fabric memory and ~2 GiB workspace; still waits on the slowest rank | P15 section 4 (lane) |
| Scheduling-only fixes for the EP straggler wait (largest-first tiles, token-split with early reduce-scatter, side-stream RS) | 1 ms / 10 ms / 10 ms per chunk against a 3% bar; the slow rank is slow on both halves | J_DESIGN section 3 (lane) |
| Column-split combine (reduce half the hidden dim early) | post-expert RMSNorm reduces over the full 4096 row before the weighted sum; needs a two-pass norm (new numerics) | J dead_ends 2 (lane) |
| Fuse `post_expert_proj` into the combine at prefill | doubles MoE all-reduce bytes (128 -> 256 MiB per layer), removes no replicated GEMM; at decode the bytes are 147 KB, so it survives as D1 (c) | P15_dead_ends (lane); TOP5 D1 |
| NCCL protocol/algorithm tuning for the 128-256 MiB prefill messages | o_proj all-reduce runs at 1.33 ms vs 1.30 ideal; the MoE AR excess is wait, not transfer | P15_dead_ends (lane) |
| In-tree `--enable-layernorm-sp` as shipped (P5a) | its Muse hook requires `--moe-a2a-backend flashinfer` + `flashinfer_trtllm` runner + Blackwell CUTLASS collectives; lane H re-implemented the region in Muse code instead | lane H dead_ends row 1 |
| P17 lever 4 (reduce-scatter the MoE output only, keep the o_proj all-reduce) | half of H's gain for most of its plumbing; superseded by H | lane H dead_ends row 6 |
| P17 "tail amortisation" by 32k chunks (-37 ms per 16k) | the tail is per-token (layers 0/61 experts and dense, draft projection), not per-forward; only the skew-averaging part (-7 to -20 ms grouped) is real | TOP5 K-eager arithmetic |
| Lane K as built (32k chunk + 32k graph buckets) | capture memory 10.5 -> 20.7 GB, max running 26 -> 16, c32 TTFT 1.90x stock vs the 2.0x gate | NEXT_STEPS section 26 (session 3) |
| `max_prefill_tokens` 32768 with chunk 16384 | same bucket requirement as K; helps only when two cold requests are queued | P17 section 3 rank 2 |
| cuBLASLt table as a first lever | 10-25 ms/chunk on 92 ms of dense GEMMs at 71-78% of peak; kept at rank 10, not dead |  |

## Prefill: scheduling, chunks, cache

| Idea | Killing number | Source / note |
|---|---|---|
| `schedule_conservativeness`, `num_continuous_decode_steps`, `schedule_policy`, `prefill_decode_interval` | admission-only or decode-only; no prefill throughput effect; continuous decode steps worsen TTFT | P17 section 3 rows 5-7 |
| Mixed chunk as a throughput lever (lane G as built) | decode rows ride as 1-token extends: throughput neutral (229 vs 229 at c16), interactivity -5%; its real value is the spec-aware form (G2) | NEXT_STEPS section 22 |
| Prefill CUDA-graph coverage gaps | partial last chunks pad <= 511 tokens: <= 1.6% on that chunk, ~0.4% overall; idle inside a forward 0.4% | P17 section 3 row 3, section 1 |
| Residual HiCache write-back host work | 0-40 ms unknown; the measured 620 ms chunk (vs P17's 629 modeled) leaves <= 10 ms for it | P17 lever 12; session 4 TTFT |
| Raising the prefix hit rate | already at its maximum: 83k computed of 326k input per 4-turn conversation = 75% (FGHIL c32); prompts are unique across conversations | metrics_after_proxy.txt |
| Host-to-device bandwidth for prefix reloads | 320 MB per 80k prefix per rank at ~20 GB/s = 16 ms vs 3.1 s recompute; never the lever | TOP5 section 2 (HiCache) |
| #48950 hybrid-SWA HiCache host ratio, #46934 sub-window L3 hits | host_hit tokens ~0 for both stock and FGHIL on the proxy; L3 storage tier unused | metrics; web research |
| Judge's 160 GB host pool as a lever for us | it changes stock's c32 (no collapse -> our c32 ~2.3x), not ours; our 75% hit rate needs only the device pool |  |

## Decode

| Idea | Killing number | Source / note |
|---|---|---|
| Draft tree inside the 8-token verify block (P16) | modeled -2 to -14% decode at c1, -7 to -19% at c16 (slot-1..3 hit rates 0.7-0.9 leave little to recover; a displaced chain node at depth 6-7 carries 0.15-0.3 of paths) | P16_TREE_FEASIBILITY section 4 (lane) |
| 16-token verify block / DDTree at all rungs | +25 to +45% step cost at c16/c32 (P3's 5-vs-8 measurement); only bs <= 4 is positive; +0 to +3% score; needs the per-slot histogram first | P16 section 4; DDTree arXiv 2604.12989 |
| Draft length 5 at c16/c32 (P3) | +0.7% throughput measured; interactivity +11% but not scored | VALUE_TABLE P3 |
| Longer draft block (16) | the released DFlash2 is block 8; a train-8 drafter does not generalise up; retraining is forbidden | DFlash paper 2602.06036; RULES.md |
| `--speculative-use-block-verification` | block verification is an exact-distribution sampling method; identical to token verification at T = 0 (greedy) | DFlash2 model card note; theory |
| Prompt lookup / n-gram (F, SuffixDecoding-style) | offline sim 0 to +2.5% at c16/c32; measured throughput-neutral at c4/c16 (229 vs 229); in the stack, done | VALUE_TABLE P4; session 1 |
| TP-split draft `fc` with multicast all-gather (#45895) | fires only at bs 1 on H200, not bitwise, 0.3-0.8% | RESOURCES section 10 |
| Finalize + all-reduce fusion (#46724), AR + residual-norm (#46137/#47174) | capability >= 10 gates, push workspace 128 KB rejects M = 9..16 on SM90, CuTeDSL on SM90 unverified; 0.6-1.8% per step on GB300 | RESOURCES section 10; web research |
| Symmetric-memory multimem all-reduce by default | the winning rows were the in-place `raw` path, unreachable through the dispatcher; the reachable `symm_mem` path loses at every size | VALUE_TABLE P1 (lane A) |
| Faster fp32 router GEMM via split-K | changes the fp32 summation order (routing ties); replaced by the column-split in D1, which is bit-exact per element | I dead_ends 9; TOP5 D1 (b) |
| Launcher thread for the 3.2 ms verify `cudaGraphLaunch` | the eager tail must be enqueued after the launch on the same stream; a pipeline restructure of the overlap worker | I dead_ends 4 (lane) |
| Rank-local "my queues are empty" HiCache sync skip; periodic safety round | not rank-symmetric: gloo collectives mismatch is a hang; replaced by the attempt-counter design | I dead_ends 2-3 (lane) |
| Small-batch attention replacement for sliding layers (P12, Gaurav's kernel) | FA3 is 6-10% of a decode step; 898 KB source to extract; 3x on the kernel moves the score ~1% | VALUE_TABLE P12 |
| Treating c1 as kernel-bound | host 16.5 ms = GPU 16.5 ms; the 1.3x MoE kernel moved c1 decode 232 -> 231 tok/s | I_AUDIT (lane) |
| Decode-only program as the path to 3x | decode is 15-46% of the wall; a decode-only stack caps at ~1.3x overall | ROADMAP_2X section 1 |

## Memory

| Idea | Killing number | Source / note |
|---|---|---|
| KV pool growth as a score lever (every GB -> KV) | pool 2.99M tokens vs 2.75M needed at 32 x 86k; capacity binds only if judge prompts run longer; kept as insurance (#48871, #48785) | session 4 |
| Lane L alternatives (held references, chunked repack, empty_cache after each layer without reordering) | 9.22-9.88 GiB still stranded in the allocator replay; only allocation order + empty_cache works | lane L dead_ends |
| Lossless weight recoding (F11/F12 style) | rules ambiguity on "the model must stay the model"; no speed gain at prefill (compute-bound) | reference-gaurav-repo memory |

## Structural rewrites

| Idea | Killing number | Source / note |
|---|---|---|
| Full EP rewrite ("switch on EP8", "experts at 8x rows") | already EP8 under the judge's flags: 32 local experts per rank on all 16k tokens' routes; nothing to switch | P15_dead_ends (lane) |
| In-tree TBO (`two_batch_overlap.py`) as shipped | DeepSeek/Qwen3/MiMo/DSV4 operation lists only, DeepEP + DP attention; Muse needs its own strategy and comm stream (that is L1, 60-100 h), not a flag | J_DESIGN (d); web research (a) |
| Prefill/decode disaggregation inside one node | the judge runs one 8xH200 node with fixed flags; PD needs separate instances and transfer engines | RULES.md |
| Draft tree as a "decode rewrite that must start now" (ROADMAP_2X step 3) | re-scored to +0 to +3% by P16; the structural prefill lever (L1) replaces it as the rewrite to start | P16 verdict |
| Hot-expert replication built blind | no routing counts exist (recorder off in `stock.json`); rules unasked; kept as R1 pending the Nov 12 instrumentation check | P15 section 3 |
| Waiting for the organisers' answer on SP-style regions before shipping H | H is in the scored day-one rung by user decision; a failure there falls back to FGH/stack/HiCache-only on day two | JUDGE_RUNBOOK section 7 |

- D2 skip-draft-for-non-consuming-rows (Nov 12): vLLM MRV2's +7.3% does not transfer; in SGLang the DFlash drafter is never invoked on a mixed or prefill chunk (draft runs only in the decode-side verify loop), so under 0.3% of chunk time is skippable. Analysis in lanes/d2/ANALYSIS.md. No code.
