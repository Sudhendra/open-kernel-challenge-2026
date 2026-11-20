# H200 SXM (`sm_90a`) Optimization Playbook

Shared context for every agent. Your track-specific file (`BREAKTHROUGH_DIRECTIVE.md` or `CLAUDE.md`) tells you *what* to build; this file tells you the hardware, the rules, and the loop you must follow.

---

## 1. Mission

We are **Overall rank 64 / 1498**. Goal: top 50 (node priority for the Serving Track), then top 10, before the **November 4** close.

| Track | Rank | Score | Speedup needed for 5th |
|---|---|---|---|
| `muse_bf16_proj` | 74 | 1.5438× | 1.0602× |
| `muse_attention_decode` | 55 | 2.6136× | 1.1686× |
| `muse_moe_w4a16` | 60 | 2.3592× | 1.1397× |
| `muse_decode_layer` | 64 | 2.1441× | 1.1207× |

### Cross-pollination rule
`muse_decode_layer` (22.62 ms baseline) ≈ `moe_w4a16` (13.45 ms) + `bf16_proj` (4.83 ms) + `attention_decode` (3.77 ms) + ~0.57 ms glue. **Design every standalone kernel so it can be imported unchanged into `muse_decode_layer/submission.py`:** clean entry points, no global state that assumes a single problem, static buffers keyed by shape.

---

## 2. We are plateaued: what is now banned

Library tuning, Triton autotuning, and basic fusion got us to 1.5×–2.6×. That phase is over.

1. **No blind parameter sweeps.** Don't tweak `num_warps` / `BLOCK_SIZE` without first computing arithmetic intensity and reading `ncu` + `-Xptxas -v` output.
2. **No Ampere-style code for compute-bound tiles.** `mma.sync` + `cp.async` + symmetric warps for large tiles leaves Hopper's TMA/`wgmma` unused. (Exception: skinny-M kernels where `m64` `wgmma` tiles would be mostly padding. See your directive.)
3. **No unprofiled submissions.** Never spend `okc check` (20/day) or `okc submit` (10/day) on a guess.

---

## 3. Hardware facts

| Property | H200 SXM |
|---|---|
| Compute die | Same Hopper die as H100 SXM |
| SMs | 132 |
| L2 cache | 50 MB |
| Shared memory | up to 228 KB / SM |
| Registers | 65,536 × 32-bit / SM |
| Dense BF16 Tensor Core | ~989 TFLOPS |
| Dense FP8 Tensor Core | ~1,979 TFLOPS |
| Memory | 141 GB HBM3e @ **4.8 TB/s** (+~43% vs H100's 3.35 TB/s) |

### Roofline consequences
- **Ridge point** ≈ 206 FLOP/byte (BF16), ≈ 412 FLOP/byte (FP8). Down from ~295 on H100.
- **Decode tracks** (M ∈ {1, 8, 16, 32, 128, 256}) are almost all **HBM-bandwidth-, L2-, or launch-latency-bound**. The 4.8 TB/s ceiling is only reachable with many bulk async (TMA) loads in flight and no launch bubbles.
- Reading 1 MB at 4.8 TB/s ≈ **0.21 µs**. An unoverlapped kernel boundary inside a CUDA graph costs a few µs of drain/fill. At small M, **launch boundaries can cost more than the math.** Fuse, go persistent, or use PDL.
- **Compute-bound kernels** need fewer pipeline stages than on H100 (data arrives faster) → spend the freed SMEM on bigger tiles.

---

## 4. `sm_90a` mechanics (for `load_inline` CUDA)

1. **Compile flags:** `-O3 -arch=sm_90a -std=c++20 --expt-relaxed-constexpr -Xptxas -v`.
   The `a` suffix is required for `wgmma`, `setmaxnreg`, and cluster multicast. Add `--use_fast_math` only after confirming `checks.py` tolerances still pass (norms and `exp` are sensitive).
2. **Zero register spills.** Read `-Xptxas -v` on every build. Any non-zero `spill stores` / `spill loads` = do not benchmark; fix register pressure first.
3. **`wgmma.mma_async`:** 4 warps (128 threads) per instruction (`m64nNk16`, N up to 256). A from SMEM or registers, B from SMEM (`ss` / `rs` modes).
4. **Warp specialization + `setmaxnreg`:** 1 producer warpgroup (issues TMA, needs few registers) + 1–2 consumer warpgroups (run `wgmma` + epilogue). Producer: `setmaxnreg.dec.sync.aligned.u32 40;`. Consumers: `setmaxnreg.inc.sync.aligned.u32 232;` (tune the numbers).
5. **TMA + transaction barriers:** build `CUtensorMap` descriptors once (in `prepare()` where available) with `CU_TENSOR_MAP_SWIZZLE_128B`. One thread issues `cp.async.bulk.tensor`; consumers wait on `mbarrier` byte counts (`mbarrier.arrive.expect_tx` / `mbarrier.try_wait.parity`), not `__syncthreads()`.
6. **Clusters + DSM:** `__cluster_dims__(2|4,1,1)`. TMA multicast (`.multicast::cluster`) loads a shared tile from L2 once into several SMs. `mapa.shared::cluster` gives cross-CTA SMEM access for Split-K / row reductions without touching global memory.
7. **`stmatrix` epilogue:** `stmatrix.sync.aligned.m8n8.x4.b16` writes accumulators to swizzled SMEM conflict-free, then a TMA store (`cp.async.bulk`) writes to HBM asynchronously.
8. **Persistent grids + L2 swizzle:** launch ~132 CTAs (×cluster size) that loop over work tiles; traverse tiles in grouped/swizzled order so concurrent SMs share L2 lines.
9. **Programmatic Dependent Launch (PDL):** when kernels must stay separate, launch the consumer with `cudaLaunchAttributeProgrammaticStreamSerialization`; the producer calls `cudaTriggerProgrammaticLaunchCompletion()` once its dependent work is issued, and the consumer calls `cudaGridDependencySynchronize()` before reading the producer's output. The consumer's preamble overlaps the producer's tail.

---

## 5. Competition constraints (never violate)

1. **Environment:** eval runs in `lmsysorg/sglang:dev-muse-spark-open`, no internet. Available: PyTorch, Triton, FlashInfer, DeepGEMM, TileLang, CuTe DSL, CUTLASS, raw CUDA via `torch.utils.cpp_extension.load_inline`.
2. **CUDA graph capture:** every timed turn replays a captured `torch.cuda.CUDAGraph` with **cold L2** between turns.
   - Inside `kernel_fn`: no `.item()`, `.cpu()`, `torch.cuda.synchronize()`, host branches on tensor values, or allocations whose addresses change across replays.
   - Pre-allocate workspaces/outputs in `prepare()` or cache statically by shape key.
3. **`prepare(weights)`:** runs once, untimed, 300 s budget, for every track **except** `muse_attention_decode`. Compile time doesn't count toward the score (only toward the job timeout). Use it to repack/swizzle weights, split the router, build TMA descriptors, allocate buffers, JIT shape-specialized kernels, and micro-benchmark candidates to fill a dispatch table.
4. **Precision rules:**
   - Weights keep their precision (NVFP4 stays NVFP4, BF16 stays BF16). Lossless reordering/interleaving is allowed; dequantizing in-kernel is allowed.
   - **Router exception:** FP32 router weights `[256, 8192]` may run as a BF16 `hi + lo` pair with FP32 accumulation at **any** batch size:
     `w_hi = w.to(bf16)`; `w_lo = (w - w_hi.float()).to(bf16)`.
   - **Attention exception:** queries may be rounded to **FP8 E4M3** (`torch.float8_e4m3fn`), nothing coarser. KV cache stays FP8.
   - Attention must attend to **every** valid key (full 2048-token window on sliding layers, full context on full layers). No top-k / sparse skipping.
   - **Activations:** do not quantize activations below what the official spec/baseline allows. If unsure whether a proposed activation quantization is legal, check the problem's `TASK.md` / rules first and log the citation in `EXPERIMENTS.md`.

---

## 6. Experiment loop (mandatory)

Never guess. Every iteration:

1. **Snapshot.** Keep `submission_best.py` = current best. Record its per-shape latencies (µs) in `EXPERIMENTS.md`.
2. **Profile.** On the H200 (Modal or RunPod), get per-shape and per-stage timing inside a CUDA graph with an L2 flush (e.g. write a ≥64 MB buffer) between replays. Use `ncu` for the top bottleneck:
   ```bash
   scripts/gpu_run.sh ncu --set full -k regex:"<kernel_name>" \
     --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_issue_stalled_long_scoreboard_per_issue_active.pct,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
     python3 local_bench.py
   ```
3. **Hypothesize first.** Append to `EXPERIMENTS.md` before editing code:
   ```markdown
   ### EXP-XX: <short title>
   - Bottleneck: <shape(s), stage, metric — e.g. "M=16 qkv_gate at 58% DRAM BW">
   - Mechanism: <what hardware feature/fusion you're applying>
   - Expected: <µs before → µs target, metric change>
   - Result: <filled in after>
   - Decision: KEEP / REVERT + why
   ```
4. **Compile gate.** A ~2 s compile-only check (`load_inline(..., verbose=True)` with `-Xptxas -v`): zero errors, zero spills. Only then run the full suite.
5. **Verify.** Pass `checks.py` / `reference.py` on fresh random seeds; capture in a CUDA graph without error; A/B against `submission_best.py` with interleaved, cold-L2 timing across **all** scored shapes.
6. **Keep or revert.** Keep only if the weighted score improves by **≥1.5%** with no correctness loss and no shape regressing meaningfully. Otherwise log the telemetry and restore `submission_best.py`.
7. **Quota gate.** Only after step 6 passes locally: `okc check`. Only after `okc check` confirms: `okc submit`.

### GPU etiquette
- **Modal:** parallel exploration and correctness. Isolated containers, run as many variants as you like.
- **RunPod H200:** final timing and `ncu` only, always via `scripts/gpu_run.sh` (holds `/tmp/h200.lock`) so concurrent agents never pollute each other's measurements.

---

## 7. Local references (no browser needed)

- `../reference_repos/`: DeepGEMM, fast.cu, flashinfer, ThunderKittens, flash-attention, CUTLASS. `grep` here before browsing.
- Installed in the container:
  ```bash
  python3 -c "import sglang, flashinfer, deep_gemm, triton; print(sglang.__file__, flashinfer.__file__, deep_gemm.__file__)"
  ```
  Inspect SGLang's Muse Spark model (`sglang/srt/models/muse_spark*`) for the exact tensor layouts the baseline uses.
- Full curated list: `../REFERENCES.md`.

---

## Claims to verify (from the source chat; unconfirmed)

Treat these as hypotheses and confirm against `TASK.md`, the container, or `ncu` before relying on them:
- Exact µs cost of a CUDA-graph kernel boundary (quoted as "1.8–3.5 µs"). Measure it.
- Whether `nvidia-smi` clock locking is permitted/effective on your RunPod instance, and the right clock values.
- Any specific shape, constant, or dimension quoted in a directive: re-read it from `TASK.md` / `reference.py` first.
