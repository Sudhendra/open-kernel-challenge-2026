# Hopper / H200 Reference Library

Curated resources for `sm_90a` kernel work. **Check `../reference_repos/` first.** Local grep is faster and more reliable than a browser on long C++ headers.

---

## 1. Worklogs & tutorials (read first)

| Resource | Link | Why |
|---|---|---|
| Pranjal Shankhdhar: *Outperforming cuBLAS on H100: a Worklog* | Blog: https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog · Code: https://github.com/pranjalssh/fast.cu | Hopper equivalent of Simon Boehm's Ampere GEMM post. Builds a GEMM step by step to beat cuBLAS: TMA descriptors, `wgmma`, warp specialization, persistent kernels, L2 swizzle. |
| Colfax Research: TMA | https://research.colfax-intl.com/tutorial-hopper-tma/ | TMA descriptors, swizzle modes, multicast. |
| Colfax Research: WGMMA | https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/ | `wgmma` operand layouts and async semantics. |
| Colfax Research: Pipelining / warp specialization | https://research.colfax-intl.com/cutlass-tutorial-design-of-an-efficient-gemm-kernel/ | `mbarrier` phase flipping, producer/consumer state machine. |
| Colfax Research: Persistent kernels & Stream-K | https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/ | Persistent tile schedulers, Stream-K for awkward shapes. |

## 2. Codebases to read / mine for few-shot examples

| Repo | Link | Use it for |
|---|---|---|
| **DeepGEMM** (DeepSeek) | https://github.com/deepseek-ai/DeepGEMM | Small, readable Hopper GEMM library. JIT shape specialization (M/N/K as compile-time constants), TMA multicast across clusters, grouped GEMM (→ MoE). |
| **FlashAttention-3** | https://github.com/Dao-AILab/flash-attention/tree/main/hopper | Ping-pong scheduling, WGMMA/softmax overlap, FP8 attention. Paper: *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*. |
| **FlashInfer** | https://github.com/flashinfer-ai/flashinfer | Pre-installed in the eval container; SGLang uses it. Look at `include/flashinfer/attention/hopper/`, paged KV loaders, fused MoE. Find where its general heuristics leave perf on the table for Muse Spark's fixed shapes. |
| **Marlin / Machete** | https://github.com/IST-DASLab/marlin · vLLM `machete` kernels | W4A16 weight interleaving done offline so runtime dequant is a couple of `lop3`/shift ops before the BF16 MMA. Machete is the Hopper variant. Directly relevant to `prepare()` repacking for `moe_w4a16`. |
| **ThunderKittens** | https://github.com/HazyResearch/ThunderKittens | Clean tile primitives wrapping `wgmma`, TMA, swizzled SMEM. Easy to port into a standalone `load_inline` file. |
| **CUTLASS 3.x / CuTe** | https://github.com/NVIDIA/cutlass | Reference for EVT (epilogue visitor trees), grouped GEMM, NVFP4/block-scaled kernels. |

## 3. Primary documentation

| Resource | Link | Use it for |
|---|---|---|
| NVIDIA PTX ISA | https://docs.nvidia.com/cuda/parallel-thread-execution/ | Authoritative syntax + register-fragment layout diagrams for `wgmma` (needed to fuse activations onto accumulators), `cp.async.bulk.tensor`, `mbarrier`, `setmaxnreg`, `stmatrix`, cluster/`mapa`. Search by instruction name rather than section number. |
| CUDA Programming Guide (Hopper sections) | https://docs.nvidia.com/cuda/cuda-c-programming-guide/ | Thread block clusters, distributed shared memory, Programmatic Dependent Launch. |

## 4. Microbenchmarking & lectures

| Resource | Why |
|---|---|
| *Dissecting the NVIDIA Hopper Architecture through Microbenchmarking* (Luo et al.) | Measured instruction latencies, DSM bandwidth, TMA overheads, L2 behaviour on Hopper. Search arXiv for the title. |
| GPU MODE lectures: https://github.com/gpu-mode/lectures | Talks on FlashAttention-3, CUTLASS 3/CuTe, and GEMM optimization. The OKC evaluator was adapted from GPU MODE's `kernelbot` / `reference-kernels`, so their harness conventions apply. |

## 5. How agents should use this

1. `grep` `../reference_repos/` for the exact instruction or pattern you need (e.g. `setmaxnreg`, `expect_tx`, `multicast::cluster`, `stmatrix`).
2. Copy the *smallest* working pattern into your `load_inline` source; don't import the whole library.
3. Cite the file you adapted from in your `EXPERIMENTS.md` entry.
