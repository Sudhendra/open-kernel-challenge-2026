# Open Kernel Challenge: Muse Spark Open on H200

Work from the [Open Kernel Challenge](https://openkernelchallenge.dev) (Oct 22 to Nov 19, 2026): make SGLang serve Muse Spark Open, an 816B-parameter NVFP4 model, faster on NVIDIA H200 GPUs.

The challenge had two halves. Five single-GPU kernel problems came first and set the queue priority for the second half, the serving track, which was the prize track: one patch to SGLang, scored on a whole 8xH200 node under multi-turn load.

## Results

| Track | Result | Where |
|---|---|---|
| **Serving** (8xH200, 1 to 32 concurrent users) | **3.073x** stock SGLang throughput at stock's interactivity; all correctness, latency and memory gates passed | [serving/](serving/) |
| Kernel problems (single H200) | 2.630x attention decode, 2.316x decoder layer, 2.418x routed experts, 1.898x prefill layer; rank 85 overall | [kernels/](kernels/) |

Serving standings are provisional until the organizers finish their code review and reruns. The board was still private when this was written, so no rank is given.

## What is original here

- **The serving patch is the original work.** It is 409 KB across 67 SGLang files: a grouped NVFP4 expert kernel, small-M decode GEMMs, FP8 prefill and verify attention kernels with FlashAttention-3's arithmetic, token-sharded prefill, an in-row branch verify for DFlash, prompt-lookup drafts, online expert placement and scheduler changes. Two of its CUDA kernels derive from public kernel-track submissions and say so. [serving/README.md](serving/README.md) has the result tables, how to apply the patch, and the reviewer notes that explain why each change is lossless.
- **The kernel-problem files are mostly other people's work.** They are stacks of public Apache-2.0 submissions by other participants, ported and re-validated with small changes. Each file's header lists its lineage. [kernels/README.md](kernels/README.md) has the details.

## Layout

| Path | What |
|---|---|
| [serving/](serving/) | the scored patch, reviewer notes, evidence logs, method, and the 8-GPU test harness |
| [kernels/](kernels/) | the four kernel-problem files and a script that checks each against its scored job |
| [okloop/](okloop/) | the unattended loop that ran agent attempts on the decoder-layer problem: gate, lanes, prompts |
| [H200_SM90A_PLAYBOOK.md](H200_SM90A_PLAYBOOK.md) | the hardware and process brief the kernel-track agents worked from (late October; its standings are of that date) |
| [REFERENCES.md](REFERENCES.md) | reading list for Hopper kernel work |

This repository is a curated cut of a much larger working directory. Experiment trees, raw run outputs, candidate patches and the daily ledger are not included.

## How it was built

Most of the code, experiments and write-ups were produced by coding agents (Claude Code and Codex) under direction, with every claim tied to a measurement: single-GPU microbenchmarks for kernels, paired 8-GPU runs on Modal for end-to-end changes, and the judge's own check runs as the final word. `okloop/` is the loop used on the kernel problems; `serving/METHOD.md` and `serving/harness/` are the serving-track equivalents.

## License

Apache-2.0, as the challenge requires of every submission. Third-party work is credited in [NOTICE](NOTICE) and in the headers of the files themselves.
