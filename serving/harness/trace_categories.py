#!/usr/bin/env python3
"""GPU time by category from torch-profiler traces, as JSON: {trace: {category: ms, ...}}.

  python trace_categories.py out.json trace1.json.gz [trace2.json.gz ...]

Categories are assigned from kernel names (first match wins).
"""
import gzip
import json
import sys
from collections import defaultdict

RULES = [
    ("moe_marlin", ("marlin_moe",)),
    ("moe_triton_bf16", ("fused_moe_kernel",)),
    ("allreduce_custom", ("all_reduce_1shot", "all_reduce_2shot", "all_reduce_")),
    ("allreduce_nccl", ("ncclDevKernel", "ncclKernel")),
    ("attention_fa3", ("FlashAttnFwd", "flash::")),
    ("router_fp32_gemm", ("s1688gemm", "cutlass_80_tensorop_s1688")),
    ("gemm_bf16", ("nvjet", "cutlass", "gemm", "splitKreduce", "Kernel2<")),
    ("d2h_copy", ("Memcpy DtoH",)),
    ("memcpy_other", ("Memcpy", "memcpy_kernel", "Memset")),
    ("norm", ("norm", "layer_norm", "rms")),
    ("routing", ("routing_kernel",)),
    ("activation", ("act_and_mul", "silu")),
    ("reduce", ("reduce_kernel",)),
    ("elementwise", ("elementwise", "vectorized", "unrolled", "index", "fill", "cat", "copy")),
]


def cat(name):
    for c, keys in RULES:
        if any(k in name for k in keys):
            return c
    return "other"


def main():
    out_path, traces = sys.argv[1], sys.argv[2:]
    result = {}
    for p in traces:
        with gzip.open(p, "rt") as f:
            ev = json.load(f)["traceEvents"]
        tot = defaultdict(float)
        n = defaultdict(int)
        for e in ev:
            if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e:
                c = cat(e["name"])
                tot[c] += e["dur"] / 1e3
                n[c] += 1
        key = p.rsplit("/", 1)[-1].split("-")[0] + "_" + ("extend" if "EXTEND" in p else "decode")
        result[key] = {"ms": dict(sorted(tot.items(), key=lambda kv: -kv[1])), "launches": dict(n), "total_ms": sum(tot.values())}
        print(key, round(sum(tot.values())), {k: round(100 * v / sum(tot.values()), 1) for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:8]}, flush=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
