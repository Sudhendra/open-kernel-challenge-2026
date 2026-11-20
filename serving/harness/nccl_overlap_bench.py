"""C2 + C4 gate microbenchmarks, one torchrun process per rank (8 ranks, NCCL over NVLink).

  torchrun --nproc_per_node 8 nccl_overlap_bench.py --mode nccl     # C2: RS/AG at the SP sizes under the env NCCL_* in force
  torchrun --nproc_per_node 8 nccl_overlap_bench.py --mode overlap  # C4 gate 1: 64 MiB AG + RS on a side stream vs a cuBLAS GEMM
  torchrun --nproc_per_node 8 nccl_overlap_bench.py --mode symm     # C4 variant: torch symm_mem fused AG+matmul / matmul+RS if present

Sizes come from lanes/l1/L1_DESIGN.md: reduce-scatter inputs [M_pad, 8192] bf16 (256 MiB total) and [M_pad, 4096] (128 MiB),
all-gather outputs of the same shapes, M_pad = 16384. Each rank prints one JSON line per measurement (rank 0 only) with
median wall ms over REPS after WARM warmups; the driver collects stdout.
"""
import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

M_PAD = 16384
WARM, REPS = 5, 20


def timed(fn, stream=None):
    out = []
    for i in range(WARM + REPS):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= WARM:
            out.append((t1 - t0) * 1e3)
    return statistics.median(out)


def say(rank, **kw):
    if rank == 0:
        print("RESULT " + json.dumps(kw), flush=True)


def mode_nccl(rank, world, tag):
    for hid in (8192, 4096):
        full = torch.randn(M_PAD, hid, dtype=torch.bfloat16, device="cuda")
        shard = torch.empty(M_PAD // world, hid, dtype=torch.bfloat16, device="cuda")
        mib = full.numel() * 2 / 2**20
        rs = timed(lambda: dist.reduce_scatter_tensor(shard, full))
        ag = timed(lambda: dist.all_gather_into_tensor(full, shard))
        # bus bandwidth as nccl-tests defines it: bytes * (n-1)/n / time
        bw = lambda ms: (full.numel() * 2 * (world - 1) / world) / (ms / 1e3) / 1e9
        say(rank, mode="nccl", tag=tag, hidden=hid, MiB=round(mib), rs_ms=round(rs, 3), ag_ms=round(ag, 3),
            rs_busbw_GBs=round(bw(rs), 1), ag_busbw_GBs=round(bw(ag), 1))


def mode_overlap(rank, world, tag):
    # compute: cuBLAS bf16 GEMM 8192 x 8192 x 2304 (the dead_ends gate spec), issued N times so it outlasts the collective
    a = torch.randn(8192, 2304, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(2304, 8192, dtype=torch.bfloat16, device="cuda")
    full = torch.randn(M_PAD, 4096, dtype=torch.bfloat16, device="cuda")      # 128 MiB total, 16 MiB per rank shard... x8 = 64 MiB moved per rank
    shard = torch.empty(M_PAD // world, 4096, dtype=torch.bfloat16, device="cuda")
    side = torch.cuda.Stream()
    n_gemm = 12

    def gemm():
        for _ in range(n_gemm):
            torch.matmul(a, b)

    def coll():
        dist.all_gather_into_tensor(full, shard)
        dist.reduce_scatter_tensor(shard, full)

    def pair():
        with torch.cuda.stream(side):
            coll()
        gemm()
        torch.cuda.current_stream().wait_stream(side)

    g = timed(gemm)
    c = timed(coll)
    p = timed(pair)
    serial = g + c
    hidden = (serial - p) / c if c > 0 else 0.0
    say(rank, mode="overlap", tag=tag, gemm_ms=round(g, 3), coll_ms=round(c, 3), pair_ms=round(p, 3),
        serial_ms=round(serial, 3), hidden_frac=round(hidden, 3), gate_pass=bool(hidden >= 0.6))


def mode_symm(rank, world, tag):
    try:
        import torch.distributed._symmetric_memory as symm  # noqa: F401
        has_fused = hasattr(torch.ops.symm_mem, "fused_all_gather_matmul") and hasattr(torch.ops.symm_mem, "fused_matmul_reduce_scatter")
    except Exception as e:  # noqa: BLE001
        say(rank, mode="symm", tag=tag, available=False, why=str(e)[:120]); return
    say(rank, mode="symm", tag=tag, available=bool(has_fused), torch=torch.__version__, nccl=".".join(map(str, torch.cuda.nccl.version())))
    if not has_fused:
        return
    group = dist.group.WORLD
    try:
        symm.enable_symm_mem_for_group(group.group_name)
        x = torch.randn(M_PAD // world, 4096, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(4096, 8192, dtype=torch.bfloat16, device="cuda")
        full = torch.empty(M_PAD, 4096, dtype=torch.bfloat16, device="cuda")

        def serial_ag_mm():
            dist.all_gather_into_tensor(full, x)
            torch.matmul(full, w)

        def fused_ag_mm():
            torch.ops.symm_mem.fused_all_gather_matmul(x, [w], gather_dim=0, group_name=group.group_name)

        s = timed(serial_ag_mm); f = timed(fused_ag_mm)
        say(rank, mode="symm", tag=tag, op="ag_matmul", serial_ms=round(s, 3), fused_ms=round(f, 3), gain=round(1 - f / s, 3))
        y = torch.randn(M_PAD, 8192, dtype=torch.bfloat16, device="cuda")
        w2 = torch.randn(8192, 4096, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(M_PAD // world, 4096, dtype=torch.bfloat16, device="cuda")

        def serial_mm_rs():
            dist.reduce_scatter_tensor(out, torch.matmul(y, w2))

        def fused_mm_rs():
            torch.ops.symm_mem.fused_matmul_reduce_scatter(y, w2, "sum", scatter_dim=0, group_name=group.group_name)

        s = timed(serial_mm_rs); f = timed(fused_mm_rs)
        say(rank, mode="symm", tag=tag, op="matmul_rs", serial_ms=round(s, 3), fused_ms=round(f, 3), gain=round(1 - f / s, 3))
    except Exception as e:  # noqa: BLE001
        say(rank, mode="symm", tag=tag, error=str(e)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["nccl", "overlap", "symm"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    {"nccl": mode_nccl, "overlap": mode_overlap, "symm": mode_symm}[args.mode](rank, world, args.tag)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
