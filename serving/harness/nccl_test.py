#!/usr/bin/env python3
"""8-GPU interconnect check: peer-access matrix and NCCL all-reduce latency/bandwidth.

Sizes are the per-layer all-reduces Muse Spark issues under TP8 ([M, 8192] bf16 after o_proj and
[M, 4096] bf16 after the MoE) for the batch sizes the judge's ladder produces, plus a large
message for peak bandwidth. Prints one JSON line per measurement from rank 0.
"""
import json
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29571"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    if rank == 0:
        peer = [[int(torch.cuda.can_device_access_peer(i, j)) if i != j else 1 for j in range(world)] for i in range(world)]
        print(json.dumps({"peer_access": peer}), flush=True)
    cases = [("oproj", m, 8192) for m in (1, 8, 32, 256)] + [("moe", m, 4096) for m in (1, 8, 32, 256)] + [("big", 4096, 8192)]
    for name, m, h in cases:
        x = torch.randn(m, h, dtype=torch.bfloat16, device="cuda")
        for _ in range(10):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        iters = 200 if m <= 256 else 20
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / iters * 1e6
        nbytes = m * h * 2
        if rank == 0:
            # Ring all-reduce moves 2*(n-1)/n of the buffer per GPU.
            print(json.dumps({"allreduce": name, "M": m, "H": h, "bytes": nbytes, "us": round(us, 1),
                              "GBps_bus": round(nbytes * 2 * (world - 1) / world / (us * 1e-6) / 1e9, 1)}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    n = torch.cuda.device_count()
    mp.spawn(worker, args=(n,), nprocs=n, join=True)
