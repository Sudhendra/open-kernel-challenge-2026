"""Lane KMICRO: router side-stream priority + GEMM2 prefetch before the spin, ONE GPU each (Modal, judge image).

  modal run harness/modal_kmicro_ubench.py::cpu_compile                       # CPU only: build base / km / kmoff, ptxas
  modal run harness/modal_kmicro_ubench.py::bench --part prio                 # 1 H200, function timeout 10 min
  modal run harness/modal_kmicro_ubench.py::bench --part pf                   # 1 H200, function timeout 10 min

Mounts lanes/kmicro python/sglang, stack-k33's .cuh as "base", lanes/kmicro_tools, lanes/moe_tools/microbench.py and
the kernel-track nvfp4 / reference generators. Results: /vol/out/kmicro/<ts>_<part>/ and stdout (RESULT_JSON line).
"""
import os
import shlex
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
ROOT = os.path.dirname(SERVING)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
CACHE = "/root/kcache"
CACHE_TAR = "/vol/out/kmicro/cache.tgz"
DEDUP_TEST = os.path.join("test", "registered", "unit", "model_executor", "runner_backend", "test_cuda_graph_dedup.py")
CUH = os.path.join("python", "sglang", "kernels", "jit", "csrc", "moe", "muse_spark_fp4_grouped.cuh")

app = modal.App("okc-kmicro-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"SGLANG_JIT_CACHE_DIR": f"{CACHE}/jit", "TORCH_EXTENSIONS_DIR": f"{CACHE}/torch",
          "OKC_KERNEL_TRACK": "/root/lane/muse_moe_w4a16", "MOE_TOOLS": "/root/lane/moe_tools",
          "KMICRO_VARIANT_DIR": f"{CACHE}/variants",
          "KMICRO_BASE_CUH": "/root/lane/base/muse_spark_fp4_grouped.cuh",
          "KMICRO_LANE_CUH": f"/root/lane/kmicro/{CUH}",
          "PYTHONPATH": "/root/lane/kmicro/python"})
    .add_local_dir(os.path.join(SERVING, "lanes", "kmicro", "python", "sglang"), "/root/lane/kmicro/python/sglang",
                   ignore=["**/__pycache__"])
    .add_local_file(os.path.join(SERVING, "lanes", "stack_k33", CUH), "/root/lane/base/muse_spark_fp4_grouped.cuh")
    .add_local_file(os.path.join(SERVING, "lanes", "kmicro_tools", "kmicro_bench.py"), "/root/lane/kmicro_tools/kmicro_bench.py")
    .add_local_file(os.path.join(SERVING, "lanes", "kmicro", DEDUP_TEST), f"/root/lane/kmicro/{DEDUP_TEST}")
    .add_local_file(os.path.join(SERVING, "lanes", "moe_tools", "microbench.py"), "/root/lane/moe_tools/microbench.py")
    .add_local_file(os.path.join(ROOT, "muse_moe_w4a16", "nvfp4.py"), "/root/lane/muse_moe_w4a16/nvfp4.py")
    .add_local_file(os.path.join(ROOT, "muse_moe_w4a16", "reference.py"), "/root/lane/muse_moe_w4a16/reference.py")
    .add_local_file(os.path.join(ROOT, "muse_moe_w4a16", "checks.py"), "/root/lane/muse_moe_w4a16/checks.py")
)
TOOLS = "/root/lane/kmicro_tools"


def _run(cmd, limit, log=None):
    print("+", " ".join(cmd), flush=True)
    t0 = time.time()
    f = open(log, "w") if log else None
    p = subprocess.Popen(cmd, cwd=TOOLS, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
            if f:
                f.write(line)
                f.flush()
            if time.time() - t0 > limit:
                p.kill()
        rc = p.wait()
    finally:
        if f:
            f.close()
    if time.time() - t0 > limit:
        rc = 124
    print(f"rc={rc} after {time.time() - t0:.0f}s", flush=True)
    return rc


def _untar():
    if os.path.exists(CACHE_TAR):
        os.makedirs(CACHE, exist_ok=True)
        subprocess.run(["tar", "-xzf", CACHE_TAR, "-C", CACHE], check=False)
        print("cache restored from", CACHE_TAR, flush=True)


@app.function(image=image, volumes={"/vol": vol}, timeout=20 * 60, cpu=16, memory=32 * 1024)
def cpu_compile():
    """CPU only: build base / km / kmoff (ptxas -v), save the JIT cache for the GPU run."""
    _untar()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kmicro/{ts}_compile"
    os.makedirs(out, exist_ok=True)
    rc = _run([sys.executable, "kmicro_bench.py", "--compile-only"], 18 * 60, f"{out}/compile.log")
    os.makedirs(CACHE, exist_ok=True)
    subprocess.run(["tar", "-czf", CACHE_TAR, "-C", CACHE, "."], check=False)
    vol.commit()
    print("cache saved to", CACHE_TAR, "rc", rc, flush=True)


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=10 * 60, cpu=8, memory=64 * 1024)
def bench(part: str = "prio", extra: str = "", limit_min: int = 9):
    _untar()
    subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.max.sm,power.limit", "--format=csv"], check=False)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kmicro/{ts}_{part}"
    os.makedirs(out, exist_ok=True)
    rc = _run([sys.executable, "kmicro_bench.py", "--part", part, "--out", f"{out}/{part}.json", *shlex.split(extra)],
              limit_min * 60 - 30, f"{out}/run.log")
    if part == "prio":  # the dedup registry's instantiate flags: its GPU unit test, kmicro and kill switch
        for kill in ("0", "1"):
            env = dict(os.environ, SGLANG_DISABLE_MUSE_ROUTER_STREAM_PRIORITY=kill)
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", f"/root/lane/kmicro/{DEDUP_TEST}"],
                               cwd="/root/lane/kmicro", env=env, capture_output=True, text=True, timeout=120)
            print(f"DEDUP_TEST kill={kill} rc={r.returncode}\n" + r.stdout[-1500:] + r.stderr[-800:], flush=True)
    vol.commit()
    print("results in", out, "rc", rc, flush=True)
