"""Lane KMOE2: decode launch of the grouped NVFP4 MoE (fold / split-K) against today's kernel, ONE GPU (Modal).

  modal run harness/modal_kmoe2_ubench.py::cpu_compile          # CPU only (free): build every arm, ptxas, cache
  modal run harness/modal_kmoe2_ubench.py::bench                # 1 GPU, 20-min cap
  modal run harness/modal_kmoe2_ubench.py::bench --extra "--sizes 8,64 --skewed '' --untouched ''"

Mounts lanes/kmoe2 (the lane's tree), stack_o2pd's .cuh as the "base" arm, lanes/kmoe2_tools, lanes/kmoe_tools
(bench helpers), lanes/moe_tools/microbench.py and the kernel-track nvfp4/reference generators.
Results: /vol/out/kmoe2/<ts>/ and stdout.
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
LANE = os.environ.get("KMOE2_LANE", "lanes/kmoe2")
CACHE = "/root/kcache"
CACHE_TAR = "/vol/out/kmoe2/cache.tgz"

app = modal.App("okc-kmoe2-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"SGLANG_JIT_CACHE_DIR": f"{CACHE}/jit", "TORCH_EXTENSIONS_DIR": f"{CACHE}/torch",
          "OKC_KERNEL_TRACK": "/root/lane/muse_moe_w4a16", "MOE_TOOLS": "/root/lane/moe_tools",
          "KMOE_TOOLS": "/root/lane/kmoe_tools", "KMOE2_BASE_CUH": "/root/lane/base/muse_spark_fp4_grouped.cuh",
          "PYTHONPATH": "/root/lane/kmoe2/python"})
    .add_local_dir(os.path.join(SERVING, LANE, "python", "sglang"), "/root/lane/kmoe2/python/sglang")
    .add_local_file(os.path.join(SERVING, "lanes", "kmoe_tools", "kmoe_ceiling.py"), "/root/lane/kmoe_tools/kmoe_ceiling.py")
    .add_local_file(os.path.join(SERVING, "lanes", "stack_o2pd", "python", "sglang", "kernels", "jit", "csrc", "moe",
                                 "muse_spark_fp4_grouped.cuh"), "/root/lane/base/muse_spark_fp4_grouped.cuh")
    .add_local_dir(os.path.join(SERVING, "lanes", "kmoe2_tools"), "/root/lane/kmoe2_tools")
    .add_local_file(os.path.join(SERVING, "lanes", "moe_tools", "microbench.py"), "/root/lane/moe_tools/microbench.py")
    .add_local_file(os.path.join(ROOT, "muse_moe_w4a16", "nvfp4.py"), "/root/lane/muse_moe_w4a16/nvfp4.py")
    .add_local_file(os.path.join(ROOT, "muse_moe_w4a16", "reference.py"), "/root/lane/muse_moe_w4a16/reference.py")
    .add_local_file(os.path.join(ROOT, "muse_moe_w4a16", "checks.py"), "/root/lane/muse_moe_w4a16/checks.py")
    .add_local_file(os.path.join(ROOT, "muse_moe_w4a16", "submission.py"), "/root/lane/muse_moe_w4a16/submission.py")
)
TOOLS = "/root/lane/kmoe2_tools"


def _run(cmd, limit, log=None):
    """Run cmd from the tools dir; stream its output to stdout and, when given, to `log` on the volume."""
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


@app.function(image=image, volumes={"/vol": vol}, timeout=40 * 60, cpu=16, memory=48 * 1024)
def cpu_compile(extra: str = ""):
    """CPU only: build tree, variants, PR #50800 and the kernel-track modules; print ptxas; save the caches."""
    _untar()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kmoe2/{ts}_compile"
    os.makedirs(out, exist_ok=True)
    rc = _run([sys.executable, "kmoe2_bench.py", "--compile-only", *shlex.split(extra)], 35 * 60, f"{out}/compile.log")
    os.makedirs(CACHE, exist_ok=True)
    subprocess.run(["tar", "-czf", CACHE_TAR, "-C", CACHE, "."], check=False)
    vol.commit()
    print("cache saved to", CACHE_TAR, "rc", rc, flush=True)


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=20 * 60, cpu=8, memory=64 * 1024)
def bench(extra: str = "", limit_min: int = 17):
    """One GPU: base / tree / fold / split on the same inputs (kmoe2_bench.py)."""
    _untar()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kmoe2/{ts}_bench"
    os.makedirs(out, exist_ok=True)
    rc = _run([sys.executable, "kmoe2_bench.py", "--out", f"{out}/bench.json", *shlex.split(extra)], limit_min * 60,
              f"{out}/run.log")
    vol.commit()
    print("results in", out, flush=True)
    if os.path.exists(f"{out}/bench.json"):
        print("BENCH_JSON " + open(f"{out}/bench.json").read().replace("\n", " "), flush=True)
    if rc:
        raise RuntimeError(f"bench rc={rc}")
