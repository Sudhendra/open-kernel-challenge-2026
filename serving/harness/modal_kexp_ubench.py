"""Lane KEXP: the fused grouped NVFP4 expert launch at prefill sizes, variants vs stack-k9's kernel, ONE GPU (Modal).

  modal run harness/modal_kexp_ubench.py::cpu_compile --extra "--arms base,g4big,prof"   # CPU only (free), caches
  modal run harness/modal_kexp_ubench.py::bench --extra "--arms base,g4big --reps 200"     # 1 GPU, 12-min hard cap

Mounts lanes/kexp (the lane's tree), stack_k9's .cuh as the "base" arm, lanes/kexp_tools, lanes/moe_tools/microbench.py
and the kernel-track nvfp4 generator. Results: /vol/out/kexp/<ts>_bench/ and stdout (BENCH_JSON line).
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
LANE = os.environ.get("KEXP_LANE", "lanes/kexp")
CACHE = "/root/kcache"
CACHE_TAR = "/vol/out/kexp/cache.tgz"
CUH = os.path.join("python", "sglang", "kernels", "jit", "csrc", "moe", "muse_spark_fp4_grouped.cuh")

app = modal.App("okc-kexp-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"SGLANG_JIT_CACHE_DIR": f"{CACHE}/jit", "TORCH_EXTENSIONS_DIR": f"{CACHE}/torch",
          "OKC_KERNEL_TRACK": "/root/lane/muse_moe_w4a16", "MOE_TOOLS": "/root/lane/moe_tools",
          "KMOE_VARIANT_DIR": f"{CACHE}/variants",
          "KEXP_BASE_CUH": "/root/lane/base/muse_spark_fp4_grouped.cuh",
          "KEXP_LANE_CUH": f"/root/lane/kexp/{CUH}",
          "PYTHONPATH": "/root/lane/kexp/python"})
    .add_local_dir(os.path.join(SERVING, LANE, "python", "sglang"), "/root/lane/kexp/python/sglang")
    .add_local_file(os.path.join(SERVING, "lanes", "stack_k9", CUH), "/root/lane/base/muse_spark_fp4_grouped.cuh")
    .add_local_dir(os.path.join(SERVING, "lanes", "kexp_tools"), "/root/lane/kexp_tools")
    .add_local_file(os.path.join(SERVING, "lanes", "moe_tools", "microbench.py"), "/root/lane/moe_tools/microbench.py")
    .add_local_file(os.path.join(ROOT, "muse_moe_w4a16", "nvfp4.py"), "/root/lane/muse_moe_w4a16/nvfp4.py")
    .add_local_file(os.path.join(ROOT, "muse_moe_w4a16", "reference.py"), "/root/lane/muse_moe_w4a16/reference.py")
)
TOOLS = "/root/lane/kexp_tools"


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
    """CPU only: build every arm, print ptxas; save the JIT caches for the GPU run."""
    _untar()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kexp/{ts}_compile"
    os.makedirs(out, exist_ok=True)
    rc = _run([sys.executable, "kexp_bench.py", "--compile-only", *shlex.split(extra)], 35 * 60, f"{out}/compile.log")
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs("/vol/out/kexp", exist_ok=True)
    subprocess.run(["tar", "-czf", CACHE_TAR, "-C", CACHE, "."], check=False)
    vol.commit()
    print("cache saved to", CACHE_TAR, "rc", rc, flush=True)


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=12 * 60, cpu=8, memory=64 * 1024)
def bench(extra: str = "", limit_min: int = 10):
    """One GPU: arms of kexp_bench.py on the same prefill-size inputs."""
    _untar()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kexp/{ts}_bench"
    os.makedirs(out, exist_ok=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.max.sm,clocks.sm,power.limit", "--format=csv"], check=False)
    rc = _run([sys.executable, "kexp_bench.py", "--out", f"{out}/bench.json", *shlex.split(extra)], limit_min * 60,
              f"{out}/run.log")
    vol.commit()
    print("results in", out, flush=True)
    if os.path.exists(f"{out}/bench.json"):
        print("BENCH_JSON " + open(f"{out}/bench.json").read().replace("\n", " "), flush=True)
    if rc:
        raise RuntimeError(f"bench rc={rc}")
