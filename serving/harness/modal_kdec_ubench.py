"""Lane KDEC: decode main loop of the grouped NVFP4 MoE vs stack-k9's kernel and the kernel-track file, ONE GPU (Modal).

  modal run harness/modal_kdec_ubench.py::cpu_compile --extra "--arms base,tree,dec,kt"   # CPU only (free), caches
  modal run harness/modal_kdec_ubench.py::bench --extra "--arms base,tree,dec,kt"          # 1 GPU, 12-min hard cap

Mounts lanes/kdec (the lane's tree), stack-k9's .cuh (lanes/knomix, byte-identical) as the "base" arm, lanes/kdec_tools,
the KMOE / KMOE2 / KEXP bench helpers, lanes/moe_tools/microbench.py and the kernel-track files.
Results: /vol/out/kdec/<ts>_bench/ and stdout (BENCH_JSON line). Every bench step runs in a subprocess so a hang in one
step (watchdog: os._exit after 20 s) keeps the earlier steps' JSON.
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
LANE = os.environ.get("KDEC_LANE", "lanes/kdec")
CACHE = "/root/kcache"
CACHE_TAR = "/vol/out/kdec/cache.tgz"
CUH = os.path.join("python", "sglang", "kernels", "jit", "csrc", "moe", "muse_spark_fp4_grouped.cuh")

app = modal.App("okc-kdec-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"SGLANG_JIT_CACHE_DIR": f"{CACHE}/jit", "TORCH_EXTENSIONS_DIR": f"{CACHE}/torch",
          "OKC_KERNEL_TRACK": "/root/lane/muse_moe_w4a16", "MOE_TOOLS": "/root/lane/moe_tools",
          "KMOE_TOOLS": "/root/lane/kmoe_tools", "KEXP_TOOLS": "/root/lane/kexp_tools",
          "KMOE2_SPIN": "/root/lane/kmoe2_tools/pdl_spin.cuh", "KMOE_VARIANT_DIR": f"{CACHE}/variants",
          "KDEC_BASE_CUH": "/root/lane/base/muse_spark_fp4_grouped.cuh",
          "KDEC_LANE_CUH": f"/root/lane/kdec/{CUH}",
          "PYTHONPATH": "/root/lane/kdec/python"})
    .add_local_dir(os.path.join(SERVING, LANE, "python", "sglang"), "/root/lane/kdec/python/sglang")
    .add_local_file(os.path.join(SERVING, "lanes", "knomix", CUH), "/root/lane/base/muse_spark_fp4_grouped.cuh")
    .add_local_dir(os.path.join(SERVING, "lanes", "kdec_tools"), "/root/lane/kdec_tools",
                   ignore=["results", "__pycache__"])
    .add_local_file(os.path.join(SERVING, "lanes", "kmoe_tools", "kmoe_ceiling.py"), "/root/lane/kmoe_tools/kmoe_ceiling.py")
    .add_local_file(os.path.join(SERVING, "lanes", "kmoe2_tools", "pdl_spin.cuh"), "/root/lane/kmoe2_tools/pdl_spin.cuh")
    .add_local_file(os.path.join(SERVING, "lanes", "kexp_tools", "variants.py"), "/root/lane/kexp_tools/variants.py")
    .add_local_file(os.path.join(SERVING, "lanes", "kexp_tools", "kexp_bench.py"), "/root/lane/kexp_tools/kexp_bench.py")
    .add_local_file(os.path.join(SERVING, "lanes", "moe_tools", "microbench.py"), "/root/lane/moe_tools/microbench.py")
    .add_local_file(os.path.join(ROOT, "muse_moe_w4a16", "nvfp4.py"), "/root/lane/muse_moe_w4a16/nvfp4.py")
    .add_local_file(os.path.join(ROOT, "muse_moe_w4a16", "reference.py"), "/root/lane/muse_moe_w4a16/reference.py")
    .add_local_file(os.path.join(ROOT, "muse_moe_w4a16", "checks.py"), "/root/lane/muse_moe_w4a16/checks.py")
    .add_local_file(os.path.join(ROOT, "muse_moe_w4a16", "submission.py"), "/root/lane/muse_moe_w4a16/submission.py")
)
TOOLS = "/root/lane/kdec_tools"


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
    """CPU only: build every arm (ptxas -v output), save the JIT / torch-extension caches for the GPU run."""
    _untar()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kdec/{ts}_compile"
    os.makedirs(out, exist_ok=True)
    rc = _run([sys.executable, "kdec_bench.py", "--compile-only", *shlex.split(extra)], 35 * 60, f"{out}/compile.log")
    os.makedirs(CACHE, exist_ok=True)
    subprocess.run(["tar", "-czf", CACHE_TAR, "-C", CACHE, "."], check=False)
    vol.commit()
    print("cache saved to", CACHE_TAR, "rc", rc, flush=True)


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=12 * 60, cpu=8, memory=64 * 1024)
def bench(extra: str = "", extra2: str = "", limit_min: int = 10):
    """One GPU, hard cap 12 min: `extra` runs first, then `extra2` (if given) in a second process."""
    _untar()
    subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.max.sm,power.limit", "--format=csv"], check=False)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kdec/{ts}_bench"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    rcs = []
    for i, ex in enumerate([e for e in (extra, extra2) if e]):
        left = limit_min * 60 - (time.time() - t0)
        if left < 30:
            print(f"step {i} skipped: {left:.0f}s left", flush=True)
            break
        rcs.append(_run([sys.executable, "kdec_bench.py", "--out", f"{out}/bench{i}.json", *shlex.split(ex)], left,
                        f"{out}/run{i}.log"))
        vol.commit()
        if os.path.exists(f"{out}/bench{i}.json"):
            print(f"BENCH_JSON{i} " + open(f"{out}/bench{i}.json").read().replace("\n", " "), flush=True)
    print("results in", out, "rcs", rcs, flush=True)
