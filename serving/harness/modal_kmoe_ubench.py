"""Lane KMOE: grouped NVFP4 MoE ceiling / variant microbench on ONE GPU inside the judge image (Modal).

  modal run harness/modal_kmoe_ubench.py::cpu_compile                 # CPU only (free): build every arm, cache to volume
  modal run harness/modal_kmoe_ubench.py::ceiling                 # 1 GPU, 20-min cap (~$1.5 worst case)
  modal run harness/modal_kmoe_ubench.py::ceiling --extra "--variants a=/root/lane/kmoe_tools/variants/a.cuh"

Mounts lanes/kmoe (worktree of stack-o2pd: "tree" is the stack's kernel), lanes/kmoe_tools, lanes/moe_tools
(microbench helpers) and the kernel-track MoE files. The CPU job tars the sglang JIT cache and the torch extension
cache to the volume; the GPU job untars them first, so it compiles only what changed.
Results: /vol/out/kmoe/<ts>/ and stdout.
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
ROOT = os.path.dirname(SERVING)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
LANE = os.environ.get("KMOE_LANE", "lanes/kmoe")
CACHE = "/root/kcache"
CACHE_TAR = "/vol/out/kmoe/cache.tgz"

app = modal.App("okc-kmoe-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"SGLANG_JIT_CACHE_DIR": f"{CACHE}/jit", "TORCH_EXTENSIONS_DIR": f"{CACHE}/torch",
          "OKC_KERNEL_TRACK": "/root/lane/muse_moe_w4a16", "MOE_TOOLS": "/root/lane/moe_tools",
          "PYTHONPATH": "/root/lane/kmoe/python"})
    .add_local_dir(os.path.join(SERVING, LANE, "python", "sglang"), "/root/lane/kmoe/python/sglang")
    .add_local_dir(os.path.join(SERVING, "lanes", "kmoe_tools"), "/root/lane/kmoe_tools")
    .add_local_file(os.path.join(SERVING, "lanes", "moe_tools", "microbench.py"), "/root/lane/moe_tools/microbench.py")
    .add_local_file(os.path.join(ROOT, "muse_moe_w4a16", "nvfp4.py"), "/root/lane/muse_moe_w4a16/nvfp4.py")
    .add_local_file(os.path.join(ROOT, "muse_moe_w4a16", "reference.py"), "/root/lane/muse_moe_w4a16/reference.py")
    .add_local_file(os.path.join(ROOT, "muse_moe_w4a16", "checks.py"), "/root/lane/muse_moe_w4a16/checks.py")
    .add_local_file(os.path.join(ROOT, "muse_moe_w4a16", "submission.py"), "/root/lane/muse_moe_w4a16/submission.py")
)
TOOLS = "/root/lane/kmoe_tools"


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
    out = f"/vol/out/kmoe/{ts}_compile"
    os.makedirs(out, exist_ok=True)
    rc = _run([sys.executable, "kmoe_ceiling.py", "--compile-only", *extra.split()], 35 * 60, f"{out}/compile.log")
    os.makedirs(CACHE, exist_ok=True)
    subprocess.run(["tar", "-czf", CACHE_TAR, "-C", CACHE, "."], check=False)
    vol.commit()
    print("cache saved to", CACHE_TAR, "rc", rc, flush=True)


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=20 * 60, cpu=8, memory=64 * 1024)
def ceiling(extra: str = ""):
    """One GPU: the ceiling table (tree / marlin / kernel-track / PR #50800, plus any --variants)."""
    _untar()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kmoe/{ts}_ceiling"
    os.makedirs(out, exist_ok=True)
    rc = _run([sys.executable, "kmoe_ceiling.py", "--out", f"{out}/ceiling.json", *extra.split()], 17 * 60, f"{out}/run.log")
    vol.commit()
    print("results in", out, flush=True)
    if os.path.exists(f"{out}/ceiling.json"):
        print(open(f"{out}/ceiling.json").read(), flush=True)
    if rc:
        raise RuntimeError(f"ceiling rc={rc}")
