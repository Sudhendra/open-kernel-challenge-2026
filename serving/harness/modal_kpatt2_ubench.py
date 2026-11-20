"""Lane KPATT2: Muse FP8 prefill attention JIT kernel vs FA3 + exp2 accuracy scan, ONE GPU in the judge image (Modal).

  modal run harness/modal_kpatt2_ubench.py::cpu_compile                 # CPU only (free): build every variant, cache
  modal run harness/modal_kpatt2_ubench.py::bench                       # 1 GPU, 20-min cap (~$1.5 worst case)
  modal run harness/modal_kpatt2_ubench.py::bench --extra "--shapes full64k --variants s4p0h24"
  modal run harness/modal_kpatt2_ubench.py::tests --files test/registered/...   # 1 GPU pytest of the lane tree

Mounts lanes/kpatt2 (worktree of stack-o2pd) and lanes/kpatt2_tools. The CPU job tars the sglang JIT cache and the
torch extension cache to the volume; the GPU job untars them first. Results: /vol/out/kpatt2/<ts>/ and stdout.
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
CACHE = "/root/kcache"
CACHE_TAR = "/vol/out/kpatt2/cache.tgz"

app = modal.App("okc-kpatt2-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"SGLANG_JIT_CACHE_DIR": f"{CACHE}/jit", "TORCH_EXTENSIONS_DIR": f"{CACHE}/torch", "MAX_JOBS": "16",
          "TORCH_CUDA_ARCH_LIST": "9.0a", "PYTHONPATH": "/root/lane/kpatt2/python", "KPATT2_PY": "/root/lane/kpatt2/python"})
    .add_local_dir(os.path.join(SERVING, "lanes", "kpatt2", "python", "sglang"), "/root/lane/kpatt2/python/sglang")
    .add_local_dir(os.path.join(SERVING, "lanes", "kpatt2", "test"), "/root/lane/kpatt2/test")
    .add_local_dir(os.path.join(SERVING, "lanes", "kpatt2_tools"), "/root/lane/kpatt2_tools")
)
TOOLS = "/root/lane/kpatt2_tools"


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
    """CPU only: build every kernel variant and the exp scan module; ptxas usage and SASS counts; save the caches."""
    _untar()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kpatt2/{ts}_compile"
    os.makedirs(out, exist_ok=True)
    rc = _run([sys.executable, "kpatt2_bench.py", "--compile-only", *extra.split()], 36 * 60, f"{out}/compile.log")
    os.makedirs(CACHE, exist_ok=True)
    subprocess.run(["tar", "-czf", CACHE_TAR, "-C", CACHE, "."], check=False)
    vol.commit()
    print("cache saved to", CACHE_TAR, "rc", rc, flush=True)


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=20 * 60, cpu=8, memory=64 * 1024)
def bench(extra: str = "", tests: str = ""):
    """One GPU: exp2 accuracy scan, FA3 vs every kernel variant at the serving prefill shapes, then optional pytest."""
    _untar()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kpatt2/{ts}_bench"
    os.makedirs(out, exist_ok=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.max.sm,power.limit", "--format=csv"], check=False)
    rc = _run([sys.executable, "kpatt2_bench.py", "--out", f"{out}/bench.json", *extra.split()], 11 * 60, f"{out}/run.log")
    if tests:
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *[f"/root/lane/kpatt2/{f}" for f in tests.split(",")]]
        rc |= _run(cmd, 6 * 60, f"{out}/pytest.log")
    vol.commit()
    print("results in", out, flush=True)
    if os.path.exists(f"{out}/bench.json"):
        print(open(f"{out}/bench.json").read(), flush=True)
    if rc:
        raise RuntimeError(f"bench rc={rc}")


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=20 * 60, cpu=8, memory=64 * 1024)
def tests(files: str):
    """One GPU: pytest the given lane test files (comma list, relative to the lane root)."""
    _untar()
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *[f"/root/lane/kpatt2/{f}" for f in files.split(",")]]
    rc = _run(cmd, 17 * 60)
    if rc:
        raise RuntimeError(f"tests rc={rc}")
