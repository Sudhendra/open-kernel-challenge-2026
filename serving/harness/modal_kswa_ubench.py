"""Lane KSWA: sliding-window / full FP8 prefill attention, stack-k9 kernel vs exact scheduling variants (Modal).

  modal run harness/modal_kswa_ubench.py::cpu_compile                    # CPU only (free): build every arm, cache
  modal run harness/modal_kswa_ubench.py::bench --extra "--arms t0,pswm"  # ONE GPU, hard 12-min cap (~$1 worst case)
  modal run harness/modal_kswa_ubench.py::bench --tests test/registered/...  # + 1-GPU pytest of the lane tree

Mounts lanes/kswa (worktree of stack-k9) and lanes/kswa_tools. The CPU job tars the JIT cache to the volume; the GPU
job untars it first, so the GPU minutes go to measurement. Results: /vol/out/kswa/<ts>/ and stdout.
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
CACHE = "/root/kcache"
CACHE_TAR = "/vol/out/kswa/cache.tgz"

app = modal.App("okc-kswa-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"SGLANG_JIT_CACHE_DIR": f"{CACHE}/jit", "TORCH_EXTENSIONS_DIR": f"{CACHE}/torch", "MAX_JOBS": "16",
          "TORCH_CUDA_ARCH_LIST": "9.0a", "PYTHONPATH": "/root/lane/kswa/python", "KSWA_PY": "/root/lane/kswa/python"})
    .add_local_dir(os.path.join(SERVING, "lanes", "kswa", "python", "sglang"), "/root/lane/kswa/python/sglang")
    .add_local_dir(os.path.join(SERVING, "lanes", "kswa", "test"), "/root/lane/kswa/test")
    .add_local_dir(os.path.join(SERVING, "lanes", "kswa_tools"), "/root/lane/kswa_tools")
)
TOOLS = "/root/lane/kswa_tools"


def _run(cmd, limit, log=None, env=None):
    print("+", " ".join(cmd), flush=True)
    t0 = time.time()
    f = open(log, "w") if log else None
    p = subprocess.Popen(cmd, cwd=TOOLS, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
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
    """CPU only: build every arm module (ptxas usage, SASS counts) and save the JIT cache."""
    os.makedirs(CACHE, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kswa/{ts}_compile"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ)
    os.makedirs("/tmp/cudastub", exist_ok=True)
    subprocess.run("ln -sf $(ls /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/targets/*/lib/stubs/libcuda.so"
                   " 2>/dev/null | head -1) /tmp/cudastub/libcuda.so.1", shell=True)
    env["LD_LIBRARY_PATH"] = "/tmp/cudastub:" + env.get("LD_LIBRARY_PATH", "")
    env["KSWA_SASS_DIR"] = f"{out}/sass"
    rc = _run([sys.executable, "kswa_bench.py", "--compile-only", *extra.split()], 36 * 60, f"{out}/compile.log", env)
    subprocess.run(["tar", "-czf", CACHE_TAR, "-C", CACHE, "."], check=False)
    vol.commit()
    print("cache saved to", CACHE_TAR, "rc", rc, flush=True)
    return rc


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=12 * 60, cpu=8, memory=64 * 1024)
def bench(extra: str = "", tests: str = ""):
    """One GPU, hard 12-minute cap: bit gate vs stack-k9, timing at production shapes, profile; optional pytest."""
    _untar()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kswa/{ts}_bench"
    os.makedirs(out, exist_ok=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.max.sm,power.limit", "--format=csv"], check=False)
    budget = 9 * 60 if tests else 10 * 60
    rc = _run([sys.executable, "kswa_bench.py", "--out", f"{out}/bench.json", *extra.split()], budget, f"{out}/run.log")
    if tests:
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *[f"/root/lane/kswa/{f}" for f in tests.split(",")]]
        rc |= _run(cmd, 90, f"{out}/pytest.log")
    vol.commit()
    print("results in", out, flush=True)
    if os.path.exists(f"{out}/bench.json"):
        print("BENCH_JSON " + open(f"{out}/bench.json").read().replace("\n", ""), flush=True)
    return rc
