"""Lane KPAGE: decode/verify FA3 (+ KATT split policy) and the prefill attention kernel at KV page sizes 1/16/64/128,
contiguous vs fragmented physical layouts, one GPU (Modal).

  modal run harness/modal_kpage_ubench.py::cpu_compile                 # CPU only (free): JIT-build the prefill
                                                                       # kernel, save the cache to the volume
  modal run harness/modal_kpage_ubench.py::bench                       # ONE GPU, hard 12-min cap (~$1 worst case)
  modal run harness/modal_kpage_ubench.py::bench --extra "--parts decode --batches 32"

Mounts lanes/kpage (worktree of lane-knomix), lanes/kpage_tools, lanes/kswa_tools. Results: /vol/out/kpage/<ts>/
and stdout (BENCH_JSON line).
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
CACHE_TAR = "/vol/out/kpage/cache.tgz"
TOOLS = "/root/lane/kpage_tools"

app = modal.App("okc-kpage-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"SGLANG_JIT_CACHE_DIR": f"{CACHE}/jit", "TORCH_EXTENSIONS_DIR": f"{CACHE}/torch", "MAX_JOBS": "16",
          "TORCH_CUDA_ARCH_LIST": "9.0a", "PYTHONPATH": "/root/lane/kpage/python", "KSWA_TOOLS": "/root/lane/kswa_tools"})
    .add_local_dir(os.path.join(SERVING, "lanes", "kpage", "python", "sglang"), "/root/lane/kpage/python/sglang")
    .add_local_dir(os.path.join(SERVING, "lanes", "kpage", "test"), "/root/lane/kpage/test")
    .add_local_dir(os.path.join(SERVING, "lanes", "kpage_tools"), TOOLS)
    .add_local_dir(os.path.join(SERVING, "lanes", "kswa_tools"), "/root/lane/kswa_tools")
)


def _run(cmd, limit, log=None, env=None, cwd=TOOLS):
    print("+", " ".join(cmd), flush=True)
    t0 = time.time()
    f = open(log, "w") if log else None
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
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


@app.function(image=image, volumes={"/vol": vol}, timeout=30 * 60, cpu=16, memory=48 * 1024)
def cpu_compile():
    """CPU only: build the prefill attention module (k9 pinned copy) and save the JIT cache."""
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs("/vol/out/kpage", exist_ok=True)
    env = dict(os.environ)
    os.makedirs("/tmp/cudastub", exist_ok=True)
    subprocess.run("ln -sf $(ls /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/targets/*/lib/stubs/libcuda.so"
                   " 2>/dev/null | head -1) /tmp/cudastub/libcuda.so.1", shell=True)
    env["LD_LIBRARY_PATH"] = "/tmp/cudastub:" + env.get("LD_LIBRARY_PATH", "")
    rc = _run([sys.executable, "kswa_bench.py", "--compile-only", "--arms", ""], 25 * 60, None, env,
              cwd="/root/lane/kswa_tools")
    subprocess.run(["tar", "-czf", CACHE_TAR, "-C", CACHE, "."], check=False)
    vol.commit()
    print("cache saved to", CACHE_TAR, "rc", rc, flush=True)
    return rc


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=12 * 60, cpu=8, memory=64 * 1024)
def bench(extra: str = ""):
    """One GPU, hard 12-minute cap: decode + prefill attention at page sizes / layouts."""
    if os.path.exists(CACHE_TAR):
        os.makedirs(CACHE, exist_ok=True)
        subprocess.run(["tar", "-xzf", CACHE_TAR, "-C", CACHE], check=False)
        print("cache restored", flush=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kpage/{ts}_bench"
    os.makedirs(out, exist_ok=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.max.sm,power.limit", "--format=csv"], check=False)
    rc = _run([sys.executable, "kpage_bench.py", "--out", f"{out}/bench.json", *extra.split()], 10 * 60,
              f"{out}/run.log")
    vol.commit()
    print("results in", out, flush=True)
    if os.path.exists(f"{out}/bench.json"):
        print("BENCH_JSON " + open(f"{out}/bench.json").read().replace("\n", ""), flush=True)
    return rc
