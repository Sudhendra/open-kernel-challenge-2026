"""Lane KWATT: short queries over a long cached prefix (warm mixed prefill attention), ONE GPU in the judge image.

  modal run harness/modal_kwatt_ubench.py::cpu_compile        # CPU only (free): build both kernel modules, cache
  modal run harness/modal_kwatt_ubench.py::bench              # 1 GPU, 12-min cap (~$1.05 worst case)
  modal run harness/modal_kwatt_ubench.py::bench --extra "--shapes q521_h75k --reps 50"

Mounts lanes/<KWATT_TREE> python (default stack_k14: today's kernel + JIT utils) and lanes/kwatt_tools (the split
variant and the bench). The CPU job tars the JIT cache to the volume; the GPU job untars it. Results: /vol/out/kwatt/.
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
CACHE_TAR = "/vol/out/kwatt/cache.tgz"
TREE = os.environ.get("KWATT_TREE", "stack_k14")

app = modal.App("okc-kwatt-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"SGLANG_JIT_CACHE_DIR": f"{CACHE}/jit", "TORCH_EXTENSIONS_DIR": f"{CACHE}/torch", "MAX_JOBS": "16",
          "TORCH_CUDA_ARCH_LIST": "9.0a", "PYTHONPATH": "/root/lane/tree/python"})
    .add_local_dir(os.path.join(SERVING, "lanes", TREE, "python", "sglang"), "/root/lane/tree/python/sglang")
    .add_local_dir(os.path.join(SERVING, "lanes", TREE, "test"), "/root/lane/tree/test")
    .add_local_dir(os.path.join(SERVING, "lanes", "kwatt_tools"), "/root/lane/kwatt_tools")
)
TOOLS = "/root/lane/kwatt_tools"


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



@app.function(image=image, volumes={"/vol": vol}, timeout=25 * 60, cpu=16, memory=48 * 1024)
def cpu_compile(extra: str = ""):
    """CPU only: build today's kernel module and the split variant; save the JIT cache."""
    _untar()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kwatt/{ts}_compile"
    os.makedirs(out, exist_ok=True)
    rc = _run([sys.executable, "kwatt_bench.py", "--compile-only", *extra.split()], 22 * 60, f"{out}/compile.log")
    os.makedirs(CACHE, exist_ok=True)
    subprocess.run(["tar", "-czf", CACHE_TAR, "-C", CACHE, "."], check=False)
    vol.commit()
    print("cache saved to", CACHE_TAR, "rc", rc, flush=True)


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=12 * 60, cpu=8, memory=64 * 1024)
def bench(extra: str = "", tests: str = ""):
    """One GPU: every shape x arm (accuracy vs FP64, timing), then optional pytest of the tree; hard 12-minute cap."""
    _untar()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kwatt/{ts}_bench"
    os.makedirs(out, exist_ok=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.max.sm,power.limit", "--format=csv"], check=False)
    rc = _run([sys.executable, "kwatt_bench.py", "--out", f"{out}/bench.json", *extra.split()], 7 * 60, f"{out}/run.log")
    if tests:
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *[f"/root/lane/tree/{f}" for f in tests.split(",")]]
        rc |= _run(cmd, 3 * 60, f"{out}/pytest.log")
    vol.commit()
    print("results in", out, flush=True)
    if os.path.exists(f"{out}/bench.json"):
        print("BENCH_JSON " + open(f"{out}/bench.json").read().replace("\n", " "), flush=True)
    if rc:
        raise RuntimeError(f"bench rc={rc}")
