"""Lane KPATT: prefill (EXTEND) attention microbench on ONE GPU inside the judge image (Modal).

  modal run harness/modal_kpatt_ubench.py::cpu_compile      # CPU only (free): build both kernel modules, cache to volume
  modal run harness/modal_kpatt_ubench.py::bench            # 1 GPU, 20-min function cap (~$1.5 worst case)
  modal run harness/modal_kpatt_ubench.py::bench --extra "--shapes full32k,swa16k --reps 300"

Arms and method: lanes/kpatt_tools/kpatt_bench.py. FA3 comes from the image's sgl_kernel (the serving binary); the
candidate kernels come from the kernel-track prefill file (../muse_prefill_layer/submission_kl_1898.py), built with
torch load_inline for sm_90a. The CPU job tars the torch extension cache to the volume; the GPU job untars it first.
Results: /vol/out/kpatt/<ts>/ and stdout.
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
CACHE_TAR = "/vol/out/kpatt/cache.tgz"
TOOLS = "/root/lane/kpatt_tools"

app = modal.App("okc-kpatt-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"TORCH_EXTENSIONS_DIR": f"{CACHE}/torch", "TORCH_CUDA_ARCH_LIST": "9.0a",
          "KPATT_KL": "/root/lane/muse_prefill_layer/submission_kl_1898.py", "MAX_JOBS": "16"})
    .add_local_dir(os.path.join(SERVING, "lanes", "kpatt_tools"), TOOLS)
    .add_local_file(os.path.join(ROOT, "muse_prefill_layer", "submission_kl_1898.py"),
                    "/root/lane/muse_prefill_layer/submission_kl_1898.py")
)


def _run(cmd, limit):
    print("+", " ".join(cmd), flush=True)
    t0 = time.time()
    try:
        rc = subprocess.run(cmd, cwd=TOOLS, timeout=limit).returncode
    except subprocess.TimeoutExpired:
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
    """CPU only: build the raw and the e43 kernel modules; save the extension cache."""
    _untar()
    rc = _run([sys.executable, "kpatt_bench.py", "--compile-only", *extra.split()], 38 * 60)
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(os.path.dirname(CACHE_TAR), exist_ok=True)
    subprocess.run(["tar", "-czf", CACHE_TAR, "-C", CACHE, "."], check=False)
    vol.commit()
    print("cache saved to", CACHE_TAR, "rc", rc, flush=True)


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=20 * 60, cpu=8, memory=64 * 1024)
def bench(extra: str = ""):
    """One GPU: FA3 (serving call + tunables) vs the kernel-track candidates at the serving prefill shapes."""
    _untar()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kpatt/{ts}_bench"
    os.makedirs(out, exist_ok=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.max.sm,power.limit", "--format=csv"], check=False)
    rc = _run([sys.executable, "kpatt_bench.py", "--out", f"{out}/bench.json", *extra.split()], 17 * 60)
    vol.commit()
    print("results in", out, flush=True)
    if os.path.exists(f"{out}/bench.json"):
        print(open(f"{out}/bench.json").read(), flush=True)
    if rc:
        raise RuntimeError(f"bench rc={rc}")
