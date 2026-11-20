"""Lane KWP: first-use GEMM block test on ONE GPU (Modal), 12-min hard cap.

  modal run harness/modal_kwp_gpu.py::bench
Runs out/codex/warm_prefill/bench/lazy_bench.py three times in separate processes:
default, CUDA_MODULE_LOADING=EAGER, and default + prewarm. Prints one RESULT line per mode.
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"

app = modal.App("okc-kwp-gpu-bench")
image = modal.Image.from_registry(IMAGE_DIGEST, add_python=None).add_local_dir(
    os.path.join(SERVING, "out", "codex", "warm_prefill", "bench"), "/root/bench"
)


@app.function(image=image, gpu="H200", timeout=12 * 60, cpu=8, memory=64 * 1024)
def bench(modes: str = "default,eager,prewarm"):
    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], check=False)
    t0 = time.time()
    for mode in modes.split(","):
        env = dict(os.environ)
        env.pop("CUDA_MODULE_LOADING", None)
        if mode == "eager":
            env["CUDA_MODULE_LOADING"] = "EAGER"
        left = 11 * 60 - (time.time() - t0)
        if left < 20:
            print("skipped", mode, flush=True)
            continue
        try:
            r = subprocess.run([sys.executable, "/root/bench/lazy_bench.py", mode, f"/tmp/{mode}.json"], env=env,
                               timeout=left, capture_output=True, text=True)
            print(r.stdout[-3000:], r.stderr[-1500:], flush=True)
        except subprocess.TimeoutExpired:
            print("timeout", mode, flush=True)
