"""CPU-only micro-timings for lane KWP (warm-prefill host path) in the judge image.

  OKC_LANE=lanes/kwp modal run harness/modal_kwp_cpu.py::bench --script match_bench.py
Scripts live in out/codex/warm_prefill/bench/ and take the lane python root as argv[1].
"""
import os
import subprocess
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = os.environ.get(
    "OKC_IMAGE", "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
)
LANE = os.environ.get("OKC_LANE", "lanes/kwp")

app = modal.App("okc-kwp-cpu-bench")
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .add_local_dir(os.path.join(SERVING, LANE, "python", "sglang"), "/root/lane/python/sglang")
    .add_local_dir(os.path.join(SERVING, "out", "codex", "warm_prefill", "bench"), "/root/bench")
)


@app.function(image=image, timeout=1200, cpu=8, memory=32 * 1024)
def bench(script: str = "match_bench.py"):
    env = dict(os.environ, PYTHONPATH="/root/lane/python")
    os.makedirs("/tmp/cudastub", exist_ok=True)
    subprocess.run(
        "ln -sf $(ls /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/targets/*/lib/stubs/libcuda.so"
        " 2>/dev/null | head -1) /tmp/cudastub/libcuda.so.1",
        shell=True,
    )
    env["LD_LIBRARY_PATH"] = "/tmp/cudastub:" + os.environ.get("LD_LIBRARY_PATH", "")
    r = subprocess.run([sys.executable, f"/root/bench/{script}", "/root/lane/python"], env=env, cwd="/root")
    print("bench rc", r.returncode, flush=True)
    return r.returncode
