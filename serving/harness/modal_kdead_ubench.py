"""Lane KDEAD GPU checks on ONE GPU inside the judge image (Modal), hard timeout 20 min.

  OKC_LANE=lanes/kdead modal run harness/modal_kdead_ubench.py::gpu_tests

Runs the lane's GPU-only unit tests: real breakable-graph capture/replay of a skipped
segment, and the FA3 K/V-only write against forward_extend's write. No timing.
Worst case ~$1.5 at the 20-min cap.
"""
import os
import subprocess
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
LANE = os.environ.get("OKC_LANE", "lanes/kdead")

app = modal.App("okc-kdead-ubench")
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .add_local_dir(os.path.join(SERVING, LANE, "python", "sglang"), "/root/lane/python/sglang")
    .add_local_dir(os.path.join(SERVING, LANE, "test"), "/root/lane/test")
)

FILES = [
    "test/registered/cuda_graph/breakable/test_run_segment_if.py",
    "test/registered/unit/layers/muse_spark/test_muse_spark_unread_last_layer.py",
]


@app.function(image=image, gpu="H200", timeout=20 * 60, cpu=4, memory=32 * 1024)
def gpu_tests(files: str = ""):
    sel = files.split(",") if files else FILES
    env = dict(os.environ, PYTHONPATH="/root/lane/python")
    subprocess.run([sys.executable, "-c", "import torch, sglang; print(torch.cuda.get_device_name(), sglang.__file__)"],
                   env=env, cwd="/root/lane", timeout=300)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", *sel],
                       env=env, cwd="/root/lane", timeout=15 * 60)
    print("gpu tests rc", r.returncode, flush=True)
    if r.returncode:
        raise RuntimeError(f"gpu tests failed rc={r.returncode}")
