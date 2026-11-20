"""Lane KBRANCH: 1-GPU check of one tree-masked verify with the lane's own code (judge image, H200).

  modal run harness/modal_kbranch_gpucheck.py::check     # 1 GPU, 10-min cap (worst case ~$0.8)

Script: lanes/kwide2_tools/kbranch_gpucheck.py; lane python tree: lanes/kbranch/python.
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
TOOLS = "/root/lane/kwide2_tools"
LANE = "/root/kbranch_nw/python"

app = modal.App("okc-kbranch-fillcost")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .add_local_dir(os.path.join(SERVING, "lanes", "kwide2_tools"), TOOLS)
    .add_local_dir(os.path.join(SERVING, "lanes", "kbranch_nw", "python", "sglang"), LANE + "/sglang")
    .add_local_dir(os.path.join(SERVING, "lanes", "kbranch", "python", "sglang"), "/root/kbranch_cuda/python/sglang")
)


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=10 * 60, cpu=8, memory=48 * 1024)
def check(lane: str = LANE):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kwide2/{ts}_fill"
    os.makedirs(out, exist_ok=True)
    for path in (lane, "/root/kbranch_cuda/python"):
        print("lane", path, flush=True)
        subprocess.run([sys.executable, "kbranch_fillcost.py", path], cwd=TOOLS, timeout=4 * 60)
    rc = 0
    vol.commit()
    print("rc", rc, "results in", out, flush=True)
