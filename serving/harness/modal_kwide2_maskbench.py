"""Lane KWIDE2 / branch: 1-GPU microbench of an exact tree mask in the FA3 verify path (judge image, H200).

  modal run harness/modal_kwide2_maskbench.py::bench     # 1 GPU, 10-min cap (worst case ~$0.8)

Script: lanes/kwide2_tools/kwide2_maskbench.py. Results: /vol/out/kwide2/<ts>_mask/ and stdout.
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

app = modal.App("okc-kwide2-maskbench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = modal.Image.from_registry(IMAGE_DIGEST, add_python=None).add_local_dir(
    os.path.join(SERVING, "lanes", "kwide2_tools"), TOOLS
)


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=10 * 60, cpu=4, memory=32 * 1024)
def bench():
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kwide2/{ts}_mask"
    os.makedirs(out, exist_ok=True)
    rc = subprocess.run([sys.executable, "kwide2_maskbench.py", f"{out}/mask.json"], cwd=TOOLS,
                        timeout=8 * 60).returncode
    vol.commit()
    print("rc", rc, "results in", out, flush=True)
