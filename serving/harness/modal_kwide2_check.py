"""Lane KWIDE2: 1-GPU attention check of the ragged wide verify layout in the judge image (Modal, H200).

  modal run harness/modal_kwide2_check.py::check     # 1 GPU, 10-min function cap (worst case ~$0.8)

Script: lanes/kwide2_tools/kwide2_check.py; the lane's lookup helpers are mounted from lanes/kwide2.
Results: /vol/out/kwide2/<ts>/check.json and stdout.
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
LOOKUP = "/root/lane/dflash_prompt_lookup.py"

app = modal.App("okc-kwide2-check")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .add_local_dir(os.path.join(SERVING, "lanes", "kwide2_tools"), TOOLS)
    .add_local_file(os.path.join(SERVING, "lanes", "kwide2", "python", "sglang", "srt", "speculative",
                                 "dflash_prompt_lookup.py"), LOOKUP)
)


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=10 * 60, cpu=4, memory=32 * 1024)
def check():
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kwide2/{ts}"
    os.makedirs(out, exist_ok=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv"], check=False)
    rc = subprocess.run([sys.executable, "kwide2_check.py", LOOKUP, f"{out}/check.json"], cwd=TOOLS,
                        timeout=8 * 60).returncode
    vol.commit()
    print("rc", rc, "results in", out, flush=True)
