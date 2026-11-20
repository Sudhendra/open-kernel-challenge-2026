"""kpick_v2: 1-GPU bench of the three prefill fold kernels, old (lanes/kpick) vs new (lanes/kpick2), judge image, H200.

  modal run harness/modal_kpick2_bench.py::bench     # 1 GPU, 10-min cap (worst case ~$0.8)

Script: lanes/kwide2_tools/kpick2_fold_bench.py.
"""
import os
import subprocess
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
TOOLS = "/root/tools"
REL = os.path.join("python", "sglang", "srt", "layers", "muse_spark", "muse_spark_sp_overlap.py")
OLD = "/root/old_overlap.py"
NEW = "/root/new_overlap.py"

app = modal.App("okc-kpick2-bench")
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .add_local_file(os.path.join(SERVING, "lanes", "kwide2_tools", "kpick2_fold_bench.py"), TOOLS + "/kpick2_fold_bench.py")
    .add_local_file(os.path.join(SERVING, "lanes", "kpick", REL), OLD)
    .add_local_file(os.path.join(SERVING, "lanes", "kpick2", REL), NEW)
)


@app.function(image=image, gpu="H200", timeout=10 * 60, cpu=8, memory=48 * 1024)
def bench():
    rc = subprocess.run([sys.executable, "kpick2_fold_bench.py", OLD, NEW], cwd=TOOLS, timeout=9 * 60).returncode
    print("rc", rc, flush=True)
