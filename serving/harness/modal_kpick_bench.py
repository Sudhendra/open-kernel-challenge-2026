"""Lane KPICK: 1-GPU bench of the pick_branch Triton kernel vs the torch body (judge image, H200).

  modal run harness/modal_kpick_bench.py::bench     # 1 GPU, 10-min cap (worst case ~$0.8)

Script: lanes/kwide2_tools/kpick_bench.py; lane: lanes/kpick (python tree + the branch test file); the torch body
comes from lanes/kbranch2w (509bbeeb8e) dflash_branch.py.
"""
import os
import subprocess
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
TOOLS = "/root/lane/kwide2_tools"
LANE = "/root/kpick/python"
TEST = "/root/kpick/test_dflash_branch.py"
OLD = "/root/kpick/old_dflash_branch.py"

app = modal.App("okc-kpick-bench")
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .add_local_dir(os.path.join(SERVING, "lanes", "kwide2_tools"), TOOLS)
    .add_local_dir(os.path.join(SERVING, "lanes", "kpick", "python", "sglang"), LANE + "/sglang")
    .add_local_file(os.path.join(SERVING, "lanes", "kpick", "test", "registered", "unit", "speculative",
                                 "test_dflash_branch.py"), TEST)
    .add_local_file(os.path.join(SERVING, "lanes", "kbranch2w", "python", "sglang", "srt", "speculative",
                                 "dflash_branch.py"), OLD)
)


@app.function(image=image, gpu="H200", timeout=10 * 60, cpu=8, memory=48 * 1024)
def bench():
    rc = subprocess.run([sys.executable, "kpick_bench.py", LANE, TEST, OLD], cwd=TOOLS, timeout=9 * 60,
                        env=dict(os.environ, PYTHONPATH=LANE)).returncode
    print("rc", rc, flush=True)
