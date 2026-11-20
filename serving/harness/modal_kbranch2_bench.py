"""Lane KBRANCH2: 1-GPU bench of branch v1 vs v2 (judge image, H200). Run only with budget.

  modal run harness/modal_kbranch2_bench.py::bench     # 1 GPU, 10-min cap (worst case ~$0.8)

Script: lanes/kwide2_tools/kbranch2_bench.py; lane: lanes/kbranch2 (python tree + the branch test file).
"""
import os
import subprocess
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
TOOLS = "/root/lane/kwide2_tools"
LANE = "/root/kbranch2/python"
TEST = "/root/kbranch2/test_dflash_branch.py"

app = modal.App("okc-kbranch2-bench")
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .add_local_dir(os.path.join(SERVING, "lanes", "kwide2_tools"), TOOLS)
    .add_local_dir(os.path.join(SERVING, "lanes", "kbranch2", "python", "sglang"), LANE + "/sglang")
    .add_local_file(os.path.join(SERVING, "lanes", "kbranch2", "test", "registered", "unit", "speculative",
                                 "test_dflash_branch.py"), TEST)
)


@app.function(image=image, gpu="H200", timeout=10 * 60, cpu=8, memory=48 * 1024)
def bench():
    rc = subprocess.run([sys.executable, "kbranch2_bench.py", LANE, TEST], cwd=TOOLS, timeout=9 * 60,
                        env=dict(os.environ, PYTHONPATH=LANE)).returncode
    print("rc", rc, flush=True)
