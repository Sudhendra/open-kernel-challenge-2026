"""KPUSH GPU proofs (judge image, own app): pair-push combine vs today's combine.

  modal run harness/modal_kpush_gpu.py::one     # 1 GPU, timeout 10 min: compile, fold exactness, row stores, e2e, kernel-alone
  modal run harness/modal_kpush_gpu.py::bench4  # 4 GPUs, timeout 10 min: 8-rank emulation layer bench D vs N + bits

Mounts lanes/<KPUSH_TREE> (default kpush) python/sglang and lanes/kpush_tools/kpush_gpu.py. RESULT lines are printed and
appended to /vol/out/kpush/<ts>/results.txt as they appear.
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
TREE = os.environ.get("KPUSH_TREE", "kpush")

app = modal.App("okc-kpush-gpu")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .add_local_dir(os.path.join(SERVING, "lanes", TREE, "python", "sglang"), "/root/lane/python/sglang")
    .add_local_file(os.path.join(SERVING, "lanes", "kpush_tools", "kpush_gpu.py"), "/root/gate/kpush_gpu.py")
)


def _stream(cmd, env, limit_s, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    t0, tail, rc = time.time(), [], None
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    with open(f"{out_dir}/results.txt", "a") as f:
        for line in p.stdout:
            print(line, end="", flush=True)
            tail = (tail + [line])[-80:]
            if line.startswith("RESULT"):
                f.write(line)
                f.flush()
                vol.commit()
            if time.time() - t0 > limit_s:
                p.kill()
                f.write(f"FAIL killed at {limit_s} s\n")
                break
        rc = p.wait()
        if rc != 0:
            f.write(f"FAIL rc={rc}\n" + "".join(tail[-40:]))
    vol.commit()
    print(f"rc={rc} after {time.time() - t0:.0f} s; results in {out_dir}", flush=True)
    return rc


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=10 * 60, cpu=8, memory=64 * 1024)
def one():
    env = dict(os.environ, PYTHONPATH="/root/lane/python", PYTHONUNBUFFERED="1")
    out = "/vol/out/kpush/" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_one"
    _stream([sys.executable, "/root/gate/kpush_gpu.py", "--one"], env, 540, out)


@app.function(image=image, gpu="H200:4", volumes={"/vol": vol}, timeout=10 * 60, cpu=12, memory=96 * 1024)
def bench4():
    env = dict(os.environ, PYTHONPATH="/root/lane/python", PYTHONUNBUFFERED="1")
    out = "/vol/out/kpush/" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_bench4"
    _stream(["torchrun", "--nproc_per_node", "4", "/root/gate/kpush_gpu.py"], env, 540, out)
