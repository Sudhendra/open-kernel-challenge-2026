"""KLFE kill test: one MoE layer, today (D) vs local-first experts (L), real grouped NVFP4 kernel, 4 GPUs (Modal, judge image).

  modal run harness/modal_klfe_ubench.py::dry      # CPU only (free): the bench's bookkeeping, masks and checker
  modal run harness/modal_klfe_ubench.py::bench4   # 4 GPUs, function timeout 10 min (~$3.5 worst case)

Mounts lanes/<KLFE_TREE> (default klfe) python/sglang (SymmetricRowSlots, the grouped kernel, routed_ring_fold) and
lanes/klfe_tools/klfe_ubench.py. No sglang communicator is built (raw torch.distributed + symmetric memory, as the
KSPARSE bench), so no world-group wiring is needed. Every RESULT line is printed and appended to
/vol/out/klfe_ubench/<ts>/results.txt (committed) as soon as it appears.
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
TREE = os.environ.get("KLFE_TREE", "klfe")

app = modal.App("okc-klfe-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .add_local_dir(os.path.join(SERVING, "lanes", TREE, "python", "sglang"), "/root/lane/python/sglang")
    .add_local_file(os.path.join(SERVING, "lanes", "klfe_tools", "klfe_ubench.py"), "/root/gate/klfe_ubench.py")
)


def _stream(cmd, env, limit_s, out_dir):
    """Run cmd; echo every line, append RESULT lines to results.txt and commit them at once; kill at limit_s."""
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


@app.function(image=image, gpu="H200:4", volumes={"/vol": vol}, timeout=10 * 60, cpu=12, memory=64 * 1024)
def bench4():
    env = dict(os.environ, PYTHONPATH="/root/lane/python", PYTHONUNBUFFERED="1")
    out = "/vol/out/klfe_ubench/" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    _stream(["torchrun", "--nproc_per_node", "4", "/root/gate/klfe_ubench.py"], env, 540, out)


@app.function(image=image, volumes={"/vol": vol}, timeout=10 * 60, cpu=4, memory=16 * 1024)
def dry():
    env = dict(os.environ, PYTHONPATH="/root/lane/python", PYTHONUNBUFFERED="1")
    out = "/vol/out/klfe_ubench/" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_dry"
    rc = _stream([sys.executable, "/root/gate/klfe_ubench.py", "--dry"], env, 500, out)
    if rc:
        raise RuntimeError(f"dry rc={rc}")
