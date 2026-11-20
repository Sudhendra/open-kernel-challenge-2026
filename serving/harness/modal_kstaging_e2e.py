"""KWATT staging removal: end-to-end bit gate of the served attention path on 4 GPUs (Modal, judge image).

  modal run harness/modal_kstaging_e2e.py::dry       # CPU only (free): one-process mocked pass of the gate script
  modal run harness/modal_kstaging_e2e.py::gate4     # 4 GPUs, function timeout 5 min (~$1.75 worst case at ~$21/h)

Mounts lanes/<KSTAGING_TREE> (default stack_k22) python/sglang and lanes/kwatt_tools/kstaging_e2e.py. Every RESULT
line is printed and appended to /vol/out/kstaging_e2e/<ts>/results.txt (committed) as soon as it appears.
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
TREE = os.environ.get("KSTAGING_TREE", "stack_k22")

app = modal.App("okc-kstaging-e2e")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .add_local_dir(os.path.join(SERVING, "lanes", TREE, "python", "sglang"), "/root/lane/python/sglang")
    .add_local_file(os.path.join(SERVING, "lanes", "kwatt_tools", "kstaging_e2e.py"), "/root/gate/kstaging_e2e.py")
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


@app.function(image=image, gpu="H200:4", volumes={"/vol": vol}, timeout=5 * 60, cpu=12, memory=64 * 1024)
def gate4():
    env = dict(os.environ, PYTHONPATH="/root/lane/python", PYTHONUNBUFFERED="1")
    out = "/vol/out/kstaging_e2e/" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    _stream(["torchrun", "--nproc_per_node", "4", "/root/gate/kstaging_e2e.py"], env, 280, out)


@app.function(image=image, volumes={"/vol": vol}, timeout=15 * 60, cpu=8, memory=32 * 1024)
def dry():
    env = dict(os.environ, PYTHONPATH="/root/lane/python", PYTHONUNBUFFERED="1")
    os.makedirs("/tmp/cudastub", exist_ok=True)
    subprocess.run(
        "ln -sf $(ls /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/targets/*/lib/stubs/libcuda.so"
        " 2>/dev/null | head -1) /tmp/cudastub/libcuda.so.1",
        shell=True,
    )
    env["LD_LIBRARY_PATH"] = "/tmp/cudastub:" + os.environ.get("LD_LIBRARY_PATH", "")
    out = "/vol/out/kstaging_e2e/" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_dry"
    rc = _stream([sys.executable, "/root/gate/kstaging_e2e.py", "--dry"], env, 600, out)
    if rc:
        raise RuntimeError(f"dry rc={rc}")
