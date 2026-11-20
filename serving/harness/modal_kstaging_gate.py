"""Lane KWATT staging removal: multicast pull all-reduce with / without staging copies, on 4 GPUs (Modal, judge image).

  modal run harness/modal_kstaging_gate.py        # 4 GPUs, hard timeout 12 min (~$4.2 worst case at ~$21/h)

Mounts lanes/kwatt/python/sglang and lanes/kwatt_tools/kstaging_bench.py; one torchrun of 4 ranks; RESULT lines on
stdout and in /vol/out/kstaging/<ts>/results.txt.
"""
import os
import subprocess
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"

app = modal.App("okc-kstaging-gate")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .add_local_dir(os.path.join(SERVING, "lanes", "kwatt", "python", "sglang"), "/root/lane/python/sglang")
    .add_local_file(os.path.join(SERVING, "lanes", "kwatt_tools", "kstaging_bench.py"), "/root/gate/kstaging_bench.py")
    .add_local_file(os.path.join(SERVING, "lanes", "kwatt_tools", "kstaging_bench2.py"), "/root/gate/kstaging_bench2.py")
)


@app.function(image=image, gpu="H200:4", volumes={"/vol": vol}, timeout=12 * 60, cpu=12, memory=48 * 1024)
def gate4(script: str = "kstaging_bench.py"):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kstaging/{ts}"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/root/lane/python", PYTHONUNBUFFERED="1")
    subprocess.run(["nvidia-smi", "topo", "-m"], check=False)
    cmd = ["torchrun", "--nproc_per_node", "4", f"/root/gate/{script}"]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=10 * 60)
        stdout, stderr, rc = r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr, rc = "timeout", -9
    lines = [l for l in stdout.splitlines() if l.startswith("RESULT")]
    with open(f"{out}/results.txt", "w") as f:
        for l in lines:
            print(l, flush=True)
            f.write(l + "\n")
        if rc != 0 or not lines:
            tail = (stderr or "")[-4000:]
            print(f"FAIL rc={rc}\n{tail}", flush=True)
            f.write(f"FAIL rc={rc}\n{tail}\n")
    vol.commit()
    print("results in", out, flush=True)
