"""Lane KPIPE gate on Modal: one vs two cold chunks in flight (lanes/kpipe_tools/kpipe_gate.py), judge image.

  modal run harness/modal_kpipe_gate.py --gpus 4     # H200:4, hard timeout 12 min (~$1.5 expected, ~$3.6 worst)
  modal run harness/modal_kpipe_gate.py --gpus 8     # H200:8, hard timeout 12 min (~$3 expected, ~$7.2 worst)
  modal run harness/modal_kpipe_gate.py --dry-run    # CPU only: threads as 2, 4 and 8 ranks, small shapes

Sections run as separate torchrun processes, most decisive first: pair (floor / ce1 / ce2 / ce2eq at post-fit
skew, bits), skew (pre-fit and no skew), nccl (nccl1 / nccl2, the L1 failure mode). RESULT json lines print to
stdout and go to out/kpipe_gate/<ts>_g<N>/results.txt on the volume, committed after each section.
Mounts lanes/kpipe/python/sglang first on PYTHONPATH (the KOVL pipelines it reuses live there).
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
SECTIONS = ("pair", "skew", "nccl")
SECTION_TIMEOUT_S = 200

app = modal.App("okc-kpipe-gate")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .add_local_dir(os.path.join(SERVING, "lanes", "kpipe", "python", "sglang"), "/root/lane/python/sglang")
    .add_local_file(os.path.join(SERVING, "lanes", "kpipe_tools", "kpipe_gate.py"), "/root/gate/kpipe_gate.py")
)


def _run(gpus: int):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kpipe_gate/{ts}_g{gpus}"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/root/lane/python", PYTHONUNBUFFERED="1")
    for section in SECTIONS:
        print(f"=== section {section} on {gpus} GPUs", flush=True)
        cmd = ["torchrun", "--nproc_per_node", str(gpus), "/root/gate/kpipe_gate.py", "--section", section]
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=SECTION_TIMEOUT_S)
            stdout, stderr, rc = r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr, rc = f"timeout {SECTION_TIMEOUT_S} s", -9
        lines = [l for l in stdout.splitlines() if l.startswith("RESULT")]
        with open(f"{out}/results.txt", "a") as f:
            for l in lines:
                print(l, flush=True)
                f.write(l + "\n")
            if rc != 0 or not lines:
                tail = (stderr or "")[-2500:]
                print(f"FAIL section={section} rc={rc}\n{tail}", flush=True)
                f.write(f"FAIL section={section} rc={rc}\n{tail}\n")
        vol.commit()
    print("results in", out, flush=True)


@app.function(image=image, gpu="H200:2", volumes={"/vol": vol}, timeout=12 * 60, cpu=8, memory=32 * 1024)
def gate2():
    _run(2)


@app.function(image=image, gpu="H200:4", volumes={"/vol": vol}, timeout=12 * 60, cpu=12, memory=48 * 1024)
def gate4():
    _run(4)


@app.function(image=image, gpu="H200:8", volumes={"/vol": vol}, timeout=12 * 60, cpu=16, memory=64 * 1024)
def gate8():
    _run(8)


@app.function(image=image, timeout=15 * 60, cpu=8, memory=16 * 1024)
def dry():
    env = dict(os.environ, PYTHONPATH="/root/lane/python", PYTHONUNBUFFERED="1")
    rc = 0
    for world in (2, 4, 8):
        print(f"=== dry run, {world} thread ranks", flush=True)
        r = subprocess.run([sys.executable, "/root/gate/kpipe_gate.py", "--dry", "--world", str(world)],
                           env=env, capture_output=True, text=True, timeout=400)
        print("\n".join(l for l in r.stdout.splitlines() if l.startswith("RESULT")), flush=True)
        if r.returncode:
            print(f"FAIL dry world={world} rc={r.returncode}\n{r.stderr[-3000:]}", flush=True)
            rc |= r.returncode
    if rc:
        raise RuntimeError(f"dry run failed rc={rc}")


@app.local_entrypoint()
def main(gpus: int = 4, dry_run: bool = False):
    if dry_run:
        dry.remote()
        return
    {2: gate2, 4: gate4, 8: gate8}[gpus].remote()
