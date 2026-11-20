"""Lane KPIPE2 gate on Modal: stage-major two/three flights on one compute stream (lanes/kpipe2_tools/kpipe2_gate.py).

  modal run harness/modal_kpipe2_gate.py              # 4 GPUs, hard function timeout 15 min (worst ~$4.5)
  modal run harness/modal_kpipe2_gate.py --dry-run    # CPU only: threads as 2, 4 and 8 ranks, small shapes

4 GPUs only (no 8-GPU function on purpose). Sections run as separate torchrun processes, most decisive first:
post (bits + all arms at post-fit skew), vol (x7/3 and x7/6 copy volume emulation), pre (pre-fit skew + KPIPE
legacy-timing reproduction). RESULT json lines print to stdout and go to out/kpipe2_gate/<ts>_g4/results.txt on the
volume, committed after each section. Mounts lanes/kpipe/python/sglang (the KOVL pipelines the gate reuses).
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
SECTIONS = ("post", "vol", "pre")
SECTION_TIMEOUT_S = 250

app = modal.App("okc-kpipe2-gate")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .add_local_dir(os.path.join(SERVING, "lanes", "kpipe", "python", "sglang"), "/root/lane/python/sglang")
    .add_local_file(os.path.join(SERVING, "lanes", "kpipe2_tools", "kpipe2_gate.py"), "/root/gate/kpipe2_gate.py")
)


def _run(gpus: int, sections):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kpipe2_gate/{ts}_g{gpus}"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/root/lane/python", PYTHONUNBUFFERED="1")
    t_start = time.time()
    for section in sections:
        print(f"=== section {section} on {gpus} GPUs (t+{time.time() - t_start:.0f} s)", flush=True)
        cmd = ["torchrun", "--nproc_per_node", str(gpus), "/root/gate/kpipe2_gate.py", "--section", section]
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
    print(f"results in {out} (t+{time.time() - t_start:.0f} s)", flush=True)


@app.function(image=image, gpu="H200:4", volumes={"/vol": vol}, timeout=15 * 60, cpu=12, memory=48 * 1024)
def gate4(sections: str = ",".join(SECTIONS)):
    _run(4, [s for s in sections.split(",") if s])


@app.function(image=image, timeout=15 * 60, cpu=8, memory=16 * 1024)
def dry():
    env = dict(os.environ, PYTHONPATH="/root/lane/python", PYTHONUNBUFFERED="1")
    rc = 0
    for world in (2, 4, 8):
        print(f"=== dry run, {world} thread ranks", flush=True)
        r = subprocess.run([sys.executable, "/root/gate/kpipe2_gate.py", "--dry", "--world", str(world)],
                           env=env, capture_output=True, text=True, timeout=400)
        print("\n".join(l for l in r.stdout.splitlines() if l.startswith("RESULT")), flush=True)
        if r.returncode:
            print(f"FAIL dry world={world} rc={r.returncode}\n{r.stderr[-3000:]}", flush=True)
            rc |= r.returncode
    if rc:
        raise RuntimeError(f"dry run failed rc={rc}")


@app.local_entrypoint()
def main(dry_run: bool = False, sections: str = ",".join(SECTIONS)):
    if dry_run:
        dry.remote()
        return
    gate4.remote(sections)
