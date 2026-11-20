"""Lanes KOVL + KOVL2 gate on Modal: copy-engine / symmetric-memory pipelines vs the NCCL path, in the judge image.

  modal run harness/modal_kovl_gate.py --gpus 4     # H200:4, hard timeout 15 min (~$2-3 expected, ~$4.5 worst)
  modal run harness/modal_kovl_gate.py --gpus 8     # H200:8, hard timeout 15 min (~$4-6 expected, ~$9 worst)
  modal run harness/modal_kovl_gate.py --gpus 2     # H200:2
  modal run harness/modal_kovl_gate.py --gpus 4 --sections core,graph,ce        # KOVL's attention sections only
  modal run harness/modal_kovl_gate.py --gpus 4 --route-frac 0.8                # busier straggler rank for rs2
  modal run harness/modal_kovl_gate.py --dry-run    # CPU only: threads as 2, 4 and 8 ranks, small shapes, no GPU billed

Default sections, most decisive first, each a separate torchrun process:
  rs2 (KOVL2 routed expert scatter: bits vs NCCL ring RS, SM store rate, straggler), ag2 (KOVL2 expert-row gather:
  bits per block count, timings), graph2 (capture/replay of both), then KOVL's core, graph, ce (attention pipelines).
RESULT json lines print to stdout and go to out/kovl_gate/<ts>_g<N>/results.txt on the volume, committed after each
section. Mounts lanes/kovl2/python/sglang (a superset of lanes/kovl: KOVL's code is unchanged there).
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
DEFAULT_SECTIONS = "rs2,ag2,graph2,core,graph,ce"
KOVL2_SECTIONS = {"rs2", "ag2", "graph2"}

app = modal.App("okc-kovl-gate")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .add_local_dir(os.path.join(SERVING, "lanes", "kovl2", "python", "sglang"), "/root/lane/python/sglang")
    .add_local_file(os.path.join(SERVING, "lanes", "kovl_tools", "kovl_gate.py"), "/root/gate/kovl_gate.py")
    .add_local_file(os.path.join(SERVING, "lanes", "kovl2_tools", "kovl2_gate.py"), "/root/gate/kovl2_gate.py")
)


def _script(section: str) -> str:
    return "/root/gate/kovl2_gate.py" if section in KOVL2_SECTIONS else "/root/gate/kovl_gate.py"


def _run(gpus: int, sections: str, extra: list):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kovl_gate/{ts}_g{gpus}"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/root/lane/python", PYTHONUNBUFFERED="1")
    for section in sections.split(","):
        print(f"=== section {section} on {gpus} GPUs", flush=True)
        cmd = ["torchrun", "--nproc_per_node", str(gpus), _script(section), "--section", section]
        if section in KOVL2_SECTIONS:
            cmd += extra
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=240)
            stdout, stderr, rc = r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired as e:
            stdout, stderr, rc = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), "timeout 240 s", -9
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


@app.function(image=image, gpu="H200:2", volumes={"/vol": vol}, timeout=15 * 60, cpu=8, memory=32 * 1024)
def gate2(sections: str, extra: list):
    _run(2, sections, extra)


@app.function(image=image, gpu="H200:4", volumes={"/vol": vol}, timeout=15 * 60, cpu=12, memory=48 * 1024)
def gate4(sections: str, extra: list):
    _run(4, sections, extra)


@app.function(image=image, gpu="H200:8", volumes={"/vol": vol}, timeout=15 * 60, cpu=16, memory=64 * 1024)
def gate8(sections: str, extra: list):
    _run(8, sections, extra)


@app.function(image=image, timeout=15 * 60, cpu=8, memory=16 * 1024)
def dry():
    env = dict(os.environ, PYTHONPATH="/root/lane/python", PYTHONUNBUFFERED="1")
    rc = 0
    for world in (2, 4, 8):
        for script in ("/root/gate/kovl2_gate.py", "/root/gate/kovl_gate.py"):
            print(f"=== dry run {os.path.basename(script)}, {world} thread ranks", flush=True)
            r = subprocess.run([sys.executable, script, "--dry", "--world", str(world)],
                               env=env, capture_output=True, text=True, timeout=300)
            print("\n".join(l for l in r.stdout.splitlines() if l.startswith("RESULT")), flush=True)
            if r.returncode:
                print(f"FAIL dry {script} world={world} rc={r.returncode}\n{r.stderr[-3000:]}", flush=True)
                rc |= r.returncode
    if rc:
        raise RuntimeError(f"dry run failed rc={rc}")


@app.local_entrypoint()
def main(gpus: int = 4, dry_run: bool = False, sections: str = DEFAULT_SECTIONS, route_frac: float = 0.66, blocks: int = 2):
    if dry_run:
        dry.remote()
        return
    extra = ["--route-frac", str(route_frac), "--blocks", str(blocks)]
    {2: gate2, 4: gate4, 8: gate8}[gpus].remote(sections, extra)
