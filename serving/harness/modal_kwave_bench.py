"""Lane KWAVE microbench on Modal, judge image: copy-engine rates with 1/2/4 copy streams, compute slowdown under
copies (grouped NVFP4 experts and a bf16 GEMM), grouped-expert cost per extra call.

  modal run harness/modal_kwave_bench.py --gpus 4     # 4 GPUs, hard timeout 12 min (~$1.5 expected, ~$3.6 worst)
  modal run harness/modal_kwave_bench.py --dry-run    # CPU only: threads as 2, 4 and 8 ranks, small shapes

Sections, each a separate torchrun process: copies, tax, experts. RESULT json lines print to stdout and go to
out/kwave_bench/<ts>_g<N>/results.txt on the volume, committed after each section. Mounts lanes/kwave/python/sglang.
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
SECTIONS = "copies,tax,experts"
TIMEOUTS = {"copies": 120, "tax": 270, "experts": 210}

app = modal.App("okc-kwave-bench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .add_local_dir(os.path.join(SERVING, "lanes", "kwave", "python", "sglang"), "/root/lane/python/sglang")
    .add_local_file(os.path.join(SERVING, "lanes", "kovl_tools", "kovl_gate.py"), "/root/gate/kovl_gate.py")
    .add_local_file(os.path.join(SERVING, "lanes", "kwave_tools", "kwave_bench.py"), "/root/gate/kwave_bench.py")
)


def _run(gpus: int, sections: str):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kwave_bench/{ts}_g{gpus}"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/root/lane/python", PYTHONUNBUFFERED="1")
    for section in sections.split(","):
        print(f"=== section {section} on {gpus} GPUs", flush=True)
        cmd = ["torchrun", "--nproc_per_node", str(gpus), "/root/gate/kwave_bench.py", "--section", section]
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=TIMEOUTS.get(section, 240))
            stdout, stderr, rc = r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr, rc = f"timeout {TIMEOUTS.get(section, 240)} s", -9
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


@app.function(image=image, gpu="H200:4", volumes={"/vol": vol}, timeout=12 * 60, cpu=12, memory=48 * 1024)
def gate4(sections: str):
    _run(4, sections)


@app.function(image=image, timeout=15 * 60, cpu=8, memory=16 * 1024)
def dry():
    env = dict(os.environ, PYTHONPATH="/root/lane/python", PYTHONUNBUFFERED="1")
    rc = 0
    for world in (2, 4, 8):
        print(f"=== dry run kwave_bench.py, {world} thread ranks", flush=True)
        r = subprocess.run([sys.executable, "/root/gate/kwave_bench.py", "--dry", "--world", str(world)],
                           env=env, capture_output=True, text=True, timeout=300)
        print("\n".join(l for l in r.stdout.splitlines() if l.startswith("RESULT")), flush=True)
        if r.returncode:
            print(f"FAIL dry world={world} rc={r.returncode}\n{r.stderr[-3000:]}", flush=True)
            rc |= r.returncode
    if rc:
        raise RuntimeError(f"dry run failed rc={rc}")


@app.local_entrypoint()
def main(gpus: int = 4, dry_run: bool = False, sections: str = SECTIONS):
    if dry_run:
        dry.remote()
        return
    if gpus != 4:
        raise SystemExit("this bench is cleared for 4 GPUs only")
    gate4.remote(sections)
