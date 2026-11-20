"""Lane KSPLIT gate on Modal, judge image: the SP region's experts in two shard halves vs one call.

  modal run harness/modal_ksplit_gate.py --gpus 4     # 4 GPUs, hard timeout 15 min (~$1.5-2 expected, ~$4.5 worst)
  modal run harness/modal_ksplit_gate.py --dry-run    # CPU only: threads as 2, 4 and 8 ranks, small shapes

Sections, each a separate torchrun process: experts (bits, eager peak memory, graph ms per layer at 2048 and 4096 rows,
replicas 2 / 0, with and without skew). The first run's oproj section (column pushes vs packed + permute) is kept in
out/ksplit/ksplit_gate_as_run_g4.py; that code was reverted. RESULT json lines print to stdout and go to
out/ksplit_gate/<ts>_g<N>/results.txt on the volume, committed after each section. Mounts lanes/ksplit/python/sglang.
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
SECTIONS = "experts"
TIMEOUTS = {"experts": 540}

app = modal.App("okc-ksplit-gate")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .add_local_dir(os.path.join(SERVING, "lanes", "ksplit", "python", "sglang"), "/root/lane/python/sglang")
    .add_local_file(os.path.join(SERVING, "lanes", "kovl_tools", "kovl_gate.py"), "/root/gate/kovl_gate.py")
    .add_local_file(os.path.join(SERVING, "lanes", "ksplit_tools", "ksplit_gate.py"), "/root/gate/ksplit_gate.py")
)


def _run(gpus: int, sections: str):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/ksplit_gate/{ts}_g{gpus}"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/root/lane/python", PYTHONUNBUFFERED="1")
    for section in sections.split(","):
        print(f"=== section {section} on {gpus} GPUs", flush=True)
        cmd = ["torchrun", "--nproc_per_node", str(gpus), "/root/gate/ksplit_gate.py", "--section", section]
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=TIMEOUTS[section])
            stdout, stderr, rc = r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr, rc = f"timeout {TIMEOUTS[section]} s", -9
        lines = [l for l in stdout.splitlines() if l.startswith("RESULT")]
        with open(f"{out}/results.txt", "a") as f:
            for l in lines:
                print(l, flush=True)
                f.write(l + "\n")
            if rc != 0 or not lines:
                tail = (stderr or "")[-3000:]
                print(f"FAIL section={section} rc={rc}\n{tail}", flush=True)
                f.write(f"FAIL section={section} rc={rc}\n{tail}\n")
        vol.commit()
    print("results in", out, flush=True)


@app.function(image=image, gpu="H200:4", volumes={"/vol": vol}, timeout=15 * 60, cpu=12, memory=48 * 1024)
def gate4(sections: str):
    _run(4, sections)


@app.function(image=image, timeout=15 * 60, cpu=8, memory=16 * 1024)
def dry():
    env = dict(os.environ, PYTHONPATH="/root/lane/python", PYTHONUNBUFFERED="1")
    # The toolkit's libcuda stub, so the model module (sgl_kernel, FA3) imports on CPU.
    os.makedirs("/tmp/cudastub", exist_ok=True)
    subprocess.run(
        "ln -sf $(ls /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/targets/*/lib/stubs/libcuda.so"
        " 2>/dev/null | head -1) /tmp/cudastub/libcuda.so.1",
        shell=True,
    )
    env["LD_LIBRARY_PATH"] = "/tmp/cudastub:" + os.environ.get("LD_LIBRARY_PATH", "")
    rc = 0
    for world in (2, 4, 8):
        print(f"=== dry run ksplit_gate.py, {world} thread ranks", flush=True)
        r = subprocess.run([sys.executable, "/root/gate/ksplit_gate.py", "--dry", "--world", str(world)],
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
        raise SystemExit("this gate is cleared for 4 GPUs only")
    gate4.remote(sections)
