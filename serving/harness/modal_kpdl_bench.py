"""Lane KPDL chain bench (norm_gate PDL trigger) on ONE H200 in the judge image (Modal), hard cap 10 min (~$0.75 worst case at $4.5/h).

  modal run harness/modal_kpdl_bench.py --chain-m 8,32,128,256 --reps 300
Prints the bench output; also saved on the volume under out/kpdl_bench/<ts>/.
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
app = modal.App("okc-kpdl-bench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"PYTHONPATH": "/root/lane/kpdl/python"})
    .add_local_dir(os.path.join(SERVING, "lanes", "kpdl", "python", "sglang"), "/root/lane/kpdl/python/sglang")
    .add_local_dir(os.path.join(SERVING, "lanes", "kfuse_tools"), "/root/lane/kfuse_tools")
    .add_local_dir(os.path.join(SERVING, "lanes", "kpdl_tools"), "/root/lane/kpdl_tools", ignore=["runs"])
)


def _stream(name, cmd, out, limit, env=None):
    print(f"===== {name}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(f"{out}/{name}.txt", "w") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd="/root/lane",
                             env={**os.environ, **(env or {})})
        try:
            for line in p.stdout:
                print(line, end="", flush=True)
                f.write(line)
                if time.time() - t0 > limit:
                    p.kill()
                    print(f"===== {name} KILLED at {limit}s", flush=True)
                    break
        finally:
            p.wait()
        print(f"===== {name} rc={p.returncode} {time.time() - t0:.0f}s", flush=True)
        f.write(f"\nrc={p.returncode}\n")
    return p.returncode != 0


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=10 * 60, cpu=8, memory=64 * 1024)
def run(only: str = "chain", reps: int = 300, chain_m: str = "8,32,128,256", tests: str = "", extra: str = "", section: str = "ng"):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kpdl_bench/{ts}"
    os.makedirs(out, exist_ok=True)
    subprocess.run("nvidia-smi -L", shell=True)
    rc = 0
    if only != "none":
        rc |= _stream("bench", [sys.executable, "/root/lane/kpdl_tools/kpdl_bench.py", "--reps",
                                str(reps), "--chain-m", chain_m, "--section", section, "--out", f"{out}/kpdl.json"], out, 8 * 60)
        vol.commit()
    if tests:
        rc |= _stream("pytest", [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                                 *[f"/root/lane/kpdl/{t}" for t in tests.split(",")]], out, 6 * 60)
    if extra:
        rc |= _stream("extra", ["bash", "-lc", extra], out, 6 * 60)
    vol.commit()
    print("results:", out, flush=True)
    return rc
