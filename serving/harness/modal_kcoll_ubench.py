"""Lane KCOLL microbench on ONE GPU in the judge image (Modal), hard cap 12 min (~$1 worst case).

  modal run harness/modal_kcoll_ubench.py                                   # bits, call, chain
  modal run harness/modal_kcoll_ubench.py --only chain --reps 300 --chain-m 8,32
  modal run harness/modal_kcoll_ubench.py --only none --tests test/registered/kernels/ops/gemm/test_muse_spark_router_mma.py
Prints the bench output; also saved on the volume under out/kcoll_ubench/<ts>/.
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
app = modal.App("okc-kcoll-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"PYTHONPATH": "/root/lane/kcoll/python"})
    .add_local_dir(os.path.join(SERVING, "lanes", "kcoll", "python", "sglang"), "/root/lane/kcoll/python/sglang")
    .add_local_dir(os.path.join(SERVING, "lanes", "kcoll_tools"), "/root/lane/kcoll_tools")
    .add_local_dir(os.path.join(SERVING, "lanes", "kcoll", "test"), "/root/lane/kcoll/test")
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


@app.function(image=image, gpu="H" + str(200), volumes={"/vol": vol}, timeout=12 * 60, cpu=8, memory=64 * 1024)
def run(only: str = "bits,call,chain", reps: int = 200, chain_m: str = "8,32", call_m: str = "8,16,32,64", cfg_arms: str = "", tests: str = "", extra: str = ""):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kcoll_ubench/{ts}"
    os.makedirs(out, exist_ok=True)
    subprocess.run("nvidia-smi -L", shell=True)
    rc = 0
    if only != "none":
        rc |= _stream("bench", [sys.executable, "/root/lane/kcoll_tools/kcoll_bench.py", "--only", only, "--reps",
                                str(reps), "--chain-m", chain_m, "--call-m", call_m, "--out", f"{out}/kcoll.json"], out, 9 * 60, env={"KCOLL_CFG_ARMS": cfg_arms})
        vol.commit()
    if tests:
        rc |= _stream("pytest", [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                                 *[f"/root/lane/kcoll/{t}" for t in tests.split(",")]], out, 3 * 60)
    if extra:
        rc |= _stream("extra", ["bash", "-lc", extra], out, 3 * 60)
    vol.commit()
    print("results:", out, flush=True)
    return rc
