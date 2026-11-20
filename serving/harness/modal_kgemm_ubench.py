"""Lane KGEMM microbench on ONE GPU in the judge image (Modal), hard cap 12 min (~$1 worst case).

  modal run harness/modal_kgemm_ubench.py                                  # all sections
  modal run harness/modal_kgemm_ubench.py --only router,window --reps 300
  modal run harness/modal_kgemm_ubench.py --only none --tests test/registered/kernels/ops/gemm/test_muse_spark_router_mma.py
Prints the bench output; also saved on the volume under out/kgemm_ubench/<ts>/.
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
app = modal.App("okc-kgemm-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"PYTHONPATH": "/root/lane/kgemm/python", "TORCH_CUDA_ARCH_LIST": "9.0a"})
    .add_local_dir(os.path.join(SERVING, "lanes", "kgemm", "python", "sglang"), "/root/lane/kgemm/python/sglang")
    .add_local_dir(os.path.join(SERVING, "lanes", "kgemm_tools"), "/root/lane/kgemm_tools")
    .add_local_dir(os.path.join(SERVING, "lanes", "kgemm", "test"), "/root/lane/kgemm/test")
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
def run(only: str = "router,moe0,window,lt,qkv,gemm", reps: int = 200, moe_cfgs: str = "", tests: str = "",
        bench_limit: int = 560):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kgemm_ubench/{ts}"
    os.makedirs(out, exist_ok=True)
    subprocess.run("nvidia-smi -L; nvidia-smi --query-gpu=clocks.max.sm,power.limit --format=csv", shell=True)
    rc = 0
    if only != "none":
        cmd = [sys.executable, "/root/lane/kgemm_tools/kgemm_bench.py", "--only", only, "--reps", str(reps),
               "--out", f"{out}/kgemm.json"]
        if moe_cfgs:
            cmd += ["--moe-cfgs", moe_cfgs]
        rc |= _stream("bench", cmd, out, bench_limit)
        vol.commit()
    if tests:
        rc |= _stream("pytest", [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                                 *[f"/root/lane/kgemm/{t}" for t in tests.split(",")]], out, 150)
    vol.commit()
    print("results:", out, flush=True)
    return rc
