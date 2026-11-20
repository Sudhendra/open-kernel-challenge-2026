"""Lane KNORM microbench on ONE GPU in the judge image (Modal), hard cap 20 min (~$1.5 worst case).

  modal run harness/modal_knorm_ubench.py                       # all sections + agnorm bit-compare
  modal run harness/modal_knorm_ubench.py --only norm --reps 300
  modal run harness/modal_knorm_ubench.py --agnorm 0
  modal run harness/modal_knorm_ubench.py --only none --tests test/registered/kernels/ops/layernorm/test_muse_spark_norm_ops.py
Prints the scripts' output; also saved on the volume under out/knorm_ubench/<ts>/.
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
app = modal.App("okc-knorm-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"PYTHONPATH": "/root/lane/knorm/python"})
    .add_local_dir(os.path.join(SERVING, "lanes", "knorm", "python", "sglang"), "/root/lane/knorm/python/sglang")
    .add_local_dir(os.path.join(SERVING, "lanes", "knorm_tools"), "/root/lane/knorm_tools")
    .add_local_dir(os.path.join(SERVING, "lanes", "agnorm_tools"), "/root/lane/agnorm_tools")
    .add_local_dir(os.path.join(SERVING, "lanes", "knorm", "test"), "/root/lane/knorm/test")
)


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=20 * 60, cpu=8, memory=64 * 1024)
def run(only: str = "", reps: int = 200, agnorm: int = 1, tests: str = ""):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/knorm_ubench/{ts}"
    os.makedirs(out, exist_ok=True)
    subprocess.run("nvidia-smi -L", shell=True)
    bench = [sys.executable, "/root/lane/knorm_tools/knorm_ubench.py", "--reps", str(reps), "--out", f"{out}/knorm.json"]
    if only:
        bench += ["--only", only]
    jobs = [("knorm", bench, 12 * 60)] if only != "none" else []
    if tests:
        jobs.append(("pytest", [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                *[f"/root/lane/knorm/{t}" for t in tests.split(",")]], 8 * 60))
    if agnorm:
        jobs.append(("agnorm_bits", [sys.executable, "/root/lane/agnorm_tools/bitcompare_ag_norm.py"], 6 * 60))
    rc = 0
    for name, cmd, limit in jobs:
        print(f"===== {name}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        with open(f"{out}/{name}.txt", "w") as f:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd="/root/lane")
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
            rc |= p.returncode != 0
        vol.commit()
    print("results:", out, flush=True)
    return rc
