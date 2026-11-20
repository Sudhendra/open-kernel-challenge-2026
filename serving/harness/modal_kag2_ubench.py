"""Lane KAG2 microbench on ONE GPU in the judge image (Modal), hard cap 12 min (~$1 worst case).

  modal run harness/modal_kag2_ubench.py            # route bits, count bits (eager + graph), tail timing
Saved on the volume under out/kag2_ubench/<ts>/.
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
app = modal.App("okc-kag2-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"PYTHONPATH": "/root/lane/kag2/python"})
    .add_local_dir(os.path.join(SERVING, "lanes", "kag2", "python", "sglang"), "/root/lane/kag2/python/sglang")
    .add_local_dir(os.path.join(SERVING, "lanes", "kag2_tools"), "/root/lane/kag2_tools")
    .add_local_dir(os.path.join(SERVING, "lanes", "kag2", "test"), "/root/lane/kag2/test")
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
def run(reps: int = 300):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kag2_ubench/{ts}"
    os.makedirs(out, exist_ok=True)
    subprocess.run("nvidia-smi -L", shell=True)
    rc = _stream("bench", [sys.executable, "/root/lane/kag2_tools/kag2_bench.py", "--reps", str(reps), "--out", f"{out}/kag2.json"], out, 10 * 60)
    vol.commit()
    print("results:", out, flush=True)
    return rc
