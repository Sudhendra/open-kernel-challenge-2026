"""Lane KGEMV microbench on ONE GPU in the judge image (Modal), hard cap 20 min (~$1.5 worst case).

  modal run harness/modal_kgemv_ubench.py::compile_check          # CPU only, free: nvcc build of the bench ext
  modal run harness/modal_kgemv_ubench.py::run                    # all sections (~6-8 min)
  modal run harness/modal_kgemv_ubench.py::run --sections ab --ms 8,32 --reps 300
Prints the bench output; also saved on the volume under out/kgemv_ubench/<ts>/.
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
LANE = os.environ.get("OKC_LANE", "")
app = modal.App("okc-kgemv-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = modal.Image.from_registry(IMAGE_DIGEST, add_python=None).add_local_dir(
    os.path.join(SERVING, "lanes", "kgemv_tools"), "/root/lane/kgemv_tools"
)
if LANE:
    image = image.env({"PYTHONPATH": "/root/lane/kgemv/python"}).add_local_dir(
        os.path.join(SERVING, LANE, "python", "sglang"), "/root/lane/kgemv/python/sglang"
    ).add_local_dir(os.path.join(SERVING, LANE, "test"), "/root/lane/kgemv/test")


def _stream(name, cmd, out, limit, env=None):
    print(f"===== {name}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(f"{out}/{name}.txt", "w") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             cwd="/root/lane/kgemv_tools", env={**os.environ, **(env or {})})
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
    return p.returncode


@app.function(image=image, volumes={"/vol": vol}, timeout=15 * 60, cpu=8, memory=32 * 1024)
def compile_check(extra: str = ""):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kgemv_ubench/{ts}_cpu"
    os.makedirs(out, exist_ok=True)
    code = (
        "import os, sys, time; sys.path.insert(0, '.');"
        "os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';"
        "import kgemv_bench as b; e = b.load_ext(); print('ext ok', [k for k in dir(e) if not k.startswith('_')]);"
        "import lane_s_small_m_gemm as t; print('triton module ok', t.triton is not None)"
    )
    rc = _stream("compile_check", [sys.executable, "-c", code], out, 12 * 60)
    if extra:
        rc |= _stream("extra", ["bash", "-lc", extra], out, 10 * 60,
                      env={"PYTHONPATH": "/root/lane/kgemv/python"} if LANE else None)
    vol.commit()
    if rc:
        raise RuntimeError(f"compile_check rc={rc}")


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=20 * 60, cpu=8, memory=64 * 1024)
def run(sections: str = "ab,c", ms: str = "8,32,64,128,192,256", reps: int = 200, shapes: str = "", extra: str = ""):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kgemv_ubench/{ts}"
    os.makedirs(out, exist_ok=True)
    subprocess.run("nvidia-smi -L", shell=True)
    rc = 0
    if sections:
        cmd = [sys.executable, "kgemv_bench.py", "--sections", sections, "--ms", ms, "--reps", str(reps),
               "--out", f"{out}/kgemv_rows.json"]
        if shapes:
            cmd += ["--shapes", shapes]
        rc |= _stream("bench", cmd, out, 16 * 60)
        vol.commit()
    if extra:
        rc |= _stream("extra", ["bash", "-lc", extra], out, 10 * 60,
                      env={"PYTHONPATH": "/root/lane/kgemv/python"} if LANE else None)
    vol.commit()
    print("results:", out, flush=True)
    return rc
