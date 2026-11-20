"""Lane KMASK (tree-mask verify attention: lagged causal FA3 + 8-key correction) in the judge image on Modal.

  modal run harness/modal_kmask_ubench.py::cpu_compile                  # CPU: JIT-build for sm_90a, ptxas
  modal run harness/modal_kmask_ubench.py::cputests --files a,b         # CPU pytest in the lane tree
  modal run harness/modal_kmask_ubench.py::run --parts abi,exact,time   # 1 H200, 10-min cap (worst case ~$0.8)
  KMASK_LANE=kbranch KMASK_EXT=/root/lane/kmask_tools/kbranch_adapter.py \
    modal run harness/modal_kmask_ubench.py::run --args "--impls fixp,proto,ext"  # gate the builder's tree
Script: lanes/kmask_tools/kmask_ubench.py. Results: /vol/out/kmask/<ts>/ and stdout.
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
TOOLS = "/root/lane/kmask_tools"
CACHE = "/root/kcache"

LANE = os.environ.get("KMASK_LANE", "kmask")  # tree whose python/sglang is under test (kbranch for the builder's path)
EXT = os.environ.get("KMASK_EXT", "")  # container path of an `ext` adapter (see lanes/kmask_tools/kmask_ext_example.py)
app = modal.App("okc-kmask-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"PYTHONPATH": "/root/lane/kmask/python", "SGLANG_JIT_CACHE_DIR": f"{CACHE}/jit",
          "KMASK_LANE_PY": "/root/lane/kmask/python", "KMASK_EXT": EXT})
    .add_local_dir(os.path.join(SERVING, "lanes", LANE, "python", "sglang"), "/root/lane/kmask/python/sglang")
    .add_local_dir(os.path.join(SERVING, "lanes", LANE, "test"), "/root/lane/kmask/test")
    .add_local_dir(os.path.join(SERVING, "lanes", "kmask_tools"), TOOLS)
    .add_local_dir(os.path.join(SERVING, "lanes", "kwide2_tools"), "/root/lane/kwide2_tools")
)
image = image.add_local_dir(os.path.join(SERVING, "lanes", "kpatt3_tools"), "/root/lane/kpatt3_tools")
LANE_B = os.environ.get("KMASK_LANE_B", "")  # optional second tree for same-container A/B timing
if LANE_B:
    image = image.add_local_dir(os.path.join(SERVING, "lanes", LANE_B, "python", "sglang"), "/root/laneb/python/sglang")


def _stream(cmd, log, limit, env=None, cwd=TOOLS):
    print(f"===== {' '.join(cmd)}  (limit {limit}s)", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd, env=env)
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()
            if time.time() - t0 > limit:
                p.kill()
                print("===== killed at limit", flush=True)
                break
        rc = p.wait()
        f.write(f"\nrc={rc}\n")
    print(f"===== rc={rc} ({time.time() - t0:.0f}s)", flush=True)
    return rc


@app.function(image=image, volumes={"/vol": vol}, timeout=15 * 60, cpu=8, memory=32 * 1024)
def cpu_compile():
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kmask/{ts}_compile"
    os.makedirs(out, exist_ok=True)
    rc = _stream([sys.executable, "kmask_compile.py"], f"{out}/compile.log", 12 * 60)
    vol.commit()
    return rc


@app.function(image=image, volumes={"/vol": vol}, timeout=30 * 60, cpu=8, memory=32 * 1024)
def cputests(files: str):
    os.makedirs("/tmp/cudastub", exist_ok=True)
    subprocess.run("ln -sf $(ls /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/targets/*/lib/stubs/libcuda.so "
                   "2>/dev/null | head -1) /tmp/cudastub/libcuda.so.1", shell=True)
    env = dict(os.environ, LD_LIBRARY_PATH="/tmp/cudastub:" + os.environ.get("LD_LIBRARY_PATH", ""))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files.split(",")], env=env,
                       cwd="/root/lane/kmask")
    print("tests rc", r.returncode, flush=True)
    return r.returncode


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=10 * 60, cpu=8, memory=64 * 1024)
def run(parts: str = "abi,exact,time", args: str = ""):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kmask/{ts}"
    os.makedirs(out, exist_ok=True)
    subprocess.run("nvidia-smi -L", shell=True)
    rc = _stream([sys.executable, "kmask_ubench.py", "--parts", parts, "--out", f"{out}/kmask.json", *args.split()],
                 f"{out}/run.log", 9 * 60 + 20)
    vol.commit()
    print("results:", out, flush=True)
    return rc


@app.function(image=image, volumes={"/vol": vol}, gpu="H200", timeout=10 * 60, cpu=8, memory=64 * 1024)
def diag(args: str = ""):
    """kdatt prefix-kernel diagnostics (lanes/kmask_tools/kdatt_diag.py)."""
    return _stream([sys.executable, "kdatt_diag.py", *args.split()], "/tmp/diag.log", 8 * 60)


@app.function(image=image, volumes={"/vol": vol}, gpu="H200", timeout=10 * 60, cpu=8, memory=64 * 1024)
def adv(args: str = ""):
    """kdatt adversarial accuracy classes at the verify shape (lanes/kmask_tools/kdatt_adv.py)."""
    return _stream([sys.executable, "kdatt_adv.py", *args.split()], "/tmp/adv.log", 9 * 60 + 20)


@app.function(image=image, volumes={"/vol": vol}, gpu="H200", timeout=10 * 60, cpu=8, memory=64 * 1024)
def gputests(files: str):
    """GPU pytest in the lane tree (paths relative to the lane root)."""
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files.split(",")], cwd="/root/lane/kmask")
    print("gpu tests rc", r.returncode, flush=True)
    return r.returncode


@app.function(image=image, volumes={"/vol": vol}, gpu="H200", timeout=10 * 60, cpu=8, memory=64 * 1024)
def sweep(args: str = ""):
    """kdatt batch-size sweep (lanes/kmask_tools/kdatt_bs_sweep.py)."""
    return _stream([sys.executable, "kdatt_bs_sweep.py", *args.split()], "/tmp/sweep.log", 8 * 60)


@app.function(image=image, volumes={"/vol": vol}, gpu="H200", timeout=10 * 60, cpu=8, memory=64 * 1024)
def abtime(args: str = ""):
    """Same-container A/B: kmask_ubench time part on the lane tree (A) then on KMASK_LANE_B (B), twice each, interleaved."""
    rc = 0
    for rep in range(2):
        for tag, py in (("A", "/root/lane/kmask/python"), ("B", "/root/laneb/python")):
            env = dict(os.environ, PYTHONPATH=py, KMASK_LANE_PY=py, SGLANG_JIT_CACHE_DIR=f"{CACHE}/jit_{tag}")
            print(f"===== AB {tag} rep {rep}", flush=True)
            rc |= _stream([sys.executable, "kmask_ubench.py", "--parts", "time", "--out", f"/tmp/ab_{tag}{rep}.json", *args.split()],
                          f"/tmp/ab_{tag}{rep}.log", 140, env=env)
    return rc


@app.function(image=image, volumes={"/vol": vol}, gpu="H200", timeout=10 * 60, cpu=8, memory=64 * 1024)
def pfab(args: str = "--modes 0 --skip-acc"):
    """Same-container prefill timing A/B: lane B / A / B / A with lanes/kpatt3_tools/kpatt3_bench.py."""
    rc = 0
    for rep in range(2):
        for tag, py in (("B", "/root/laneb/python"), ("A", "/root/lane/kmask/python")):
            env = dict(os.environ, PYTHONPATH=py, SGLANG_JIT_CACHE_DIR=f"{CACHE}/jitp_{tag}")
            print(f"===== PF {tag} rep {rep}", flush=True)
            rc |= _stream([sys.executable, "kpatt3_bench.py", *args.split()], f"/tmp/pf_{tag}{rep}.log", 150, env=env,
                          cwd="/root/lane/kpatt3_tools")
    return rc


@app.function(image=image, volumes={"/vol": vol}, gpu="H200", timeout=10 * 60, cpu=8, memory=64 * 1024)
def tool(script: str, args: str = ""):
    """Run any lanes/kmask_tools script on one GPU."""
    return _stream([sys.executable, script, *args.split()], "/tmp/tool.log", 9 * 60)
