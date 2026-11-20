"""CPU-only jobs inside the judge image: compile gates and checkpoint scans, no GPU billed.

  modal run harness/modal_cpu.py::gate --lane lanes/moe        # nvcc compile gate + scale histogram
  modal run harness/modal_cpu.py::gate --lane lanes/moe --skip-hist

The lane tree's python/sglang is mounted first on PYTHONPATH; moe_tools scripts run from /root/moe_tools.
Results print to stdout and are copied to the volume under out/cpu/<ts>/.
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = os.environ.get(
    "OKC_IMAGE", "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
)
LANE = os.environ.get("OKC_LANE", "lanes/moe")
CKPT = "/vol/hf/hub/models--meta-models--Muse-Spark-Open-816B-A42B-NVFP4"

app = modal.App("okc-cpu")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)

image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"HF_HOME": "/vol/hf", "HF_HUB_OFFLINE": "1"})
    .add_local_dir(os.path.join(SERVING, LANE, "python", "sglang"), "/root/lane/python/sglang")
    .add_local_dir(os.path.join(SERVING, "lanes", "moe_tools"), "/root/moe_tools")
    .add_local_dir(os.path.join(SERVING, LANE, "test"), "/root/lane/test")
    # Kernel-track original of the same kernel, for the ptxas spill baseline.
    .add_local_file(
        os.path.join(os.path.dirname(SERVING), "muse_prefill_layer", "submission.py"),
        "/root/kernel_track/muse_prefill_layer/submission.py",
    )
)


@app.function(image=image, volumes={"/vol": vol}, timeout=3600, cpu=8, memory=32 * 1024)
def gate(skip_hist: bool = False, skip_gate: bool = False, baseline: bool = False, bisect: bool = False):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/cpu/{ts}"
    os.makedirs(out, exist_ok=True)
    rc = 0
    if not skip_gate:
        print("=== compile gate ===", flush=True)
        env = dict(os.environ, LANE_PY="/root/lane/python")
        r = subprocess.run(["bash", "/root/moe_tools/compile_gate.sh", f"{out}/compile_gate.log"], env=env)
        print("compile gate rc", r.returncode, flush=True)
        rc |= r.returncode
    if baseline:
        print("=== spill baseline (kernel-track build) ===", flush=True)
        env = dict(os.environ, SUBMISSION="/root/kernel_track/muse_prefill_layer/submission.py")
        r = subprocess.run(["bash", "/root/moe_tools/spill_baseline.sh", f"{out}/spill_baseline.log"], env=env)
        print("spill baseline rc", r.returncode, flush=True)
    if bisect:
        print("=== spill bisect (per kernel-track hunk) ===", flush=True)
        env = dict(os.environ, LANE_PY="/root/lane/python")
        r = subprocess.run([sys.executable, "/root/moe_tools/spill_bisect.py"], env=env)
        print("spill bisect rc", r.returncode, flush=True)
    if not skip_hist:
        print("=== scale histogram ===", flush=True)
        r = subprocess.run(
            [sys.executable, "/root/moe_tools/scale_hist.py", CKPT, "--json", f"{out}/scale_hist.json", "--check-codes"]
        )
        print("scale hist rc", r.returncode, flush=True)
        rc |= r.returncode
    vol.commit()
    print("results in", out, flush=True)
    if rc:
        raise RuntimeError(f"cpu job failed rc={rc}")
    return rc


CPU_TESTS = [
    "test/registered/kernels/ops/kvcache/test_hicache.py",
    "test/registered/unit/sampling/test_renorm_policy.py",
]


@app.function(image=image, volumes={"/vol": vol}, timeout=1800, cpu=8, memory=32 * 1024)
def tests(files: str = ""):
    """CPU unit tests of the lane tree (pytest), lane python first on PYTHONPATH."""
    sel = files.split(",") if files else CPU_TESTS
    env = dict(os.environ, PYTHONPATH="/root/lane/python")
    r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *sel], env=env, cwd="/root/lane")
    print("tests rc", r.returncode, flush=True)
    if r.returncode:
        raise RuntimeError(f"tests failed rc={r.returncode}")
