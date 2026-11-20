"""Lane KREP GPU checks on ONE GPU inside the judge image (Modal), hard timeout 20 min.

  OKC_LANE=lanes/krep modal run harness/modal_krep_ubench.py::gpu_tests
  OKC_LANE=lanes/krep modal run harness/modal_krep_ubench.py::bench      (router cost per layer call, ~3 min)
  OKC_LANE=lanes/krep modal run harness/modal_krep_ubench.py::all        (gpu_tests, then bench if they pass)

Runs the lane's GPU kernel test: the replica router equals the reference rule on every simulated rank, the cross-rank
sum with replicas matches the owner layout within bf16 ulps (stats printed as "[krep gpu]"), graph replay reads the
live table. First compile of the changed .cuh happens here (~2-4 min). No timing.
Worst case ~$1.5 at the 20-min cap.
"""
import os
import subprocess
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
LANE = os.environ.get("OKC_LANE", "lanes/krep")
# Bench scripts mounted at /root/krep_tools (OKC_KREP_TOOLS=lanes/stack_k3_krep_tools for ::slots).
TOOLS = os.environ.get("OKC_KREP_TOOLS", "lanes/krep_tools")

app = modal.App("okc-krep-ubench")
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .add_local_dir(os.path.join(SERVING, LANE, "python", "sglang"), "/root/lane/python/sglang")
    .add_local_dir(os.path.join(SERVING, LANE, "test"), "/root/lane/test")
    .add_local_dir(os.path.join(SERVING, TOOLS), "/root/krep_tools")
)

FILES = ["test/registered/kernels/ops/moe/test_muse_spark_expert_replicas_gpu.py"]


@app.function(image=image, gpu="H200", timeout=20 * 60, cpu=4, memory=32 * 1024)
def gpu_tests(files: str = ""):
    sel = files.split(",") if files else FILES
    env = dict(os.environ, PYTHONPATH="/root/lane/python")
    subprocess.run([sys.executable, "-c", "import torch, sglang; print(torch.cuda.get_device_name(), sglang.__file__)"],
                   env=env, cwd="/root/lane", timeout=300)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-s", "-p", "no:cacheprovider", *sel],
                       env=env, cwd="/root/lane", timeout=15 * 60)
    print("gpu tests rc", r.returncode, flush=True)
    if r.returncode:
        raise RuntimeError(f"gpu tests failed rc={r.returncode}")


@app.function(image=image, gpu="H200", timeout=20 * 60, cpu=4, memory=32 * 1024)
def bench():
    env = dict(os.environ, PYTHONPATH="/root/lane/python")
    r = subprocess.run([sys.executable, "/root/krep_tools/bench_router.py"], env=env, cwd="/root/lane", timeout=15 * 60)
    print("bench rc", r.returncode, flush=True)
    if r.returncode:
        raise RuntimeError(f"bench failed rc={r.returncode}")


@app.function(image=image, gpu="H200", timeout=20 * 60, cpu=4, memory=32 * 1024)
def all():
    env = dict(os.environ, PYTHONPATH="/root/lane/python")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-s", "-p", "no:cacheprovider", *FILES],
                       env=env, cwd="/root/lane", timeout=10 * 60)
    print("gpu tests rc", r.returncode, flush=True)
    if r.returncode:
        raise RuntimeError(f"gpu tests failed rc={r.returncode}")
    r = subprocess.run([sys.executable, "/root/krep_tools/bench_router.py"], env=env, cwd="/root/lane", timeout=8 * 60)
    print("bench rc", r.returncode, flush=True)
    if r.returncode:
        raise RuntimeError(f"bench failed rc={r.returncode}")


@app.function(image=image, gpu="H200", timeout=15 * 60, cpu=4, memory=32 * 1024)
def slots():
    """GPU tests (1/2/4/8 slots), then bench_router_slots.py; hard cap 15 min.

    OKC_LANE=lanes/stack_k3_krep OKC_KREP_TOOLS=lanes/stack_k3_krep_tools modal run harness/modal_krep_ubench.py::slots
    """
    env = dict(os.environ, PYTHONPATH="/root/lane/python")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-s", "-p", "no:cacheprovider", *FILES],
                       env=env, cwd="/root/lane", timeout=8 * 60)
    print("gpu tests rc", r.returncode, flush=True)
    b = subprocess.run([sys.executable, "/root/krep_tools/bench_router_slots.py"], env=env, cwd="/root/lane",
                       timeout=5 * 60)
    print("bench rc", b.returncode, flush=True)
    if r.returncode or b.returncode:
        raise RuntimeError(f"slots failed tests rc={r.returncode} bench rc={b.returncode}")
