"""MoE grouped-vs-Marlin microbench on ONE H200 inside the judge image (Modal), hard timeout 90 min.

  modal run harness/modal_ubench.py            # full: spill baseline, compile gate, preflight, seeds 0 and 1
  modal run harness/modal_ubench.py --preflight-only
Results land on the volume under out/ubench/<ts>/ and are also printed. Worst case ~$7 at the 90-min cap.
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
ROOT = os.path.dirname(SERVING)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"

app = modal.App("okc-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .add_local_dir(os.path.join(SERVING, "lanes", "moe", "python", "sglang"), "/root/lane/moe/python/sglang")
    .add_local_dir(os.path.join(SERVING, "lanes", "moe_tools"), "/root/lane/moe_tools")
    .add_local_file(os.path.join(ROOT, "muse_moe_w4a16", "nvfp4.py"), "/root/lane/muse_moe_w4a16/nvfp4.py")
    .add_local_file(os.path.join(ROOT, "muse_moe_w4a16", "reference.py"), "/root/lane/muse_moe_w4a16/reference.py")
    .add_local_file(os.path.join(ROOT, "muse_moe_w4a16", "checks.py"), "/root/lane/muse_moe_w4a16/checks.py")
    .add_local_file(
        os.path.join(ROOT, "muse_prefill_layer", "submission.py"), "/root/lane/muse_prefill_layer/submission.py"
    )
)


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=90 * 60, cpu=8, memory=64 * 1024)
def ubench(preflight_only: bool = False):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/ubench/{ts}"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/root/lane/moe/python", OKC_KERNEL_TRACK="/root/lane/muse_moe_w4a16")
    tools = "/root/lane/moe_tools"

    def run(name, cmd, extra=None, limit=1800):
        print(f"=== {name} {time.strftime('%H:%M:%SZ', time.gmtime())} ===", flush=True)
        e = dict(env, **(extra or {}))
        try:
            r = subprocess.run(cmd, env=e, cwd=tools, timeout=limit)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = 124
        print(f"{name} rc={rc}", flush=True)
        vol.commit()
        return rc

    run("env", [sys.executable, "-c",
        "import torch, sglang; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability(), sglang.__file__); "
        "assert sglang.__file__.startswith('/root/lane/moe/python')"], limit=120)
    run("spill_baseline", ["bash", "./spill_baseline.sh", f"{out}/spill_baseline.log"],
        {"SUBMISSION": "/root/lane/muse_prefill_layer/submission.py"}, limit=900)
    rc = run("compile_gate", ["bash", "./compile_gate.sh", f"{out}/compile_gate.log"], {"LANE_PY": "/root/lane/moe/python"}, limit=900)
    rc = run("preflight", [sys.executable, "microbench.py", "--sizes", "9", "--reps", "2", "--out", f"{out}/preflight.json"], limit=1500)
    if rc != 0:
        raise RuntimeError(f"preflight failed rc={rc}; results in {out}")
    if preflight_only:
        print("results in", out, flush=True)
        return
    for seed in (0, 1):
        run(f"seed{seed}", [sys.executable, "microbench.py", "--sizes", "1,9,18,36,72,144,288,1000,16384", "--reps", "20",
            "--seed", str(seed), "--out", f"{out}/microbench_seed{seed}.json"], limit=1800)
    print("results in", out, flush=True)


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=20 * 60, cpu=8, memory=64 * 1024)
def refcheck(sizes: str = "9,144"):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/ubench/{ts}_ref"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/root/lane/moe/python", OKC_KERNEL_TRACK="/root/lane/muse_moe_w4a16")
    r = subprocess.run([sys.executable, "ref_compare.py", "--sizes", sizes, "--out", f"{out}/ref_compare.json"],
                       env=env, cwd="/root/lane/moe_tools", timeout=900)
    vol.commit()
    print("refcheck rc", r.returncode, "results in", out, flush=True)
    if r.returncode:
        raise RuntimeError("refcheck failed")


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=20 * 60, cpu=8, memory=64 * 1024)
def profile(sizes: str = "1,9,18,36,72,144", extra: str = ""):
    """Small-M time breakdown (moe_tools/small_m_profile.py); ~6-8 min, worst case ~$1 at the 20-min cap."""
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/ubench/{ts}_profile"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/root/lane/moe/python", OKC_KERNEL_TRACK="/root/lane/muse_moe_w4a16")
    r = subprocess.run([sys.executable, "small_m_profile.py", "--sizes", sizes, "--out", f"{out}/profile.json", *extra.split()],
                       env=env, cwd="/root/lane/moe_tools", timeout=1000)
    vol.commit()
    print("profile rc", r.returncode, "results in", out, flush=True)
    if r.returncode:
        raise RuntimeError("profile failed")
