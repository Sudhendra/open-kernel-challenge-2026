"""Lane KATT on ONE GPU in the judge image (Modal): FA3 split sweep at the serving verify shapes, then the
kernel-track attention arm vs the serving FA3 chain.

  modal run harness/modal_katt_ubench.py::prebuild          # CPU only (free): builds the kernel-track modules
                                                            # into the volume's torch-extensions dir, prints seconds
  modal run harness/modal_katt_ubench.py::run               # 1 GPU, hard cap 20 min (worst case ~$1.5)
  modal run harness/modal_katt_ubench.py::run --only sweep  # or --only kt, or --only pol (split policy check)
Results print to stdout and land on the volume under out/katt/<ts>/.
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
KT = os.path.join(ROOT, "muse_attention_decode")
EXT_DIR = "/vol/katt/torch_ext"

app = modal.App("okc-katt-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"PYTHONPATH": "/root/lane/katt/python", "TORCH_EXTENSIONS_DIR": EXT_DIR,
          "OKC_KT_ATTN": "/root/kt/muse_attention_decode"})
    .add_local_dir(os.path.join(SERVING, "lanes", "katt", "python", "sglang"), "/root/lane/katt/python/sglang")
    .add_local_dir(os.path.join(SERVING, "lanes", "katt_tools"), "/root/lane/katt_tools")
    .add_local_file(os.path.join(KT, "submission.py"), "/root/kt/muse_attention_decode/submission.py")
    .add_local_file(os.path.join(KT, "reference.py"), "/root/kt/muse_attention_decode/reference.py")
    .add_local_file(os.path.join(KT, "checks.py"), "/root/kt/muse_attention_decode/checks.py")
    .add_local_dir(os.path.join(SERVING, "lanes", "katt", "test"), "/root/lane/katt/test")
)


def _stream(cmd, log, limit, cwd="/root/lane/katt_tools"):
    print(f"===== {' '.join(cmd)}  (limit {limit}s) {time.strftime('%H:%M:%SZ', time.gmtime())}", flush=True)
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd)
        t0 = time.time()
        try:
            for line in p.stdout:
                print(line, end="", flush=True)
                f.write(line)
                if time.time() - t0 > limit:
                    p.kill()
                    print("===== killed at limit", flush=True)
                    break
            p.wait(timeout=30)
        except Exception as exc:  # noqa: BLE001
            p.kill()
            print("===== error", exc, flush=True)
        rc = p.returncode
        f.write(f"\nrc={rc}\n")
    print(f"===== rc={rc} ({time.time() - t0:.0f}s)", flush=True)
    return rc


@app.function(image=image, volumes={"/vol": vol}, timeout=40 * 60, cpu=8, memory=32 * 1024)
def prebuild():
    os.makedirs(EXT_DIR, exist_ok=True)
    code = (
        "import time,sys; sys.path.insert(0,'/root/kt/muse_attention_decode'); t=time.time()\n"
        "import submission as S\nS._zh_build_wst(); S._zh_build_vfg(); print('zh builds', round(time.time()-t), 's', "
        "{k: v[-300:] for k, v in S._ZH.items() if k.endswith('_err')}, flush=True)\n"
        "t=time.time()\ntry:\n    S._build()\nexcept Exception as e:\n    print('vjeux import (expected on CPU):', repr(e)[-300:])\n"
        "print('vjeux build', round(time.time()-t), 's', flush=True)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], timeout=35 * 60)
    subprocess.run(f"ls -la {EXT_DIR} {EXT_DIR}/* | head -60", shell=True)
    vol.commit()
    return r.returncode


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=20 * 60, cpu=8, memory=64 * 1024)
def run(only: str = "", sweep_args: str = "", kt_args: str = ""):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/katt/{ts}"
    os.makedirs(out, exist_ok=True)
    subprocess.run("nvidia-smi -L; nvcc --version | tail -2", shell=True)
    t0 = time.time()
    if only in ("", "sweep"):
        _stream([sys.executable, "fa3_split_ubench.py", "--out", f"{out}/sweep.json", *sweep_args.split()],
                f"{out}/sweep.log", limit=9 * 60)
        vol.commit()
    if only == "pol":
        _stream([sys.executable, "policy_gpu_check.py", "--out", f"{out}/policy.json", *sweep_args.split()],
                f"{out}/policy.log", limit=17 * 60)
        vol.commit()
    if only in ("", "kt"):
        left = 19 * 60 - (time.time() - t0) - 30
        _stream([sys.executable, "kt_arm_ubench.py", "--out", f"{out}/kt.json", "--splits", "1,2,4,8,16",
                 *kt_args.split()], f"{out}/kt.log", limit=max(60, int(left)))
        vol.commit()
    print("results:", out, flush=True)


@app.function(image=image, timeout=30 * 60, cpu=8, memory=32 * 1024)
def cputests(files: str):
    """CPU pytest of lane tests that import the attention backend: libcuda.so.1 resolved to the
    toolkit's stub so sgl_kernel imports (no CUDA call runs)."""
    os.makedirs("/tmp/cudastub", exist_ok=True)
    subprocess.run("ln -sf $(ls /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/targets/*/lib/stubs/libcuda.so 2>/dev/null | head -1) /tmp/cudastub/libcuda.so.1", shell=True)
    env = dict(os.environ, LD_LIBRARY_PATH="/tmp/cudastub:" + os.environ.get("LD_LIBRARY_PATH", ""))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files.split(",")], env=env,
                       cwd="/root/lane/katt")
    print("tests rc", r.returncode, flush=True)
    return r.returncode
