"""Lane KATT2 (deterministic FP8 verify attention) in the judge image on Modal.

  modal run harness/modal_katt2_ubench.py::cpu_compile             # CPU only (free): JIT-build the kernel with the
                                                                   # judge nvcc for sm_90a, print ptxas + seconds,
                                                                   # save the JIT cache to the volume
  modal run harness/modal_katt2_ubench.py::cputests --files a,b    # CPU pytest in the lane tree (libcuda stub)
  modal run harness/modal_katt2_ubench.py::run --only core         # 1 GPU, 20-min cap: kernel vs FA3 (+KATT policy)
  modal run harness/modal_katt2_ubench.py::run --only chain        # 1 GPU: vs the kernel-track atomics kernel
  modal run harness/modal_katt2_ubench.py::run --only pytest       # 1 GPU: the lane's kernel unit tests (incl. GPU)
Mounts lanes/katt2 (the lane tree), lanes/katt2_tools, lanes/katt_tools and the kernel-track attention files.
Results print to stdout and land on the volume under out/katt2/<ts>/.
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
CACHE = "/root/kcache"
CACHE_TAR = "/vol/out/katt2/cache.tgz"
TOOLS = "/root/lane/katt2_tools"

app = modal.App("okc-katt2-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"PYTHONPATH": "/root/lane/katt2/python:/root/lane/katt_tools", "SGLANG_JIT_CACHE_DIR": f"{CACHE}/jit",
          "TORCH_EXTENSIONS_DIR": "/vol/katt/torch_ext", "OKC_KT_ATTN": "/root/kt/muse_attention_decode"})
    .add_local_dir(os.path.join(SERVING, "lanes", "katt2", "python", "sglang"), "/root/lane/katt2/python/sglang")
    .add_local_dir(os.path.join(SERVING, "lanes", "katt2", "test"), "/root/lane/katt2/test")
    .add_local_dir(os.path.join(SERVING, "lanes", "katt2_tools"), TOOLS)
    .add_local_dir(os.path.join(SERVING, "lanes", "katt_tools"), "/root/lane/katt_tools")
    .add_local_file(os.path.join(KT, "submission.py"), "/root/kt/muse_attention_decode/submission.py")
    .add_local_file(os.path.join(KT, "reference.py"), "/root/kt/muse_attention_decode/reference.py")
    .add_local_file(os.path.join(KT, "checks.py"), "/root/kt/muse_attention_decode/checks.py")
)


def _stream(cmd, log, limit, env=None):
    print(f"===== {' '.join(cmd)}  (limit {limit}s) {time.strftime('%H:%M:%SZ', time.gmtime())}", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=TOOLS, env=env)
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


def _untar():
    if os.path.exists(CACHE_TAR):
        os.makedirs(CACHE, exist_ok=True)
        subprocess.run(["tar", "-xzf", CACHE_TAR, "-C", CACHE], check=False)
        print("JIT cache restored", flush=True)


@app.function(image=image, volumes={"/vol": vol}, timeout=30 * 60, cpu=8, memory=32 * 1024)
def cpu_compile(stages: str = "4,3", variants: str = "v4,v5p"):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/katt2/{ts}_compile"
    os.makedirs(out, exist_ok=True)
    subprocess.run("nvcc --version | tail -2", shell=True)
    rc = _stream([sys.executable, "katt2_compile.py", "--stages", stages, "--variants", variants],
                 f"{out}/compile.log", 25 * 60)
    os.makedirs(CACHE, exist_ok=True)
    subprocess.run(["tar", "-czf", CACHE_TAR, "-C", CACHE, "."], check=False)
    vol.commit()
    return rc


@app.function(image=image, volumes={"/vol": vol}, timeout=30 * 60, cpu=8, memory=32 * 1024)
def cputests(files: str):
    os.makedirs("/tmp/cudastub", exist_ok=True)
    subprocess.run("ln -sf $(ls /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/targets/*/lib/stubs/libcuda.so "
                   "2>/dev/null | head -1) /tmp/cudastub/libcuda.so.1", shell=True)
    env = dict(os.environ, LD_LIBRARY_PATH="/tmp/cudastub:" + os.environ.get("LD_LIBRARY_PATH", ""))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files.split(",")], env=env,
                       cwd="/root/lane/katt2")
    print("tests rc", r.returncode, flush=True)
    return r.returncode


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=20 * 60, cpu=8, memory=64 * 1024)
def run(only: str = "core", args: str = ""):
    _untar()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/katt2/{ts}"
    os.makedirs(out, exist_ok=True)
    subprocess.run("nvidia-smi -L; nvcc --version | tail -2", shell=True)
    t0 = time.time()
    for part in only.split(","):
        left = int(19 * 60 - (time.time() - t0) - 20)
        if left < 60:
            print("no time left for", part, flush=True)
            break
        if part == "pytest":
            cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                   "/root/lane/katt2/test/registered/unit/layers/attention/test_fp8_verify_attention.py"]
        else:
            cmd = [sys.executable, "katt2_ubench.py", "--part", part, "--out", f"{out}/{part}.json", *args.split()]
        _stream(cmd, f"{out}/{part}.log", limit=left)
        vol.commit()
    print("results:", out, flush=True)


@app.function(image=image, volumes={"/vol": vol}, timeout=15 * 60, cpu=4, memory=16 * 1024)
def sass(pattern: str = "fp8_verify_attention_4_true"):
    """CPU: SASS statistics of a cached JIT build (run cpu_compile first)."""
    _untar()
    so = subprocess.run(f"find {CACHE} -name '*{pattern}*.so' | head -1", shell=True, capture_output=True,
                        text=True).stdout.strip()
    print("module:", so, flush=True)
    script = (f"cuobjdump -sass {so} > /tmp/k.sass; "
              "for k in attention_kernel combine_kernel; do echo == $k; "
              "awk -v k=$k '/Function :/{f=($0 ~ k)} f' /tmp/k.sass > /tmp/f.sass; "
              "echo instructions $(grep -cE '^ +/\\*[0-9a-f]{4}\\*/' /tmp/f.sass); "
              "for op in QGMMA MUFU.EX2 F2FP SHFL BAR SYNCS LDS STS LDGSTS LDG STL LDL ST.E LD.E FMNMX FFMA FADD FMUL PRMT WARPSYNC; do "
              "echo $op $(grep -c \" $op\" /tmp/f.sass); done; done")
    subprocess.run(script, shell=True)
