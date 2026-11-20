"""Lane KPATT3: kMode variants of the Muse FP8 prefill attention kernel vs FA3, ONE H200 in the judge image (Modal).

  modal run harness/modal_kpatt3_ubench.py::bench --extra "--modes 0,1,2,4,3"     # 10-min cap
  modal run harness/modal_kpatt3_ubench.py::bench --tests test/registered/kernels/ops/attention/test_muse_spark_prefill_attention_kernel.py

Mounts lanes/kpatt3 and lanes/kpatt3_tools. Results: /vol/out/kpatt3/<ts>/ and stdout.
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
CACHE = "/root/kcache"

app = modal.App("okc-kpatt3-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"SGLANG_JIT_CACHE_DIR": f"{CACHE}/jit", "TORCH_EXTENSIONS_DIR": f"{CACHE}/torch", "MAX_JOBS": "16",
          "TORCH_CUDA_ARCH_LIST": "9.0a", "PYTHONPATH": "/root/lane/kpatt3/python"})
    .add_local_dir(os.path.join(SERVING, "lanes", "kpatt3", "python", "sglang"), "/root/lane/kpatt3/python/sglang")
    .add_local_dir(os.path.join(SERVING, "lanes", "kpatt3", "test"), "/root/lane/kpatt3/test")
    .add_local_dir(os.path.join(SERVING, "lanes", "kpatt3_tools"), "/root/lane/kpatt3_tools")
)
TOOLS = "/root/lane/kpatt3_tools"


def _run(cmd, limit, log=None):
    print("+", " ".join(cmd), flush=True)
    t0 = time.time()
    f = open(log, "w") if log else None
    p = subprocess.Popen(cmd, cwd=TOOLS, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
            if f:
                f.write(line)
                f.flush()
            if time.time() - t0 > limit:
                p.kill()
        rc = p.wait()
    finally:
        if f:
            f.close()
    if time.time() - t0 > limit:
        rc = 124
    print(f"rc={rc} after {time.time() - t0:.0f}s", flush=True)
    return rc


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=10 * 60, cpu=8, memory=64 * 1024)
def bench(extra: str = "", tests: str = ""):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/kpatt3/{ts}_bench"
    os.makedirs(out, exist_ok=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.max.sm,power.limit", "--format=csv"], check=False)
    rc = 0
    if extra != "none":
        rc = _run([sys.executable, "kpatt3_bench.py", "--out", f"{out}/bench.json", *extra.split()], 7 * 60, f"{out}/run.log")
    if tests:
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *[f"/root/lane/kpatt3/{f}" for f in tests.split(",")]]
        rc |= _run(cmd, 4 * 60, f"{out}/pytest.log")
    vol.commit()
    print("results in", out, flush=True)
    if os.path.exists(f"{out}/bench.json"):
        print("BENCH_JSON " + open(f"{out}/bench.json").read().replace("\n", " "), flush=True)
    if rc:
        raise RuntimeError(f"bench rc={rc}")


@app.function(image=image, timeout=10 * 60, cpu=16, memory=32 * 1024)
def cpu_compile(modes: str = "0,1,2,4,3"):
    """CPU only: build each kMode module with the judge nvcc (dlopen fails without a GPU; the build is what counts)."""
    code = (
        "import os,sys,time\n"
        "os.environ['SGLANG_JIT_LOG_RESOURCE_USAGE']='1'\n"
        "from sglang.kernels.jit.utils import override_jit_cuda_arch\n"
        "from sglang.kernels.jit.utils.compile import ninja\n"
        "ninja._log_resource_usage = lambda name, output: print('=== ptxas', name, '===', output, flush=True)\n"
        "from sglang.kernels.ops.attention import muse_spark_prefill_attention as ops\n"
        "rc=0\n"
        "with override_jit_cuda_arch(9, 0, 'a'):\n"
        f"  for m in [{modes}]:\n"
        "    t0=time.time()\n"
        "    try:\n"
        "      ops._module(8, m); print('BUILD_OK', m, flush=True)\n"
        "    except Exception as e:\n"
        "      s=str(e); ok=('libcuda' in s or 'dlopen' in s.lower() or 'CUDA driver' in s or 'no CUDA' in s.lower())\n"
        "      print('BUILD_'+('LOADFAIL_ONLY' if ok else 'ERROR'), m, round(time.time()-t0), s[-3000:], flush=True); rc|= (not ok)\n"
        "sys.exit(rc)\n"
    )
    rc = _run([sys.executable, "-c", code], 9 * 60)
    if rc:
        raise RuntimeError(f"compile rc={rc}")
