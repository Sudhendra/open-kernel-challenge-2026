"""KREP2 expert copies: 4-GPU microbench of one MoE layer's expert phase (Modal, judge image).

  modal run harness/modal_krep2_bench.py::dry       # CPU only (free): imports, route data, copy table, steer reference
  modal run harness/modal_krep2_bench.py::bench     # 4 GPUs, function timeout 10 min (~$3.5 worst case)

Mounts lanes/<KREP2_TREE> (default krep2) python/sglang and lanes/krep2_tools. The bench needs no communicator (8
virtual EP ranks, 2 per GPU, each timed on its own GPU); gloo only gathers the results. Every RESULT line is printed
and appended to /vol/out/krep2_bench/<ts>/results.txt (committed) as soon as it appears.
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
TREE = os.environ.get("KREP2_TREE", "krep2")
PASS_ENV = {k: v for k, v in os.environ.items() if k.startswith("KREP2_") and k != "KREP2_TREE"}

app = modal.App("okc-krep2-bench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .add_local_dir(os.path.join(SERVING, "lanes", TREE, "python", "sglang"), "/root/lane/python/sglang")
    .add_local_file(os.path.join(SERVING, "lanes", "krep2_tools", "krep2_bench.py"), "/root/gate/krep2_bench.py")
    .add_local_file(
        os.path.join(SERVING, "lanes", "krep2_tools", "routes_dumpA_placed.pt"), "/root/gate/routes_dumpA_placed.pt"
    )
)


def _stream(cmd, env, limit_s, out_dir):
    """Run cmd; echo every line, append RESULT lines to results.txt and commit them at once; kill at limit_s."""
    os.makedirs(out_dir, exist_ok=True)
    t0, tail, rc = time.time(), [], None
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    with open(f"{out_dir}/results.txt", "a") as f:
        for line in p.stdout:
            print(line, end="", flush=True)
            tail = (tail + [line])[-80:]
            if line.startswith("RESULT"):
                f.write(line)
                f.flush()
                vol.commit()
            if time.time() - t0 > limit_s:
                p.kill()
                f.write(f"FAIL killed at {limit_s} s\n")
                break
        rc = p.wait()
        if rc != 0:
            f.write(f"FAIL rc={rc}\n" + "".join(tail[-40:]))
    vol.commit()
    print(f"rc={rc} after {time.time() - t0:.0f} s; results in {out_dir}", flush=True)
    return rc


@app.function(image=image, gpu="H200:4", volumes={"/vol": vol}, timeout=10 * 60, cpu=12, memory=96 * 1024)
def bench4(env_extra: dict):
    env = dict(os.environ, PYTHONPATH="/root/lane/python", PYTHONUNBUFFERED="1", **env_extra)
    out = "/vol/out/krep2_bench/" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    _stream(["torchrun", "--nproc_per_node", "4", "/root/gate/krep2_bench.py"], env, 560, out)


@app.function(image=image, volumes={"/vol": vol}, timeout=15 * 60, cpu=8, memory=32 * 1024)
def dry_run(env_extra: dict):
    env = dict(os.environ, PYTHONPATH="/root/lane/python", PYTHONUNBUFFERED="1", **env_extra)
    os.makedirs("/tmp/cudastub", exist_ok=True)
    subprocess.run(
        "ln -sf $(ls /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/targets/*/lib/stubs/libcuda.so"
        " 2>/dev/null | head -1) /tmp/cudastub/libcuda.so.1",
        shell=True,
    )
    env["LD_LIBRARY_PATH"] = "/tmp/cudastub:" + os.environ.get("LD_LIBRARY_PATH", "")
    out = "/vol/out/krep2_bench/" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_dry"
    rc = _stream([sys.executable, "/root/gate/krep2_bench.py", "--dry"], env, 600, out)
    if rc:
        raise RuntimeError(f"dry rc={rc}")


@app.local_entrypoint()
def dry():
    dry_run.remote(PASS_ENV)


@app.local_entrypoint()
def bench():
    bench4.remote(PASS_ENV)
