"""Lane D 1-GPU gates on ONE H200 in the judge image (Modal): G5 bit-exactness + graph replay + microbench, G9 tile sweep.

  modal run harness/modal_dpkg_ubench.py            # both, ~10 min, ~$1-2 (hard cap 25 min)
Prints the scripts' output; also saved on the volume under out/dpkg_ubench/<ts>/.
"""
import os, subprocess, sys, time
import modal

HERE = os.path.dirname(os.path.abspath(__file__)); SERVING = os.path.dirname(HERE)
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
app = modal.App("okc-dpkg-ubench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"PYTHONPATH": "/root/lane/dpkg/python"})
    .add_local_dir(os.path.join(SERVING, "lanes", "dpkg", "python", "sglang"), "/root/lane/dpkg/python/sglang")
    .add_local_dir(os.path.join(SERVING, "lanes", "dpkg_tools"), "/root/lane/dpkg_tools")
)

@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=25 * 60, cpu=8, memory=64 * 1024)
def run(quick: str = "", only: str = ""):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()); out = f"/vol/out/dpkg_ubench/{ts}"; os.makedirs(out, exist_ok=True)
    subprocess.run("nvidia-smi -L", shell=True)
    rc = 0
    for name, cmd in [
        ("g5", ["python", "/root/lane/dpkg_tools/g5_fa3_kv_store_gpu.py", "--check", "--bench"]),
        ("g9", ["python", "/root/lane/dpkg_tools/g9_bf16_moe_sweep.py"] + (["--quick"] if quick else [])),
    ]:
        if only and name != only: continue
        print(f"===== {name}: {' '.join(cmd)}", flush=True)
        with open(f"{out}/{name}.txt", "w") as f:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd="/root/lane")
            for line in p.stdout:
                print(line, end="", flush=True); f.write(line)
            p.wait(); print(f"===== {name} rc={p.returncode}", flush=True); f.write(f"\nrc={p.returncode}\n"); rc |= p.returncode
    vol.commit(); print("results:", out)
    return rc
