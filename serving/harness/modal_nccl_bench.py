"""C2 (NCCL RS/AG settings) and C4 gate 1 (collective/compute overlap) on ONE 8xH200 node in the judge image.

  modal run harness/modal_nccl_bench.py            # all configs, ~5 min of GPU time; hard cap 12 min (~$7 worst case at 8xH200)
  modal run harness/modal_nccl_bench.py --quick    # default-env nccl + overlap only

Each config is its own torchrun (NCCL reads its env at init). Results: printed RESULT lines and out/nccl_bench/<ts>/ on the volume.
"""
import os
import subprocess
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIGEST = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"

app = modal.App("okc-nccl-bench")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = modal.Image.from_registry(IMAGE_DIGEST, add_python=None).add_local_file(
    os.path.join(HERE, "nccl_overlap_bench.py"), "/root/bench/nccl_overlap_bench.py"
)

# (tag, env overrides) for --mode nccl; the first is the stock environment the judge launches with
NCCL_CONFIGS = [
    ("default", {}),
    ("ring", {"NCCL_ALGO": "Ring"}),
    ("nvls", {"NCCL_ALGO": "NVLS", "NCCL_NVLS_ENABLE": "1"}),
    ("tree", {"NCCL_ALGO": "Tree"}),
    ("ch16", {"NCCL_MIN_NCHANNELS": "16", "NCCL_MAX_NCHANNELS": "16"}),
    ("ch32", {"NCCL_MIN_NCHANNELS": "32", "NCCL_MAX_NCHANNELS": "32"}),
    ("nvls_ch32", {"NCCL_ALGO": "NVLS", "NCCL_NVLS_ENABLE": "1", "NCCL_MIN_NCHANNELS": "32", "NCCL_MAX_NCHANNELS": "32"}),
]
OVERLAP_CONFIGS = [
    ("default", {}),
    ("ctas", {"NCCL_MIN_CTAS": "16", "NCCL_MAX_CTAS": "32"}),
    ("nvls", {"NCCL_ALGO": "NVLS", "NCCL_NVLS_ENABLE": "1"}),
]


@app.function(image=image, gpu="H200:8", volumes={"/vol": vol}, timeout=12 * 60, cpu=16, memory=64 * 1024)
def bench(quick: bool = False):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"/vol/out/nccl_bench/{ts}"
    os.makedirs(out, exist_ok=True)
    base = dict(os.environ, PYTHONUNBUFFERED="1")
    log = open(f"{out}/results.txt", "a")

    def run(mode, tag, env, extra_env=None):
        e = dict(base, **env, **(extra_env or {}))
        cmd = ["torchrun", "--nproc_per_node", "8", "/root/bench/nccl_overlap_bench.py", "--mode", mode, "--tag", tag]
        print(f"=== {mode} {tag} {env}", flush=True)
        r = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=240)
        lines = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
        for l in lines:
            print(l, flush=True); log.write(l + "\n")
        if tag.endswith("_debug"):
            open(f"{out}/nccl_debug_{tag}.txt", "w").write(r.stderr or "")
            picks = sorted({l.split("NCCL INFO")[-1].strip()[:160] for l in (r.stderr or "").splitlines() if "NCCL INFO" in l and ("algo" in l.lower() or "channel" in l.lower() or "nvls" in l.lower())})
            for l in picks[:40]:
                print("NCCLINFO " + l, flush=True); log.write("NCCLINFO " + l + "\n")
        if r.returncode != 0 or not lines:
            tail = (r.stderr or "")[-1500:]
            print(f"rc={r.returncode} stderr tail:\n{tail}", flush=True)
            log.write(f"FAIL {mode} {tag} rc={r.returncode}\n{tail}\n")
        log.flush()

    # one NCCL_DEBUG pass to see what the stock env picks (algo/proto/channels) at these sizes
    run("nccl", "default_debug", {}, {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "TUNING,GRAPH,NVLS"})
    # overlap and symm first: they decide the prefill overlap lane, so they must not be the part a timeout cuts
    for tag, env in (OVERLAP_CONFIGS[:1] if quick else OVERLAP_CONFIGS):
        run("overlap", tag, env)
        vol.commit()
    run("symm", "default", {})
    vol.commit()
    for tag, env in (NCCL_CONFIGS[:1] if quick else NCCL_CONFIGS):
        run("nccl", tag, env)
        vol.commit()
    log.close()
    vol.commit()
    print("results in", out, flush=True)


@app.local_entrypoint()
def main(quick: bool = False):
    bench.remote(quick=quick)
