"""Modal driver for the serving track: weights on a persistent Volume, sessions on H200:8.

  modal run harness/modal_serve.py::smoke      # ~1 min on H200:8: topology, image, volume
  modal run harness/modal_serve.py::download   # CPU only: 533 GB target + drafter into the volume
  modal run --detach harness/modal_serve.py::session   # one 8xH200 session running profile_session.sh
  modal volume get okc-muse-weights out/<ts>.tgz .   # fetch results

Always launch `session` with --detach (see harness/README.md): the function takes no stdin, writes
its log to the volume (out/<ts>/session.log), and commits the volume itself, so the session runs
to completion and keeps its results even if the local client disconnects or the laptop sleeps.

Needs a Modal secret named `huggingface` holding HF_TOKEN (create it yourself:
`modal secret create huggingface HF_TOKEN=...`). The image is the judge's SGLang image by
digest, so what runs here is what the judge runs; our checkout is put first on PYTHONPATH.
"""
import os
import signal
import subprocess
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
# The judge's image by digest: lmsysorg/sglang:v0.5.23, SGLang 297e54fd (organisers' pin, posted Nov 9).
# Override per launch to run on a different base (the code comes from the serving/sglang and
# serving/sglang_stock checkouts, so put them at matching commits; README has the details):
#   OKC_IMAGE=lmsysorg/sglang:dev@sha256:... modal run --detach harness/modal_serve.py::session
IMAGE_DIGEST = os.environ.get(
    "OKC_IMAGE", "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
)
TARGET = "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"
DRAFTER = "RadixArk/Muse-Spark-Open-DFlash2"

app = modal.App("okc-serving")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

STOCK_PY = os.path.join(SERVING, "sglang_stock", "python", "sglang")   # git worktree at the image's commit
# The patched tree: serving/sglang by default; OKC_PATCHED=lanes/stack_fg mounts a lane worktree instead.
PATCHED_ROOT = os.path.join(SERVING, os.environ.get("OKC_PATCHED", "sglang"))
UNIT_TEST_DIR = os.path.join("test", "registered", "unit", "mem_cache")  # the PRE_TESTS files (~4 MB), per tree

# OKC_EXTRA_TREES="l1=lanes/l1_fghil,keager=lanes/keager,g2=lanes/g2": more patched trees, mounted below as
# /root/serving/sglang_<name>/python/sglang; variants select them with SRC=<name> (profile_session.sh reads
# SRC_<NAME>). The SRC_* values are baked into the image env here because this module is re-imported inside the
# container without OKC_EXTRA_TREES (a dict built at import time is empty there: Nov 13 09:37 CST session died on
# "bad SRC in l1"), and Modal refuses an .env() build step after add_local_dir.
EXTRA_TREE_DIRS = {}
for _kv in filter(None, os.environ.get("OKC_EXTRA_TREES", "").split(",")):
    _n, _w = _kv.split("=", 1)
    _py = os.path.join(SERVING, _w, "python", "sglang")
    if not os.path.isdir(_py):
        sys.exit(f"OKC_EXTRA_TREES: {_py} missing")
    EXTRA_TREE_DIRS[_n] = _py
EXTRA_TREES = {"SRC_" + n.upper().replace("-", "_"): f"/root/serving/sglang_{n}/python" for n in EXTRA_TREE_DIRS}
judge_image = (
    modal.Image.from_registry(IMAGE_DIGEST, add_python=None)
    .env({"HF_HOME": "/vol/hf", "HF_HUB_OFFLINE": "1", "PYTHONPATH": "/root/serving/sglang/python", **EXTRA_TREES})
    # NVTX ranges for SGLANG_ENABLE_NVTX_SCHEDULER; installed only if the judge image lacks it.
    # (Observability only: profile_session.sh still checks `import nvtx` before enabling it.)
    .run_commands("python -c 'import nvtx' 2>/dev/null || pip install -q nvtx")
    .add_local_dir(os.path.join(SERVING, "harness"), "/root/serving/harness")
    .add_local_dir(os.path.join(PATCHED_ROOT, "python", "sglang"), "/root/serving/sglang/python/sglang")
    .add_local_dir(os.path.join(PATCHED_ROOT, UNIT_TEST_DIR), "/root/serving/sglang/" + UNIT_TEST_DIR)
)
# EXTRA_TESTS (comma-separated test files relative to the patched tree root, e.g. a lane's GPU kernel
# test) run once on the patched tree before boot when PRE_TESTS=1; their folders are mounted here. The env var only
# selects what to MOUNT (read locally); pass --extra-tests <same list> to session() so the container runs them
# (Nov 12: the env alone never reached the container, so the R2 session A ran without the kernel test).
EXTRA_TESTS = os.environ.get("OKC_EXTRA_TESTS", "")
for _t in filter(None, EXTRA_TESTS.split(",")):
    _d = os.path.dirname(_t)
    if os.path.isfile(os.path.join(PATCHED_ROOT, _t)):
        judge_image = judge_image.add_local_dir(os.path.join(PATCHED_ROOT, _d), "/root/serving/sglang/" + _d)
    else:
        print(f"warning: OKC_EXTRA_TESTS file {_t} missing under {PATCHED_ROOT}; skipped", file=sys.stderr)
# Optional second tree for SRC=stock (profile_session.sh falls back to sglang/python without it). The
# three unit-test files import only sglang.* (checked Nov 8), so the mem_cache test folder is all that
# is needed from test/; when the stock worktree lacks it, the script runs the patched tree's test files.
if os.path.isdir(STOCK_PY):
    judge_image = judge_image.add_local_dir(STOCK_PY, "/root/serving/sglang_stock/python/sglang")
    stock_tests = os.path.join(SERVING, "sglang_stock", UNIT_TEST_DIR)
    if os.path.isdir(stock_tests):
        judge_image = judge_image.add_local_dir(stock_tests, "/root/serving/sglang_stock/" + UNIT_TEST_DIR)
    else:
        print(f"warning: {stock_tests} missing; PRE_TESTS runs the patched tree's test files against the stock tree", file=sys.stderr)
else:
    print(f"warning: {STOCK_PY} missing; SRC=stock will use serving/sglang (set up the worktree first)", file=sys.stderr)
for _n, _py in EXTRA_TREE_DIRS.items():   # same layout as make_bundle.sh EXTRA_TREES for the Nebius path
    judge_image = judge_image.add_local_dir(_py, f"/root/serving/sglang_{_n}/python/sglang")
# HF_HOME must be set before huggingface_hub is imported (it reads it at import), so it goes in the image env.
dl_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=dl_image, volumes={"/vol": vol}, secrets=[hf_secret], timeout=6 * 3600, cpu=4)
def download():
    from huggingface_hub import snapshot_download

    for repo in (DRAFTER, TARGET):
        print("downloading", repo, flush=True)
        path = snapshot_download(repo, max_workers=16)
        # Modal resolves /vol to /__modal/volumes/<id>, so check the realpath, not the prefix.
        assert os.path.realpath(path).startswith(os.path.realpath("/vol/hf")), f"not on the volume: {path}"
        print("done", repo, "->", path, flush=True)
        vol.commit()
    print(subprocess.run(["du", "-sh", "/vol/hf"], capture_output=True, text=True).stdout)


@app.function(image=judge_image, gpu="H200:8", volumes={"/vol": vol}, timeout=600)
def smoke():
    for cmd in (
        "nvidia-smi --query-gpu=index,name,power.limit,memory.total --format=csv",
        "nvidia-smi topo -m",
        "python -c 'import sglang, torch; print(sglang.__file__, torch.__version__, torch.cuda.device_count())'",
        "python -c 'import aiohttp; print(aiohttp.__version__)'",
        "du -sh /vol/hf/hub/* 2>/dev/null || echo 'volume empty'",
        "nvcc --version | tail -1",
    ):
        print("$", cmd, flush=True)
        print(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout, flush=True)


@app.function(image=judge_image, gpu="H200:8", volumes={"/vol": vol}, timeout=8 * 3600, memory=512 * 1024, cpu=32)
def session(
    trace: str = "",
    extra_args: str = "--hicache-size 42",
    prof_conc: str = "1 4 16 32",
    skip_replay: str = "",
    max_min: int = 75,
    variants: str = "",
    stock_mode: str = "",
    pre_tests: str = "",
    ar_bench: str = "",
    proxy_rungs: str = "",
    stock_self: str = "",
    stock_self_restart: str = "",
    extra_tests: str = "",
    boot_timeout_min: int = 18,
    server_env_extra: str = "",
    queue: str = "",
    queue_idle_min: int = 30,
    queue_hold_min: int = 360,
    reuse_out: str = "",
    proxy_shape: str = "judge",
    proxy_corpus: str = "",
    mem_gate: str = "1",
    mem_probe: str = "1",
    mem_loaded_probe: str = "",
    proxy_client: str = "",
):
    """profile_session.sh with results committed to the volume under out/.

    Modal's sandbox has no cgroup-v1 memory.stat, so SGLang cannot auto-size the HiCache host
    pool, and the judge's --hicache-size 160 (GB per rank) does not fit 512 GiB, so extra_args
    overrides it (it comes after the judge flags; last flag wins). Default 42 GB per rank (Nov 15,
    research/proxy_rebase_2026-11-15.md section 2): the smallest size at which STOCK keeps all 32
    judge-shape conversations (32 x ~86.8k tokens + 10%) in its host full-KV pool (30.3% of the
    hicache size goes to full KV at 4096 B/token/rank, the rest to SWA), i.e. cache-hot at c32 like
    the judge. 8 x 42 GB = 336 GB pinned of 512 GiB. The old proxy used 16 (pass
    --extra-args "--hicache-size 16" with --proxy-shape short to reproduce a pre-Nov-15 run).
    proxy_shape: judge (13 turns, the judge's shape) or short (the old 4-turn proxy).
    proxy_corpus: directory in the image for ladder.py --corpus (default sglang/srt), e.g. a site-packages tree.
    mem_gate / mem_probe: profile_session.sh MEM_GATE / MEM_PROBE (per-variant GPU memory over stock).
    mem_loaded_probe: profile_session.sh MEM_LOADED_PROBE, e.g. "90,150" (logprob probes mid-c32; off by default).
    proxy_client: profile_session.sh PROXY_CLIENT: "" / chat (text to /v1/chat/completions) or ids (input_ids to /generate).

    Detach-safe: no stdin, the script tees its own log to the volume, and a SIGTERM to this
    function (modal app stop, preemption) is forwarded to the script, whose EXIT trap tars the
    results before the volume is committed here. max_min is the script's watchdog budget; the
    Modal timeout above (3 h) is only a backstop: keep max_min <= 150 so the watchdog fires first.

    pre_tests="1" runs the three HiCache CPU unit tests per tree before boot; ar_bench="1" runs
    harness/ar_bench.py while the GPUs are free. variants accepts SRC=stock|patched and ARGS=<flags>
    keys (see README). proxy_rungs="CxRxS,..." overrides the proxy preset (16x1x2) for stock and
    every variant replay, and the budget estimates scale with it. stock_self="1" replays the stock
    proxy once with the same prompts for the agreement noise floor (agree.py --self);
    stock_self_restart="1" restarts the stock server first, so both sides are fresh boots.
    boot_timeout_min stops a boot whose /health is not up by then, or sooner on a fatal server.log
    line (the stock boot then exits 3, a variant gets FAILED.txt). server_env_extra="K=V,K=V" is
    added to every server launch, stock and variants.
    `modal run` exposes every parameter as --with-hyphens (--pre-tests 1).

    queue="queue/<name>" keeps the container after the planned run and chains more experiments pushed to the
    volume with `harness/modal_queue.py push` (one .env file per item: VARIANTS, PROXY_RUNGS, PROF_CONC, MAX_MIN,
    EXTRA_ARGS, SERVER_ENV_EXTRA, STOCK_REUSE, PROF=warm knobs WARM_*). Each item runs profile_session.sh with STOCK_MODE=reuse against the
    first run's proxy.json (no stock boot, no tests), results in their own out/<ts>/. The container exits after
    queue_idle_min minutes with no item, after queue_hold_min minutes of queue time, or when <queue>/STOP exists.
    Only trees mounted at launch are usable (SRC=<name> from OKC_EXTRA_TREES). The 8 h function timeout is the
    hard cap; the H200:8 bill runs while the container idles, so push items promptly or stop it."""
    env = dict(
        os.environ,
        OUT_ROOT="/vol/out",
        SGLANG_SRC_MODE="pythonpath",
        EXTRA_ARGS=extra_args,
        PROF_CONC=prof_conc,
        SKIP_REPLAY=skip_replay,
        MAX_MIN=str(max_min),
        VARIANTS=variants,
        STOCK_MODE=stock_mode,
        PRE_TESTS=pre_tests,
        AR_BENCH=ar_bench,
        PROXY_RUNGS=proxy_rungs,
        STOCK_SELF=stock_self,
        STOCK_SELF_RESTART=stock_self_restart,
        BOOT_TIMEOUT_MIN=str(boot_timeout_min),
        SERVER_ENV_EXTRA=server_env_extra,
        PROXY_SHAPE=proxy_shape,
        PROXY_CORPUS=proxy_corpus,
        MEM_GATE=mem_gate,
        MEM_PROBE=mem_probe,
        MEM_LOADED_PROBE=mem_loaded_probe,
        PROXY_CLIENT=proxy_client,
        EXTRA_TESTS=extra_tests or EXTRA_TESTS,
        **EXTRA_TREES,
    )
    cmd = ["bash", "/root/serving/harness/profile_session.sh"] + ([trace] if trace else [])
    if reuse_out:
        # No planned run: serve the queue straight away against an earlier run's stock proxy (agreement only;
        # throughput reads come from each item's own control variant on this node).
        first_out = reuse_out if reuse_out.startswith("/") else os.path.join("/vol/out", reuse_out); rc = 0
        print("reuse_out", first_out, "exists" if os.path.isfile(os.path.join(first_out, "proxy.json")) else "MISSING proxy.json", flush=True)
    else:
        rc, first_out = _run_once(cmd, env)
        print("session exit", rc, "out", first_out, flush=True)
    if queue:
        _serve_queue(queue, env, first_out, queue_idle_min, queue_hold_min)
    if rc != 0:
        raise RuntimeError(f"profile_session.sh exited {rc}")
    return rc


def _newest_out():
    ds = sorted(d for d in os.listdir("/vol/out") if d.endswith("Z") and os.path.isdir(os.path.join("/vol/out", d)))
    return os.path.join("/vol/out", ds[-1]) if ds else ""


def _run_once(cmd, env):
    """Run profile_session.sh once, forwarding TERM; returns (rc, the run's out dir)."""
    before = _newest_out()
    proc = subprocess.Popen(cmd, cwd="/root/serving", env=env, stdin=subprocess.DEVNULL)

    def forward(signum, _frame):
        print(f"session got signal {signum}, forwarding TERM to profile_session.sh", flush=True)
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    try:
        rc = proc.wait()
    finally:
        try:
            proc.wait(timeout=300)  # let the EXIT trap finish the tarball if we were interrupted
        except subprocess.TimeoutExpired:
            proc.kill()
        _vol_sync("commit")
    after = _newest_out()
    return rc, (after if after != before else "")


def _parse_item(path):
    kv = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv


def _vol_sync(what):
    """vol.reload()/commit() with the open-file conflict handled: profile_session.sh leaves `tee -a .../session.log`
    (process substitution) alive after it exits, and Modal refuses to reload while a file is open (Nov 13 15:50 CST:
    this killed the first queue-mode container right after its planned run). Kill the writers, retry."""
    import time
    for attempt in range(6):
        try:
            (vol.reload if what == "reload" else vol.commit)(); return True
        except Exception as e:  # modal.exception.ConflictError and friends
            print(f"[queue] volume {what} failed ({type(e).__name__}: {str(e)[:160]}); killing stray writers, retry {attempt + 1}", flush=True)
            subprocess.run("pkill -f 'tee -a /vol/out' ; pkill -f 'tee /vol/out' ; sync", shell=True)
            time.sleep(5)
    print(f"[queue] volume {what} still failing; continuing without it", flush=True); return False


def _gpus_quiet(timeout_s=600):
    """Between queue items: the previous item's server must be gone from the GPUs before the next boots. Nov 13 23:44 UTC
    the 03_rlg control booted with 68 GB free per GPU (the 02_d server was still unloading) and the next variant's rank 0
    aborted at init; host RAM went the same way. Kill leftovers, then wait for every GPU to be under 2 GB used."""
    import time
    subprocess.run("pkill -f 'sglang.launch_server' ; pkill -f 'sglang::scheduler' ; pkill -f 'tee -a /vol/out'", shell=True)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            used = [int(x) for x in subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).split()]
        except Exception as e:
            print(f"[queue] nvidia-smi failed: {e}", flush=True); return
        if max(used) < 2048:
            print(f"[queue] GPUs quiet after {time.time() - t0:.0f}s: used MiB {used}", flush=True); return
        if int(time.time() - t0) % 60 == 0: print(f"[queue] waiting for GPUs to free: used MiB {used}", flush=True)
        time.sleep(5)
    print(f"[queue] GPUs still busy after {timeout_s}s: used MiB {used}; running anyway", flush=True)


def _serve_queue(queue, base_env, first_out, idle_min, hold_min):
    import glob, shutil, time
    qdir = os.path.join("/vol", queue); done = os.path.join(qdir, "done"); os.makedirs(done, exist_ok=True); _vol_sync("commit")
    t0 = time.time(); last = time.time(); n = 0
    print(f"[queue] serving {qdir}: idle {idle_min} min, hold {hold_min} min; first run {first_out}", flush=True)
    while True:
        _vol_sync("reload")
        if os.path.exists(os.path.join(qdir, "STOP")):
            print("[queue] STOP found", flush=True); break
        items = sorted(glob.glob(os.path.join(qdir, "*.env")))
        if not items:
            if (time.time() - last) / 60 > idle_min: print("[queue] idle timeout", flush=True); break
            if (time.time() - t0) / 60 > hold_min: print("[queue] hold timeout", flush=True); break
            time.sleep(60); continue
        item = items[0]; kv = _parse_item(item); name = os.path.basename(item)[:-4]
        reuse = kv.pop("STOCK_REUSE", "") or first_out
        if reuse and not reuse.startswith("/"): reuse = os.path.join("/vol/out", reuse)
        env = dict(base_env, STOCK_MODE="reuse", STOCK_REUSE_DIR=reuse, PRE_TESTS="", AR_BENCH="", STOCK_SELF="", EXTRA_TESTS="")
        for k in ("VARIANTS", "PROXY_RUNGS", "PROF_CONC", "MAX_MIN", "EXTRA_ARGS", "SERVER_ENV_EXTRA", "PROXY_SHAPE", "PROXY_CORPUS", "MEM_GATE", "MEM_PROBE", "MEM_LOADED_PROBE",
                  "PROXY_CLIENT",   # chat (default) | ids: ladder.py --client (modal_queue.py push --proxy-client)
                  # PROF=warm knobs (profile_session.sh profile_loads; modal_queue.py push --prof-env); keep in step with modal_queue.WARM_KEYS
                  "WARM_TURNS", "WARM_EXTEND", "WARM_SKIP", "WARM_MAX_STEPS", "WARM_MIN_CACHED", "WARM_WITH_STACK", "WARM_PYSPY", "WARM_PYSPY_S", "WARM_PYSPY_ARGS", "WARM_WAIT_S"):
            if k in kv: env[k] = kv[k]
        print(f"[queue] item {name}: {kv} (reuse {reuse})", flush=True)
        if not os.path.isfile(os.path.join(reuse, "proxy.json")):
            print(f"[queue] item {name}: no proxy.json under {reuse}; skipped", flush=True)
            shutil.move(item, os.path.join(done, name + ".SKIPPED.env")); _vol_sync("commit"); last = time.time(); continue
        _gpus_quiet()
        rc, out = _run_once(["bash", "/root/serving/harness/profile_session.sh"], env); n += 1
        with open(os.path.join(done, name + ".result"), "w") as f: f.write(f"rc={rc}\nout={out}\n")
        shutil.move(item, os.path.join(done, name + ".env")); _vol_sync("commit"); last = time.time()
        print(f"[queue] item {name} done rc={rc} out={out}", flush=True)
    print(f"[queue] exit after {n} items, {(time.time() - t0) / 60:.0f} min", flush=True)
