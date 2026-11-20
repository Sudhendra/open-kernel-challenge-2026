#!/usr/bin/env python3
"""TP8 all-reduce path sweep for Muse Spark message sizes on 8xH200 (SM90).

Run inside the SGLang image (pinned tree first on PYTHONPATH):
    python harness/ar_bench.py --out /vol/out/ar_bench.json [--paths a,b] [--sizes 1,9] [--iters 200]

Every row is one JSON line (stdout and --out) keyed by (path, M, graph):
  path   nccl | pynccl | car_1shot_push | car_1shot_pull | car_2shot_pull | car_2shot_push
         | car_2shot_pull_mc | symm_mem | symm_mem_multimem_raw | symm_mem_2shot_raw
  M      tokens; message = [M, hidden] bf16, so bytes = M * 16 KB at hidden 8192
  graph  false = eager launches; true = 10 calls captured in one CUDA graph, replayed,
         per-call time = replay time / 10 (decode runs this way)
  status ok | skip (size not eligible for that path) | error (exception text in "error")
         | timeout (path exceeded --path-timeout here or at an earlier size; reason in "error")
  median_us / p10_us / p90_us   rank-0 per-call latency over 3 interleaved blocks of --iters
         calls (paths rotate order inside each size to cancel clock drift)
  max_rank_median_us   slowest rank's median: the number a TP8 layer actually waits for
  max_abs_diff   vs torch NCCL on small-integer inputs; must be 0.0 (bf16-exact sums)
  pick_eager / pick_graph   what CustomAllReduceV2._pick_config would choose for this size
         with the shipped SM90 table and the default 16 MB workspace cap
         (SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB); "nccl" = custom AR declines,
         "_mc" = multicast, "_zerocopy" = graph pointer-table pull (no staging copy).
The last line is {"summary": ...}; stdout ends with a table of medians and the winner per
(M, graph) among rows with status ok and max_abs_diff == 0.

Mapping to the SM90 world-8 table (configs/custom_all_reduce_v2.py `_sm90_configs`):
  graph: 1shot_push <= 128 KB (M <= 8); 1shot_pull threshold == 1shot_push, so never picked;
         multicast 2shot_pull 512 KB..128 MB (M >= 32) when the mc handle exists;
         plain 2shot_pull <= 32 MB otherwise (M 9..2048 without multicast).
  eager: 1shot_push <= 128 KB; multicast range 0..128 MB; 2shot_pull <= 128 MB.
  2shot_push threshold is 0 (off) on SM90, so car_2shot_push is forced via override_algo.
  The default workspace cap clips every pull threshold to 16 MB, so in a stock server
  M=1024 (16 MB) is the last custom-AR size; 4096 and 16384 go to pynccl / torch NCCL.
  Read a crossover as: the smallest M where a later path's median beats the earlier one
  in both median and p90; a win under 3% is noise at these iteration counts.
  M=9 (one request + 8 drafts, 144 KB) sits just above the 128 KB 1shot_push ceiling:
  compare car_1shot_push vs car_2shot_pull (or _mc) there first.

Approach: the communicators are built standalone, the way the in-tree kernel test and
bench do it (test/registered/kernels/{ops,benchmark}/communication/*custom_all_reduce.py):
gloo default group + parallel_state.init_world_group, then CustomAllReduceV2(cpu_group)
with explicit workspace sizes and `override_algo` to force each algorithm. No env flag
forces an algo (SGLANG_OPT_CUSTOM_ALL_REDUCE_V2_FORCE does not exist in this tree);
going through tensor_model_parallel_all_reduce would only ever measure the table's pick.
Multicast is the direct kernel entry custom_all_reduce(..., use_multicast=True), the same
call _pick_config makes. symm_mem is TorchSymmMemCommunicator (server flag
--enable-torch-symm-mem; --enable-symm-mem is NCCL window registration, not benched).

Hang handling: a hung kernel blocks inside torch.cuda.synchronize, which no in-process timer
can interrupt, and the GPU stream stays poisoned afterwards. So the parent process watches a
heartbeat from rank 0, SIGKILLs every rank when one path sits in a guarded section longer
than --path-timeout, and respawns at the same (size, mode) without that path. The path gets
a status=timeout row for that point and everything after it.
"""
import argparse
import contextlib
import ctypes
import faulthandler
import json
import logging
import os
import shutil
import signal
import statistics
import sys
import tempfile
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DEFAULT_SIZES = (1, 8, 9, 32, 72, 144, 288, 1024, 4096, 16384)
ALL_PATHS = (
    "nccl",
    "pynccl",
    "car_1shot_push",
    "car_1shot_pull",
    "car_2shot_pull",
    "car_2shot_push",
    "car_2shot_pull_mc",
    "symm_mem",
    "symm_mem_multimem_raw",
    "symm_mem_2shot_raw",
)
DTYPE = torch.bfloat16
MB = 1024 * 1024
_GRAPH_INPUTS_KEEPALIVE = []


class Path:
    """One all-reduce variant: call(operand) -> reduced tensor."""

    def __init__(self, name, call, max_bytes, *, prepare=None, graph=True, capture_ctx=None,
                 shared_buffer=False, note=""):
        self.name = name
        self.call = call
        self.max_bytes = max_bytes
        # prepare(x) returns the operand the call reduces (a clone or a symm-buffer view).
        self.prepare = prepare or (lambda x: x)
        self.graph = graph
        self.capture_ctx = capture_ctx or contextlib.nullcontext
        # True when every call works in place on one symmetric buffer.
        self.shared_buffer = shared_buffer
        self.note = note

    def eligible(self, nbytes):
        if nbytes > self.max_bytes:
            return f"{nbytes} B > path max {self.max_bytes} B"
        return None


# ----------------------------------------------------------------------------------------
# Distributed setup
# ----------------------------------------------------------------------------------------


def _init_dist(rank, world, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    _precompile(rank)
    import sglang.srt.distributed.parallel_state as ps
    from sglang.srt.runtime_context import get_parallel

    ps._WORLD = coord = ps.init_world_group(ranks=list(range(world)), local_rank=rank, backend="nccl")
    get_parallel().override_permanently(world_group=coord)
    device = torch.device(f"cuda:{rank}")
    nccl_group = dist.new_group(backend="nccl", device_id=device)
    # Non-default stream, as the in-tree custom-AR test and bench do.
    torch.cuda.set_stream(torch.cuda.Stream())
    return coord.cpu_group, nccl_group, device


def _precompile(rank):
    """Rank 0 JIT-builds the custom-AR modules first so 8 ranks don't race on the cache."""
    if rank == 0:
        try:
            from sglang.kernels.ops.communication import all_reduce as ar_ops

            ar_ops._init_communicator()
            # Build the IPC module by constructing the class. _init_ipc_manager has no
            # cache_once; only lazy_register_class may call it, or the second ObjectDef
            # raises "Overriding sgl.IPCManager" when CustomAllReduceV2 constructs one.
            ar_ops.IPCManager()
            ar_ops.get_all_reduce_module(DTYPE, dist.get_world_size())
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"precompile_error": repr(e)}), flush=True)
    dist.barrier()


def _agree(ok, cpu_group):
    flags = [None] * dist.get_world_size(cpu_group)
    dist.all_gather_object(flags, bool(ok), group=cpu_group)
    return all(flags)


def _gather(obj, cpu_group):
    out = [None] * dist.get_world_size(cpu_group)
    dist.all_gather_object(out, obj, group=cpu_group)
    return out


def _sync(cpu_group):
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)


# ----------------------------------------------------------------------------------------
# Per-path watchdog
# ----------------------------------------------------------------------------------------


class _Beat:
    """Rank 0's record of the guarded section it is in; the parent process reads it."""

    def __init__(self, file):
        self.file = file

    def _write(self, state):
        if self.file:
            tmp = f"{self.file}.tmp"
            with open(tmp, "w") as f:
                json.dump(state, f)
            os.replace(tmp, self.file)

    @contextlib.contextmanager
    def guard(self, paths, phase, M=None, graph=None):
        self._write({"paths": list(paths), "phase": phase, "M": M, "graph": graph, "t": time.time()})
        try:
            yield
        finally:
            self._write(None)


def _read_beat(file):
    try:
        with open(file) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _die_with_parent():
    """Linux: without this, ranks outlive a killed parent and keep the GPUs busy."""
    try:
        ctypes.CDLL(None).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


def _supervise(cfg):
    """mp.spawn the ranks; on a stuck path, kill them and resume without that path."""
    if cfg["out"]:
        open(cfg["out"], "w").close()
    state_dir = tempfile.mkdtemp(prefix="ar_bench_")
    beat_file = os.path.join(state_dir, "beat.json")
    dead, start, attempt = {}, (0, False), 0
    try:
        while True:
            with contextlib.suppress(FileNotFoundError):
                os.remove(beat_file)
            run = dict(cfg, port=cfg["port"] + attempt, beat=beat_file, state_dir=state_dir,
                       dead=dict(dead), start=start)
            ctx = mp.spawn(worker, args=(cfg["nprocs"], run), nprocs=cfg["nprocs"], join=False)
            stuck = None
            while stuck is None:
                if ctx.join(timeout=2):
                    return
                b = _read_beat(beat_file)
                if cfg["path_timeout"] and b and time.time() - b["t"] > cfg["path_timeout"]:
                    stuck = b
            where = f"{stuck['phase']} at M={stuck['M']} graph={stuck['graph']}"
            print(f"[ar_bench] {stuck['paths']} exceeded {cfg['path_timeout']} s in {where}; "
                  "killing ranks and continuing without it", flush=True)
            for proc in ctx.processes:
                proc.kill()
            for proc in ctx.processes:
                proc.join(30)
            for name in stuck["paths"]:
                dead[name] = f"exceeded {cfg['path_timeout']} s in {where}"
            if stuck["M"] is not None:
                start = (cfg["sizes"].index(stuck["M"]), bool(stuck["graph"]))
            attempt += 1
    finally:
        shutil.rmtree(state_dir, ignore_errors=True)


# ----------------------------------------------------------------------------------------
# Path construction
# ----------------------------------------------------------------------------------------


def _build_paths(wanted, cpu_group, nccl_group, device, max_bytes, push_cap_bytes, setup_rows, beat):
    world = dist.get_world_size(cpu_group)
    paths = {}

    def record(name, err):
        setup_rows.append({"path": name, "setup_error": err})

    if "nccl" in wanted:
        def nccl_call(x):
            dist.all_reduce(x, group=nccl_group)
            return x

        paths["nccl"] = Path("nccl", nccl_call, 1 << 62, prepare=lambda x: x.clone(),
                             note="torch.distributed NCCL, in place")

    if "pynccl" in wanted:
        try:
            from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

            with beat.guard(["pynccl"], "setup"):
                pn = PyNcclCommunicator(group=cpu_group, device=device)
            if not pn.available:
                raise RuntimeError("PyNcclCommunicator unavailable (NCCL library not loaded)")

            def pynccl_call(x, pn=pn):
                with pn.change_state(enable=True):
                    return pn.outplace_all_reduce(x)

            paths["pynccl"] = Path("pynccl", pynccl_call, 1 << 62, note="sglang PyNccl outplace")
        except Exception as e:  # noqa: BLE001
            record("pynccl", _err(e))

    car_names = [p for p in wanted if p.startswith("car_")]
    car = None
    if car_names:
        try:
            with beat.guard(car_names, "setup"):
                car = _build_car(cpu_group, device, max_bytes, push_cap_bytes, world)
            paths.update(_car_paths(car, car_names, world, setup_rows))
        except Exception as e:  # noqa: BLE001
            for n in car_names:
                record(n, _err(e))

    sm_names = [p for p in wanted if p.startswith("symm_mem")]
    if sm_names:
        try:
            with beat.guard(sm_names, "setup"):
                paths.update(_symm_paths(cpu_group, device, sm_names, have_car=car is not None,
                                         setup_rows=setup_rows))
        except Exception as e:  # noqa: BLE001
            for n in sm_names:
                record(n, _err(e))

    # Every rank must bench the same list. A setup error on one rank used to leave it on a
    # different path from the others, each spinning forever on peers that never arrive.
    built = _gather(sorted(paths), cpu_group)
    common = set.intersection(*map(set, built))
    errored = {r["path"] for r in setup_rows}
    for name in wanted:
        if name not in common and name not in errored:
            bad = [i for i, names in enumerate(built) if name not in names]
            record(name, f"setup failed on ranks {bad}")
    return [paths[p] for p in wanted if p in common], car


def _build_car(cpu_group, device, max_bytes, push_cap_bytes, world):
    from sglang.srt.distributed.device_communicators.configs.custom_all_reduce_v2 import (
        two_shot_push_slot_bytes,
    )
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import CustomAllReduceV2

    push_slot = max(min(max_bytes, push_cap_bytes), two_shot_push_slot_bytes(max_bytes, world))
    car = CustomAllReduceV2(cpu_group, device, max_pull_size=max_bytes, max_push_size=push_slot)
    if car.disabled:
        raise RuntimeError("CustomAllReduceV2 disabled (can_use_custom_all_reduce_v2 is False)")
    return car


def _car_paths(car, names, world, setup_rows):
    from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo, custom_all_reduce
    from sglang.srt.distributed.device_communicators.configs.custom_all_reduce_v2 import (
        two_shot_push_slot_bytes,
    )

    def forced(algo):
        def call(x):
            car.override_algo = algo
            return car.custom_all_reduce(x)

        return call

    # Largest n with two_shot_push_slot_bytes(n) <= slot; same bound as Heuristic.clip.
    push2_max = world * 16 * (car.max_push_size // 32)
    assert two_shot_push_slot_bytes(push2_max, world) <= car.max_push_size

    specs = {
        "car_1shot_push": (AllReduceAlgo.ONE_SHOT_PUSH, car.max_push_size),
        "car_1shot_pull": (AllReduceAlgo.ONE_SHOT_PULL, car.max_pull_size),
        "car_2shot_pull": (AllReduceAlgo.TWO_SHOT_PULL, car.max_pull_size),
        "car_2shot_push": (AllReduceAlgo.TWO_SHOT_PUSH, push2_max),
    }
    out = {}
    for name in names:
        if name in specs:
            algo, cap = specs[name]
            out[name] = Path(name, forced(algo), cap, capture_ctx=car.capture,
                             note=f"CustomAllReduceV2 override_algo={algo.algo_name}")
    if "car_2shot_pull_mc" in names:
        if car.config.num_mc_blocks is None:
            setup_rows.append({"path": "car_2shot_pull_mc", "setup_error":
                               f"multicast unavailable (has_multicast={car.has_multicast}); "
                               "symm-mem multicast_ptr is 0 without NVLS/CU_DEVICE_ATTRIBUTE_MULTICAST"})
        else:
            def mc_call(x):
                return custom_all_reduce(car.obj, x, algo=AllReduceAlgo.TWO_SHOT_PULL, use_multicast=True)

            out["car_2shot_pull_mc"] = Path(
                "car_2shot_pull_mc", mc_call, car.max_pull_size,
                note=f"2shot_pull multicast, {car.config.num_mc_blocks} blocks")
    return out


def _symm_paths(cpu_group, device, names, have_car, setup_rows):
    from sglang.srt.distributed.device_communicators.torch_symm_mem import TorchSymmMemCommunicator
    from sglang.srt.environ import envs

    if not have_car and torch.__version__ < "2.11.0":
        import torch.distributed._symmetric_memory as tsm

        tsm.enable_symm_mem_for_group(cpu_group.group_name)
    note = ""
    try:
        sm = TorchSymmMemCommunicator(group=cpu_group, device=device)
    except Exception as e:  # noqa: BLE001
        # The Inkling buffer sizing reads the runtime exec config, absent standalone.
        note = f"retried with SGLANG_OPT_USE_INKLING_CUSTOM_AR=0 after {_err(e)}"
        with envs.SGLANG_OPT_USE_INKLING_CUSTOM_AR.override(False):
            sm = TorchSymmMemCommunicator(group=cpu_group, device=device)
    if sm.disabled:
        raise RuntimeError("TorchSymmMemCommunicator disabled (multicast_ptr == 0 or unsupported arch)")
    group_name = sm.group.group_name
    buf_bytes = sm.buffer.numel() * sm.buffer.element_size()
    out = {}
    if "symm_mem" in names:
        # should_torch_symm_mem_allreduce uses a strict '<' against max_size.
        out["symm_mem"] = Path("symm_mem", sm.all_reduce, sm.max_size - 2,
                               note=f"TorchSymmMemCommunicator.all_reduce (copy in/out), max {sm.max_size} B {note}")

    def raw(op):
        def prepare(x):
            view = sm.buffer[: x.numel()]
            view.copy_(x.view(-1))
            return view

        def call(view):
            op(view, "sum", group_name)
            return view

        return prepare, call

    if "symm_mem_multimem_raw" in names:
        prep, call = raw(torch.ops.symm_mem.multimem_all_reduce_)
        out["symm_mem_multimem_raw"] = Path("symm_mem_multimem_raw", call, buf_bytes, prepare=prep,
                                            shared_buffer=True,
                                            note="multimem_all_reduce_ in place on the symm buffer, no staging")
    if "symm_mem_2shot_raw" in names:
        prep, call = raw(torch.ops.symm_mem.two_shot_all_reduce_)
        out["symm_mem_2shot_raw"] = Path("symm_mem_2shot_raw", call, buf_bytes, prepare=prep,
                                         shared_buffer=True,
                                         note="two_shot_all_reduce_ in place on the symm buffer, no staging")
    return out


def _sglang_picks(car, world):
    """What the shipped table picks per size with the default 16 MB workspace cap."""
    if car is None:
        return None
    from sglang.srt.distributed.device_communicators.configs.custom_all_reduce_v2 import (
        get_all_reduce_config,
    )
    from sglang.srt.environ import envs

    base = get_all_reduce_config(world)
    cap = envs.SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB.get() * 1024
    prod = base.clip(max_push_bytes=min(base.max_push_bytes(world), cap),
                     max_pull_bytes=min(base.max_pull_bytes, cap), world_size=world)
    if not car.has_multicast:
        prod = prod._replace(num_mc_blocks=None)

    def pick(nbytes, graph):
        saved = car.config
        car.config = prod
        try:
            cfg = car._pick_config(nbytes, graph)
        finally:
            car.config = saved
        if cfg is None:
            return "nccl"
        return cfg.algo.algo_name + ("_mc" if cfg.use_multicast else "") + ("_zerocopy" if cfg.use_graph else "")

    return pick


# ----------------------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------------------


def _err(e):
    return f"{type(e).__name__}: {e}"[:600]


def _time_calls(fn, warmup, iters, calls_per_fn=1):
    for _ in range(warmup):
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) * 1000.0 / calls_per_fn for s, e in zip(starts, ends)]


def _diff(out, ref):
    return float((out.float() - ref.float()).abs().max().item())


def _capture(path, operands, calls):
    graph = torch.cuda.CUDAGraph()
    outs = []
    with path.capture_ctx():
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            for i in range(calls):
                outs.append(path.call(operands[i]))
    torch.cuda.synchronize()
    return graph, outs


def _pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def _bench_size(M, paths, cfg, cpu_group, nccl_group, device, emit, picks, beat, skip_eager=False):
    H = cfg["hidden"]
    nbytes = M * H * DTYPE.itemsize
    rank = dist.get_rank()
    gen = torch.Generator(device=device).manual_seed(1000 * rank + M)
    # Small integers: 8 ranks * 15 = 120 < 256, so bf16 sums are exact in any order.
    x_corr = torch.randint(0, 16, (M, H), device=device, generator=gen).to(DTYPE)
    ref = x_corr.clone()
    dist.all_reduce(ref, group=nccl_group)
    # Zeros keep in-place paths from overflowing to inf across hundreds of calls.
    x_time = torch.zeros(M, H, dtype=DTYPE, device=device)
    base = {"M": M, "H": H, "bytes": nbytes, "kb": nbytes / 1024}
    if picks is not None:
        base.update(pick_eager=picks(nbytes, False), pick_graph=picks(nbytes, True))

    modes = ([] if skip_eager else [False]) + ([True] if cfg["graph"] else [])
    for graph_mode in modes:
        live, rows = [], {}
        for p in paths:
            row = dict(base, path=p.name, graph=graph_mode, note=p.note)
            reason = p.eligible(nbytes) or (None if (p.graph or not graph_mode) else "not graph-capturable")
            if reason:
                rows[p.name] = dict(row, status="skip", error=reason)
                continue
            rows[p.name] = dict(row, status="ok", samples=[], max_abs_diff=None)
            live.append(p)

        state = _prepare_mode(live, rows, graph_mode, cfg, x_corr, ref, x_time, cpu_group, beat, M)
        live = [p for p in live if rows[p.name]["status"] == "ok"]

        for block in range(cfg["repeats"]):
            k = block % max(1, len(live))
            for p in live[k:] + live[:k]:
                if rows[p.name]["status"] != "ok":
                    continue
                ok = True
                # The guard spans _agree: rank 0 waits there for any rank still stuck.
                with beat.guard([p.name], "timed", M, graph_mode):
                    try:
                        _sync(cpu_group)
                        if graph_mode:
                            g = state[p.name]["graph"]
                            times = _time_calls(g.replay, max(2, cfg["warmup"] // cfg["graph_calls"]),
                                                cfg["graph_replays"], calls_per_fn=cfg["graph_calls"])
                        else:
                            operand = state[p.name]["operand"]
                            times = _time_calls(lambda: p.call(operand), cfg["warmup"], cfg["iters"])
                        rows[p.name]["samples"].extend(times)
                    except Exception as e:  # noqa: BLE001
                        ok = False
                        rows[p.name].update(status="error", error=_err(e))
                    if not _agree(ok, cpu_group) and rows[p.name]["status"] == "ok":
                        rows[p.name].update(status="error", error="failed on another rank")

        if graph_mode:
            _graph_correctness(live, rows, state, x_corr, ref, cpu_group, beat, M)
        state.clear()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        for p in paths:
            emit(_finish_row(rows[p.name], cpu_group))
        for name, reason in cfg["dead"].items():
            emit(dict(base, path=name, graph=graph_mode, note="", status="timeout", error=reason))
    del x_corr, ref, x_time
    torch.cuda.empty_cache()


def _prepare_mode(live, rows, graph_mode, cfg, x_corr, ref, x_time, cpu_group, beat, M):
    """Eager: correctness + timing operand. Graph: capture --graph-calls calls per path."""
    state = {}
    gin = None
    if graph_mode and live:
        gin = torch.zeros(cfg["graph_calls"], x_time.numel(), dtype=DTYPE, device=x_time.device)
        # Custom-AR caches IPC mappings of registered graph inputs by address; never free
        # one and let a later size reuse that address.
        _GRAPH_INPUTS_KEEPALIVE.append(gin)
    for p in live:
        ok = True
        with beat.guard([p.name], "prepare", M, graph_mode):
            try:
                if graph_mode:
                    if p.shared_buffer:
                        operands = [p.prepare(gin[0])] * cfg["graph_calls"]
                    else:
                        operands = [gin[i] for i in range(cfg["graph_calls"])]
                    graph, outs = _capture(p, operands, cfg["graph_calls"])
                    state[p.name] = {"graph": graph, "outs": outs, "gin": gin}
                else:
                    out = p.call(p.prepare(x_corr))
                    torch.cuda.synchronize()
                    rows[p.name]["max_abs_diff"] = _diff(out.view(ref.shape), ref)
                    state[p.name] = {"operand": p.prepare(x_time)}
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                ok = False
                rows[p.name].update(status="error", error=_err(e) + "\n" + traceback.format_exc(limit=3)[-400:])
            if not _agree(ok, cpu_group) and rows[p.name]["status"] == "ok":
                rows[p.name].update(status="error", error="failed on another rank")
    return state


def _graph_correctness(live, rows, state, x_corr, ref, cpu_group, beat, M):
    for p in live:
        if rows[p.name]["status"] != "ok" or p.name not in state:
            continue
        if p.shared_buffer:
            rows[p.name]["max_abs_diff"] = None
            rows[p.name]["note"] += "; graph diff skipped (calls chain on one buffer), see eager row"
            continue
        ok = True
        with beat.guard([p.name], "graph_correctness", M, True):
            try:
                st = state[p.name]
                st["gin"].zero_()
                st["gin"][0].copy_(x_corr.view(-1))
                _sync(cpu_group)
                st["graph"].replay()
                torch.cuda.synchronize()
                rows[p.name]["max_abs_diff"] = _diff(st["outs"][0].view(ref.shape), ref)
                st["gin"].zero_()
            except Exception as e:  # noqa: BLE001
                ok = False
                rows[p.name].update(status="error", error="graph correctness: " + _err(e))
            if not _agree(ok, cpu_group) and rows[p.name]["status"] == "ok":
                rows[p.name].update(status="error", error="graph correctness failed on another rank")


def _finish_row(row, cpu_group):
    samples = row.pop("samples", None) or []
    med = statistics.median(samples) if samples else None
    meds = _gather(med, cpu_group)
    world = len(meds)
    if samples and row["status"] == "ok":
        row.update(n=len(samples), median_us=round(med, 2), p10_us=round(_pct(samples, 0.10), 2),
                   p90_us=round(_pct(samples, 0.90), 2), mean_us=round(statistics.fmean(samples), 2),
                   max_rank_median_us=round(max(m for m in meds if m is not None), 2),
                   busbw_GBps=round(row["bytes"] * 2 * (world - 1) / world / (med * 1e-6) / 1e9, 1))
    return row


# ----------------------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------------------


def _summary(rows, path_names, sizes):
    lines, winners = [], {}
    for graph_mode in (False, True):
        sub = [r for r in rows if r.get("graph") is graph_mode]
        if not sub:
            continue
        hdr = f"{'M':>6} {'KB':>8} " + " ".join(f"{n[:14]:>14}" for n in path_names) + f"  {'winner':<22} pick"
        lines += ["", f"== {'CUDA graph (per call)' if graph_mode else 'eager'}: median us (x = error, - = skip) ==", hdr]
        for M in sizes:
            by = {r["path"]: r for r in sub if r["M"] == M}
            if not by:
                continue
            cells, best = [], None
            for n in path_names:
                r = by.get(n)
                if r is None or r["status"] == "skip":
                    cells.append(f"{'-':>14}")
                elif r["status"] != "ok" or "median_us" not in r:
                    cells.append(f"{'x':>14}")
                else:
                    bad = r.get("max_abs_diff") not in (None, 0.0)
                    cells.append(f"{r['median_us']:>13.1f}{'!' if bad else ' '}")
                    if not bad and (best is None or r["median_us"] < best[1]):
                        best = (n, r["median_us"])
            any_row = next(iter(by.values()))
            pick = any_row.get("pick_graph" if graph_mode else "pick_eager", "")
            win = f"{best[0]} {best[1]:.1f}" if best else "none"
            winners[f"{'graph' if graph_mode else 'eager'}:{M}"] = best[0] if best else None
            lines.append(f"{M:>6} {any_row['kb']:>8.0f} " + " ".join(cells) + f"  {win:<22} {pick}")
    lines.append("('!' = max_abs_diff != 0 vs NCCL; excluded from winner)")
    return "\n".join(lines), winners


def worker(rank, world, cfg):
    _die_with_parent()
    if rank != 0:
        logging.disable(logging.WARNING)
    cpu_group, nccl_group, device = _init_dist(rank, world, cfg["port"])
    # _supervise truncated --out; a respawned attempt appends after the rows already written.
    out_f = open(cfg["out"], "a") if rank == 0 and cfg["out"] else None
    # Rows of all attempts, so the closing summary covers the whole run.
    rows_file = os.path.join(cfg["state_dir"], "rows.jsonl")
    beat = _Beat(cfg["beat"] if rank == 0 else None)

    def emit(row):
        if rank == 0:
            line = json.dumps(row)
            print(line, flush=True)
            if out_f:
                out_f.write(line + "\n")
                out_f.flush()
            with open(rows_file, "a") as f:
                f.write(line + "\n")

    sizes = cfg["sizes"]
    max_bytes = max(sizes) * cfg["hidden"] * DTYPE.itemsize
    setup_rows = []
    wanted = [p for p in cfg["paths"] if p not in cfg["dead"]]
    paths, car = _build_paths(wanted, cpu_group, nccl_group, device, max_bytes,
                              cfg["push_cap_mb"] * MB, setup_rows, beat)
    picks = _sglang_picks(car, world)
    # A respawned attempt repeats setup; keep one meta line and one set of setup rows.
    meta_file = os.path.join(cfg["state_dir"], "meta_written")
    first_attempt = not os.path.exists(meta_file)
    meta = {"meta": {"world": world, "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
                   "nccl": str(torch.cuda.nccl.version()), "hidden": cfg["hidden"],
                   "iters": cfg["iters"], "repeats": cfg["repeats"], "graph_calls": cfg["graph_calls"],
                   "paths": [p.name for p in paths], "car_max_push": car.max_push_size if car else None,
                   "car_max_pull": car.max_pull_size if car else None,
                   "car_has_multicast": car.has_multicast if car else None,
                   "time": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    if first_attempt:
        emit(meta)
        for r in setup_rows:
            emit(r)
        if rank == 0:
            open(meta_file, "w").close()
    start_size, start_graph = cfg["start"]
    for i, M in enumerate(sizes):
        if i < start_size:
            continue
        faulthandler.dump_traceback_later(cfg["size_timeout"], exit=True)
        _bench_size(M, paths, cfg, cpu_group, nccl_group, device, emit, picks, beat,
                    skip_eager=i == start_size and start_graph)
        faulthandler.cancel_dump_traceback_later()
    if rank == 0:
        with open(rows_file) as f:
            all_rows = [r for r in map(json.loads, f) if "M" in r]
        names = [n for n in cfg["paths"] if any(r["path"] == n for r in all_rows)]
        table, winners = _summary(all_rows, names, sizes)
        emit({"summary": {"winners": winners}})
        print(table, flush=True)
        if out_f:
            out_f.close()
    _sync(cpu_group)
    if car is not None:
        car.close()
    dist.destroy_process_group()


def _parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="ar_bench.json", help="JSON-lines output path (rank 0)")
    ap.add_argument("--paths", default="all", help="comma list from: " + ",".join(ALL_PATHS))
    ap.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)), help="comma list of token counts M")
    ap.add_argument("--hidden", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=200, help="timed calls per block (eager)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3, help="interleaved blocks per size")
    ap.add_argument("--graph-calls", type=int, default=10, help="all-reduces captured per CUDA graph")
    ap.add_argument("--graph-replays", type=int, default=0, help="replays per block (0 = iters / graph-calls)")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--push-cap-mb", type=int, default=64,
                    help="cap on the 1shot_push slot (x16 slots of symmetric memory)")
    ap.add_argument("--size-timeout", type=int, default=1200, help="seconds per size before a rank dumps and exits")
    ap.add_argument("--path-timeout", type=int, default=120,
                    help="seconds one path may spend in setup, prepare or a timed block (0 = off)")
    ap.add_argument("--port", type=int, default=29573)
    ap.add_argument("--nprocs", type=int, default=0, help="ranks (0 = all visible GPUs)")
    a = ap.parse_args()
    paths = list(ALL_PATHS) if a.paths == "all" else [p.strip() for p in a.paths.split(",") if p.strip()]
    unknown = [p for p in paths if p not in ALL_PATHS]
    if unknown:
        ap.error(f"unknown paths {unknown}; choose from {ALL_PATHS}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    return {
        "out": a.out, "paths": paths, "sizes": [int(s) for s in a.sizes.split(",") if s.strip()],
        "hidden": a.hidden, "iters": a.iters, "warmup": a.warmup, "repeats": a.repeats,
        "graph_calls": a.graph_calls, "graph": not a.no_graph,
        "graph_replays": a.graph_replays or max(10, a.iters // a.graph_calls),
        "push_cap_mb": a.push_cap_mb, "size_timeout": a.size_timeout, "port": a.port,
        "path_timeout": a.path_timeout,
        "nprocs": a.nprocs or torch.cuda.device_count(),
    }


if __name__ == "__main__":
    cfg = _parse()
    if cfg["nprocs"] < 2:
        sys.exit("need at least 2 GPUs")
    _supervise(cfg)
