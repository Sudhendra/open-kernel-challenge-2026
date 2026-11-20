# Serving harness

One paid 8xH200 Modal session boots SGLang with the judge's flags, warms up, runs a ladder,
profiles via `/start_profile`, and optionally A/Bs env-var variants on a proxy workload. Results
are tarred and committed to the `okc-muse-weights` volume under `out/`.

**Pin (organisers, Nov 9):** SGLang `297e54fd` (v0.5.23), image
`lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f` (the `v0.5.23` tag),
which is the `modal_serve.py` default. `serving/sglang_stock` is detached at that commit and `serving/sglang`
is on `main-track-pin` (that commit plus our two HiCache commits). The patches are in `serving/patches/pin-v0.5.23/`.
Judge flags new at this pin: `--speculative-draft-attention-backend fa4 --mm-feature-transport cpu --hicache-size 160`.
Modal's 512 GiB container cannot hold 160 GB per rank, so `extra_args` (default `--hicache-size 16`) comes after
the judge flags and wins (argparse keeps the last occurrence).

Next session, from `serving/` (no `OKC_IMAGE` override needed):

```
modal run --detach harness/modal_serve.py::session --max-min 80 --stock-mode proxy --pre-tests 1 --stock-self 1 --stock-self-restart 1 --proxy-rungs "4x1x2,16x2x2" --variants "patched:SRC=patched;symm:ARGS=--enable-torch-symm-mem"
```

Its budget line is 79 of 80 minutes (tests 3, stock/proxy 19, stock_self 19, variants 2 x 19; 288 requests ~ 11 min).
That counts the first boot as 4 min and it is about 13 min when the page and JIT caches are cold, so the last
variant can be skipped by the `elapsed + 19 <= max_min` check; raise `--max-min` to about 90 to be sure it runs.

| file | what |
|---|---|
| `modal_serve.py` | Modal app: `smoke`, `download`, `session` (runs `profile_session.sh`) |
| `profile_session.sh` | the session itself (also runs on any 8xH200 box from `serving/`) |
| `ar_bench.py` | TP8 all-reduce path sweep (NCCL, PyNccl, custom AR, symm-mem) at Muse message sizes; `AR_BENCH=1` runs it |
| `ladder.py` | synthetic multi-turn load generator; `--proxy` preset |
| `agree.py` | speed ratios, TTFT/length gates, output agreement between two ladder runs |
| `nccl_test.py` | peer-access matrix + all-reduce at the per-layer message sizes |
| `dbg_live.sh` | one request + server-log errors; run automatically when warm-up fails |
| `report_charts.py`, `trace_*.py` | post-processing of a fetched session |
| `comment_trim.py` | trims comments/docstrings OUR patch added until the reduced judge diff fits: `python3 harness/comment_trim.py lanes/<tree> --base 297e54fd --target-bytes 385000 [--dry]`, then commit the printed files by name, `--verify-against <pre-trim head>`, `min_diff.py`, `judge.sh lint`. Re-run it on each new tree (never cherry-pick an old trim commit). Report: `research/comment_trim_2026-11-19.md` |

## Budget

- Modal cap is **$200/month**. One session is H200:8 for its whole wall time, boot included.
- Sessions are capped at **75 minutes** (`max_min`, default 75). The watchdog writes the elapsed
  minutes to `watchdog.log` and the session log every 5 minutes, and at `max_min` it TERMs the
  script. The EXIT trap still stops the server, writes the NCCL summaries and the tarball, and
  `modal_serve.py` commits the volume afterwards. The Modal function `timeout` (2 h) is only a
  backstop: keep `max_min` at or below 105 so the watchdog always fires first.
- Server boot is about **11 minutes** (startup_s 633-714 on Nov 7). A full stock run (warm-up,
  ladder at c1/c4/c16/c32, four profiles, replay) took about 35 minutes end to end on Nov 7.
- One session at a time. Run `smoke` first after changing the image.

## Running a session

Always launch with `--detach`, from `serving/`:

```
modal run --detach harness/modal_serve.py::session
modal run --detach harness/modal_serve.py::session --variants "nolog:SGLANG_LOG_DECODE_GRAPH_KEY=0"
modal run --detach harness/modal_serve.py::session --stock-mode proxy --max-min 40
modal run --detach harness/modal_serve.py::session --max-min 90 --stock-mode proxy --pre-tests 1 --ar-bench 1 \
    --variants "patched:SRC=patched;d5:ARGS=--speculative-num-draft-tokens 5"
```

`modal run` turns every `session` parameter into a `--with-hyphens` flag (`pre_tests` -> `--pre-tests`;
checked with `modal run harness/modal_serve.py::session --help`, modal 1.6.3).

`--detach` keeps the app running when the local client disconnects (laptop sleep, network drop,
Ctrl-C). `session` is written for that: it takes no stdin, `profile_session.sh` tees its own
output to `out/<ts>/session.log` on the volume, a SIGTERM to the function (`modal app stop`,
preemption) is forwarded to the script so its EXIT trap can tar the results, and the volume is
committed in a `finally`. Follow along with `modal app logs okc-serving`, or read
`session.log` from the volume.

Fetch results:

```
modal volume ls okc-muse-weights out/
modal volume get okc-muse-weights out/<ts>.tgz .
```

`session` arguments (all optional): `trace` (agentic-trace JSON path inside the container),
`extra_args` (local-only server flags, default `--hicache-size 16`), `prof_conc` (default
`"1 4 16 32"`), `skip_replay`, `max_min` (75), `variants` (see below), `stock_mode`
(`full` or `proxy`), `pre_tests` (`1` = the three HiCache CPU unit tests per tree before boot),
`ar_bench` (`1` = `ar_bench.py` before boot, while the GPUs are free), `proxy_rungs` (`CxRxS,...`
override of the proxy preset for stock and every variant replay), `stock_self` (`1` = replay the
stock proxy once for the agreement noise floor), `stock_self_restart` (`1` = restart the stock
server before that replay). The last five default off, so an old launch line behaves as before.
`boot_timeout_min` (default 18) and `server_env_extra` (`K=V,K=V`, default empty) are described under
"Boot failures" below.

Outside Modal: `bash harness/profile_session.sh [trace.json]` from `serving/` with the same
knobs as env vars (`MAX_MIN`, `VARIANTS`, `STOCK_MODE`, `EXTRA_ARGS`, `PROF_CONC`,
`SKIP_REPLAY`, `OUT_ROOT`, `SGLANG_SRC_MODE`, `PRE_TESTS`, `AR_BENCH`, `PROXY_RUNGS`, `STOCK_SELF`,
`STOCK_SELF_RESTART`, `BOOT_TIMEOUT_MIN`, `SERVER_ENV_EXTRA`, `SRC_STOCK`, `SRC_PATCHED`).

### Boot failures

A rank can hang while the server process stays alive, so waiting for the process to exit is not
enough (Nov 9: 58 paid minutes of polling after one rank's multicast setup failed). Every boot now has
two limits:

- `BOOT_TIMEOUT_MIN` (default 18; the slowest cold boot so far is 793 s): `/health` is not up by then.
- Every 30 s while waiting, `server.log` is searched for `Setup Custom allreduce failed`,
  `CUDA_ERROR_UNKNOWN` or `Traceback (most recent call last)`. None of the logged boots that reached
  `/health` had one of these before it came up.

Either one writes `BOOT_TIMEOUT.txt` (the reason, then the last 80 lines of `server.log`) into that
boot's directory and stops the server. The stock boot then exits 3 and Modal reports
`profile_session.sh exited 3`. The stock self-restart and variants get `FAILED.txt` and the session
carries on.

`SERVER_ENV_EXTRA="K=V,K=V"` (Modal: `--server-env-extra`) is added to every server launch, stock and
variants; a variant's own env wins on the same key. It is recorded in `session_args.txt`.

To skip the multicast path that failed on Nov 9, use `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0`: at the
pin that flag selects the all-reduce whose setup does the symmetric-memory rendezvous, and `0` falls back
to the legacy `CustomAllreduce`. `SGLANG_OPT_USE_INKLING_CUSTOM_AR=0` does not reach that setup; it only
turns off the Inkling kernels built on top of `--enable-torch-symm-mem`. Either setting moves the all-reduce
path away from what the judge runs, so treat absolute numbers from such a session with care. The same
multicast setup worked on Modal in the 18:19 UTC session that day (`multicast = True` in its startup facts).

### Proxy rungs and the stock self-replay

`PROXY_RUNGS="4x1x2,16x2x2,32x1x1"` appends `--rungs` to the proxy ladder (`ladder.py` lets an
explicit `--rungs` win over the `--proxy` preset `16x1x2`), for the stock proxy run and for every
variant's `--replay` run, so one session can measure c4, c16 twice and c32 instead of the fixed c16.
Each rung is concurrency x reps x conversations per slot, at 4 turns.

`STOCK_SELF=1` runs the proxy ladder once more against the stock server after the stock proxy run,
with `--replay proxy.json --out proxy_self.json` and the proxy's own `--seed 4242 --offset 4000`.
The prompts are already prefix-cached, so it takes about half the proxy time and measures only the
stock-vs-stock agreement noise floor (`self_agreement.txt`). Every variant's `agree.py` then gets
`--self proxy_self.json`, so `agree.txt` prints the noise floor next to the candidate's agreement.
The Nov 9 session replayed with `--seed 0`, which redraws the user turns, so its "stock vs stock"
compared different prompts; ignore that session's noise floor.

`STOCK_SELF_RESTART=1` (with `STOCK_SELF=1`) stops the stock server and boots it again before the
self pass, with its logs in `stock_self/`, so both sides of the comparison are fresh boots. It
costs as much as a variant. If that boot or its warm-up fails, `stock_self/FAILED.txt` says so, the
session carries on and variants get no `--self`.

### Two source trees

The image carries two SGLang trees, both put on `PYTHONPATH` ahead of the image's own install:

| tree | local | in the image | default |
|---|---|---|---|
| stock | `serving/sglang_stock` (git worktree at the image's commit, `297e54fd`) | `/root/serving/sglang_stock/python/sglang` | `SRC_STOCK=sglang_stock/python` if that dir exists, else `sglang/python` |
| patched | `serving/sglang` (branch `main-track-pin`) | `/root/serving/sglang/python/sglang` | `SRC_PATCHED=sglang/python` |

The stock phase always runs from the stock tree; a variant picks with `SRC=`. `modal_serve.py`
skips the stock mount with a warning when `serving/sglang_stock` is missing (then both names
point at `serving/sglang`). The session records `commit_stock.txt`, `commit_patched.txt`,
`patch_patched.diff` (and `patch_stock.diff`) from `git rev-parse HEAD` / `git diff` of each
tree; on Modal only `python/sglang` is mounted, so there is no `.git` and the files say `no git`
(the local launch log is the record of the commits in that case). `sglang_path_<tree>.txt` shows
where `import sglang` resolved for each tree.

### Pre-boot steps (`PRE_TESTS=1`, `AR_BENCH=1`)

Both run before the server boots, are non-fatal, and are capped at 10 minutes each by a
background-run-and-poll helper (`capped`), because the image may not ship `timeout`.

- `PRE_TESTS=1`: once per tree, with `PYTHONPATH=<tree>/python` and cwd `<tree>` so the relative
  test paths resolve (a file the tree lacks is skipped with a NOTE, because pytest aborts on a missing
  path; the v0.5.23 pin has no `test_page_unified_staging.py`, so two files run there):
  `python -m pytest -q -p no:cacheprovider test/registered/unit/mem_cache/{test_asymmetric_mha_pool_host_unit,test_page_unified_staging,test_hicache_staged_write_back_dispatch}.py`.
  Output in `unit_tests_stock.txt` / `unit_tests_patched.txt`, exit codes in `unit_tests_rc.txt`
  (`<tree> <rc>` per line; 124 = cap hit). If `pytest` does not import, it falls back to
  `python -m unittest -v` on the same files. If a tree has no `test/` dir (only `python/sglang`
  was mounted), the patched tree's test files are run against that tree's `PYTHONPATH`, and the
  output file says so. `modal_serve.py` mounts `test/registered/unit/mem_cache` from both trees
  (about 4 MB each); the three files import only `sglang.*` (checked Nov 8), nothing else under
  `test/`.
- `AR_BENCH=1`: `python harness/ar_bench.py --out $OUT/ar_bench.jsonl` with stdout/stderr in
  `ar_bench.txt`, on the stock tree's `PYTHONPATH`. Defaults are kept (10 sizes x 10 paths x
  eager+graph x 3 blocks x 200 timed + 20 warm-up calls, about 130k all-reduce calls): the
  per-call latency summed over the ten sizes is about 2 ms (dominated by the 64 MB and 256 MB
  messages), so the timed part is well under a minute, and spawn + NCCL/CAR/symm-mem set-up,
  graph captures and correctness checks add a few minutes. Estimate 3-5 minutes; the 10-minute
  cap only catches hangs (its own `--size-timeout` is 20 minutes per size, which the cap
  shortens).

The session log starts with the planned budget, e.g.
`[budget] tests+ar_bench+stock+self+2 variants ~ 83 min of MAX_MIN 90 (tests 3, ar_bench 8, stock/proxy 24, stock_self 0, variants 2 x 24; proxy rungs 16x1x2 = 128 requests ~ 12 min)`,
and warns when it exceeds `MAX_MIN`. The proxy estimate is 4 min plus 1 min per 16 requests over
the rung set (capped at 40 min); a variant or the stock proxy phase is that plus 12 min (boot,
warm-up, teardown); the self-replay is half the proxy estimate. With the default rungs that gives
the familiar 24 min per variant. The per-variant skip check uses the same derived number.

## What a session does

1. Node facts, commits/diffs of both trees, the planned-budget line.
2. `PRE_TESTS=1`: unit tests per tree. `AR_BENCH=1`: `ar_bench.py`. Then `nccl_test.py`.
3. Boot the server (stock tree) with the judge's flags plus ours: `--enable-metrics --decode-log-interval 20`,
   the timing env below, and the NCCL start-up log. `extra_args` replaces the judge's `--hicache-size 160`.
4. Warm-up (1x1x1, 2 turns). If every warm-up request fails, it runs `dbg_live.sh` and exits.
5. Stock phase:
   - `STOCK_MODE=full` (the default when there are no variants): ladder at c1/c4/c16/c32, a
     40-step `/start_profile` at each `prof_conc` load, then a replay of the ladder for the
     agreement noise floor. Same as the Nov 7 session.
   - `STOCK_MODE=proxy` (the default when `VARIANTS` is set): the proxy workload only (rungs from
     `PROXY_RUNGS` if set), then with `STOCK_SELF=1` the self-replay for the noise floor (after a
     server restart with `STOCK_SELF_RESTART=1`).
6. Each variant: stop the server, start it again with the extra env, the selected tree (`SRC=`)
   and extra flags (`ARGS=`), warm up, then run the proxy workload with `--replay` of the stock
   proxy run, so prompts are byte-identical. Then `agree.py stock-proxy variant-proxy`, with
   `--self stock-self-proxy` when `STOCK_SELF=1` produced one.
7. EXIT trap (every exit path): stop the server, write `nccl_summary.txt`, then tar `out/<ts>/`.

### Always-on timing env

All of these exist in `sglang/python/sglang/srt/environ.py` (checked Nov 7; none dropped):

| var | effect |
|---|---|
| `SGLANG_ENABLE_METRICS_DEVICE_TIMER=1` | GPU-event forward timing; adds forward occupancy to the prefill/decode log lines and metrics |
| `SGLANG_RECORD_STEP_TIME=1` | step-time per running-request count (gap / decode-log-interval) |
| `SGLANG_LOG_FORWARD_ITERS=1` | `[iter]` id on every prefill/decode log line |
| `SGLANG_LOG_MS=1` | millisecond timestamps in the server log |
| `SGLANG_LOG_DECODE_GRAPH_KEY` | off by default (LOG_GRAPH_KEY=1 to enable); logs the graph key on every replay, not free |
| `SGLANG_ENABLE_NVTX_SCHEDULER=1` | NVTX ranges on scheduler stages (no-ops unless nsys is attached) |
| `SGLANG_PROFILE_WITH_STACK=False` | `/start_profile` traces without Python stacks (smaller, faster export) |
| `SGLANG_PROFILE_RECORD_SHAPES=False` | `/start_profile` traces without shapes |

`SGLANG_ENABLE_NVTX_SCHEDULER` is set only if `python -c 'import nvtx'` works; otherwise the
script writes `nvtx_missing.txt`. The Modal image installs `nvtx` if the judge image lacks it.

`SGLANG_LOG_DECODE_GRAPH_KEY` is not free: it calls `logger.info` on every graph replay, for
target and draft, on every rank. That makes `server.log` large and adds Python overhead per
step. If the stock numbers move compared with Nov 7, drop it first; a variant with
`SGLANG_LOG_DECODE_GRAPH_KEY=0` measures its cost.

### NCCL start-up log

The server runs with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,NVLS` and
`NCCL_DEBUG_FILE=<dir>/nccl_%h_%p.log`. Every rank and process writes its own small file.
`nccl_summary.txt` greps the tuning and NVLS lines from all of them, prefixed with the file name.
The grep pattern is a guess at NCCL's wording, so read the raw files if the summary is thin.

## Variant syntax

```
VARIANTS="name:ENV=val,ENV=val;name2:ENV=val"
```

- Variants are separated by `;`. Each one is `name:` followed by `KEY=val` pairs separated by `,`.
- Names use `[A-Za-z0-9_.-]`, and `stock` is reserved. Values cannot contain `,` or `;` (spaces are fine).
- Two keys are reserved and are not exported to the server:
  - `SRC=stock|patched`: which tree the variant's server imports (first on `PYTHONPATH`). Default
    `stock`. Anything else is rejected by the up-front check.
  - `ARGS=<server flags>`: appended after `EXTRA_ARGS` on the launch line. argparse keeps the
    last occurrence, so `ARGS=--speculative-num-draft-tokens 5` overrides the judge's `8`.
  - `PROF=1`: after the proxy, torch-profile the variant (`profile_by_stage`, `PROF_STEPS` 20) at each `PROF_CONC`
    load, posted a fixed delay after the load starts (warm extends + decode). `PROF=cold` (Nov 16): the profile is
    posted on the idle server BEFORE fresh judge-shape first turns (~74.2k tokens), so the EXTEND trace holds the
    cold 16,384-token chunks (4 full per conversation; prefix `cold_c<C>`) and ends at the first decode step.
    `PROF=warm` (Nov 17): fresh judge-shape conversations (`WARM_TURNS` 5); `harness/warm_trigger.py` follows
    server.log and posts `/start_profile` (CPU+GPU, `with_stack`, no `profile_by_stage`) at the first later-turn
    `Prefill batch` (`#cached-token` >= 1024, `#queue-req` 0) after a `Decode batch` line, then `/stop_profile` after
    `WARM_EXTEND` (12) more prefill forwards; decode steps between them are in the window. Prefix `warm_c<C>`, plus a
    py-spy sidecar on the rank-0 scheduler (`prof/pyspy_rank0_warm_c<C>.txt`) and `warm_trigger_warm_c<C>.json`.
    Knobs `WARM_*` (profile_session.sh header; queue items: `modal_queue.py push --prof-env`).
  - `HOOK=<script>`: run after warm-up with `--url <url> --out <variant dir>`.
- Every other `KEY=val` is a server env var, added after the default timing/NCCL env, so it can
  override it.
- The whole string is checked before boot, and a malformed one fails at once (before `nvidia-smi`).
- Results go to `out/<ts>/variants/<name>/` with `env.txt` (env pairs only), `src.txt`
  (`<tree> <python dir>`) and `args.txt` (`EXTRA_ARGS` + `ARGS`).

Examples: `patched:SRC=patched` (our HiCache commits, otherwise the judge's flags);
`d5:ARGS=--speculative-num-draft-tokens 5` (stock tree, 5 draft tokens);
`mix:SGLANG_X=1,SRC=patched,ARGS=--foo 1 --bar 2`.

Budget: with the default rungs each variant costs about 24 minutes (11 min boot, warm-up, about
10 min proxy, teardown). Stock in proxy mode is the same. **Stock plus two variants is the most
that fits in 75 minutes**; with `PRE_TESTS` (~3 min) and `AR_BENCH` (~8 min allowed) as well, that
plan is about 83 minutes, so launch with `--max-min 90`. A larger `PROXY_RUNGS` set raises the
per-variant cost (the `[budget]` line prints the derived figure): `4x1x2,16x2x2,32x1x1` is 416
requests, about 30 min of proxy, so 42 min per phase and 15 min for the self-replay. Before each variant the script checks `elapsed + 24 <= max_min`. A variant that
would not fit is skipped, and the script writes `SKIPPED.txt` for it instead of being killed
half way through boot. If you also want the full stock flow (`stock_mode=full` with variants),
that is about 35 + 10 minutes before the first variant, so only one variant fits.

## Workload shape

`ladder.py` defaults: `--base-chars 280000` (first prompt) and `--turn-chars 2150` (new user
text per later turn, drawn uniformly from [1350, 5400)). These counts assume the SGLang-source
corpus tokenises at about 3.5 chars/token, which gives about 80k first-prompt tokens and about
770 new tokens for a typical turn. The mean draw is about 960 tokens, plus each turn carries the
previous 220-token answer. Check `prompt_tokens` in the records to confirm on another corpus.
Output is fixed at 220 tokens (`ignore_eos`), as the judge does.

`--proxy` means rungs `16x1x2` (c16, 1 rep, 2 conversations per slot) and 4 turns. That is 128
requests, with 32 cold 80k prefills. It is meant to finish in about 10 minutes after warm-up.
An explicit `--rungs` or `--turns` overrides the preset. It prints the same per-rung JSON summary.

## Output files (`out/<ts>/`)

| file | content |
|---|---|
| `session.log` | everything the script printed (watchdog, rungs, agreement) |
| `session_args.txt`, `timing_env.txt` | knobs and timing env actually used |
| `timeline.txt` | `<event> HH:MM:SS` marks (unit_tests_*, ar_bench_*, server_up, warmup_done, proxy_done, proxy_self_done, stock_done, prof_cN_done, variant_*, session_done) |
| `exit_code.txt` | script exit code (143 = TERM: watchdog or Modal stop) |
| `watchdog.log` | elapsed minutes every 5 min; `ABORTED.txt` if the budget was hit |
| `gpus.csv`, `topo.txt`, `nvidia_smi.txt`, `pip_freeze.txt` | node facts |
| `sglang_path.txt`, `commit.txt`, `patch.diff` | `import sglang` under the image env; copies of the patched tree's commit and diff (legacy names) |
| `commit_stock.txt`, `commit_patched.txt`, `patch_stock.diff`, `patch_patched.diff`, `sglang_path_<tree>.txt` | per-tree commit, diff (`no git` when only `python/sglang` was mounted) and import path |
| `src.txt`, `args.txt` | tree and flags the stock server ran with (`EXTRA_ARGS`) |
| `unit_tests_stock.txt`, `unit_tests_patched.txt`, `unit_tests_rc.txt` | `PRE_TESTS=1` output and exit codes per tree |
| `ar_bench.jsonl`, `ar_bench.txt` | `AR_BENCH=1` rows and log |
| `nccl.jsonl`, `nccl.err` | `nccl_test.py` peer access + all-reduce timings |
| `nccl_<host>_<pid>.log`, `nccl_summary.txt` | NCCL INIT/GRAPH/TUNING/NVLS start-up log per process, and the grep of it |
| `gpu_util.csv` | 1 Hz util/power/memory per GPU for the whole session |
| `server.log`, `server_full.log` | server output (with the timing env lines) |
| `startup.txt`, `startup_facts.txt` | boot seconds; grep of KV size, graphs, backends, spec, allreduce |
| `BOOT_TIMEOUT.txt` | only when a boot was aborted: the reason and the last 80 lines of `server.log` (`exit_code.txt` is 3 for the stock boot) |
| `config_*.json`, `server_info.json` | model/drafter configs and `/get_server_info` |
| `warmup.json` | warm-up ladder |
| `proxy.json`, `proxy.rungs` | stock proxy run (proxy mode or when variants are set; rungs from `PROXY_RUNGS`) |
| `proxy_self.json`, `proxy_self.rungs`, `self_agreement.txt` | `STOCK_SELF=1`: replay of the stock proxy and its agreement (noise floor) |
| `stock_self/` | `STOCK_SELF_RESTART=1`: `server.log`, `startup*.txt`, `warmup.json`, NCCL logs of the restarted stock server, `FAILED.txt` if it did not come up |
| `stock.json`, `stock.rungs` | stock ladder (full mode) |
| `prof_load_cN.json`, `prof/`, `profile_replies.txt` | load during each profile, torch traces, `/start_profile` replies |
| `stock2.json`, `self_agreement.txt` | replay run and stock-vs-stock agreement (full mode; in proxy mode `self_agreement.txt` comes from `STOCK_SELF`) |
| `metrics_after_*.txt` | Prometheus `/metrics` snapshots |
| `dbg_live.txt` | only when warm-up failed |
| `variants/<name>/` | `env.txt`, `src.txt`, `args.txt`, `server.log`, `startup*.txt`, `warmup.json`, `proxy.json`, `proxy.rungs`, `agree.txt`, `metrics_after_proxy.txt`, NCCL logs + summary, `FAILED.txt`/`SKIPPED.txt` |

The tarball is `out/<ts>.tgz`, next to the directory.

## Running on a different base image

`modal_serve.py` builds from the judge's image digest by default (v0.5.23 since Nov 9). To run on another base,
set `OKC_IMAGE` and put the two checkouts at matching commits first, because the checkouts are what land on
PYTHONPATH: `serving/sglang_stock` at the image's build commit (a worktree:
`git -C serving/sglang worktree add ../sglang_stock <commit>`), `serving/sglang` on a branch of the same commit
plus our patches. The Nov 8 and Nov 9 commands below ran on `main-track` over the stand-in `9d547e19`.

The `dev` tag of 2026-11-08 (manifest-list digest, built from SGLang `9d547e19`, pushed 2026-11-08T00:47Z;
`curl -s https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags/dev | python -c 'import json,sys; print(json.load(sys.stdin)["digest"])'`):

    sha256:d6d1cb8d2e10d8be4a7cb3eab6434660a8be115fc511d9501fb3f0df36a2b067

Nov 8 plan (unit tests on both trees, `ar_bench`, stock proxy, patched-tree variant, 5-draft-token variant;
about 83 planned minutes), launched from `serving/`:

    OKC_IMAGE=lmsysorg/sglang:dev@sha256:d6d1cb8d2e10d8be4a7cb3eab6434660a8be115fc511d9501fb3f0df36a2b067 modal run --detach harness/modal_serve.py::session --max-min 90 --stock-mode proxy --pre-tests 1 --ar-bench 1 --variants "patched:SRC=patched;d5:ARGS=--speculative-num-draft-tokens 5"

Nov 9 plan (unit tests, `ar_bench`, stock proxy at c4/c16x2/c32 plus the self-replay, patched-tree variant):

    OKC_IMAGE=lmsysorg/sglang:dev@sha256:d6d1cb8d2e10d8be4a7cb3eab6434660a8be115fc511d9501fb3f0df36a2b067 modal run --detach harness/modal_serve.py::session --max-min 70 --stock-mode proxy --pre-tests 1 --ar-bench 1 --stock-self 1 --proxy-rungs "4x1x2,16x2x2,32x1x1" --variants "patched:SRC=patched"

Its derived budget line is
`[budget] tests+ar_bench+stock+self+1 variants ~ 110 min of MAX_MIN 70 (tests 3, ar_bench 8, stock/proxy 42, stock_self 15, variants 1 x 42; proxy rungs 4x1x2,16x2x2,32x1x1 = 416 requests ~ 30 min)`:
by the 1-min-per-16-requests rule this plan does not fit in 70 minutes, and the variant would be skipped
at about 68 min elapsed (`elapsed + 42 > 70`). Raise `--max-min` (105 is the ceiling under the 2 h Modal
timeout), drop `--stock-self 1`, or shrink the rung set (for example `4x1x2,16x1x2,32x1x1` is 288 requests,
22 min of proxy, 34 per phase, 11 for the self-replay: 3+8+34+11+34 = 90).

Docker Hub tag -> commit: the `nightly-dev-YYYYMMDD-<sha>` tag names the commit the `dev` tag was built from.
