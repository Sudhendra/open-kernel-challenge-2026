#!/usr/bin/env bash
# One paid 8xH200 session: stock baseline + profiles + the runtime facts MAP.md lists as unknown,
# optionally followed by env-var / source-tree / server-flag variants on the proxy workload.
# See harness/README.md.
# Run inside the judge's image (lmsysorg/sglang:v0.5.23), from the serving/ directory, with the
# model weights already on a persistent volume (HF_HOME) so no time is spent downloading.
#
#   bash harness/profile_session.sh [trace.json]
#
# Knobs (env):
#   MAX_MIN=75        hard budget in minutes; watchdog logs elapsed time every 5 min, then aborts
#   VARIANTS=""       "name:ENV=val,ENV=val;name2:ENV=val" -- each restarts the server with the
#                     extra env and runs the proxy workload into $OUT/variants/<name>/.
#                     Reserved keys: SRC=stock|patched (which tree the server imports; default
#                     stock) and ARGS=<server flags> (appended after EXTRA_ARGS; last flag wins).
#   STOCK_MODE=       full (warmup, ladder, profiles, replay), proxy (warmup + proxy only), or reuse: no stock boot at all,
#                     proxy.json (+ proxy_self.json) copied from STOCK_REUSE_DIR (an earlier run on the same node/weights);
#                     variants only. Used by the Modal queue mode to chain experiments on one container.
#                     Default: full without VARIANTS, proxy with VARIANTS (boot is ~11 min, so
#                     full stock + variants does not fit in 75 min).
#   SRC_STOCK=        python dir of the stock tree (default sglang_stock/python if it exists, else
#                     sglang/python). The stock phase and SRC=stock variants import from it.
#   SRC_PATCHED=      python dir of the patched tree (default sglang/python).
#   PRE_TESTS=1       before boot: the three HiCache CPU unit tests, once per tree (10 min cap each)
#   AR_BENCH=1        before boot (GPUs free): harness/ar_bench.py all-reduce sweep (10 min cap)
#   PROXY_RUNGS=      "CxRxS,..." override of the proxy preset (16x1x2) for the stock proxy run and
#                     every variant's replay; the budget estimates scale with it (see below)
#   PROXY_SHAPE=judge conversation shape of the proxy (Nov 15): judge = 13 turns, ~74k first turn, ~753 new tokens per
#                     later turn, 220 out (ladder.py --proxy, the judge JSON's `shape` block); short = the pre-Nov-15
#                     4-turn proxy (ladder.py --proxy-short; reproduces old runs). Reuse mode refuses a stock proxy.json
#                     of the other shape (replay keys and prompts would not match).
#   MEM_GATE=1        per boot: sample nvidia-smi memory.used per GPU every 0.5 s from /health up to the end of the
#                     proxy (mem_run.csv -> mem_peak.json); per variant, harness/mem_gate.py compares the peak to stock's
#                     on the same node -> variants/<v>/mem_over_stock.txt, PASS/WARN/FAIL against 2.0 GB with a 0.3 GB
#                     margin (the judge's gpu_memory.over_stock_gb; research/proxy_rebase_2026-11-15.md section 3).
#   MEM_PROBE=1       with MEM_GATE: after warmup, stock and every variant run harness/hooks/logprob_probe.py (one
#                     16,400-token prompt with return_logprob, the judge's KL-turn shape) inside the memory window, so
#                     both sides of the comparison see that transient. MEM_PROBE= turns it off.
#   MEM_LOADED_PROBE= off by default; "90,150" fires the same logprob probe in the background that many seconds into
#                     the proxy (the first rung, c32 with the judge rungs), i.e. while ~31 users decode. That is the
#                     judge's KL turns interleaved with load, the one shape that can read +2.2 GB for S5 @32 and -0.8
#                     @16 with identical static memory (proxy_rebase section 6.2). Writes mem_probe_loaded_<t>.txt per
#                     boot; mem_gate.py adds "loaded KL transient over stock" as a second number. It perturbs the c32
#                     rung by one 16k prefill (<1% of its prefill work) on both sides; stock must run fresh (not reuse).
#   STOCK_SELF=1      after the stock proxy run, replay it once with the same prompts
#                     (proxy_self.json): the agreement noise floor, passed to agree.py --self
#   STOCK_SELF_RESTART=1  with STOCK_SELF: restart the stock server first (logs in stock_self/), so
#                     both sides of the agreement comparison are fresh boots
#   BOOT_TIMEOUT_MIN=18  per boot: /health not up by then, or a fatal line in server.log (checked every
#                     30 s), writes BOOT_TIMEOUT.txt and stops the server. Stock boot: exit 3. Stock
#                     self-restart and variants: FAILED.txt and the session continues.
#   SERVER_ENV_EXTRA= "K=V,K=V" added to every server launch, stock and variants (a variant's own env
#                     still wins), e.g. to switch off a path that does not work on Modal
#   EXTRA_ARGS, PROF_CONC, SKIP_REPLAY, OUT_ROOT, SGLANG_SRC_MODE: as before.
#   WARM_*            knobs of the PROF=warm profile window (variant key, see profile_loads): WARM_TURNS=5 turns per conversation
#                     of the load, WARM_EXTEND=12 prefill forwards in the window, WARM_SKIP=0 warm prefill batches passed over
#                     before arming, WARM_MAX_STEPS=240 num_steps backstop (forwards), WARM_MIN_CACHED=1024, WARM_WITH_STACK=1
#                     (Python stacks; the variant's server gets SGLANG_PROFILE_WITH_STACK=True), WARM_PYSPY=1 py-spy sidecar on
#                     the rank-0 scheduler for WARM_PYSPY_S=20 s with WARM_PYSPY_ARGS="-r 100 --idle --threads", WARM_WAIT_S=900.
#
# Everything lands in out/<timestamp>/ and is tarred on every exit path (success, failure,
# watchdog abort), so the Modal driver's volume commit always has something to keep.
set -euo pipefail
cd "$(dirname "$0")/.."
TRACE="${1:-}"
PORT=30000
URL="http://127.0.0.1:$PORT"
MAX_MIN="${MAX_MIN:-75}"
VARIANTS="${VARIANTS:-}"
PRE_TESTS="${PRE_TESTS:-}"
AR_BENCH="${AR_BENCH:-}"
PROXY_RUNGS="${PROXY_RUNGS:-}"
STOCK_SELF="${STOCK_SELF:-}"
STOCK_SELF_RESTART="${STOCK_SELF_RESTART:-}"
BOOT_TIMEOUT_MIN="${BOOT_TIMEOUT_MIN:-18}"
[[ "$BOOT_TIMEOUT_MIN" =~ ^[1-9][0-9]*$ ]] || { echo "bad BOOT_TIMEOUT_MIN: $BOOT_TIMEOUT_MIN (want minutes > 0)" >&2; exit 1; }
SERVER_ENV_EXTRA="${SERVER_ENV_EXTRA:-}"
PROXY_SHAPE="${PROXY_SHAPE:-judge}"
case "$PROXY_SHAPE" in judge) PROXY_TURNS=13;; short) PROXY_TURNS=4;; *) echo "bad PROXY_SHAPE: $PROXY_SHAPE (judge|short)" >&2; exit 1;; esac
# PROXY_CORPUS: directory the proxy ladder samples first/later turns from (ladder.py --corpus; default sglang/srt).
# Checked before any boot, so a wrong path costs no server start.
PROXY_CORPUS="${PROXY_CORPUS:-}"
[ -z "$PROXY_CORPUS" ] || [ -d "$PROXY_CORPUS" ] || { echo "PROXY_CORPUS is not a directory: $PROXY_CORPUS" >&2; exit 1; }
# PROXY_CLIENT (Nov 17): how the proxy runs (stock, self, variant replays) talk to the server. Unset or chat = the
# OpenAI chat endpoint with text messages (unchanged); ids = the judge's client, /generate with client-built input_ids
# (harness/ladder.py --client ids). Profile loads and warmup stay on chat.
PROXY_CLIENT="${PROXY_CLIENT:-}"
case "$PROXY_CLIENT" in ""|chat|ids) ;; *) echo "bad PROXY_CLIENT: $PROXY_CLIENT (chat|ids)" >&2; exit 1;; esac
MEM_GATE="${MEM_GATE-1}"
MEM_PROBE="${MEM_PROBE-1}"
MEM_LOADED_PROBE="${MEM_LOADED_PROBE-}"
STOCK_FULL_MIN=45 TESTS_MIN=3 AR_BENCH_MIN=8 STEP_CAP_S=600   # planning estimates; cap for tests/ar_bench
# Proxy time estimate: 2 min fixed + (0.096 + 0.0072 x turns) min per conversation over the rung set (conc x reps x
# convs), capped at 60 min. Fit: Nov 8 the 4-turn 16x1x2 preset (32 conversations) took 3.5 min (formula 6, as the old
# "1 min per 32 requests"); the judge shape at 32x1x1,16x1x1,4x1x2,1x1x2 (58 conversations) is ~11 min for stock from the
# judge's own rung rates (stock c32 wave ~330 s, c16 ~170 s, c4 ~53 s, c1 ~20 s per conversation; ours ~2.3x faster) ->
# formula 13. A variant's boot took 4 min once the first server had warmed the page and JIT caches, so a variant is
# ~8 min + proxy (+3 with MEM_PROBE). The first boot is ~13 min; STOCK_FULL_MIN covers that. A variant is skipped if it cannot fit.
proxy_convs() {   # proxy_convs "CxRxS,..." -> conversation count
  local total=0 r c
  IFS=',' read -ra _rr <<< "$1"
  for r in "${_rr[@]}"; do
    [[ "$r" =~ ^[0-9]+x[0-9]+x[0-9]+$ ]] || { echo "bad PROXY_RUNGS rung: $r (want CxRxS)" >&2; return 1; }
    IFS='x' read -ra c <<< "$r"; total=$(( total + c[0] * c[1] * c[2] ))
  done
  echo "$total"
}
PROXY_CONVS=$(proxy_convs "${PROXY_RUNGS:-16x1x2}") || exit 1
PROXY_REQS=$(( PROXY_CONVS * PROXY_TURNS ))
PROXY_MIN=$(( 2 + (PROXY_CONVS * (96 + 7 * PROXY_TURNS) + 999) / 1000 )); [ "$PROXY_MIN" -gt 60 ] && PROXY_MIN=60
PROBE_MIN=0; [ -n "$MEM_GATE" ] && [ -n "$MEM_PROBE" ] && PROBE_MIN=3
VARIANT_MIN=$(( 8 + PROXY_MIN + PROBE_MIN ))
SELF_MIN=$(( (PROXY_MIN + 1) / 2 ))   # replay of cached prompts: about half the proxy time
if [ -n "$VARIANTS" ]; then STOCK_MODE="${STOCK_MODE:-proxy}"; else STOCK_MODE="${STOCK_MODE:-full}"; fi
# Source trees. Modal mounts sglang_stock/python/sglang next to sglang/python/sglang (both under
# /root/serving); locally sglang_stock is a git worktree. Without a stock tree both point at sglang/.
if [ -d sglang_stock/python ]; then _stock_default="sglang_stock/python"; else _stock_default="sglang/python"; fi
SRC_STOCK="$(cd "${SRC_STOCK:-$_stock_default}" && pwd)"
SRC_PATCHED="$(cd "${SRC_PATCHED:-sglang/python}" && pwd)"
OUT="${OUT_ROOT:-out}/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT/prof"
OUT_ABS="$(cd "$OUT" && pwd)"
T_START=$(date +%s)
# Session stdout/stderr also go to the volume, so a detached run needs no client to read logs.
exec > >(tee -a "$OUT/session.log") 2>&1

LADDER=(python harness/ladder.py --url "$URL" --turns 4)
[ -n "$TRACE" ] && LADDER+=(--trace "$TRACE")
# Proxy preset (PROXY_SHAPE: judge = 13 turns, short = the old 4). Fixed seed/offset so stock and every
# variant see the same first-turn prompts; variants also --replay stock's assistant turns.
if [ "$PROXY_SHAPE" = short ]; then _pflag=--proxy-short; else _pflag=--proxy; fi
PROXY=(python harness/ladder.py --url "$URL" "$_pflag" --seed 4242 --offset 4000)
[ -n "$TRACE" ] && PROXY+=(--trace "$TRACE")
[ -n "$PROXY_RUNGS" ] && PROXY+=(--rungs "$PROXY_RUNGS")   # explicit --rungs wins over the preset
[ -n "$PROXY_CORPUS" ] && PROXY+=(--corpus "$PROXY_CORPUS")
[ "$PROXY_CLIENT" = ids ] && PROXY+=(--client ids)

SERVER="" SAMPLER="" WATCHDOG="" FG="" MEMS="" LPP="" WW="" VFAILED=""
# timeline.txt stays "<event> HH:MM:SS" (report_charts.py splits each line into exactly two fields).
mark() { echo "$1 $(date -u +%T)" >> "$OUT/timeline.txt"; echo "[session] $1 +$(( ($(date +%s) - T_START) / 60 ))m"; }
# Long commands run in the background and are waited on: bash defers traps while a foreground
# child runs, so this is what lets a watchdog TERM interrupt a 10-minute ladder immediately.
run() { "$@" & FG=$!; wait "$FG"; local rc=$?; FG=""; return $rc; }
nap() { local d="${NAP_DIV:-1}"; sleep "$(( d > 1 ? ($1 + d - 1) / d : $1 ))" & wait $!; }   # NAP_DIV: dryrun_session.sh only
# capped <seconds> <cmd...>: like run, but TERM (then KILL) the command past the cap. Returns the
# command's rc, or 124 on timeout (what GNU timeout returns). Used instead of `timeout`, which the
# image may not ship. Polls every 5 s, so a watchdog TERM is seen within 5 s.
capped() { local cap=$1; shift; "$@" & FG=$!; capped_wait "$cap"; }
# capped_wait <seconds>: the wait half of capped(), for a job already started into $FG.
capped_wait() {
  local cap=$1 t0 rc=0; t0=$(date +%s)
  while kill -0 "$FG" 2>/dev/null; do
    if [ $(( $(date +%s) - t0 )) -ge "$cap" ]; then
      echo "[capped] ${cap}s cap hit"; kill -TERM "$FG" 2>/dev/null; sleep 10; kill -KILL "$FG" 2>/dev/null
      wait "$FG" 2>/dev/null || true; FG=""; return 124
    fi
    sleep 5
  done
  wait "$FG" || rc=$?; FG=""; return $rc
}
# Memory window (MEM_GATE): a 0.5 s per-GPU memory.used sampler from /health up to the end of the proxy.
mem_start() {   # mem_start <dir>
  [ -n "$MEM_GATE" ] || return 0
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -lms 500 > "$1/mem_run.csv" 2>/dev/null &
  MEMS=$!
}
mem_stop() {   # mem_stop <dir>: stop the sampler, write <dir>/mem_peak.json
  [ -n "$MEMS" ] || return 0
  kill "$MEMS" 2>/dev/null; wait "$MEMS" 2>/dev/null || true; MEMS=""
  python harness/mem_gate.py peak "$1/mem_run.csv" "$1/mem_peak.json" | tee "$1/mem_peak.txt" || true
}
mem_probe() {   # mem_probe <dir>: the judge's KL-turn shape (16,400-token prompt, return_logprob) inside the window
  [ -n "$MEM_GATE" ] && [ -n "$MEM_PROBE" ] || return 0
  capped 240 python harness/hooks/logprob_probe.py --url "$URL" > "$1/mem_probe.txt" 2>&1 || echo "mem_probe rc=$? (non-fatal)" >> "$1/mem_probe.txt"
  tail -3 "$1/mem_probe.txt"
}
mem_loaded_probe() {   # mem_loaded_probe <dir>: background probes at MEM_LOADED_PROBE seconds into the proxy (LPP pid)
  LPP=""
  [ -n "$MEM_GATE" ] && [ -n "$MEM_LOADED_PROBE" ] || return 0
  ( t0=$(date +%s)
    for t in ${MEM_LOADED_PROBE//,/ }; do
      while [ $(( $(date +%s) - t0 )) -lt "$t" ]; do sleep 1; done
      python harness/hooks/logprob_probe.py --url "$URL" --budget 150 > "$1/mem_probe_loaded_$t.txt" 2>&1 || true   # --budget caps it
    done ) &
  LPP=$!
}
mem_loaded_probe_done() {   # after the proxy: a probe still pending past the proxy end is cut (its file says so)
  [ -n "${LPP:-}" ] || return 0
  pkill -P "$LPP" 2>/dev/null; kill "$LPP" 2>/dev/null; wait "$LPP" 2>/dev/null || true; LPP=""
  ls "$1"/mem_probe_loaded_*.txt >/dev/null 2>&1 && grep -h "max peak-before" "$1"/mem_probe_loaded_*.txt | sed 's/^/  loaded /' || true
}
host_mem() {   # host_mem <dir> <label>: host RAM after a boot (sizes --hicache-size for the next run); Linux only
  awk -v l="$2" '/MemTotal|MemAvailable/ {printf "%s %s %.1f GiB\n", l, $1, $2/1048576}' /proc/meminfo >> "$1/host_mem.txt" 2>/dev/null || true
}

# python dir of a tree: stock | patched | <name> (extra tree given as env SRC_<NAME>=<python dir>, e.g. SRC_L1=/root/serving/sglang_l1/python)
src_dir() { case "$1" in stock) echo "$SRC_STOCK";; patched) echo "$SRC_PATCHED";; *) local _v="SRC_$(echo "$1" | tr '[:lower:]-' '[:upper:]_')"; echo "${!_v:-}";; esac; }
tree_root() { dirname "$1"; }   # <tree>/python -> <tree>

# ---------------------------------------------------------------- validate VARIANTS up front
# VENVS: the ENV=val pairs (comma-joined), VSRCS: stock|patched, VARGS: extra server flags.
VNAMES=() VENVS=() VSRCS=() VARGS=() VPROF=() VHOOK=()   # VHOOK: HOOK=<script> runs after warmup with <url> <variant dir> (e.g. a memory probe)   # VPROF: PROF=1 -> torch-profile the variant at PROF_CONC loads (c1/c4 by default); PROF=cold -> the same loads, but the profile is armed before fresh cold judge-shape first turns (cold 16k prefill chunks, see profile_loads); PROF=warm -> fresh judge-shape conversations, profile armed from server.log at the first later-turn prefill after the cold storm (harness/warm_trigger.py)
if [ -n "$VARIANTS" ]; then
  IFS=';' read -ra _vl <<< "$VARIANTS"
  for v in "${_vl[@]}"; do
    [ -z "${v// /}" ] && continue
    case "$v" in *:*) ;; *) echo "bad variant (no name:): $v"; exit 1;; esac
    name="${v%%:*}"; spec="${v#*:}"
    [[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad variant name: $name"; exit 1; }
    [ "$name" = stock ] && { echo "variant name 'stock' is reserved"; exit 1; }
    IFS=',' read -ra _kv <<< "$spec"
    envs="" src=stock args="" prof="" hook=""
    for kv in "${_kv[@]}"; do
      [[ "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || { echo "bad variant env in $name: $kv"; exit 1; }
      case "${kv%%=*}" in
        SRC) src="${kv#*=}"
             case "$src" in stock|patched) ;; *) [ -n "$(src_dir "$src")" ] && [ -d "$(src_dir "$src")" ] || { echo "bad SRC in $name: $src (stock|patched|<name> with SRC_<NAME>=<python dir> set)"; exit 1; };; esac;;
        ARGS) args="${kv#*=}";;
        PROF) prof="${kv#*=}";;
        HOOK) hook="${kv#*=}";;
        *) envs="${envs:+$envs,}$kv";;
      esac
    done
    VNAMES+=("$name"); VENVS+=("$envs"); VSRCS+=("$src"); VARGS+=("$args"); VPROF+=("$prof"); VHOOK+=("$hook")
  done
  echo "variants: ${VNAMES[*]}"
  [ "${#VNAMES[@]}" -gt 2 ] && echo "WARNING: ${#VNAMES[@]} variants; only 2 + stock fit in 75 min with the default rungs, later ones will be skipped by the budget check"
fi
SERVER_ENV=()
if [ -n "$SERVER_ENV_EXTRA" ]; then
  IFS=',' read -ra _kv <<< "$SERVER_ENV_EXTRA"
  for kv in "${_kv[@]}"; do
    [[ "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || { echo "bad SERVER_ENV_EXTRA entry: $kv (want K=V,K=V)"; exit 1; }
    SERVER_ENV+=("$kv")
  done
  echo "server env extra: ${SERVER_ENV[*]}"
fi
# Planned budget, from the per-step estimates above (boot included in stock/variant figures).
if [ "$STOCK_MODE" = full ]; then _stock_est=$(( STOCK_FULL_MIN + PROXY_MIN )); elif [ "$STOCK_MODE" = reuse ]; then _stock_est=0; else _stock_est=$VARIANT_MIN; fi
_tests_est=0; [ -n "$PRE_TESTS" ] && _tests_est=$TESTS_MIN
_ar_est=0; [ -n "$AR_BENCH" ] && _ar_est=$AR_BENCH_MIN
_self_est=0; [ -n "$STOCK_SELF" ] && _self_est=$SELF_MIN
# A restarted self pass has a boot and cold prompts, so it costs what a variant costs.
[ -n "$STOCK_SELF" ] && [ -n "$STOCK_SELF_RESTART" ] && _self_est=$VARIANT_MIN
_plan=$(( _tests_est + _ar_est + _stock_est + _self_est + ${#VNAMES[@]} * VARIANT_MIN ))
echo "[budget] tests+ar_bench+stock+self+${#VNAMES[@]} variants ~ ${_plan} min of MAX_MIN ${MAX_MIN} (tests ${_tests_est}, ar_bench ${_ar_est}, stock/${STOCK_MODE} ${_stock_est}, stock_self ${_self_est}, variants ${#VNAMES[@]} x ${VARIANT_MIN}; proxy ${PROXY_SHAPE} shape, rungs ${PROXY_RUNGS:-16x1x2} = ${PROXY_CONVS} conversations x ${PROXY_TURNS} turns = ${PROXY_REQS} requests ~ ${PROXY_MIN} min; mem probe ${PROBE_MIN} min/boot)"
[ "$_plan" -gt "$MAX_MIN" ] && echo "WARNING: planned ${_plan} min exceeds MAX_MIN ${MAX_MIN}; late variants will be skipped by the budget check"
# Host-RAM preflight for the HiCache pool: 8 ranks pin <hicache-size> GB each (the last --hicache-size in EXTRA_ARGS
# wins, as in argparse). Over 75% of MemTotal leaves too little for 8 rank processes + loader: warn (Linux only).
_hc=$(printf '%s\n' ${EXTRA_ARGS:-} | awk 'p {v=$0; p=0} /^--hicache-size$/ {p=1} END {print v}')
_memtot=$(awk '/MemTotal/ {printf "%d", $2/1048576*1.073741824}' /proc/meminfo 2>/dev/null || echo "")
if [ -n "$_hc" ] && [ -n "$_memtot" ]; then
  echo "[host] hicache ${_hc} GB/rank x 8 = $(( _hc * 8 )) GB pinned of MemTotal ${_memtot} GB"
  [ $(( _hc * 8 * 100 )) -gt $(( _memtot * 75 )) ] && echo "WARNING: hicache pool over 75% of host RAM; expect host OOM kills"
fi

# ---------------------------------------------------------------- timing / NCCL env for the server
# Always-on timing: per-step device timer, step-time histogram, forward-iter ids, ms timestamps,
# decode graph keys, NVTX scheduler ranges (only if the nvtx package imports; without nsys attached
# the ranges are no-ops). Profiler stack/shape recording off: /start_profile traces stay small.
TIMING_ENV=(
  SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 SGLANG_RECORD_STEP_TIME=1 SGLANG_LOG_FORWARD_ITERS=1
  SGLANG_LOG_MS=1 SGLANG_LOG_DECODE_GRAPH_KEY=${LOG_GRAPH_KEY:-0}
  SGLANG_PROFILE_WITH_STACK=False SGLANG_PROFILE_RECORD_SHAPES=False
)
if python -c 'import nvtx' 2>/dev/null; then
  TIMING_ENV+=(SGLANG_ENABLE_NVTX_SCHEDULER=1)
else
  echo "nvtx not importable: SGLANG_ENABLE_NVTX_SCHEDULER left off" | tee "$OUT/nvtx_missing.txt"
fi
printf '%s\n' "${TIMING_ENV[@]}" > "$OUT/timing_env.txt"

start_server() {   # start_server <dir> <stock|patched> "<extra flags>" [ENV=val ...]
  local dir=$1 src=$2 vargs=$3; shift 3
  local dabs; dabs="$(cd "$dir" && pwd)"
  mkdir -p "$dir/prof"
  # NCCL start-up log: every rank writes its own small file (%h host, %p pid).
  # The judge's launch command (organisers' Nov 9 listing for the v0.5.23 pin). --enable-metrics and
  # --decode-log-interval are ours, for stage timings. The judge uses --hicache-size 160; Modal's
  # 512 GiB container cannot hold 160 GB per rank, so EXTRA_ARGS (default --hicache-size 16 in
  # modal_serve.py) must stay after the judge flags. Variant ARGS come after it (argparse: last
  # occurrence wins).
  # SERVER_ENV_EXTRA and then the variant env come last so they override the defaults above them. The
  # selected tree goes first on PYTHONPATH so `import sglang` resolves to it rather than the image's install.
  # shellcheck disable=SC2086
  env "${TIMING_ENV[@]}" \
    PYTHONPATH="$(src_dir "$src")${PYTHONPATH:+:$PYTHONPATH}" \
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,NVLS NCCL_DEBUG_FILE="$dabs/nccl_%h_%p.log" \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 SGLANG_TIMEOUT_KEEP_ALIVE=1800 \
    SGLANG_TORCH_PROFILER_DIR="$dabs/prof" \
    ${SERVER_ENV[@]+"${SERVER_ENV[@]}"} \
    ${1+"$@"} \
  python -m sglang.launch_server \
    --model-path meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
    --speculative-draft-model-path RadixArk/Muse-Spark-Open-DFlash2 --tp-size 8 --ep-size 8 \
    --quantization modelopt_fp4 --moe-runner-backend marlin --mem-fraction-static 0.85 \
    --speculative-algorithm DFLASH --speculative-num-draft-tokens 8 \
    --speculative-draft-attention-backend fa4 --mm-feature-transport cpu \
    --enable-hierarchical-cache --hicache-size 160 --chunked-prefill-size 16384 --context-length 131072 \
    --enable-cache-report --incremental-streaming-output --random-seed 1234 \
    --max-running-requests 32 --cuda-graph-max-bs-decode 32 \
    --enable-metrics --decode-log-interval 20 --port "$PORT" ${EXTRA_ARGS:-} ${vargs} > "$dir/server.log" 2>&1 < /dev/null &
  SERVER=$!
  echo "$src $(src_dir "$src")" > "$dir/src.txt"
  echo "${EXTRA_ARGS:-} ${vargs}" > "$dir/args.txt"
}

# Lines that mean this boot will never finish. A hung rank keeps the process alive, so the kill -0
# below cannot see it (Nov 9: 58 min of polling after a multicast failure on one rank). Past boots
# had none of these before /health came up. Fixed strings, passed to grep -F.
BOOT_FATAL=(-e "Setup Custom allreduce failed" -e "CUDA_ERROR_UNKNOWN" -e "Traceback (most recent call last)")
BOOT_SCAN_S="${BOOT_SCAN_S:-30}"

boot_abort() {   # boot_abort <dir> "<reason>": record why, stop the server and any leftover ranks
  local dir=$1 reason=$2
  echo "[boot] aborted: $reason"
  { echo "$reason"; echo "--- last 80 lines of server.log"; tail -80 "$dir/server.log"; } > "$dir/BOOT_TIMEOUT.txt"
  stop_server
  # A rank that never allocated leaves the GPUs "empty", so stop_server's own pkill may not have run.
  { command -v pkill > /dev/null && pkill -KILL -f 'sglang::|sglang.launch_server'; } || true
}

boot_failed() {   # boot_failed <dir> <wait_healthy rc>: FAILED.txt for a boot the session survives
  local dir=$1 rc=$2
  if [ "$rc" -eq 3 ]; then echo "boot aborted: $(head -1 "$dir/BOOT_TIMEOUT.txt")" > "$dir/FAILED.txt"; else echo "server died" > "$dir/FAILED.txt"; fi
  stop_server
}

wait_healthy() {   # wait_healthy <dir>; returns 1 if the server died, 3 on boot timeout or a fatal log line
  local dir=$1 t0 el last_scan=0 hit hit_el="" hit_lines="" nl; t0=$(date +%s)
  until curl -sf --max-time 10 "$URL/health" > /dev/null; do
    kill -0 "$SERVER" 2>/dev/null || { echo "server died"; tail -50 "$dir/server.log"; return 1; }
    el=$(( $(date +%s) - t0 ))
    if [ $(( el - last_scan )) -ge "$BOOT_SCAN_S" ]; then
      last_scan=$el
      hit=$(grep -n -m1 -F "${BOOT_FATAL[@]}" "$dir/server.log" 2>/dev/null || true)
      # A fatal-looking line only aborts the boot once the log has stopped growing for BOOT_STALL_S after it:
      # Nov 13 a rank's cuMulticastAddDevice CUDA_ERROR_UNKNOWN (torch symmetric-memory multicast init, falls
      # back) was logged and the boot carried on; the old immediate abort cost a Modal session.
      if [ -n "$hit" ]; then
        nl=$(wc -l < "$dir/server.log")
        if [ -z "$hit_el" ] || [ "$nl" != "$hit_lines" ]; then hit_el=$el; hit_lines=$nl; echo "[boot] suspicious line at ${el}s, watching for a stall: $hit"
        elif [ $(( el - hit_el )) -ge "${BOOT_STALL_S:-180}" ]; then boot_abort "$dir" "fatal server.log line, log stalled ${BOOT_STALL_S:-180}s after it (at ${hit_el}s): $hit"; return 3; fi
      fi
    fi
    if [ "$el" -ge $(( BOOT_TIMEOUT_MIN * 60 )) ]; then
      boot_abort "$dir" "/health not up after ${el}s (BOOT_TIMEOUT_MIN ${BOOT_TIMEOUT_MIN})"; return 3
    fi
    nap 10
  done
  echo "startup_s $(( $(date +%s) - t0 ))" | tee "$dir/startup.txt"
  grep -iE "max_total_num_tokens|KV cache|mem_fraction|cuda graph|capture|jit|compil|attention backend|dflash|spec|multicast|custom allreduce|warning" "$dir/server.log" > "$dir/startup_facts.txt" || true
}

stop_server() {
  [ -n "$SERVER" ] || return 0
  kill -TERM "$SERVER" 2>/dev/null || true   # SGLang handles TERM: graceful stop of all ranks
  local i used
  for i in $(seq 1 60); do kill -0 "$SERVER" 2>/dev/null || break; sleep 2; done
  kill -KILL "$SERVER" 2>/dev/null || true
  wait "$SERVER" 2>/dev/null || true
  SERVER=""
  # A restart needs the GPUs empty: wait for scheduler processes to release memory.
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    [ "${used:-0}" -lt 2000 ] && return 0
    [ "$i" -eq 30 ] && command -v pkill > /dev/null && pkill -KILL -f 'sglang::|sglang.launch_server' || true
    sleep 2
  done
  echo "WARNING: GPU memory still ${used} MiB after server stop"
}

nccl_summary() {   # TUNING and NVLS lines from every rank's NCCL log in <dir>
  local dir=$1
  ls "$dir"/nccl_*.log > /dev/null 2>&1 || return 0
  grep -HE "TUNING|Tuning|NVLS|nvls|Algorithm|Protocol|Max NThreads|Latency|Bandwidth" "$dir"/nccl_*.log \
    | sed "s|^$dir/||" > "$dir/nccl_summary.txt" 2>/dev/null || true
}

warmup() {   # warmup <dir>; returns 1 if every warmup request failed
  local dir=$1
  run "${LADDER[@]}" --rungs 1x1x1 --seed 999 --offset 900 --turns 2 --out "$dir/warmup.json" || return 1
  # A failing ladder must not burn the session.
  python - "$dir/warmup.json" <<'PY' || return 1
import json, sys
d = json.load(open(sys.argv[1])); r = d["records"]
bad = [x for x in r if not x.get("completion_tokens")]
print("warmup records", len(r), "bad", len(bad))
for x in bad[:2]: print(json.dumps({k: x.get(k) for k in ("status", "error", "raw_head", "chunks")})[:800])
sys.exit(1 if bad and len(bad) == len(r) else 0)
PY
}

tree_facts() {   # tree_facts <stock|patched>: commit + diff of that tree ("no git" when only python/sglang was mounted)
  local src=$1 root; root="$(tree_root "$(src_dir "$src")")"
  if git -C "$root" rev-parse HEAD > "$OUT/commit_$src.txt" 2>/dev/null; then
    git -C "$root" diff > "$OUT/patch_$src.diff" 2>/dev/null || true
  else
    echo "no git" > "$OUT/commit_$src.txt"; : > "$OUT/patch_$src.diff"
  fi
  PYTHONPATH="$(src_dir "$src")${PYTHONPATH:+:$PYTHONPATH}" python -c 'import sglang; print(sglang.__file__)' > "$OUT/sglang_path_$src.txt" 2>&1 || true
  echo "$src: $(cat "$OUT/commit_$src.txt") $(cat "$OUT/sglang_path_$src.txt")"
}

# The HiCache CPU unit tests from the mem_cache folder, against one tree's PYTHONPATH. Files a tree
# lacks are dropped (the v0.5.23 pin has no test_page_unified_staging.py): pytest aborts on a missing path.
UNIT_TESTS=(test_asymmetric_mha_pool_host_unit.py test_page_unified_staging.py test_hicache_staged_write_back_dispatch.py)
unit_tests() {   # unit_tests <stock|patched>; non-fatal, STEP_CAP_S cap, output in $OUT/unit_tests_<src>.txt
  local src=$1 pydir root testdir cwd rc=0 f runner present=()
  pydir="$(src_dir "$src")"; root="$(tree_root "$pydir")"
  testdir="test/registered/unit/mem_cache"; cwd="$root"
  {
    if [ ! -d "$root/$testdir" ]; then
      # Only python/sglang of this tree is mounted: run the patched tree's test files against it.
      cwd="$(tree_root "$SRC_PATCHED")"
      echo "NOTE: $root has no $testdir; running the patched tree's test files ($cwd/$testdir) against PYTHONPATH=$pydir"
    fi
    for f in "${UNIT_TESTS[@]}"; do
      if [ -f "$cwd/$testdir/$f" ]; then present+=("$f"); else echo "NOTE: skipping missing test file $cwd/$testdir/$f"; fi
    done
    if PYTHONPATH="$pydir" python -m pytest --version > /dev/null 2>&1; then
      runner=(python -m pytest -q -p no:cacheprovider)
    else
      echo "NOTE: pytest not importable; falling back to python -m unittest"
      runner=(python -m unittest -v)
    fi
    echo "cwd=$cwd PYTHONPATH=$pydir"; echo "+ ${runner[*]} ${present[*]/#/$testdir/}"
    ( cd "$cwd" && PYTHONPATH="$pydir${PYTHONPATH:+:$PYTHONPATH}" exec "${runner[@]}" "${present[@]/#/$testdir/}" ) &
    FG=$!; capped_wait "$STEP_CAP_S" || rc=$?
    echo "unit_tests_$src rc=$rc"
  } > "$OUT/unit_tests_$src.txt" 2>&1
  echo "$src $rc" >> "$OUT/unit_tests_rc.txt"
  tail -5 "$OUT/unit_tests_$src.txt"
  return 0
}

finish() {   # EXIT trap: runs on success, error, and watchdog/Modal TERM
  local rc=$?
  set +e
  trap - EXIT TERM INT
  [ -n "$FG" ] && kill -TERM "$FG" 2>/dev/null
  [ -n "$WATCHDOG" ] && kill "$WATCHDOG" 2>/dev/null
  stop_server
  [ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null
  [ -n "$MEMS" ] && kill "$MEMS" 2>/dev/null
  [ -n "${LPP:-}" ] && kill "$LPP" 2>/dev/null
  [ -n "${WW:-}" ] && kill -TERM "$WW" 2>/dev/null   # PROF=warm watcher: on TERM it posts /stop_profile if armed
  for d in "$OUT" "$OUT/stock_self" "$OUT"/variants/*/; do [ -d "$d" ] && nccl_summary "${d%/}"; done
  cp "$OUT/server.log" "$OUT/server_full.log" 2>/dev/null
  echo "$rc" > "$OUT/exit_code.txt"
  mark session_done
  tar czf "$OUT.tgz" "$OUT"
  echo "done: $OUT.tgz (rc=$rc)"
  exit $rc
}
trap finish EXIT
trap 'echo "[session] caught signal, finishing"; exit 143' TERM INT

# ---------------------------------------------------------------- node facts
nvidia-smi --query-gpu=index,name,memory.total --format=csv > "$OUT/gpus.csv"
[ "$(wc -l < "$OUT/gpus.csv")" -eq 9 ] || { echo "need 8 GPUs"; cat "$OUT/gpus.csv"; exit 1; }

# Our checkout replaces the image's sglang: pip -e normally, or PYTHONPATH where the tree is read-only.
# (Modal uses pythonpath; start_server also puts the selected tree first on PYTHONPATH explicitly.)
if [ "${SGLANG_SRC_MODE:-pip}" = pip ]; then pip install -e "$SRC_PATCHED" --no-deps -q; fi
python -c 'import sglang; print(sglang.__file__)' | tee "$OUT/sglang_path.txt"
echo "SRC_STOCK=$SRC_STOCK"; echo "SRC_PATCHED=$SRC_PATCHED"
tree_facts stock
tree_facts patched
cp "$OUT/commit_patched.txt" "$OUT/commit.txt"; cp "$OUT/patch_patched.diff" "$OUT/patch.diff"   # legacy names
nvidia-smi topo -m > "$OUT/topo.txt" 2>&1 || true
pip freeze > "$OUT/pip_freeze.txt" 2>/dev/null || true
nvidia-smi > "$OUT/nvidia_smi.txt" || true
{ echo "MAX_MIN=$MAX_MIN"; echo "STOCK_MODE=$STOCK_MODE"; echo "VARIANTS=$VARIANTS"; echo "EXTRA_ARGS=${EXTRA_ARGS:-}"
  echo "SRC_STOCK=$SRC_STOCK"; echo "SRC_PATCHED=$SRC_PATCHED"; echo "PRE_TESTS=$PRE_TESTS"; echo "AR_BENCH=$AR_BENCH"
  echo "PROXY_RUNGS=$PROXY_RUNGS"; echo "STOCK_SELF=$STOCK_SELF"; echo "STOCK_SELF_RESTART=$STOCK_SELF_RESTART"; echo "BOOT_TIMEOUT_MIN=$BOOT_TIMEOUT_MIN"; echo "SERVER_ENV_EXTRA=$SERVER_ENV_EXTRA"; echo "PROXY_MIN=$PROXY_MIN VARIANT_MIN=$VARIANT_MIN"; echo "PROXY_SHAPE=$PROXY_SHAPE PROXY_TURNS=$PROXY_TURNS PROXY_CORPUS=$PROXY_CORPUS"; echo "MEM_GATE=$MEM_GATE MEM_PROBE=$MEM_PROBE MEM_LOADED_PROBE=$MEM_LOADED_PROBE"; [ -z "$PROXY_CLIENT" ] || echo "PROXY_CLIENT=$PROXY_CLIENT"; } > "$OUT/session_args.txt"

# Hard budget: log elapsed minutes every 5 min; past MAX_MIN, TERM the session (finish() tars).
(
  next=5
  while sleep 60; do
    m=$(( ($(date +%s) - T_START) / 60 ))
    if [ "$m" -ge "$next" ]; then echo "[watchdog] elapsed ${m} min of MAX_MIN ${MAX_MIN}" | tee -a "$OUT/watchdog.log"; next=$(( next + 5 )); fi
    if [ "$m" -ge "$MAX_MIN" ]; then
      echo "BUDGET EXCEEDED at ${m} min (MAX_MIN ${MAX_MIN}), aborting" | tee -a "$OUT/ABORTED.txt" "$OUT/watchdog.log"
      kill -TERM $$
      exit 0
    fi
  done
) &
WATCHDOG=$!

# ---------------------------------------------------------------- pre-boot: unit tests, ar_bench
if [ -n "$PRE_TESTS" ]; then
  : > "$OUT/unit_tests_rc.txt"
  for src in stock patched; do mark "unit_tests_${src}_start"; unit_tests "$src"; done
  mark unit_tests_done; cat "$OUT/unit_tests_rc.txt"
  # EXTRA_TESTS: comma-separated files relative to the patched tree root (GPU kernel tests of a lane);
  # patched tree only, non-fatal, same cap. Output in $OUT/extra_tests.txt.
  if [ -n "${EXTRA_TESTS:-}" ]; then
    mark extra_tests_start
    _xroot="$(tree_root "$SRC_PATCHED")"; _xrc=0
    {
      echo "cwd=$_xroot PYTHONPATH=$SRC_PATCHED"; echo "+ python -m pytest -q -p no:cacheprovider ${EXTRA_TESTS//,/ }"
      ( cd "$_xroot" && PYTHONPATH="$SRC_PATCHED${PYTHONPATH:+:$PYTHONPATH}" exec python -m pytest -q -p no:cacheprovider ${EXTRA_TESTS//,/ } ) &
      FG=$!; capped_wait "$STEP_CAP_S" || _xrc=$?
      echo "extra_tests rc=$_xrc"
    } > "$OUT/extra_tests.txt" 2>&1
    echo "extra $_xrc" >> "$OUT/unit_tests_rc.txt"; tail -5 "$OUT/extra_tests.txt"; mark extra_tests_done
  fi
fi
if [ -n "$AR_BENCH" ]; then
  # GPUs are still free here. Defaults (10 sizes x 10 paths x eager+graph x 3 blocks x 220 calls)
  # are a few minutes; the cap is for hangs (ar_bench's own --size-timeout is 20 min per size).
  mark ar_bench_start
  PYTHONPATH="$SRC_STOCK${PYTHONPATH:+:$PYTHONPATH}" capped "$STEP_CAP_S" python harness/ar_bench.py --out "$OUT/ar_bench.jsonl" > "$OUT/ar_bench.txt" 2>&1 \
    && echo "ar_bench ok" || echo "ar_bench rc=$? (non-fatal)"
  grep -c '"M"' "$OUT/ar_bench.jsonl" 2>/dev/null | sed 's/^/ar_bench rows: /' || true
  mark ar_bench_done
fi

# Interconnect: peer access + NCCL all-reduce at the exact per-layer message sizes.
run python harness/nccl_test.py > "$OUT/nccl.jsonl" 2> "$OUT/nccl.err" || true

# 1 Hz GPU utilisation/power/memory for the whole session: idle gaps show up here.
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used --format=csv -l 1 > "$OUT/gpu_util.csv" &
SAMPLER=$!

profile_loads() {   # profile_loads <dir> <mark prefix> "<loads>" [cold|warm]: torch profiler under a load at each C
  # Default: delay before profiling = time for every slot's cold first-turn prefill to finish (measured run 2:
  # ~7 s per 64k-token conversation, queued, so c32 needs ~3 min), so the window holds warm extends + decode only.
  # cold (variant key PROF=cold, Nov 16): /start_profile (profile_by_stage) is posted BEFORE the load starts, on an idle
  # server, with fresh judge-shape first turns (~74.2k tokens: 4 full 16,384-token chunks + a partial per conversation).
  # profile_by_stage starts the EXTEND trace at the first prefill batch (cold chunk 1) and stops it at the first
  # decode batch, which comes only after every queued first-turn prefill (prefill is scheduled before decode), capped at
  # PROF_STEPS prefill batches; then PROF_STEPS decode steps. Files: <prefix>-...-TP-<r>-EP-<r>-EXTEND.trace.json.gz.
  # warm (variant key PROF=warm, Nov 17): fresh judge-shape conversations (WARM_TURNS turns) at each C. harness/warm_trigger.py
  # follows server.log from its end at load start and POSTs /start_profile (CPU+GPU, with_stack, no profile_by_stage, num_steps
  # WARM_MAX_STEPS as a backstop) at the first "Prefill batch" with #cached-token >= WARM_MIN_CACHED and #queue-req 0 after a
  # "Decode batch" line (= the cold first-turn storm is over), then /stop_profile after WARM_EXTEND more prefill forwards, so the
  # window holds later-turn prefill forwards and the decode steps between them on every rank. Optional py-spy sidecar on the
  # rank-0 scheduler from the same trigger. Files: <dir>/prof/warm_c<C>-<id>-TP-<r>-EP-<r>.trace.json.gz (no stage suffix),
  # <dir>/prof/pyspy_rank0_warm_c<C>.txt (raw folded stacks), <dir>/warm_trigger_warm_c<C>.{json,log} (arm/stop lines, forward ids).
  # Traces land in <dir>/prof (SGLANG_TORCH_PROFILER_DIR is set per server dir by start_server).
  local d=$1 pfx=$2 loads=$3 mode=${4:-} C DELAY freegb tag lflags
  for C in $loads; do
    case $C in 1) DELAY=9;; 4) DELAY=14;; 16) DELAY=70;; 32) DELAY=185;; *) DELAY=60;; esac
    tag="c$C"; lflags=(--turns 8 --seed "$((100 + C))" --offset "$((100 * C))")
    if [ "$mode" = cold ]; then tag="cold_c$C"; lflags=(--shape judge --turns 2 --seed "$((7100 + C))" --offset "$((7100 * C))"); fi
    if [ "$mode" = warm ]; then tag="warm_c$C"; lflags=(--shape judge --turns "${WARM_TURNS:-5}" --seed "$((8100 + C))" --offset "$((8100 * C))"); fi
    # Host-RAM guard (Nov 13: the 8-rank trace export OOM-killed the server twice in a 512 GiB container when several
    # profiled variants ran back to back). Skip the profile instead of losing the server; PROF_STEPS shortens the window.
    freegb=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 999)
    if [ "${freegb:-999}" -lt "${PROF_MIN_FREE_GB:-150}" ]; then
      echo "[prof] $tag skipped: host MemAvailable ${freegb} GB < PROF_MIN_FREE_GB ${PROF_MIN_FREE_GB:-150}" | tee -a "$d/profile_replies.txt"; mark "${pfx}prof_${tag}_skipped_lowmem"; continue
    fi
    prof_post() { curl -s -X POST "$URL/start_profile" -H 'Content-Type: application/json' \
      -d "{\"num_steps\": ${PROF_STEPS:-20}, \"activities\": [\"CPU\", \"GPU\"], \"profile_by_stage\": true, \"record_shapes\": false, \"profile_prefix\": \"$tag\"}" | tee -a "$d/profile_replies.txt"; echo " [$tag]" >> "$d/profile_replies.txt"; }
    [ "$mode" = cold ] && prof_post   # armed on the idle server; the first prefill batch of the load below starts the trace
    [ "$mode" = warm ] && warm_watch_start "$d" "$tag"   # background watcher on server.log; it arms the profile itself
    "${LADDER[@]}" --rungs "${C}x1x1" "${lflags[@]}" --out "$d/prof_load_$tag.json" > /dev/null &
    FG=$!
    if [ -z "$mode" ]; then nap $DELAY; prof_post; fi
    wait "$FG" || true; FG=""
    [ "$mode" = warm ] && warm_watch_done "$d" "$tag"
    nap "${PROF_EXPORT_NAP:-120}"   # trace export of 8 ranks is slow and memory-hungry; let it finish (and free host RAM) before the next load
    echo "[prof] $tag host MemAvailable after export: $(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null) GB" | tee -a "$d/profile_replies.txt"
    curl -s "$URL/metrics" > "$d/metrics_after_prof_$tag.txt" || true
    mark "${pfx}prof_${tag}_done"
  done
}

warm_watch_start() {   # warm_watch_start <dir> <tag>: harness/warm_trigger.py in the background (pid WW), from server.log's current end
  local d=$1 tag=$2 off spy=()
  off=$(wc -c < "$d/server.log" | tr -d ' ')
  rm -f "$d/.load_done_$tag"
  [ "${WARM_PYSPY:-1}" != 0 ] && spy=(--pyspy-out "$d/prof/pyspy_rank0_$tag.txt" --pyspy-s "${WARM_PYSPY_S:-20}" "--pyspy-args=${WARM_PYSPY_ARGS:--r 100 --idle --threads}")
  python harness/warm_trigger.py watch --log "$d/server.log" --offset "$off" --url "$URL" --prefix "$tag" \
    --extend "${WARM_EXTEND:-12}" --skip "${WARM_SKIP:-0}" --max-steps "${WARM_MAX_STEPS:-240}" --min-cached "${WARM_MIN_CACHED:-1024}" \
    --with-stack "${WARM_WITH_STACK:-1}" --done-file "$d/.load_done_$tag" --summary "$d/warm_trigger_$tag.json" ${spy[@]+"${spy[@]}"} \
    > "$d/warm_trigger_$tag.log" 2>&1 &
  WW=$!
  echo "[prof] $tag warm trigger armed on $d/server.log from byte $off (pid $WW)" | tee -a "$d/profile_replies.txt"
}
warm_watch_done() {   # warm_watch_done <dir> <tag>: the load ended; the watcher stops an open window, waits for the export + py-spy (capped)
  local d=$1 tag=$2 rc=0
  : > "$d/.load_done_$tag"
  [ -n "$WW" ] || return 0
  FG=$WW; WW=""; capped_wait "${WARM_WAIT_S:-900}" || rc=$?
  tee -a "$d/profile_replies.txt" < "$d/warm_trigger_$tag.log" | grep -E "ARM|STOP|start reply|stop reply|py-spy|outcome|nothing profiled" || true
  echo "[prof] $tag warm trigger rc=$rc" | tee -a "$d/profile_replies.txt"
}

# ---------------------------------------------------------------- stock
if [ "$STOCK_MODE" = reuse ]; then
  [ -n "${STOCK_REUSE_DIR:-}" ] && [ -f "$STOCK_REUSE_DIR/proxy.json" ] || { echo "STOCK_MODE=reuse needs STOCK_REUSE_DIR with proxy.json"; exit 1; }
  cp "$STOCK_REUSE_DIR/proxy.json" "$OUT/proxy.json"; cp "$STOCK_REUSE_DIR/proxy.rungs" "$OUT/proxy.rungs" 2>/dev/null || true
  [ -f "$STOCK_REUSE_DIR/proxy_self.json" ] && cp "$STOCK_REUSE_DIR/proxy_self.json" "$OUT/proxy_self.json"
  for f in server_info.json config_*.json self_agreement.txt mem_peak.json mem_peak.txt host_mem.txt; do cp "$STOCK_REUSE_DIR"/$f "$OUT/" 2>/dev/null || true; done
  # The reused stock must be the same conversation shape as this session's proxy (pre-Nov-15 files have no "shape": short).
  _rshape=$(python -c 'import json,sys; a=json.load(open(sys.argv[1])).get("args") or {}; print(a.get("shape") or ("short" if a.get("turns", 4) == 4 else "legacy"))' "$OUT/proxy.json" 2>/dev/null || echo unreadable)
  if [ "$_rshape" != "$PROXY_SHAPE" ]; then
    echo "STOCK_REUSE_DIR proxy.json is shape '$_rshape' but PROXY_SHAPE=$PROXY_SHAPE: refusing (set PROXY_SHAPE=$_rshape, or re-run stock)"; exit 1
  fi
  [ -n "$MEM_GATE" ] && [ ! -f "$OUT/mem_peak.json" ] && echo "WARNING: $STOCK_REUSE_DIR has no mem_peak.json: variants' mem_over_stock will read UNKNOWN"
  echo "$STOCK_REUSE_DIR" > "$OUT/stock_reused_from.txt"; echo "stock proxy reused from $STOCK_REUSE_DIR"; mark stock_reused
else
start_server "$OUT" stock ""
wait_healthy "$OUT" || exit $?
mark server_up
mem_start "$OUT"; host_mem "$OUT" stock

# Runtime facts.
for m in meta-models/Muse-Spark-Open-816B-A42B-NVFP4 RadixArk/Muse-Spark-Open-DFlash2; do
  python - "$m" > "$OUT/config_$(basename "$m").json" <<'PY' || true
import sys
from huggingface_hub import hf_hub_download
print(open(hf_hub_download(sys.argv[1], "config.json")).read())
PY
done
curl -s "$URL/get_server_info" > "$OUT/server_info.json" || true

warmup "$OUT" || { echo "WARMUP FAILED, aborting"; bash harness/dbg_live.sh "$OUT" > "$OUT/dbg_live.txt" 2>&1; exit 2; }
mark warmup_done
curl -s "$URL/metrics" > "$OUT/metrics_after_warmup.txt" || true
mem_probe "$OUT"

if [ "$STOCK_MODE" = proxy ] || [ -n "$VARIANTS" ]; then
  # The reference every variant is compared against (same prompts, fresh server each time).
  mem_loaded_probe "$OUT"
  run "${PROXY[@]}" --out "$OUT/proxy.json" > "$OUT/proxy.rungs"; cat "$OUT/proxy.rungs"
  curl -s "$URL/metrics" > "$OUT/metrics_after_proxy.txt" || true
  mem_loaded_probe_done "$OUT"
  mem_stop "$OUT"
  mark proxy_done
  if [ -n "$STOCK_SELF" ]; then
    # The stock-vs-stock agreement noise floor that every variant's agree.txt is read against; not a
    # speed number. PROXY carries the seed and offset: another seed redraws the user turns. Without a
    # restart the prompts are already prefix-cached on this server.
    SELF_READY=1
    if [ -n "$STOCK_SELF_RESTART" ]; then
      stop_server
      mark stock_self_restart_start
      mkdir -p "$OUT/stock_self"
      start_server "$OUT/stock_self" stock ""
      if wait_healthy "$OUT/stock_self"; then
        warmup "$OUT/stock_self" || { echo "warmup failed" > "$OUT/stock_self/FAILED.txt"; SELF_READY=""; }
      else
        boot_failed "$OUT/stock_self" $?; SELF_READY=""
      fi
    fi
    if [ -n "$SELF_READY" ]; then
      run "${PROXY[@]}" --replay "$OUT/proxy.json" --out "$OUT/proxy_self.json" > "$OUT/proxy_self.rungs" || true
      python harness/agree.py "$OUT/proxy.json" "$OUT/proxy_self.json" > "$OUT/self_agreement.txt" 2>&1 || true
      cat "$OUT/self_agreement.txt"
      mark proxy_self_done
    fi
  fi
fi


if [ "$STOCK_MODE" = full ]; then
  # A ladder at each load. Fresh seed so the prompts are not already cached.
  run "${LADDER[@]}" --rungs 1x1x1,4x1x1,16x1x1,32x1x1 --out "$OUT/stock.json" > "$OUT/stock.rungs"; cat "$OUT/stock.rungs"
  curl -s "$URL/metrics" > "$OUT/metrics_after_stock.txt" || true
  mem_stop "$OUT"   # no-op if the proxy already closed the window; profiling below would inflate the peak
  mark stock_done

  profile_loads "$OUT" "" "${PROF_CONC:-1 4 16 32}"

  # The replay re-sends stock's prompts (already prefix-cached): agreement noise floor only, not speed.
  if [ -z "${SKIP_REPLAY:-}" ]; then
    run "${LADDER[@]}" --rungs 1x1x1,4x1x1,16x1x1,32x1x1 --replay "$OUT/stock.json" --seed 0 --out "$OUT/stock2.json"
  fi
  [ -f "$OUT/stock2.json" ] && python harness/agree.py "$OUT/stock.json" "$OUT/stock2.json" 2>&1 | tee "$OUT/self_agreement.txt" || true
  nap 60   # let trace files finish writing
fi
cp "$OUT/server.log" "$OUT/server_full.log" 2>/dev/null || true
nccl_summary "$OUT"

fi
# ---------------------------------------------------------------- variants
for i in ${VNAMES[@]+"${!VNAMES[@]}"}; do
  name="${VNAMES[$i]}"; VD="$OUT/variants/$name"
  el=$(( ($(date +%s) - T_START) / 60 ))
  mkdir -p "$VD"
  if [ $(( el + VARIANT_MIN )) -gt "$MAX_MIN" ]; then
    echo "skipped: elapsed ${el} min + ~${VARIANT_MIN} min > MAX_MIN ${MAX_MIN}" | tee "$VD/SKIPPED.txt"
    continue
  fi
  kv=()
  [ -n "${VENVS[$i]}" ] && IFS=',' read -ra kv <<< "${VENVS[$i]}"
  # PROF=warm records Python stacks: TIMING_ENV's SGLANG_PROFILE_WITH_STACK=False would veto the request's with_stack (the
  # tokenizer reads it per /start_profile), so this server gets True (last env wins). Nothing else reads it.
  [ "${VPROF[$i]:-}" = warm ] && [ "${WARM_WITH_STACK:-1}" != 0 ] && kv+=(SGLANG_PROFILE_WITH_STACK=True)
  printf '%s\n' ${kv[@]+"${kv[@]}"} > "$VD/env.txt"
  echo "variant $name: SRC=${VSRCS[$i]} ARGS='${VARGS[$i]}' ENV=${VENVS[$i]}"
  stop_server
  mark "variant_${name}_start"
  start_server "$VD" "${VSRCS[$i]}" "${VARGS[$i]}" ${kv[@]+"${kv[@]}"}
  wait_healthy "$VD" || { boot_failed "$VD" $?; continue; }
  mem_start "$VD"; host_mem "$VD" "$name"
  if ! warmup "$VD"; then
    mem_stop "$VD"; echo "warmup failed" > "$VD/FAILED.txt"; bash harness/dbg_live.sh "$VD" > "$VD/dbg_live.txt" 2>&1; continue
  fi
  mem_probe "$VD"
  if [ -n "${VHOOK[$i]:-}" ]; then
    echo "variant $name: hook ${VHOOK[$i]}"; mark "variant_${name}_hook_start"
    capped 300 python "${VHOOK[$i]}" --url "$URL" --out "$VD" > "$VD/hook.txt" 2>&1 || echo "hook rc=$? (non-fatal)" | tee -a "$VD/hook.txt"
    tail -5 "$VD/hook.txt"; mark "variant_${name}_hook_done"
  fi
  mem_loaded_probe "$VD"
  _prc=0; run "${PROXY[@]}" --replay "$OUT/proxy.json" --out "$VD/proxy.json" > "$VD/proxy.rungs" || _prc=$?
  cat "$VD/proxy.rungs"
  # A client that crashed or produced no rung must not look green (Nov 17: an ids-client crash read as "0 rungs", rc=0):
  # FAILED.txt here, and the session exits 4 after the last variant (queue .result rc=4).
  _nr=$(grep -c '"concurrency"' "$VD/proxy.rungs" 2>/dev/null || true)
  if [ "$_prc" -ne 0 ] || [ "${_nr:-0}" -eq 0 ]; then
    echo "PROXY FAILED: variant $name client rc=$_prc, ${_nr:-0} rungs" | tee "$VD/FAILED.txt"; VFAILED="$VFAILED $name"
  fi
  curl -s "$URL/metrics" > "$VD/metrics_after_proxy.txt" || true
  mem_loaded_probe_done "$VD"
  SELF_ARG=(); [ -f "$OUT/proxy_self.json" ] && SELF_ARG=(--self "$OUT/proxy_self.json")
  python harness/agree.py "$OUT/proxy.json" "$VD/proxy.json" ${SELF_ARG[@]+"${SELF_ARG[@]}"} > "$VD/agree.txt" 2>&1 || true
  cat "$VD/agree.txt"
  mem_stop "$VD"
  MEMLINE="mem gate off"
  if [ -n "$MEM_GATE" ]; then
    MEMLINE=$(python harness/mem_gate.py compare "$OUT/mem_peak.json" "$VD/mem_peak.json" "$VD/mem_over_stock.txt" --limit 2.0 --margin 0.3 --stock-dir "$OUT" --cand-dir "$VD" 2>&1 | tail -1)
  fi
  # The per-variant summary: throughput rungs, agreement verdict, memory gate. A FAIL/WARN here means no judge check.
  echo "[variant $name] $(grep -c '"concurrency"' "$VD/proxy.rungs" 2>/dev/null || echo 0) rungs; $(grep -m1 -i 'agreement cand vs stock' "$VD/agree.txt" | cut -c1-80); $MEMLINE" | tee "$VD/summary.txt"
  if [ -n "${VPROF[$i]:-}" ]; then
    _pmode=""; _pdesc=""
    [ "${VPROF[$i]}" = cold ] && _pmode=cold && _pdesc=" (cold first-turn prefill)"   # PROF=cold: profile armed before fresh cold first turns
    [ "${VPROF[$i]}" = warm ] && _pmode=warm && _pdesc=" (warm later-turn prefill, log-triggered)"   # PROF=warm: armed after the cold storm
    echo "variant $name: profiling at loads ${PROF_CONC:-1 4}${_pdesc}"; profile_loads "$VD" "variant_${name}_" "${PROF_CONC:-1 4}" "$_pmode"
    ls "$VD/prof" 2>/dev/null | head -20 > "$VD/prof_files.txt" || true
  fi
  nccl_summary "$VD"
  mark "variant_${name}_done"
done
if [ -n "$VFAILED" ]; then echo "PROXY FAILED for variants:$VFAILED (see variants/<name>/FAILED.txt); exit 4"; exit 4; fi
# finish() (EXIT trap) stops the server, writes NCCL summaries and the tarball.
