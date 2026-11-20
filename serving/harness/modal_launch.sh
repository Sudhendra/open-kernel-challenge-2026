#!/usr/bin/env bash
# modal_launch.sh -- the ONLY way to launch an 8xH200 session on Modal. Runs the pre-flight gates, refuses to stop a
# running container, then launches. Every gate must pass or nothing is launched.
#   bash harness/modal_launch.sh --variants "<VARIANTS>" [--reuse-out <ts>] [--replace <app-id>] [-- <extra modal run args>]
#   env: OKC_PATCHED, OKC_EXTRA_TREES (same as modal_serve.py); PROXY_RUNGS, PROF_CONC, MAX_MIN, PROXY_SHAPE (judge|short,
#   default judge), PROXY_CORPUS (ladder.py --corpus dir inside the image) for the planned run. Host cache: modal_serve.py's --extra-args default (--hicache-size 42) unless passed after --.
# Gates: 1 bash -n + python ast of the harness; 2 harness/test_modal_queue.py; 3 harness/dryrun_session.sh in the exact mode
# (reuse when --reuse-out, else proxy) with the exact VARIANTS; 4 every SRC=<name> in VARIANTS is a mounted tree; 5 --replace
# only stops an app whose tasks==0 at the moment of the stop (read in the same second; a running job is never touched).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; SERVING="$(dirname "$HERE")"; cd "$SERVING"
VARIANTS=""; REUSE=""; REPLACE=""; EXTRA=()
while [ $# -gt 0 ]; do case "$1" in
  --variants) VARIANTS="$2"; shift 2;; --reuse-out) REUSE="$2"; shift 2;; --replace) REPLACE="$2"; shift 2;;
  --) shift; EXTRA=("$@"); break;; *) echo "unknown arg $1"; exit 2;; esac; done
log() { echo "[launch $(TZ=America/Chicago date '+%H:%M:%S CST')] $*"; }
log "gate 1: syntax"; bash -n harness/profile_session.sh; python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" harness/modal_serve.py harness/modal_queue.py harness/warm_trigger.py
log "gate 2: queue loop unit test"; python3 -I harness/test_modal_queue.py > /tmp/okc_launch_qtest.log 2>&1 || { cat /tmp/okc_launch_qtest.log; exit 1; }
python3 -I harness/test_warm_trigger.py > /tmp/okc_launch_wtest.log 2>&1 || { cat /tmp/okc_launch_wtest.log; exit 1; }   # PROF=warm trigger
log "gate 4: SRC trees"; for v in $(printf '%s' "$VARIANTS" | tr ';' '\n' | grep -o 'SRC=[A-Za-z0-9_-]*' | cut -d= -f2 | sort -u); do
  case "$v" in stock|patched) ;; *) printf '%s' "${OKC_EXTRA_TREES:-}" | tr ',' '\n' | grep -q "^$v=" || { echo "SRC=$v is not in OKC_EXTRA_TREES=${OKC_EXTRA_TREES:-}"; exit 1; };; esac; done
# gate 0: --reuse-out means "no planned run": modal_serve.py runs ONLY queue items then, so --variants without --queue would
# boot a node and exit (Nov 15 11:33 CST, ap-6e04NBKAd7OwgBXCsEMZys). Push the line as a queue item instead.
if [ -n "$REUSE" ] && [ -n "$VARIANTS" ] && ! printf '%s\n' "${EXTRA[@]}" | grep -q -- '--queue'; then
  echo "refused: --reuse-out with --variants but no --queue runs nothing (push the variants as a queue item: harness/modal_queue.py push)"; exit 1
fi
for kv in $(printf '%s' "${OKC_EXTRA_TREES:-}" | tr ',' ' '); do d="$SERVING/${kv#*=}/python/sglang"; [ -d "$d" ] || { echo "tree missing: $d"; exit 1; }; done
MODE=proxy; [ -n "$REUSE" ] && MODE=reuse
if [ -n "$VARIANTS" ]; then log "gate 3: dry run ($MODE) of the exact variant line"
  DRY_OUT="$(mktemp -d /tmp/okc_launch_dry.XXXXXX)" PROXY_RUNGS="${PROXY_RUNGS:-1x1x1}" PROF_CONC="${PROF_CONC:-1 4}" MAX_MIN="${MAX_MIN:-160}" STOCK_SELF="${STOCK_SELF:-}" \
    bash harness/dryrun_session.sh "$MODE" "$VARIANTS" > /tmp/okc_launch_dry.log 2>&1 || { tail -20 /tmp/okc_launch_dry.log; exit 1; }
  grep -q "rc=0" /tmp/okc_launch_dry.log || { tail -20 /tmp/okc_launch_dry.log; exit 1; }
fi
if [ -n "$REPLACE" ]; then
  tasks=$(modal app list --json 2>/dev/null | python3 -c 'import json,sys
for a in json.load(sys.stdin):
    if a["app_id"]==sys.argv[1] and a["state"]!="stopped": print(int(a["tasks"] or 0)); break
else: print("gone")' "$REPLACE")
  case "$tasks" in gone) log "gate 5: $REPLACE already stopped";; 0) log "gate 5: $REPLACE has 0 tasks, stopping it"; modal app stop -y "$REPLACE" >/dev/null;;
    *) echo "gate 5: $REPLACE has $tasks running task(s): NOT stopping, NOT launching"; exit 1;; esac
fi
ARGS=(--max-min "${MAX_MIN:-160}" --stock-mode proxy --proxy-rungs "${PROXY_RUNGS:-1x1x2,4x1x2,16x1x2}" --prof-conc "${PROF_CONC:-1 4}" --proxy-shape "${PROXY_SHAPE:-judge}")
[ -n "$VARIANTS" ] && ARGS+=(--variants "$VARIANTS"); [ -n "$REUSE" ] && ARGS+=(--reuse-out "$REUSE")
[ -n "${PROXY_CORPUS:-}" ] && ARGS+=(--proxy-corpus "$PROXY_CORPUS")
[ -n "${MEM_LOADED_PROBE:-}" ] && ARGS+=(--mem-loaded-probe "$MEM_LOADED_PROBE")
log "launching: modal run --detach harness/modal_serve.py::session ${ARGS[*]} ${EXTRA[*]:-}"
modal run --detach harness/modal_serve.py::session "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | grep -v -i "hf_[A-Za-z0-9]" | grep -i "modal.com/apps\|Error" | tail -3
modal app list --json 2>/dev/null | python3 -c 'import json,sys; print([(a["app_id"],a["state"],a["tasks"],a["created_at"][11:16]) for a in json.load(sys.stdin) if a["state"]!="stopped"])'
