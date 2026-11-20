#!/usr/bin/env bash
# dryrun_session.sh -- run the REAL profile_session.sh end to end on a laptop with a mock server/ladder/GPU (harness/mock/).
# Catches control-flow bugs (undefined functions, wrong branches, bad variant lines) before a GPU container does.
#   bash harness/dryrun_session.sh proxy  "VARIANTS..."  [extra ENV=val ...]
#   bash harness/dryrun_session.sh reuse  "VARIANTS..."  [extra ENV=val ...]     # STOCK_MODE=reuse against a mock stock dir
#   bash harness/dryrun_session.sh full   ""             [extra ENV=val ...]
# Exit code = profile_session.sh's. Output under $DRY_OUT (default a temp dir), printed at the end. $DRY_OUT/argv.log holds every
# mocked python invocation (argv, plus SGLANG_*/PYTHONPATH env of server launches) for before/after diffs of the command lines.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; SERVING="$(dirname "$HERE")"
MODE="${1:?proxy|reuse|full}"; VARIANTS_IN="${2:-}"; shift 2 || true
T="${DRY_OUT:-$(mktemp -d /tmp/okc_dry.XXXXXX)}"; mkdir -p "$T/out" "$T/trees"
# fake trees for every SRC name used (stock, patched, and any SRC=<name>)
mk() { mkdir -p "$T/trees/$1/python/sglang" "$T/trees/$1/test/registered/unit/mem_cache"; touch "$T/trees/$1/python/sglang/__init__.py"; }
mk stock; mk patched; ENVS=()
for v in $(printf '%s' "$VARIANTS_IN" | tr ';' '\n' | grep -o 'SRC=[A-Za-z0-9_-]*' | cut -d= -f2 | sort -u); do
  case "$v" in stock|patched) ;; *) mk "$v"; ENVS+=("SRC_$(echo "$v" | tr '[:lower:]-' '[:upper:]_')=$T/trees/$v/python");; esac
done
# PROXY_CORPUS names a directory inside the image; the mock run checks the plumbing with a local stand-in (harness/).
# reuse: a mock stock dir of the session's shape (PROXY_SHAPE, default judge) with a stock memory peak for the gate
REUSE=(); if [ "$MODE" = reuse ]; then mkdir -p "$T/stock_prev"; _sh="${PROXY_SHAPE:-judge}"
  echo "{\"records\":[],\"mock\":true,\"args\":{\"shape\":\"$_sh\"}}" > "$T/stock_prev/proxy.json"; echo mock > "$T/stock_prev/proxy.rungs"
  echo '{"samples":8,"peak_mib":{"0":119000,"1":119001,"2":119002,"3":119003,"4":119004,"5":119005,"6":119006,"7":119007}}' > "$T/stock_prev/mem_peak.json"
  REUSE=(STOCK_MODE=reuse "STOCK_REUSE_DIR=$T/stock_prev"); fi
[ "$MODE" = full ] && REUSE=(STOCK_MODE=full); [ "$MODE" = proxy ] && REUSE=(STOCK_MODE=proxy)
echo "[dryrun] mode=$MODE out=$T"
cd "$SERVING" && env PATH="$HERE/mock:$PATH" NAP_DIV="${NAP_DIV:-30}" BOOT_SCAN_S=2 MOCK_BOOT_S=1 MOCK_ARGV_LOG="$T/argv.log" \
  OUT_ROOT="$T/out" SRC_STOCK="$T/trees/stock/python" SRC_PATCHED="$T/trees/patched/python" ${ENVS[@]+"${ENVS[@]}"} "${REUSE[@]}" \
  VARIANTS="$VARIANTS_IN" MAX_MIN="${MAX_MIN:-30}" PRE_TESTS="${PRE_TESTS:-}" AR_BENCH="" STOCK_SELF="${STOCK_SELF:-}" \
  PROXY_RUNGS="${PROXY_RUNGS:-1x1x1}" PROXY_SHAPE="${PROXY_SHAPE:-judge}" PROF_CONC="${PROF_CONC:-1 4}" EXTRA_ARGS="--hicache-size 16" \
  PROXY_CORPUS="${PROXY_CORPUS:+$HERE}" "$@" \
  SGLANG_SRC_MODE=pythonpath bash harness/profile_session.sh > "$T/dryrun.log" 2>&1; rc=$?
echo "[dryrun] rc=$rc"; grep -n -i "mem over stock\|command not found\|unbound variable\|syntax error\|No such file\|bad variant\|bad SRC\|FAILED\|skipped\|stock_reused\|profiling at loads\|prof_c.*done\|prof_.*_c.*done\|\[warm\]\|done:" "$T/dryrun.log" | head -40
echo "[dryrun] log: $T/dryrun.log"; exit $rc
