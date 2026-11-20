#!/usr/bin/env bash
# The one verify command. usage: gate.sh [--stage lint|compile|correct|time|all] candidate.py
# exit 0 = keep (or stage passed), 1 = not a keep / failed, 2 = infrastructure error.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/config.env"
STAGE=all; [[ "${1:-}" == "--stage" ]] && { STAGE=$2; shift 2; }
CAND=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
ID=${ATTEMPT:-$(basename "$(dirname "$CAND")")}
ADIR=$(dirname "$CAND"); mkdir -p "$ADIR"
BAND=$(cat "$HERE/state/aa_band.txt" 2>/dev/null || echo $AA_BAND_DEFAULT)
THR=$(python3 -c "print(max($KEEP_MIN, 2*$BAND))")
mark(){ echo "$1" > "$ADIR/.stage"; echo "[$(date +%H:%M:%S)] $ID $1" | tee -a "$ADIR/gate.log" >&2; }
box(){ "$HERE/box.sh" "$@"; }
BOXC=A; [[ -n "$BOX_B" && "${GATE_USE_B:-1}" == 1 ]] && BOXC=B   # compile/correct box (GATE_USE_B=0 keeps the gate on A)
score_of(){ grep "^SCORE $2:" "$1" | tail -1 | awk '{print $3}'; }

run_lint(){ mark lint; python3 "$HERE/lint.py" "$CAND" "$HERE/best/submission.py" | tee -a "$ADIR/gate.log"; return ${PIPESTATUS[0]}; }
run_sync(){ box $BOXC put "$CAND" "work/$ID/candidate.py" || return 2; box A put "$CAND" "work/$ID/candidate.py" || return 2; }
run_compile(){ mark compile
  box $BOXC run "$ID-compile" COLD=1 "work/$ID/candidate.py --cases 0 --reps 1" > "$ADIR/compile.log" 2>&1; rc=$?
  secs=$(grep -o "^prepare .*: [0-9]*s" "$ADIR/compile.log" | tail -1 | grep -o "[0-9]*s$" | tr -d s)
  [[ $rc -ne 0 || -z "$secs" ]] && { tail -30 "$ADIR/compile.log" >&2; echo "compile: FAILED" | tee -a "$ADIR/gate.log"; return 1; }
  echo "compile: prepare ${secs}s cold (limit $COLD_COMPILE_S)" | tee -a "$ADIR/gate.log"
  (( secs > COLD_COMPILE_S )) && return 1; return 0; }
run_correct(){ mark correct
  box $BOXC run "$ID-correct" "work/$ID/candidate.py --check --reps 3" > "$ADIR/correct.log" 2>&1; rc=$?
  grep -E "^CHECK .*FAIL|^CHECKS:" "$ADIR/correct.log" | tee -a "$ADIR/gate.log"
  grep -q "^CHECKS: ALL PASS" "$ADIR/correct.log" || { tail -20 "$ADIR/correct.log" >&2; return 1; }; return 0; }
run_time(){ mark timing
  local pass=0 r1 r2 b1 c1 b2 c2
  box A run "$ID-t1" "best/submission.py work/$ID/candidate.py --reps $REPS" > "$ADIR/time1.log" 2>&1 || return 2
  b1=$(score_of "$ADIR/time1.log" submission.py); c1=$(score_of "$ADIR/time1.log" candidate.py)
  [[ -z "$b1" || -z "$c1" ]] && { tail -20 "$ADIR/time1.log" >&2; return 2; }
  r1=$(python3 -c "print($c1/$b1-1)")
  echo "time1: best $b1 cand $c1 ratio $(printf %+.4f $r1) (need > $THR)" | tee -a "$ADIR/gate.log"
  python3 -c "import sys; sys.exit(0 if $r1 > $THR else 1)" || { echo "verdict: NO (run 1)" | tee -a "$ADIR/gate.log"; write_result "$r1" "" no; return 1; }
  box A run "$ID-t2" "work/$ID/candidate.py best/submission.py --reps $REPS" > "$ADIR/time2.log" 2>&1 || return 2
  b2=$(score_of "$ADIR/time2.log" submission.py); c2=$(score_of "$ADIR/time2.log" candidate.py)
  r2=$(python3 -c "print($c2/$b2-1)")
  echo "time2: best $b2 cand $c2 ratio $(printf %+.4f $r2) (need > $THR)" | tee -a "$ADIR/gate.log"
  if python3 -c "import sys; sys.exit(0 if $r2 > $THR else 1)"; then echo "verdict: KEEP" | tee -a "$ADIR/gate.log"; write_result "$r1" "$r2" keep; return 0
  else echo "verdict: NO (run 2)" | tee -a "$ADIR/gate.log"; write_result "$r1" "$r2" no; return 1; fi; }
write_result(){ python3 - "$ADIR" "$1" "$2" "$3" "$THR" "$BAND" <<'PY'
import json,sys,glob,re
d,r1,r2,v,thr,band=sys.argv[1:]
def sc(f):
    try: return {l.split()[1].rstrip(':'):float(l.split()[2]) for l in open(f) if l.startswith('SCORE ')}
    except Exception: return {}
json.dump({"verdict":v,"r1":float(r1) if r1 else None,"r2":float(r2) if r2 else None,"threshold":float(thr),"aa_band":float(band),
           "scores1":sc(d+"/time1.log"),"scores2":sc(d+"/time2.log")}, open(d+"/result.json","w"), indent=1)
PY
}
case "$STAGE" in
  lint) run_lint ;;
  compile) run_lint && run_sync && run_compile ;;
  correct) run_lint && run_sync && run_correct ;;
  time) run_sync && run_time ;;
  all) run_lint || exit 1; run_sync || exit 2; run_compile || exit 1; run_correct || exit 1; run_time; rc=$?; mark "done:$rc"; exit $rc ;;
  *) echo "bad stage"; exit 2 ;;
esac
