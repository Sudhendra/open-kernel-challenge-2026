#!/bin/bash
# Judge-day helper: pin guard, patch rules, submit, log. Usage:
#   harness/judge.sh rules                 # re-fetch the live API; fail if pin/image/args moved
#   harness/judge.sh lint  <patch.diff>    # local patch rules (bytes, line length, prefixes, words)
#   harness/judge.sh check <patch.diff> [note]   # rules + lint, then okc serve check --no-wait, logged
#   harness/judge.sh submit <patch.diff> [note]  # same for the scored run
#   harness/judge.sh status [job]          # okc serve status --json saved under out/judge/
#   harness/judge.sh queue                 # public queue backlog/running right now
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); S=$(dirname "$HERE")
API=https://openkernelchallenge.dev/api/v1
REF="$S/rules_live/serving_2026-11-10.json"
LOG="$S/out/judge_log.tsv"; mkdir -p "$S/out/judge"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

rules() {
  local live="$S/out/judge/serving_$(date -u +%Y%m%dT%H%M%SZ).json"
  curl -sS -m 30 "$API/serving" -o "$live"
  python3 -I - "$REF" "$live" <<'PY'
import json, sys
ref, live = (json.load(open(p)) for p in sys.argv[1:3])
bad = 0
for k in ("base", "rules", "launch", "concurrency", "rungs", "check", "reps"):
    if ref.get(k) != live.get(k):
        bad = 1; print("CHANGED:", k); print("  was:", json.dumps(ref.get(k))[:400]); print("  now:", json.dumps(live.get(k))[:400])
t = live.get("tracks", {}).get("serving", {})
print("pin", live["base"]["commit"][:10], "| serving open:", t.get("open"), "| opens", t.get("opens"), "| closes", t.get("closes"), "| priority:", live.get("priority"))
sys.exit(bad)
PY
}

lint() {
  local p="$1"; [ -s "$p" ] || { echo "empty or missing patch: $p (an empty diff is only valid for the baseline check)"; }
  local bytes longest out ml fw
  bytes=$(wc -c < "$p"); longest=$(awk '{ if (length($0)>m) m=length($0) } END {print m+0}' "$p")
  out=$(grep '^diff --git' "$p" | grep -vc ' b/python/sglang/' || true)
  ml=$(grep '^diff --git' "$p" | grep -c model_loader || true)
  fw=$(grep -cE 'kc-secret|KC_RESULT|KC_SERVING|KC_GATE|/kc-job|/kc-root|result_secret|serviceaccount|SIMULATE' "$p" || true)
  local sfx; sfx=$(grep '^diff --git' "$p" | awk '{print $4}' | grep -vE '\.(py|cu|cuh|h|hpp|c|cc|cpp|json)$' | wc -l | tr -d ' ' || true)
  local ren; ren=$(grep -cE '^(rename|similarity index|old mode|new mode|Binary files)' "$p" || true)
  echo "bytes=$bytes/409600 longest=$longest/400 files=$(grep -c '^diff --git' "$p" || true) outside=$out model_loader=$ml bad_suffix=$sfx rename_mode_binary=$ren forbidden=$fw"
  [ "$bytes" -le 409600 ] && [ "$longest" -le 400 ] && [ "$out" = 0 ] && [ "$ml" = 0 ] && [ "$sfx" = 0 ] && [ "$ren" = 0 ] && [ "$fw" = 0 ] || { echo "LINT FAIL"; return 1; }
  echo "LINT OK"
}

run() {  # mode patch note
  local mode="$1" p="$2" note="${3:-}"
  rules || { echo "pin or rules moved: STOP and re-read before submitting"; return 2; }
  lint "$p"
  local sha; sha=$(shasum -a 256 "$p" | cut -c1-12)
  echo "submitting $mode $p ($sha) $(ts)"
  local outf="$S/out/judge/${mode}_$(date -u +%Y%m%dT%H%M%SZ)_$sha.txt"
  okc serve "$mode" "$p" --no-wait | tee "$outf"
  [ -s "$LOG" ] || printf 'utc\tmode\tpatch\tsha\tnote\tsubmit_output\n' > "$LOG"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(ts)" "$mode" "$p" "$sha" "$note" "$(tr '\n' ' ' < "$outf" | cut -c1-300)" >> "$LOG"
}

case "${1:-}" in
  rules) rules ;;
  lint) lint "$2" ;;
  check) run check "$2" "${3:-}" ;;
  submit) run submit "$2" "${3:-}" ;;
  status) f="$S/out/judge/status_$(date -u +%Y%m%dT%H%M%SZ).json"; okc serve status ${2:-} --json | tee "$f"; echo "saved $f" ;;
  queue) curl -sS -m 20 "$API/queue"; echo ;;
  *) sed -n 2,9p "$0" ;;
esac
