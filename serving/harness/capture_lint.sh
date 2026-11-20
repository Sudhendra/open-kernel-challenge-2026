#!/usr/bin/env bash
# Grep a lane diff (or files) for host<->device patterns that break or freeze under CUDA graph capture/replay.
# Usage: harness/capture_lint.sh <patch.diff | file.py ...>   (added lines only for a .diff)
# Not a proof: a hit needs reading; a clean run still needs the review in LANE_TEMPLATE section 5b.
pat='\.item\(\)|\.tolist\(\)|\.cpu\(\)|\.numel\(\) *[<>=!]|if [a-z_.]+\.any\(\)|if [a-z_.]+\.all\(\)|bool\([a-z_.]+\)|\[[^]]*\] *= *-?[0-9A-Z_]+ *$|torch\.tensor\(|device="cpu"|\.nonzero\(|torch\.unique|\.bincount\(|synchronize\('
hits=0
for f in "$@"; do
  case "$f" in
    *.diff|*.patch) out=$(grep -E '^\+[^+]' "$f" | grep -nE "$pat") ;;
    *) out=$(grep -nE "$pat" "$f") ;;
  esac
  if [ -n "$out" ]; then echo "== $f"; echo "$out"; hits=$((hits + $(echo "$out" | wc -l))); fi
done
if [ "$hits" = 0 ]; then echo "capture_lint: no host/device scalar patterns found (read the review anyway)"; else echo "capture_lint: $hits hit(s) to read"; fi
