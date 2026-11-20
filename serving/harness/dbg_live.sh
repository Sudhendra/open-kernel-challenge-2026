#!/usr/bin/env bash
# Run inside the live session container: one small streaming request + recent server errors.
#   bash harness/dbg_live.sh [dir-with-server.log]    (default: newest /vol/out/<ts>/)
# profile_session.sh passes the stock or variant directory whose server is running.
DIR="${1:-$(ls -td /vol/out/*/ 2>/dev/null | head -1)}"
LOG="${DIR%/}/server.log"
URL="${URL:-http://127.0.0.1:30000}"
curl -s -m 120 "$URL/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Say hi"}],"max_tokens":20,"stream":true,"stream_options":{"include_usage":true}}' | head -c 1500
echo; echo "--- server log tail ($LOG)"
tail -5 "$LOG" | cut -c1-300
echo "--- errors"
grep -iE "error|exception|traceback|400|422" "$LOG" | grep -v FutureWarning | tail -8 | cut -c1-300
