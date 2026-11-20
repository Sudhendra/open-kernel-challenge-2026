#!/usr/bin/env bash
# One fresh attempt. usage: attempt.sh <lane> <provider> [id]. exit 0 = candidate.py produced, 3 = no candidate, 2 = error
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/config.env"
LANE=$1; PROV=$2; ID=${3:-$LANE-$(date +%m%d-%H%M%S)}
A="$HERE/attempts/$ID"; mkdir -p "$A"; cd "$A"
cp "$HERE/best/submission.py" best.py; cp "$HERE/lanes/$LANE.md" lane.md; cp "$HERE/LEARNINGS.md" .
[[ "$LANE" == "ports" ]] && cp -R "$HERE/queue" ./queue
[[ "$LANE" == "research" ]] && cp -R "$HERE/queue" ./queue
printf '#!/usr/bin/env bash\nATTEMPT=%s exec %s/gate.sh "$@"\n' "$ID" "$HERE" > gate; chmod +x gate
sed -e "s/{{ID}}/$ID/g" -e "s/{{LANE}}/$LANE/g" -e "s/{{MINUTES}}/$ATTEMPT_MINUTES/g" "$HERE/prompts/attempt.md" > prompt.md
[[ "$LANE" == "research" ]] && cp "$HERE/prompts/research.md" prompt.md
echo "lane=$LANE provider=$PROV start=$(date -Iseconds)" > meta.txt
TO=$(( ATTEMPT_MINUTES * 60 + 300 ))
# portable timeout (macOS has no coreutils timeout)
tmo(){ perl -e 'alarm shift; exec @ARGV' "$@"; }
TOOLS="Read,Edit,Write,Grep,Glob,Bash(./gate *),Bash(python3 -c *),Bash(python3 -m py_compile *),Bash(grep *),Bash(diff *)"
[[ "$LANE" == "research" ]] && CLAUDE_MODEL=${RESEARCH_MODEL:-$CLAUDE_MODEL}
[[ "$LANE" == "research" ]] && TOOLS="WebSearch,WebFetch,Read,Write,Grep,Glob,Bash(gh search *),Bash(gh api *),Bash(curl -s *)"
case "$PROV" in
  claude)   tmo $TO claude -p --model "$CLAUDE_MODEL" --effort high --permission-mode acceptEdits \
              --allowedTools "$TOOLS" \
              < prompt.md > agent.log 2>&1 ;;
  codex)    tmo $TO codex exec --model "$CODEX_MODEL" -c model_reasoning_effort=high --sandbox workspace-write \
              -c sandbox_workspace_write.network_access=true --skip-git-repo-check -o last_message.txt - < prompt.md > agent.log 2>&1 ;;
  opencode) tmo $TO opencode --standalone run --model "$OPENCODE_MODEL" --file prompt.md \
              'Do the task in the attached file. Work only in the current directory.' > agent.log 2>&1 ;;
  *) echo "unknown provider $PROV"; exit 2 ;;
esac
rc=$?; echo "end=$(date -Iseconds) rc=$rc" >> meta.txt
if [[ -s candidate.py ]]; then echo "candidate: yes"; exit 0; else echo "candidate: no"; exit 3; fi
