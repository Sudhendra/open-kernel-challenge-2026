#!/usr/bin/env bash
# ssh / rsync helpers. usage: box.sh <A|B> <check|sync|run|setup> [args]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/config.env"
which=${1:?A or B}; shift; cmd=${1:?check|sync|run|setup}; shift || true
host=$BOX_A; [[ "$which" == "B" && -n "$BOX_B" ]] && host=$BOX_B; [[ "$which" == "C" && -n "${BOX_C:-}" ]] && host=$BOX_C
SSH="ssh -i $SSH_KEY -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 $host"
case "$cmd" in
  check)  # box alive, right GPU, container up
    $SSH "nvidia-smi --query-gpu=name,power.limit --format=csv,noheader; (docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null || sudo docker ps --format '{{.Names}} {{.Status}}') | grep '^judge ' || echo 'judge container: DOWN'; cat $REMOTE_DIR/.py 2>/dev/null || echo 'py: unprobed'" ;;
  sync)   # track essentials + best + remote scripts (never the whole repo)
    rsync -az -e "ssh -i $SSH_KEY -o BatchMode=yes" --delete \
      "$ROOT/$TRACK/reference.py" "$ROOT/$TRACK/checks.py" "$ROOT/$TRACK/nvfp4.py" "$ROOT/$TRACK/baseline.py" "$host:$REMOTE_DIR/track/"
    rsync -az -e "ssh -i $SSH_KEY -o BatchMode=yes" "$ROOT/$TRACK/pod/bench.py" "$host:$REMOTE_DIR/track/pod/"
    rsync -az -e "ssh -i $SSH_KEY -o BatchMode=yes" "$HERE/best/submission.py" "$HERE/best/aa_copy.py" "$host:$REMOTE_DIR/best/"
    rsync -az -e "ssh -i $SSH_KEY -o BatchMode=yes" "$HERE/remote/" "$host:$REMOTE_DIR/bin/"
    $SSH "chmod +x $REMOTE_DIR/bin/*.sh" ;;
  put)    # put <local file> <remote relative path>
    rdir=$(dirname "$2")
    rsync -az -e "ssh -i $SSH_KEY -o BatchMode=yes" --rsync-path="mkdir -p $REMOTE_DIR/$rdir && rsync" "$1" "$host:$REMOTE_DIR/$2" ;;
  run)    # run <job id> <bench args...>  -> serialized on the GPU lock, log streamed back
    id=$1; shift; $SSH "$REMOTE_DIR/bin/judge_run.sh $id $*" ;;
  setup)  # one-time box preparation
    $SSH "mkdir -p $REMOTE_DIR/bin" && rsync -az -e "ssh -i $SSH_KEY -o BatchMode=yes" "$HERE/remote/" "$host:$REMOTE_DIR/bin/" && $SSH "chmod +x $REMOTE_DIR/bin/*.sh && REMOTE_DIR=$REMOTE_DIR IMAGE=$IMAGE $REMOTE_DIR/bin/setup_box.sh" ;;
  ssh)    exec $SSH "$@" ;;
  *) echo "unknown: $cmd"; exit 2 ;;
esac
