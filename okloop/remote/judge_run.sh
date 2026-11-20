#!/usr/bin/env bash
# Runs ON the box: serialize on the GPU lock, run bench inside the judge container, stream + save the log.
# usage: judge_run.sh <job id> [COLD=1] <args to bench.py>
set -uo pipefail
REMOTE_DIR=$HOME/okc; ID=$1; shift
COLD=0; [[ "${1:-}" == "COLD=1" ]] && { COLD=1; shift; }
D=docker; docker info >/dev/null 2>&1 || D="sudo docker"
PY=$(cat $REMOTE_DIR/.py 2>/dev/null || echo python3)
LOG=$REMOTE_DIR/logs/$ID.log
ENV=""; [[ $COLD == 1 ]] && ENV="-e TORCH_EXTENSIONS_DIR=/root/okc/.cache/cold_$ID"
export LD_LIBRARY_PATH_IN="/opt/sglang/lib/python3.12/site-packages/torch/lib"
flock -w 14400 $REMOTE_DIR/.gpu.lock \
  $D exec $ENV judge bash -lc "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH_IN:\${LD_LIBRARY_PATH:-}; cd /root/okc && $PY track/pod/bench.py $*" 2>&1 | tee $LOG
exit ${PIPESTATUS[0]}
