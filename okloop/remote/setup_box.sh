#!/usr/bin/env bash
# Runs ON the box once: judge container with the repo dir mounted, persistent compile caches, python probe.
set -euo pipefail
REMOTE_DIR=${REMOTE_DIR:-$HOME/okc}; IMAGE=${IMAGE:-lmsysorg/sglang:dev-muse-spark-open}
mkdir -p $REMOTE_DIR/{track/pod,best,work,logs,.cache/torch_ext,.cache/triton}
D=docker; docker info >/dev/null 2>&1 || D="sudo docker"
$D pull $IMAGE
if ! $D ps --format '{{.Names}}' | grep -q '^judge$'; then
  $D rm -f judge >/dev/null 2>&1 || true
  $D run -d --name judge --gpus all --ipc=host --ulimit memlock=-1 --restart unless-stopped \
    -v $REMOTE_DIR:/root/okc -w /root/okc \
    -e TORCH_EXTENSIONS_DIR=/root/okc/.cache/torch_ext -e TRITON_CACHE_DIR=/root/okc/.cache/triton \
    $IMAGE sleep infinity
fi
# which python inside the image has torch+triton
PY=""
for p in python3 /opt/sglang/bin/python /usr/bin/python3; do
  if $D exec judge bash -lc "$p -c 'import torch, triton; print(torch.__version__)'" >/dev/null 2>&1; then PY=$p; break; fi
done
[[ -z "$PY" ]] && { echo "no python with torch+triton found in image"; exit 1; }
echo "$PY" > $REMOTE_DIR/.py
$D exec judge bash -lc "$PY -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, torch.cuda.get_device_name(0))'"
echo "setup ok: judge container up, python=$PY"
