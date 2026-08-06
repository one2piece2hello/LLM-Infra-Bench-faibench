#!/usr/bin/env bash
set -euo pipefail
export PATH=/opt/kernelbench-venv/bin:/root/.cargo/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

if [ -x /app/timer.sh ]; then
  /app/timer.sh >/app/.timer/timer.log 2>&1 &
fi

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

exec /bin/bash -l
