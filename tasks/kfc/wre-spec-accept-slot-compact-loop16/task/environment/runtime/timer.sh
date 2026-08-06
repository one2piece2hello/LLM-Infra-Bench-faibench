#!/usr/bin/env bash
set -euo pipefail
export PATH=/opt/kernelbench-venv/bin:/root/.cargo/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
mkdir -p /app/.timer
date -u +%FT%TZ > /app/.timer/start.txt
remaining=${FRONTIER_REMAINING_SECS:-21600}
printf '%s\n' "$remaining" > /app/.timer/remaining_secs
printf '%s\n' "$remaining" > /app/.timer/budget_secs
printf '%s\n' "$$" > /app/.timer/timer.pid
while true; do
  sleep 60
done
