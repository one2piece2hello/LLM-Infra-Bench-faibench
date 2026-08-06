#!/usr/bin/env bash
set -euo pipefail
mkdir -p /app/.timer
/app/timer.sh >/tmp/frontier-timer.log 2>&1 &
exec /bin/bash -l
