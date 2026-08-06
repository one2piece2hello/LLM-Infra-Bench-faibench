#!/usr/bin/env bash
set -e
mkdir -p /app/.timer /logs/agent
export FRONTIER_TIMER_BOOTSTRAP=1
( env -u BASH_ENV -u ENV /app/timer.sh ) &
echo $! > /app/.timer/timer.pid
exec "$@"
