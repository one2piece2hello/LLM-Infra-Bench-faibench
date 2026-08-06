#!/usr/bin/env bash
# Matches the proven ready-task entrypoint (kernel-opt-p0a4-003): start the timer
# in the background and hand the container to a login shell so the orchestrator can
# keep it alive and exec /tests/test.sh into it. `exec "$@"` is honored when a
# command is passed (e.g. local smoke).
set -euo pipefail
export PATH=/opt/kernelbench-venv/bin:$PATH
mkdir -p /app/.timer
/app/timer.sh >/tmp/task-timer.log 2>&1 &
if [ "$#" -gt 0 ]; then
  exec "$@"
fi
exec /bin/bash -l
