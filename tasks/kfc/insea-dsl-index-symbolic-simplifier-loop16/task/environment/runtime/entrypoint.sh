#!/usr/bin/env bash
# Canonical entrypoint: start the timer in the background and hand the container to
# a login shell so the orchestrator can keep it alive and exec /tests/test.sh into it.
# `exec "$@"` is honored when a command is passed (e.g. local smoke). CPU task: no
# GPU/venv assumptions — pin a portable PATH that resolves python3.
set -euo pipefail
export PATH=/opt/kernelbench-venv/bin:/usr/local/bin:/usr/bin:/bin:$PATH
mkdir -p /app/.timer
/app/timer.sh >/tmp/task-timer.log 2>&1 &
if [ "$#" -gt 0 ]; then
  exec "$@"
fi
exec /bin/bash -l
