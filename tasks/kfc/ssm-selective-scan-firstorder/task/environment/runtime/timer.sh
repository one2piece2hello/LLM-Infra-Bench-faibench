#!/usr/bin/env bash
# Minimal global timer stub; the harness replaces it with the real budget timer.
set -euo pipefail
export PATH=/opt/kernelbench-venv/bin:$PATH
mkdir -p /app/.timer
echo 10800 > /app/.timer/remaining_secs
echo 10800 > /app/.timer/budget_secs
