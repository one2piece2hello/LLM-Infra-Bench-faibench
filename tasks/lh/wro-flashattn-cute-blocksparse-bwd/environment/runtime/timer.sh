#!/usr/bin/env bash
set -euo pipefail
mkdir -p /app/.timer
remaining=${FRONTIER_REMAINING_SECS:-14400}
printf '%s\n' "$remaining" > /app/.timer/remaining_secs
while true; do sleep 60; done
