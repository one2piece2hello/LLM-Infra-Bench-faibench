#!/usr/bin/env bash
set -u
TIMER_DIR="/app/.timer"
mkdir -p "$TIMER_DIR"
START_EPOCH=$(date +%s)
BUDGET_SECS="${TASK_BUDGET_SECS:-72000}"
echo "$START_EPOCH" > "$TIMER_DIR/start_epoch"
echo "$BUDGET_SECS" > "$TIMER_DIR/budget_secs"
echo $$ > "$TIMER_DIR/timer.pid"
while true; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - START_EPOCH))
  REMAINING=$((BUDGET_SECS - ELAPSED))
  [ "$REMAINING" -lt 0 ] && REMAINING=0
  echo "$REMAINING" > "$TIMER_DIR/remaining_secs"
  echo "$ELAPSED" > "$TIMER_DIR/elapsed_secs"
  [ "$REMAINING" -le 0 ] && break
  sleep 10
done
