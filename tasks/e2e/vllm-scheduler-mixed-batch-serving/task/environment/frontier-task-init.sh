# Timer daemon auto-start on login shells (frozen-swe runtime contract). Baked via COPY (a
# RUN heredoc silently no-ops on the classic docker builder); the entrypoint also starts it.
if [ -x /app/timer.sh ] && [ "${FRONTIER_TIMER_BOOTSTRAP:-0}" != "1" ]; then
  timer_pid_file=/app/.timer/timer.pid
  if [ ! -s "$timer_pid_file" ] || ! kill -0 "$(cat "$timer_pid_file" 2>/dev/null)" 2>/dev/null; then
    FRONTIER_TIMER_BOOTSTRAP=1 env -u BASH_ENV -u ENV /app/timer.sh >/dev/null 2>&1 &
  fi
fi
