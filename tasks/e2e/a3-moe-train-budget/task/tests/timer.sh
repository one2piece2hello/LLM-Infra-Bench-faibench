#!/bin/bash
# FROZEN harness-owned wall-clock timer. Runs the solver's training under a HARD wall-clock cap
# the solver cannot disable or self-report: the child is started in its OWN SESSION and the
# whole PROCESS GROUP is signalled at the cap, so a daemonised background trainer cannot keep
# running (and keep overwriting the scored checkpoint) past the budget.
#   usage: timer.sh <seconds> <cmd> [args...]
# SIGTERM at the cap so the training's periodic atomic checkpoint is the scored artifact, then
# SIGKILL after a grace period.
set -u
SECS="$1"; shift
mkdir -p /app/.timer
date +%s > /app/.timer/start
echo "$SECS" > /app/.timer/budget_sec
# background elapsed/remaining publisher (read-only to the solver)
(
  start=$(date +%s)
  while :; do
    now=$(date +%s); el=$((now-start)); rem=$((SECS-el))
    echo "$el" > /app/.timer/elapsed; echo "$rem" > /app/.timer/remaining
    [ "$rem" -le 0 ] && break
    sleep 2
  done
) &
PUB=$!

# --- hard cap, external and process-group wide + PRIVILEGE DROP --------------------------
# The child must be in its OWN process group so the whole group can be signalled, and it is
# launched under an UNPRIVILEGED uid when E2E_TRAIN_UID is set by test.sh.
#
# 🔴 WHY THE UID MATTERS. The frozen surface (/tests: the held-out corpus, the calibrated
# constants) is quarantined chmod 0000 while the training runs — but chmod means NOTHING to
# uid 0, which bypasses DAC entirely. Running the candidate's training as a non-root uid is what
# turns "quarantined" from nominal into real: only then can the candidate not read the held-out
# split it is scored on. The drop is verified from inside the child (it writes its euid to
# /app/.timer/train_uid) so the harness records the TRUTH rather than the intent.
TRAIN_UID="${E2E_TRAIN_UID:-}"
MYPGID=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
python3 - "$TRAIN_UID" "$@" <<'PYLAUNCH' &
import os, sys
uid = sys.argv[1].strip()
cmd = sys.argv[2:]
# fork FIRST so setsid() can never fail with EPERM (it does when the caller is already a process
# group leader). MEASURED on NVIDIA H20 2026-07-27: if setsid silently failed, the training shared the
# HARNESS's process group and the timer's group-wide kill took test.sh down with it -- the run
# produced ZERO result files and looked like a training hang.
if os.fork() != 0:
    os._exit(0)
os.setsid()
try:
    os.makedirs("/app/.timer", exist_ok=True)
except Exception:
    pass
if uid:
    try:
        u = int(uid)
        os.setgroups([])
        os.setgid(u)
        os.setuid(u)
    except Exception as exc:
        sys.stderr.write("[timer] privilege drop to uid %s FAILED: %s\n" % (uid, exc))
try:
    with open("/app/.timer/train_uid", "w") as fh:
        fh.write("%d %d\n" % (os.getuid(), os.geteuid()))
except Exception:
    pass
os.execvp(cmd[0], cmd)
PYLAUNCH
LAUNCHER=$!
wait "$LAUNCHER" 2>/dev/null || true      # the launcher forks and exits; the grandchild trains
sleep 1
# locate the training's OWN session leader (pgid == sid) outside the harness's group
CHILD=$(ps -eo pid,pgid,sid,args 2>/dev/null | awk -v me="$MYPGID" '$2==$3 && $2!=me && /run_training/ {print $1; exit}')
[ -n "$CHILD" ] || CHILD=$(ps -eo pid,pgid,sid,args 2>/dev/null | awk -v me="$MYPGID" '$2==$3 && $2!=me && /train_gpt/ {print $1; exit}')
PGID=""
[ -n "$CHILD" ] && PGID=$(ps -o pgid= -p "$CHILD" 2>/dev/null | tr -d ' ')
# 🔴 SELF-KILL GUARD: never signal our own process group.
if [ -n "$PGID" ] && [ "$PGID" = "$MYPGID" ]; then
  echo "[timer] WARNING: training shares the harness process group; using per-pid signals"
  PGID=""
fi
echo "[timer] training pid=${CHILD:-unknown} pgid=${PGID:-none} harness_pgid=$MYPGID"
rc=0
deadline=$(( $(date +%s) + SECS ))
timed_out=0
while :; do
  if [ -z "$CHILD" ] || ! kill -0 "$CHILD" 2>/dev/null; then
    rc=0
    break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    timed_out=1
    echo "[timer] wall-clock budget ${SECS}s reached; stopping the training process group (checkpoint = latest atomic save)"
    if [ -n "$PGID" ]; then kill -TERM -"$PGID" 2>/dev/null || true
    else kill -TERM "$CHILD" 2>/dev/null || true; fi
    sleep 8
    if [ -n "$PGID" ]; then kill -KILL -"$PGID" 2>/dev/null || true
    else kill -KILL "$CHILD" 2>/dev/null || true; fi
    rc=0
    break
  fi
  sleep 1
done
# --- reap EVERY straggler ------------------------------------------------------------------
# 🔴 MEASURED on NVIDIA H20 2026-07-28 with solution/probe_budget_bite: a process-group kill is NOT
# enough. The probe forks a grandchild that calls setsid() ITSELF, so the grandchild lands in a NEW
# session/group that `kill -- -$PGID` never reaches. It outlived the budget, kept rewriting the
# scored checkpoint into the eval phase, AND held the harness's log pipe open so test.sh never
# finished — the run HUNG instead of scoring.
# The reliable reaper is BY UID: the training runs unprivileged (E2E_TRAIN_UID), so killing every
# process owned by that uid reaches any descendant regardless of session or group. This is a
# second, independent payoff from the privilege drop.
if [ -n "$PGID" ]; then kill -KILL -"$PGID" 2>/dev/null || true; fi
if [ -n "${E2E_TRAIN_UID:-}" ]; then
  for _ in 1 2 3; do
    pkill -KILL -u "$E2E_TRAIN_UID" 2>/dev/null || true
    sleep 1
    # count only LIVE processes: a killed child sits as <defunct> (state Z) until its parent
    # reaps it, and a zombie holds no resources and cannot touch the checkpoint. Counting zombies
    # produced a spurious "survived the reaper" warning on the first measured run.
    _live=$(ps -o stat= -u "$E2E_TRAIN_UID" 2>/dev/null | grep -cv '^[[:space:]]*Z' || true)
    [ "${_live:-0}" -eq 0 ] && break
  done
  _live=$(ps -o stat= -u "$E2E_TRAIN_UID" 2>/dev/null | grep -cv '^[[:space:]]*Z' || true)
  if [ "${_live:-0}" -gt 0 ]; then
    echo "[timer] WARNING: processes owned by uid $E2E_TRAIN_UID survived the reaper:"
    ps -o pid,pgid,sid,stat,args -u "$E2E_TRAIN_UID" 2>/dev/null | head -5
  else
    echo "[timer] reaped every LIVE process owned by uid $E2E_TRAIN_UID (zombies ignored)"
  fi
fi
kill "$PUB" 2>/dev/null || true
# timing out is EXPECTED when the recipe uses the full budget; the periodic checkpoint is valid.
[ "$timed_out" -eq 1 ] && exit 0
exit "$rc"
