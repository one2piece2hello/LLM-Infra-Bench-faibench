#!/usr/bin/env bash
# =============================================================================
# solve.sh -- e2e/vllm-scheduler-mixed-batch-serving
#   vLLM scheduler tuning for mixed prefill/decode batches (serving, H20)
#
# ORACLE FORM      : LAUNCH SCRIPT (+ sidecar module for the negative). The entry contract is a shell
#                    script the verifier executes to bring up the candidate server.
# EDITABLE SURFACE : $SUBMISSION_DIR (default /app/submission) inside the 1-commit git tree /app/repo
# GRADED UNIT      : /app/submission/launch_server.sh (+ any sidecar it invokes from /app/submission)
# PRIMARY METRIC   : ABBA paired serving speedup -> min(1, ln(speedup/1.2857108)/ln(1.2857108)), 0 unless speedup > 1.2857108
# MODE VARIABLE    : NONE in tests/test.sh -- it always grades /app/submission/launch_server.sh.
#                    (`/app/.oracle_solution` only flips a --oracle flag compute_reward.py declares and
#                    never reads; solve.sh does not create that marker.)
# REFERENCE ASSETS : solution/oracle_v2_launch_server.sh (= manifest oracle_impl),
#                    solution/negative_fake_launch_server.sh + negative_fake_server.py,
#                    solution/oracle_sweep_launch_server.sh (ORACLE_VARIANT sweep harness)
# SHA-PINNED PATHS : tests/reward_manifest.json holds the frozen ref_speedup; test.sh additionally pins
#                    sha256(/app/timer.sh) and cross-checks the published wall-clock budget. solve.sh
#                    writes ONLY under /app/submission.
#
# GRADE AFTER LANDING (one command, run from the task/ dir of the package):
#   docker run --rm --gpus all -v "$PWD:/task:ro" -v "$PWD/tests:/tests:ro" \
#     fai/vllm-scheduler-mixed-batch-serving:oss bash -lc '
#       bash /task/solution/solve.sh && bash /tests/test.sh'
#   -> /logs/verifier/reward.json
#
# CLI: (default)=oracle | --negative | --noop | --variant <name> | --list | --help
#
# --noop SEMANTICS : --noop 把可编辑面复位到烤入基线（与 verifier 取基线的方式一致，HEAD 不动），
#                    使对照组能量到真正的 no-op 值。
#                    (`--noop` resets the editable surface to the baked baseline -- the same way the verifier
#                    materialises its own timing/scoring baseline -- with HEAD untouched, so the control arm
#                    measures the REAL no-op value instead of whatever a previous arm left behind.)
#                    HOW, here: rm -f $SUB/negative_fake_server.py (a negative sidecar), then restore
#                    launch_server.sh from the pre-landing snapshot under ${SOLVE_BACKUP_DIR:-/tmp/.fai_solve_baseline}/<task-id>/
#                    (/app/submission is NOT git-tracked here -- only /app/repo is).
#
# HEAD MUST NOT MOVE. solve.sh only writes files into the editable surface -- no git add,
# no commit, no branch. The verifier grades the WORK TREE and (where that surface is a git
# tree) gates on `git status --porcelain`, so a commit turns a correct solution into
# reward 0 (RUNNABLE_SPEC H3). HEAD is re-read after landing; the script aborts if it moved.
#
# NOTE: 🔴 test.sh source-scans /app/submission for /tests/|compute_reward|launch_baseline|
#       /logs/verifier|reward.json|reward.txt|prompts.jsonl|reward_manifest|/opt/loop/private|
#       hidden_suite and zeroes the run (forbidden_edit_path) on a hit. Verified against that
#       regex: oracle_v2_launch_server.sh and both negative files are CLEAN;
#       oracle_sweep_launch_server.sh (line 7) and baseline_relaxed_launch.sh (line 4) mention
#       `launch_baseline` in a COMMENT and would therefore be rejected. So:
#         * --variant oracle_sweep is implemented but prints a loud gate warning;
#         * baseline_relaxed_launch.sh is deliberately NOT offered -- it is a candidate for
#           tests/launch_baseline.sh (the frozen 1.0 anchor), not a submission launcher, so
#           landing it in /app/submission would be wrong on top of tripping the gate.
#       The probe/diag helpers (probe_pair.py, probe_burst_ladder.py, client_bound_diag.py)
#       also reference the evaluation surface and are reviewer-side tools -- run them from
#       outside /app/submission, never land them.
# =============================================================================
set -uo pipefail

TASK_ID='vllm-scheduler-mixed-batch-serving'
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(dirname "$SELF")"
BK="${SOLVE_BACKUP_DIR:-/tmp/.fai_solve_baseline}/${TASK_ID}"

say()  { printf '[solve:%s] %s\n' "$TASK_ID" "$*"; }
warn() { printf '[solve:%s] WARN: %s\n' "$TASK_ID" "$*" >&2; }
die()  { printf '[solve:%s] ERROR: %s\n' "$TASK_ID" "$*" >&2; exit 1; }

# Reference assets live OUTSIDE the image on purpose. Look for them wherever the
# operator may have mounted the package; SOLVE_ASSET_DIR overrides everything.
ASSET_ROOTS=("${SOLVE_ASSET_DIR:-}" "$SELF" "$PKG" "$PKG/tests" "$PKG/solution" \
             /solution /solution/tests /task /task/solution /task/tests /tests /opt/solution)
asset() { local rel="$1" r
  for r in "${ASSET_ROOTS[@]}"; do [ -n "$r" ] || continue
    [ -e "$r/$rel" ] && { printf '%s\n' "$r/$rel"; return 0; }
  done
  return 1
}
need() { asset "$1" || die "reference asset '$1' not found.
  searched: ${ASSET_ROOTS[*]}
  mount the task package (e.g. -v <pkg>:/task:ro) or set SOLVE_ASSET_DIR=<dir>."; }

# Pristine snapshot, kept OUTSIDE every graded surface (/tmp), so --noop can undo a
# landing inside a long-lived container. Taken once, right before the first overwrite.
_k() { printf '%s' "$1" | tr '/' '_'; }
snap() { local f="$1" k; k="$(_k "$1")"; mkdir -p "$BK" 2>/dev/null || return 0
  [ -e "$BK/$k" ] || [ -e "$BK/$k.ABSENT" ] || {
    if [ -e "$f" ]; then cp -p "$f" "$BK/$k" 2>/dev/null || true
    else : > "$BK/$k.ABSENT" 2>/dev/null || true; fi; }; }
unsnap() { local f="$1" k; k="$(_k "$1")"
  if   [ -e "$BK/$k" ];        then cp -p "$BK/$k" "$f" && say "restored $f from the pristine snapshot"
  elif [ -e "$BK/$k.ABSENT" ]; then rm -f "$f" && say "removed $f (absent in the pristine image)"
  else return 1; fi; }

put() { # put <src> <dst> : idempotent copy, snapshots the pristine <dst> first
  local s="$1" d="$2"
  [ -f "$s" ] || die "missing reference file: $s"
  mkdir -p "$(dirname "$d")" || die "cannot create $(dirname "$d")"
  snap "$d"
  if cmp -s "$s" "$d"; then say "already landed (identical): $d"
  else cp -f "$s" "$d" || die "copy failed: $s -> $d"; say "landed $(basename "$s") -> $d"; fi
  [ -x "$s" ] && { chmod +x "$d" 2>/dev/null || true; }
  return 0
}

head_of() { git -C "$1" rev-parse HEAD 2>/dev/null || echo NO_GIT; }
assert_head() { # assert_head <dir> <before>
  local now; now="$(head_of "$1")"
  [ "$now" = "$2" ] || die "HEAD of $1 moved ($2 -> $now).
  The verifier grades the WORK TREE and asserts HEAD is still the single baked baseline
  commit (RUNNABLE_SPEC H3) -- a commit makes a correct solution score 0."
  [ "$now" = NO_GIT ] || say "HEAD unchanged: $1 @ $(printf '%.12s' "$now")"
}

MODE=oracle
while [ $# -gt 0 ]; do
  case "$1" in
    --oracle)    MODE=oracle ;;
    --negative)  MODE=negative ;;
    --baseline2) MODE=baseline2 ;;
    --noop)      MODE=noop ;;
    --variant)   shift; [ $# -gt 0 ] || die "--variant needs a name"; MODE="$1" ;;
    --list)      MODE=__list ;;
    -h|--help)   MODE=__help ;;
    *)           die "unknown argument '$1' (try --help)" ;;
  esac
  shift
done

list_variants() {
  printf 'variants for %s:\n' "$TASK_ID"
  printf '  %-26s %s\n' '(default) / --oracle' 'oracle_v2_launch_server.sh -- async scheduling + 4 API servers'
  printf '  %-26s %s\n' '--negative' 'negative_fake_launch_server.sh + negative_fake_server.py -> 0'
  printf '  %-26s %s\n' '--variant oracle_sweep' 'the ORACLE_VARIANT sweep harness (⚠ trips the source-scan gate)'
  printf '  %-26s %s\n' '--noop' 'reset launch_server.sh to the baked baseline -- the TRUE no-op arm'
}
if [ "$MODE" = __list ]; then list_variants; exit 0; fi
if [ "$MODE" = __help ]; then
  sed -n '2,/^set -uo pipefail$/p' "${BASH_SOURCE[0]}" | sed '$d'
  list_variants
  exit 0
fi

SUB="${SUBMISSION_DIR:-/app/submission}"
GITDIR="${REPO_DIR:-/app/repo}"

land() { # land <asset-relpath=>dest-name> ...
  local before spec s d
  [ -d "$SUB" ] || mkdir -p "$SUB" || die "cannot create $SUB"
  before="$(head_of "$GITDIR")"
  for spec in "$@"; do
    s="${spec%%=>*}"; d="${spec##*=>}"
    put "$(need "$s")" "$SUB/$d"
  done
  assert_head "$GITDIR" "$before"
  [ "$before" = NO_GIT ] || { say "work-tree state of $GITDIR:"
    git -C "$GITDIR" status --porcelain 2>/dev/null | sed 's/^/    /'; }
}

restore() { # restore <dest-name> ...
  local before f; before="$(head_of "$GITDIR")"
  for f in "$@"; do
    if [ "$before" != NO_GIT ] \
       && git -C "$GITDIR" ls-files --error-unmatch -- "$SUB/$f" >/dev/null 2>&1; then
      git -C "$GITDIR" checkout -- "$SUB/$f" \
        && say "restored $SUB/$f from the baked baseline commit"
    elif ! unsnap "$SUB/$f"; then
      warn "no pristine copy of $SUB/$f is recoverable here -- use a fresh container for a true no-op arm"
    fi
  done
  assert_head "$GITDIR" "$before"
}

case "$MODE" in
  oracle)    land 'solution/oracle_v2_launch_server.sh=>launch_server.sh' ;;
  negative)  land 'solution/negative_fake_launch_server.sh=>launch_server.sh' \
                  'solution/negative_fake_server.py=>negative_fake_server.py' ;;
  oracle_sweep)
             warn "solution/oracle_sweep_launch_server.sh mentions 'launch_baseline' in a comment;"
             warn "test.sh's source scan over /app/submission will reject it with"
             warn "forbidden_edit_path (reward 0). Land it only for a sweep you score by hand,"
             warn "or strip that comment in your own copy first (the shipped artifact is read-only"
             warn "for solve.sh by contract)."
             land 'solution/oracle_sweep_launch_server.sh=>launch_server.sh' ;;
  baseline2) die "no baseline2 submission arm: solution/baseline_relaxed_launch.sh is a candidate for
  the FROZEN tests/launch_baseline.sh anchor, not a /app/submission launcher (and it mentions
  'launch_baseline' in a comment, which test.sh's source scan rejects). See the NOTE in the header." ;;
  noop)      rm -f "$SUB/negative_fake_server.py"
             restore launch_server.sh ;;
  *) die "no such variant: $MODE (see --list)" ;;
esac
say "done. GRADE WITH:"
say '  bash /tests/test.sh          # no mode variable; test.sh grades /app/submission/launch_server.sh'
