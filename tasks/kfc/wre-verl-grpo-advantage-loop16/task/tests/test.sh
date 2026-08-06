#!/usr/bin/env bash
# Canonical test.sh — mode dispatch + correctness HARD gate + isolated perf + reward.
# The scored unit is submission/advantage_estimators.py (no repo tree / git-diff gate).
# MODE (env or $1): candidate | noop | oracle | baseline2 | negative
set -uo pipefail
export PATH="/opt/kernelbench-venv/bin:/usr/local/bin:$PATH"

MODE="${1:-${WRE_MODE:-candidate}}"
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WRE_WORKSPACE:-/app/workspace}"
SUB="$WORK/submission/advantage_estimators.py"

# reviewer-only assets (uploaded fresh with tests/, never in the solver workspace)
ORACLE="$TESTS_DIR/oracle_advantage.py"
NAIVE="$TESTS_DIR/naive_advantage.py"
NEG="$TESTS_DIR/negative_advantage.py"
STUB="$TESTS_DIR/stub_advantage.py"

OUT="${WRE_OUT:-/logs/verifier/wre_result.json}"
mkdir -p "$(dirname "$OUT")"

case "$MODE" in
  candidate)  SCORED="$SUB" ;;
  noop)       SCORED="$STUB" ;;       # empty stub -> NotImplementedError -> correctness fail -> 0
  oracle)     SCORED="$ORACLE" ;;
  baseline2)  SCORED="$NAIVE" ;;
  negative)   SCORED="$NEG" ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac

WRE_MODE="$MODE" WRE_SCORED_MODULE="$SCORED" WRE_ORACLE_MODULE="$ORACLE" WRE_OUT="$OUT" \
  python3 "$TESTS_DIR/verify_core.py"
# verify_core always writes reward.json-equivalent; test.sh exits 0 (reward.json is the truth)
python3 "$TESTS_DIR/compute_reward.py" "$OUT" || true
exit 0
