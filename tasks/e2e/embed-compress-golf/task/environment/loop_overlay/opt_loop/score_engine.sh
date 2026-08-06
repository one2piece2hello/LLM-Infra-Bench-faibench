#!/usr/bin/env bash
# ============================================================================
# e2e-g2-embed-compress-golf — loop16 score_engine.sh   0700 root
# ============================================================================
# Invoked ONLY by /opt/loop/submit.sh. Scores the LIVE /app/submission encoder by running THIS
# task's OWN verifier (compute_reward.py, baked verbatim under /opt/loop/private/tests/) on the
# PUBLIC, DISJOINT dev split shipped at /data/retrieval/dev_* — the SAME two-stage retrieve+refine
# nDCG@10 pipeline, the SAME budget gate and anti-degenerate probes the runner grades, but on public
# data. It then normalizes the verifier's /logs/verifier JSON into the dev files submit.sh/sanitize
# read.
#
# 🔴 LEAK-FREE BY CONSTRUCTION: the overlay bakes NO held-out corpus/queries/qrels answer-key and NO
# calibrated anchor. This engine points the verifier's data-path env overrides at the PUBLIC dev
# split and hands it an UNCALIBRATED manifest (ref_speedup=null, strong_baseline_ndcg=0), so the
# private compute_reward emits a RAW dev_ndcg (its dev_score), used ONLY to rank best-of-k — never a
# score normalized against the graded anchor. The graded anchor is uploaded fresh to /tests at
# grading and is NOT in this container; this engine never reads /tests.
export PATH=/opt/kernelbench-venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
set -uo pipefail
git config --global --add safe.directory '*' 2>/dev/null || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV_OUT="${LOOP_DEV_OUT:-/logs/loop/dev}"; export LOOP_DEV_OUT="$DEV_OUT"; mkdir -p "$DEV_OUT"
TESTS="$HERE/private/tests"
VOUT=/logs/verifier            # the verifier writes its own 6-file contract here
mkdir -p "$VOUT"
rm -f "$DEV_OUT/verifier_state.json" "$DEV_OUT/reward.json" "$DEV_OUT/harness_error.txt" 2>/dev/null || true

PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ]; then
  echo "harness_error: no python on PATH" > "$DEV_OUT/harness_error.txt"; exit 3
fi
if [ ! -f "$TESTS/compute_reward.py" ]; then
  echo "baked dev verifier missing at $TESTS/compute_reward.py" > "$DEV_OUT/harness_error.txt"; exit 3
fi

# The PUBLIC dev split baked by the base image (disjoint from the graded held-out split).
DATA="${DATA_PATH:-/data/retrieval}"
if [ ! -f "$DATA/dev_corpus.jsonl" ] || [ ! -f "$DATA/dev_queries.jsonl" ] || [ ! -f "$DATA/dev_qrels.json" ]; then
  echo "harness_error: public dev split not found under $DATA (dev_corpus/dev_queries/dev_qrels)" > "$DEV_OUT/harness_error.txt"
  exit 3
fi

# Deterministic thread pin (mirror the graded env; nDCG is exact arithmetic so this cannot move it).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}" \
       TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""

# Point the verifier at the PUBLIC dev split + the UNCALIBRATED dev manifest (no anchor).
export E2E_CORPUS_PATH="$DATA/dev_corpus.jsonl"
export E2E_QUERIES_PATH="$DATA/dev_queries.jsonl"
export E2E_QRELS_PATH="$DATA/dev_qrels.json"
export E2E_MANIFEST_PATH="$TESTS/dev_manifest.json"
export SUBMISSION_DIR="${SUBMISSION_DIR:-/app/submission}"

# Run the verifier in a clean, path-hardened process (same isolation flags as tests/test.sh).
unset PYTHONPATH PYTHONHOME
cp "$TESTS/compute_reward.py" /tmp/e2e_embed_dev_verifier.py 2>/dev/null || true
SUBMISSION_DIR="$SUBMISSION_DIR" PYTHONSAFEPATH=1 "$PY" -I /tmp/e2e_embed_dev_verifier.py \
  > "$DEV_OUT/verdict.out" 2>"$DEV_OUT/verdict.err"
vr=$?

# Normalize /logs/verifier JSON -> dev files. On a genuine crash (no reward.json), refund.
"$PY" - "$DEV_OUT" "$VOUT" "$vr" <<'PYNORM'
import json, sys
from pathlib import Path
DEV = Path(sys.argv[1]); VOUT = Path(sys.argv[2]); vr = sys.argv[3]

def _load(p):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception: return None

metrics = _load(VOUT / "metrics.json") or _load(VOUT / "reward.json")
if metrics is None:
    err = ""
    try: err = (DEV / "verdict.err").read_text(errors="replace")[:400]
    except Exception: pass
    (DEV / "harness_error.txt").write_text(
        f"dev verifier produced no /logs/verifier/metrics.json (exit {vr}); "
        f"stderr head: {err.splitlines()[0] if err.strip() else '(none)'}\n")
    (DEV / "verifier_state.json").write_text(json.dumps({"correctness_ok": False, "hard_fail_reasons": ["harness_error"]}))
    (DEV / "reward.json").write_text(json.dumps({"dev_score": 0.0, "dev_ndcg": None}))
    sys.exit(3)

# correctness_ok := all HARD gates passed (budget / anti-degenerate / dim-match / refiner / plausibility).
# On the UNCALIBRATED dev manifest the reward field is 0 with hard_fail_reasons like
# 'ref_speedup_invalid_or_missing' / 'speedup_not_above_baseline' — those are ANCHOR-only fails and are
# NOT correctness fails, so we key correctness off all_hard_gates_pass, and rank on the RAW dev nDCG.
_ANCHOR_ONLY = {"ref_speedup_invalid_or_missing", "speedup_not_above_baseline",
                "build_or_entry_contract_failed"}
all_hard = bool(metrics.get("all_hard_gates_pass"))
lb = metrics.get("leaderboard") or {}
ndcg = lb.get("ndcg")
if ndcg is None:
    ndcg = metrics.get("ndcg")
# a build/entry-contract fail (e.g. ratio None because anchor absent) is NOT a correctness defect if the
# real hard gates all passed; correctness_ok reflects the genuine gates only.
hard = [str(r) for r in (metrics.get("hard_fail_reasons") or [])]
genuine_fail = [r for r in hard if r not in _ANCHOR_ONLY]
correctness_ok = all_hard and not genuine_fail and isinstance(ndcg, (int, float))

failing = ""
if not correctness_ok:
    failing = (genuine_fail[0] if genuine_fail else (hard[0] if hard else "encoder_or_budget_gate_failed"))

dev_ndcg = float(ndcg) if isinstance(ndcg, (int, float)) else None
dev_score = dev_ndcg if (correctness_ok and dev_ndcg is not None) else 0.0

(DEV / "verifier_state.json").write_text(json.dumps({
    "correctness_ok": correctness_ok,
    "hard_fail_reasons": hard,
    "failing_invariant": failing,
}, sort_keys=True))
(DEV / "reward.json").write_text(json.dumps({
    "dev_score": dev_score,
    "dev_metric": "nDCG@10 (PUBLIC dev proxy; higher is better; NOT your graded score)",
    "dev_ndcg": dev_ndcg,
    "correctness_ok": correctness_ok,
}, sort_keys=True))
sys.exit(0)
PYNORM
rc=$?
if [ "$rc" -ne 0 ] && [ "$rc" -ne 3 ] && [ ! -f "$DEV_OUT/harness_error.txt" ]; then
  { echo "score_engine_exit_${rc}"; tail -n 20 "$DEV_OUT/verdict.err" 2>/dev/null; } > "$DEV_OUT/harness_error.txt"
fi
exit "$rc"
