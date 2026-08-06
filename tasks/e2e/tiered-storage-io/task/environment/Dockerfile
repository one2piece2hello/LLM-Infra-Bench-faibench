# syntax=docker/dockerfile:1
# =============================================================================
# correctness-e2e-e5-tiered-storage-io — OPEN-SOURCE, self-contained task image
#
#   Tiered storage I/O (multi-tier cache with eviction/promotion semantics) — correctness task.
#   The agent edits ONLY  /app/submission/tiered_store.py
#   Scoring: the OPEN verifier under /tests (MOUNTED, never baked) drives that module.
#
#   FULLY PUBLIC — no internal mirror, no private base image, no prebuilt image.
#   This task is PURE PYTHON STDLIB: no torch, no CUDA, no GPU (task.toml: gpus = 0).
#   Platform: linux/amd64 or arm64 (pure Python; nothing is compiled at build time).
#
#   NO PERFORMANCE ANCHOR: this is a CORRECTNESS task — the reward is the fraction of hidden
#   cases passed, computed entirely inside the container. There is no hardware-calibrated
#   constant, so results are hardware-independent and need NO re-calibration on other machines.
#   Reference implementations live in this package's tests/oracles/ (when present) and can be
#   scored with -e VERIFIER_MODE=<oracle-name>; see tests/test.sh.
#
# -----------------------------------------------------------------------------
# BUILD   (context = the task/ directory; this file lives in task/environment/)
#     cd <...>/tiered-storage-io/task
#     docker build -f environment/Dockerfile -t fai/e5-tiered-storage-io:oss .
# RUN
#     docker run --rm -v "$PWD/tests:/tests:ro" -it fai/e5-tiered-storage-io:oss
# SCORE   (inside the container; tests/ is mounted, not baked)
#     bash /tests/test.sh                     # -> /logs/verifier/reward.json
#
#   The original private overlay is kept beside this file as Dockerfile.orig_private_base.
# =============================================================================
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC \
    PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# ---- minimal OS deps. git is REQUIRED: the verifier's scope gate runs
#      `git -C /app/submission status --porcelain` to list what the agent changed. ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates procps coreutils findutils tar gzip \
        vim less file jq tini \
    && rm -rf /var/lib/apt/lists/*

# ---- venv at the exact path tests/test.sh prepends to PATH
#      (`export PATH=/opt/kernelbench-venv/bin:$PATH`) so the interpreter the verifier
#      picks up is deterministic. This task needs no third-party packages. ----
RUN python3 -m venv /opt/kernelbench-venv
ENV PATH=/opt/kernelbench-venv/bin:$PATH

# ---- solver-editable submission + git scope-gate base (ONE commit, upstream history stripped) ----
RUN mkdir -p /app/submission /logs/verifier /logs/artifacts
COPY environment/workspace/submission/tiered_store.py /app/submission/tiered_store.py
COPY instruction.md /app/submission/instruction.md
COPY instruction.md /app/instruction.md
RUN cd /app/submission \
 && git init -q \
 && git config user.email "reviewer@fai-bench.local" \
 && git config user.name "fai-bench" \
 && git config --global --add safe.directory /app/submission \
 && printf '__pycache__/\n*.pyc\n' > .gitignore \
 && git add -A \
 && git commit -qm "baseline submission starter" \
 && test "$(git rev-list --count HEAD)" = "1"

# ---- runtime contract: entrypoint + elapsed-time daemon (kept for family consistency) ----
RUN printf '#!/bin/bash\nset -e\nmkdir -p /app/.timer\nnohup /app/timer.sh >/app/.timer/timer.log 2>&1 &\nif [ "$#" -gt 0 ]; then exec "$@"; fi\nexec /bin/bash -l\n' > /app/entrypoint.sh \
 && chmod +x /app/entrypoint.sh \
 && printf '#!/bin/bash\nmkdir -p /app/.timer\nSTART=$(date +%%s)\nwhile true; do echo $(( $(date +%%s) - START )) > /app/.timer/elapsed; sleep 5; done\n' > /app/timer.sh \
 && chmod 755 /app/timer.sh

# ---- OPEN grading suite is NOT baked. tests/ (test.sh + workload.py + ...) ships in this
#      package and is MOUNTED at /tests when scoring, so the solver cannot read/edit it while
#      solving. Nothing must live under /opt/verifier. ----
RUN rm -rf /opt/verifier

ENV E2E_SUBMISSION_DIR=/app/submission SUBMISSION_DIR=/app/submission
WORKDIR /app

# ---- build-time smoke (all HARD; this task has no heavy runtime to probe) ----
RUN set -eu \
 && python3 -c "import sys; print('python', sys.version.split()[0])" \
 && python3 -c "import struct, zlib, threading, hashlib, importlib.util; print('stdlib ok')" \
 && test -f /app/submission/tiered_store.py \
 && test -d /app/submission/.git \
 && test ! -e /opt/verifier \
 && test -x /app/entrypoint.sh \
 && echo "BUILD-ASSERTS OK: submission entry + git scope base present, no baked verifier"

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["/bin/bash","-l"]
