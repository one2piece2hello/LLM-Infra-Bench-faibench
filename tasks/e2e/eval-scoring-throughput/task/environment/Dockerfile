# syntax=docker/dockerfile:1
# =============================================================================
# e2e-h3-eval-harness-throughput-quality — OPEN-SOURCE, self-contained task image
#
#   Task: make lm-evaluation-harness' post-generation SCORING path (filter ensemble +
#         metric computation) as fast as possible over a large record set while
#         reproducing EVERY per-sample score EXACTLY. The agent may rewrite
#         /app/submission/scoring_pipeline.py and/or edit lm_eval inside /app/repo.
#
#   FULLY PUBLIC — no internal mirror, no private base image, no prebuilt image.
#   CPU ONLY — task.toml declares gpus = 0; a CPU-only torch wheel is installed on purpose
#              (the graded scoring path is pure Python/numpy; torch is only an lm_eval import).
#   Platform: linux/amd64 or arm64.
#   Build context = the task/ directory (this Dockerfile lives in task/environment/).
#
# -----------------------------------------------------------------------------
# BUILD  (needs outbound network: PyPI + github.com)
#     cd <...>/eval-scoring-throughput/task
#     docker build -f environment/Dockerfile -t fai/e2e-h3-eval-scoring-throughput:oss .
#
# RUN
#     docker run --rm -it fai/e2e-h3-eval-scoring-throughput:oss
#
# SCORE (inside the container; tests/ is MOUNTED, never baked into the image)
#     docker run --rm -v "$PWD/tests:/tests:ro" \
#       fai/e2e-h3-eval-scoring-throughput:oss bash /tests/test.sh   # -> /logs/verifier/reward.json
#
# 🔴 RE-CALIBRATE THE TIMING ANCHOR ON YOUR MACHINE
#   reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0 with the FROZEN constants in
#   tests/verifier-correctness-manifest.json:
#       ref_speedup = 2.24419   (median ABBA speedup of the strongest reviewer reference)
#       strong_baseline_time_sec = 0.174827
#   Both were measured on the authoring CPU lane (Intel Sapphire Rapids, AVX-512-VNNI). The
#   verifier re-times the strong baseline live in ABBA pairs, so the RATIO is fairly portable —
#   but a machine with very different single-core throughput or vector width can still shift it.
#   The reviewer reference implementations ship in this package under tests/oracles/, so you can
#   re-measure ref_speedup directly:
#     docker run --rm -v "$PWD/tests:/tests:ro" -e VERIFIER_MODE=<oracle-name> \
#       fai/e2e-h3-eval-scoring-throughput:oss bash /tests/test.sh
#     # read the measured speedup from /logs/verifier/benchmark_results.json (median of >=3 runs)
#     # and write it into tests/verifier-correctness-manifest.json -> ref_speedup.value
#   (ref_speedup <= 1 or missing => HARD FAIL 0 by design.)
#
# VERSION ANCHORS
#     Python 3.10 · lm-evaluation-harness @ 97a5e2c (0.4.13.dev0) · torch 2.7.0+cpu · numpy>=1.26,<3
#
# THE DATASET IS REGENERATED, NOT SHIPPED
#   The original private base baked /data/eval_harness/dev_samples.jsonl as an opaque blob.
#   This build instead runs the package's OWN generator (environment/build_dataset.py) — pure
#   stdlib, fixed seed 20240725, no network — and then ASSERTS that the held-out split it
#   produces is byte-identical to the digest the anchors were calibrated on. If that assert
#   passes, your corpus is provably the calibrated one. The held-out blob is DELETED IN THE SAME
#   LAYER, so it never exists in the image; the graded copy comes from tests/ at scoring time.
#
#   The original private overlay is kept beside this file as Dockerfile.orig_private_base.
# =============================================================================
FROM python:3.10-slim-bookworm

ARG LMEVAL_REF=97a5e2c710e2b56b9dd48f367bb6fe87bbb2c176
ARG HELDOUT_SHA256=b387c1747ee312ac400ebf7e9ea5138cf336a57b0d8b661448f3193cff61eb55

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates curl procps coreutils \
        tar gzip vim less file jq \
    && rm -rf /var/lib/apt/lists/*

# ---- venv. Keep this exact path: tests/test.sh pins /opt/kernelbench-venv/bin ----
ENV VENV=/opt/kernelbench-venv
RUN python3 -m venv "$VENV"
ENV PATH=$VENV/bin:$PATH
RUN pip install --no-cache-dir -U pip setuptools wheel

# ---- CPU-only torch first, so lm_eval's dependency resolution cannot pull a multi-GB CUDA wheel
#      into a task that declares gpus = 0. ----
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.7.0" \
 && python -c "import torch; print('torch', torch.__version__); assert torch.__version__.startswith('2.7.0')"

# ---- /app/repo: the WHOLE lm-evaluation-harness tree at the recorded commit, editable-installed.
#      The task is whole-repo editable: the agent may rewrite lm_eval's filter/metric/batching path.
#      The commit is the one recorded in tests/verifier-correctness-manifest.json
#      (source_commit_app_repo) — fetched by SHA so the tree is exactly the calibrated one.
#      Installed WITH its declared dependencies (unlike the original overlay, whose private base
#      already carried them) and with torch constrained so the CPU wheel above is kept. ----
RUN set -eu; \
    mkdir -p /app/repo; cd /app/repo; \
    git init -q; \
    git remote add origin https://github.com/EleutherAI/lm-evaluation-harness.git; \
    git fetch -q --depth 1 origin "${LMEVAL_REF}"; \
    git checkout -q FETCH_HEAD; \
    test -f /app/repo/lm_eval/api/task.py || { echo "BUILD_ASSERT_FATAL: lm_eval marker missing"; exit 1; }; \
    test -f /app/repo/pyproject.toml || { echo "BUILD_ASSERT_FATAL: pyproject.toml missing"; exit 1; }; \
    n="$(find /app/repo -type f -not -path '*/.git/*' | wc -l)"; \
    echo "[build] /app/repo file count = $n"; \
    test "$n" -ge 500 || { echo "BUILD_ASSERT_FATAL: /app/repo has only $n files - clone mutilated"; exit 1; }
RUN printf 'torch==2.7.0\n' > /tmp/constraints.txt \
 && pip install --no-cache-dir --constraint /tmp/constraints.txt "numpy>=1.26,<3" \
 && pip install --no-cache-dir --constraint /tmp/constraints.txt -e /app/repo \
 && python -c "import lm_eval, pathlib; p=pathlib.Path(lm_eval.__file__).resolve(); print('lm_eval', lm_eval.__version__, p); assert str(p).startswith('/app/repo/'), p" \
 && python -c "import torch; assert torch.__version__.startswith('2.7.0'), torch.__version__"

# ---- the scoring-record dataset. ONLY the agent-visible DEV split survives, under /data.
#      The HELD-OUT split is generated, its digest is ASSERTED against the calibrated constant,
#      and then DELETED IN THE SAME LAYER, so no blob of it ever exists in this image.
#      Deterministic, pure stdlib, NO network. ----
COPY environment/build_dataset.py /tmp/e2e_build_dataset.py
RUN python3 /tmp/e2e_build_dataset.py \
 && got="$(sha256sum /tmp/heldout_samples.jsonl | awk '{print $1}')" \
 && echo "held-out digest: ${got}" \
 && { test "${got}" = "${HELDOUT_SHA256}" \
      || { echo "BUILD_ASSERT_FATAL: regenerated held-out split != the CALIBRATED timed workload."; \
           echo "  expected ${HELDOUT_SHA256}"; echo "  got      ${got}"; \
           echo "  ref_speedup=2.24419 was calibrated on the expected one - re-calibrate (see header)"; \
           exit 1; }; } \
 && echo "OK: generated held-out split is byte-identical to the CALIBRATED timed workload" \
 && rm -f /tmp/e2e_build_dataset.py /tmp/heldout_samples.jsonl \
 && test ! -e /tmp/heldout_samples.jsonl \
 && test ! -e /tmp/e2e_build_dataset.py \
 && test -f /data/eval_harness/dev_samples.jsonl \
 && echo "[build] agent-visible DEV split: $(wc -l < /data/eval_harness/dev_samples.jsonl) rows"

# ---- solver-owned submission skeleton + the starter template + the task statement ----
RUN mkdir -p /app/submission /logs/verifier /logs/artifacts
COPY environment/workspace/scoring_pipeline_template.py /app/submission/scoring_pipeline_template.py
COPY instruction.md /app/instruction.md

# ---- git baseline for the editable tree: ONE commit as the diff base, no upstream remote ----
RUN git config --global user.email "reviewer@fai-bench.local" \
 && git config --global user.name "fai-bench" \
 && git config --global init.defaultBranch main \
 && git config --global --add safe.directory /app/repo \
 && cd /app/repo \
 && rm -rf .git \
 && printf '__pycache__/\n*.pyc\n' > .gitignore \
 && git init -q \
 && git add -A \
 && git commit -qm "lm-evaluation-harness @ pinned commit (upstream history stripped)" \
 && test "$(git rev-list --count HEAD)" = "1" \
 && test -z "$(git remote)" \
 && echo "git diff base ready: $(git rev-parse --short HEAD)"

# ---- OPEN grading suite is NOT baked into the image (by design).
#   tests/ (test.sh + compute_reward.py + the held-out split + the frozen manifest carrying the
#   calibrated anchors plus the reviewer reference implementations under tests/oracles/) is MOUNTED FRESH at /tests when scoring. It ships in this
#   package — open-source, but deliberately outside the image so the solver can read neither the
#   held-out data nor the anchors while solving.

# ---- 🔴 ANTI-LEAK ASSERTS: any regression that re-bakes the frozen surface or the held-out
#      split dies HERE, at build time, instead of silently invalidating the task. ----
RUN set -eu; \
    for p in /opt/verifier /opt/verifier-correctness-manifest.json /opt/strong_baseline \
             /opt/negative /opt/ceiling /tests; do \
      if [ -e "$p" ]; then echo "BUILD_ASSERT_FATAL: frozen surface leaked into the image at $p"; exit 3; fi; \
    done; \
    hits="$(find / -xdev -type f \( -name 'heldout_*' -o -name compute_reward.py \
              -o -name 'verifier-correctness-manifest.json' \) 2>/dev/null | head -20)"; \
    if [ -n "$hits" ]; then echo "BUILD_ASSERT_FATAL: baked frozen-surface files:"; echo "$hits"; exit 3; fi; \
    echo "[build] ANTI-LEAK ASSERTS PASS: no frozen surface, no held-out split, no anchor in the image"

# ---- runtime contract: minimal entrypoint + elapsed stub (this family has no wall-clock budget) ----
RUN printf '#!/bin/bash\nset -e\nif [ "$#" -gt 0 ]; then exec "$@"; fi\nexec /bin/bash -l\n' > /app/entrypoint.sh \
 && chmod +x /app/entrypoint.sh \
 && printf '#!/bin/bash\nSTART=$(date +%%s)\nwhile true; do echo "elapsed=$(( $(date +%%s) - START ))s"; sleep 60; done\n' > /app/timer.sh \
 && chmod +x /app/timer.sh

ENV DATA_PATH=/data/eval_harness \
    SUBMISSION_DIR=/app/submission \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app/submission

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["tail","-f","/dev/null"]
