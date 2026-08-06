# syntax=docker/dockerfile:1
# =============================================================================
# wro-sp24mm-matmul-sol — OPEN-SOURCE, self-contained task image
#
#   Task: restore a Triton GEMM that exploits 2:4 semi-structured sparsity instead of densifying the weight.
#         The working tree at /app/repo ships a SLOW-BUT-CORRECT implementation; the
#         agent may edit ONLY sp24mm/matmul.py and must make it fast without
#         changing the public signature or the numerical contract.
#
#   FULLY PUBLIC — no internal mirror, no private base image, no prebuilt image.
#   Needs 1 NVIDIA GPU to RUN (task.toml: gpus = 1, calibrated on H20); no GPU needed to BUILD.
#   Platform: linux/amd64 ONLY (official torch wheels are manylinux x86_64).
#   Build context = this task directory (the Dockerfile lives in environment/).
#
# -----------------------------------------------------------------------------
# BUILD  (needs outbound network for PyPI and download.pytorch.org)
#     cd <...>/wro-sp24mm-matmul-sol
#     docker build -f environment/Dockerfile -t fai/lh-sp24mm-matmul:oss .
#
# RUN
#     docker run --rm --gpus all -it fai/lh-sp24mm-matmul:oss
#
# SCORE (inside the container; tests/ is MOUNTED, never baked into the image)
#     docker run --rm --gpus all -v "$PWD/tests:/tests:ro" fai/lh-sp24mm-matmul:oss bash /tests/test.sh
#     # -> /logs/verifier/reward.json   (modes: candidate | noop | oracle | negative)
#
# 🔴 RE-CALIBRATE THE ANCHOR ON YOUR HARDWARE
#   reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; ref_speedup is a FROZEN
#   constant read from /opt/verifier-correctness-manifest.json (baked below from this
#   package's tests/verifier-correctness-manifest.json) and was calibrated on H20 (see the calibrated_on field in that JSON for the exact shape).
#   Anchors on this bench are LARGE (this one: 10.28458 x) because they compare an eager
#   Python/dense reference against a real kernel; the ratio is dominated by the algorithmic
#   difference and is therefore fairly portable, but it is still a single-machine measurement.
#   The verifier natively supports mode dispatch, so re-measuring is ONE command — the
#   reference patches ship in this package and are passed in by PATH, never baked:
#     docker run --rm --gpus all -v "$PWD/tests:/tests:ro" -v "$PWD:/patches:ro" \
#       -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_PATCH=/patches/oracle.patch \
#       fai/lh-sp24mm-matmul:oss bash /tests/test.sh
#     # read the measured speedup from /logs/verifier/ (median of >=3 runs) and write it into
#     # tests/verifier-correctness-manifest.json -> ref_speedup
#   Sanity modes: noop (unmodified tree) should score ~the no-op value; negative
#   (KERNELBENCH_NEGATIVE_PATCH=/patches/negative.patch) must score 0.
#
# WHY THE WORKING TREE IS VENDORED IN THIS PACKAGE (and not cloned)
#   /app/repo is a CLEAN-ROOM package authored for this task — there is no upstream
#   repository to clone, and the reference implementation exists only as this package's
#   oracle.patch. The degraded tree under environment/repo/ was recovered from the
#   originally shipped image and verified: `git apply --check oracle.patch` succeeds
#   against it, which proves it is byte-exact the graded starting point. It is a few tens
#   of KB, so vendoring keeps the package small anyway.
#   environment/repo/ deliberately carries NO .git — the single baseline commit is created
#   at build time (with core.fileMode=false, see the note at that step).
#
#   oracle.patch / negative.patch are reviewer-only: excluded by the context-root .dockerignore
#   so they can never end up in an image layer, and supplied at scoring time by env var.
#
# PROVENANCE of environment/repo/ (so anyone can re-derive it independently)
#   recovered from: <internal registry>/kernelbench/wro-sp24mm-matmul-sol-loop16:v1-ms1@sha256:4e3141eb3e4c642d4e40ae1b454b59b1cec99f4c1c31346a925c9937944c5446
#   method: `crane manifest` -> the small crane-appended layers -> `crane blob` + tar,
#           overlaid in manifest order; .git and any .wh..wh..opq marker dropped.
#   verification: `git apply --check -p1 oracle.patch` succeeds against these bytes,
#           and `git apply --check -p1 negative.patch` does too — i.e. this tree is the
#           exact carved baseline both reference patches were generated against.
# =============================================================================
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

# Compute capability of the GPU you will RUN on (Triton JIT targets the live device, this
# only matters for hand-built CUDA extensions): 8.0 A100 / 8.9 L40S-4090 / 9.0 H100-H20 / 10.0 B200
ARG TORCH_CUDA_ARCH_LIST="9.0"
ARG TORCH_VERSION=2.7.0

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc-10 g++-10 cmake ninja-build pkg-config \
        python3.10 python3.10-dev python3.10-venv python3-pip python-is-python3 \
        git ca-certificates curl procps coreutils tar gzip vim less file jq \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100 \
    && rm -rf /var/lib/apt/lists/*

# ---- venv. Keep this exact path: tests/test.sh puts /opt/kernelbench-venv/bin at the head
#      of PATH, and a bare python3 from anywhere else would not have torch. ----
ENV VENV=/opt/kernelbench-venv
RUN python3.10 -m venv "$VENV"
ENV PATH=$VENV/bin:/usr/local/cuda/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN pip install --no-cache-dir -U pip setuptools wheel

# ---- torch 2.7.0 from the OFFICIAL public index; it pins its own matching triton (3.3.x).
#      The original private base shipped torch 2.5.1 / triton 3.1.0 and the task overlay then
#      upgraded the venv to torch 2.7.0 — 2.7.0 is therefore the version the anchor was
#      calibrated under, so it is pinned here directly. ----
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
        "torch==${TORCH_VERSION}" \
 && python -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__); assert torch.__version__.startswith('2.7.0'), torch.__version__"

# ---- the graded working tree. A REAL directory (not a symlink): the scope gate runs
#      `git -C /app/repo status --porcelain`, which cannot see through a symlink. ----
RUN mkdir -p /app/repo /app/.timer /app/workspace /logs/verifier /logs/agent /logs/artifacts
COPY environment/repo/ /app/repo/
COPY instruction.md /app/instruction.md

# ---- the baseline commit = the diff base the scope gate compares against.
#      🔴 core.fileMode=false is LOAD-BEARING, not hygiene: the originally shipped image had
#      core.fileMode=true while some files were mode 0775 on disk but 100644 in the index, so
#      `git status --porcelain` reported phantom out-of-scope paths in EVERY mode and forced
#      reward to 0. Setting it false at repo creation makes that class of defect impossible. ----
RUN git config --global user.email "reviewer@fai-bench.local" \
 && git config --global user.name "fai-bench" \
 && git config --global init.defaultBranch main \
 && git config --global --add safe.directory /app/repo \
 && cd /app/repo \
 && git init -q \
 && git config core.fileMode false \
 && printf '__pycache__/\n*.pyc\n' >> .gitignore \
 && git add -A \
 && git commit -qm "degraded baseline (graded starting point)" \
 && rm -rf .git/logs \
 && test "$(git rev-list --count HEAD)" = "1" \
 && test -z "$(git remote)" \
 && test "$(git config --get core.fileMode)" = "false" \
 && test -z "$(git status --porcelain)" \
 && echo "[build] /app/repo: 1 commit, no remote, fileMode=false, clean tree"

# ---- runtime contract + the 16-round self-improvement harness, exactly as the original image
#      shipped it. /opt/loop/private/ holds the IN-SESSION DEV proxy scorer: by this bench's
#      design the timed workload is NOT a secret (the dev proxy needs it, and its manifest says
#      "dev reward is an in-session proxy; the runner end-of-session /tests score is
#      authoritative"). What is never present is the reference implementation. ----
COPY environment/runtime/entrypoint.sh /app/entrypoint.sh
COPY environment/runtime/timer.sh /app/timer.sh
COPY environment/runtime/run_dev_bench.sh /app/workspace/run_dev_bench.sh
COPY environment/loop/ /opt/loop/
RUN chmod 0755 /app/entrypoint.sh /app/timer.sh /app/workspace/run_dev_bench.sh /opt/loop/submit.sh \
 && chown -R root:root /app/entrypoint.sh /app/timer.sh /opt/loop \
 && chmod 0700 /opt/loop/score_engine.sh /opt/loop/sanitize_feedback.py \
 && chmod -R go-rwx /opt/loop/private \
 && test -f /opt/loop/private/manifest.json \
 && test -f /opt/loop/private/tests/test.sh

# ---- the FROZEN anchor. tests/test.sh reads ref_speedup from this exact path and silently
#      falls back to 1.0 (which would make every reward 0) if it is missing, so it must be
#      baked. It carries no reference implementation — only the calibrated constant. ----
COPY tests/verifier-correctness-manifest.json /opt/verifier-correctness-manifest.json

# ---- OPEN grading suite is NOT baked: tests/ (test.sh + compute_reward.py + workload.py + h20.py (roofline/SOL helpers))
#   is MOUNTED FRESH at /tests when scoring. It ships in this package — open-source, but
#   outside the image so the graded harness cannot be edited from inside the session.

# ---- 🔴 ANTI-LEAK ASSERTS. The reference implementation must not exist anywhere in the image,
#      and the scope files must still be the DEGRADED ones. ----
#      (The filename list below is an AUTHOR-SIDE ARTIFACT DENYLIST: names that only ever
#      exist in the authoring workspace. Asserting they are absent proves nothing leaked.)
RUN set -eu; rc=0; \
    hits="$(find / -xdev -type f \( -name 'oracle*.patch' -o -name 'negative*.patch' \
              -o -name 'baseline2.patch' -o -name 'scope_card.json' \) 2>/dev/null | head -10)"; \
    if [ -n "$hits" ]; then echo "BUILD_ASSERT_FAIL: a reference/authoring file is baked:"; echo "$hits"; rc=1; fi; \
    for p in /opt/verifier /opt/strong_baseline /opt/negative /opt/ceiling /tests; do \
      if [ -e "$p" ]; then echo "BUILD_ASSERT_FAIL: frozen-surface path baked: $p"; rc=1; fi; \
    done; \
    for s in sp24mm/matmul.py; do \
      if grep -qE "@triton\.jit|tl\.dot" "/app/repo/$s"; then \
        echo "BUILD_ASSERT_FAIL: $s already contains a Triton kernel - the tree is NOT the degraded baseline"; rc=1; fi; \
    done; \
    test "$rc" = 0; \
    echo "[build] ANTI-LEAK ASSERTS PASS: no reference impl, no authoring artefact, no /tests"

ENV PYTHONPATH=/app/repo \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    LD_LIBRARY_PATH=/opt/kernelbench-venv/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64
WORKDIR /app/repo

# ---- build-time smoke: the graded package imports FROM /app/repo (the verifier asserts the
#      same thing at scoring time via its import-origin gate) and the scope files are present. ----
RUN python - <<'PY'
import importlib, os, sys
sys.path.insert(0, "/app/repo")
m = importlib.import_module("sp24mm")
loc = os.path.dirname(getattr(m, "__file__", "") or "")
print("sp24mm imported from", loc)
assert loc.startswith("/app/repo"), loc
for f in ('sp24mm/matmul.py',):
    p = os.path.join("/app/repo", f)
    assert os.path.isfile(p), f"scope file missing: {p}"
    print("scope file present:", f, os.path.getsize(p), "B")
import json
mf = json.load(open("/opt/verifier-correctness-manifest.json"))
print("baked anchor ref_speedup =", mf.get("ref_speedup"))
assert isinstance(mf.get("ref_speedup"), (int, float)) and mf["ref_speedup"] > 1, mf
print("OK: import origin, scope files and anchor all in place")
PY

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["tail","-f","/dev/null"]
