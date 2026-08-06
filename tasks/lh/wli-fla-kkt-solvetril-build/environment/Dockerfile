# syntax=docker/dockerfile:1
# =============================================================================
# wli-fla-kkt-solvetril-build — OPEN-SOURCE, self-contained task image
#
#   Task: implement, from a hollow start, the K K^T scaled-dot kernel and the lower-triangular solve used by the chunked linear-attention path.
#         Editable scope (the ONLY files the scope gate allows to change):
##     - fla/ops/common/chunk_scaled_dot_kkt.py
#     - fla/ops/utils/solve_tril.py
#
#   FULLY PUBLIC — no internal mirror, no private base image, no prebuilt image.
#   Needs 1 NVIDIA GPU to RUN (task.toml: gpus = 1). No GPU needed to BUILD.
#   Platform: linux/amd64 ONLY (official torch wheels are manylinux x86_64).
#   Build context = this task directory (the Dockerfile lives in environment/).
#
# -----------------------------------------------------------------------------
# BUILD  (needs outbound network for PyPI only — the working tree ships in this package)
#     cd <...>/wli-fla-kkt-solvetril-build
#     docker build -f environment/Dockerfile -t fai/lh-fla-kkt-solvetril-build:oss .
#
# RUN     docker run --rm --gpus all -it fai/lh-fla-kkt-solvetril-build:oss
# SCORE   docker run --rm --gpus all -v "$PWD/tests:/tests:ro" fai/lh-fla-kkt-solvetril-build:oss bash /tests/test.sh
#         # -> /logs/verifier/reward.json
#         # modes via -e KERNELBENCH_VERIFY_MODE=: candidate | noop | oracle | negative | baseline2
#
# REWARD: this is a CORRECTNESS task — primary_metric = graded_suite_fraction, the
#   fraction of hidden graded cases that pass, computed entirely inside the container.
#   There is NO hardware-calibrated constant, so results are hardware-independent and
#   need no re-calibration. Score the reference with:
#     docker run --rm --gpus all -v "$PWD/tests:/tests:ro" -v "$PWD:/patches:ro" \
#       -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_PATCH=/patches/oracle.patch \
#       fai/lh-fla-kkt-solvetril-build:oss bash /tests/test.sh
#   negative (KERNELBENCH_NEGATIVE_PATCH=...) MUST score 0.
#
# NOTE ON sitecustomize: tests/test.sh creates its OWN throwaway sitecustomize (a
#   tilelang import blocker) in a mktemp dir and puts it first on PYTHONPATH, so unlike
#   the sibling fla tasks this image does not need to bake /opt/wro_sitecustomize.
#
# HOW THE GRADED WORKING TREE IS REPRODUCED
#   /app/repo is the flash-linear-attention repository with the target implementation CARVED OUT (a slow-but-correct or
#   stubbed reference in its place). It is VENDORED under environment/repo/ rather than cloned:
#   the shipped image built it from a pre-staged tarball, so no upstream commit was ever recorded,
#   and these bytes were RECOVERED FROM THE SHIPPED IMAGE AND VERIFIED — this package's
#   oracle.patch applies cleanly to them (`git apply --check`), which proves they are the exact
#   starting point every graded run saw. Vendoring also makes the build independent of GitHub.
#   environment/repo/ carries NO .git; the single baseline commit is created at build time.
#
#   oracle.patch / negative.patch are reviewer-only and excluded by the context-root .dockerignore.
#
# PROVENANCE of environment/repo/ (so anyone can re-derive it independently)
#   recovered from: <internal registry>/kernelbench/wli-fla-kkt-solvetril-build:v3y@sha256:9eab3170a359b786ab0d3ce3a9ce8a91820eb132cb0746c95d9003521e1c8b55
#   method: `crane manifest` -> the small crane-appended layers -> `crane blob` + tar,
#           overlaid in manifest order; .git and any .wh..wh..opq marker dropped.
#   verification: `git apply --check -p1 oracle.patch` succeeds against these bytes,
#           and `git apply --check -p1 negative.patch` does too — i.e. this tree is the
#           exact carved baseline both reference patches were generated against.
# =============================================================================
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

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

ENV VENV=/opt/kernelbench-venv
RUN python3.10 -m venv "$VENV"
ENV PATH=$VENV/bin:/usr/local/cuda/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
        "torch==${TORCH_VERSION}" \
 && python -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__); assert torch.__version__.startswith('2.7.0'), torch.__version__"

# ---- 🔴 NO pip-installed twin of the graded package: it would reinstate the reference impl. ----
RUN python - <<'PY'
import importlib.util, sys
for name in ("fla", "flash_linear_attention"):
    if importlib.util.find_spec(name) is not None:
        sys.exit(f"BUILD_ASSERT_FATAL: {name} is installed in site-packages - that reinstates the reference implementation")
print("OK: no pip-installed fla twin; /app/repo will be the only copy")
PY

RUN mkdir -p /app/repo /app/.timer /app/workspace /logs/verifier /logs/agent /logs/artifacts
COPY environment/repo/ /app/repo/
COPY instruction.md /app/instruction.md

# ---- baseline commit = the diff base for the scope gate (`git -C /app/repo status --porcelain`).
#      🔴 core.fileMode=false is LOAD-BEARING: a shipped image in this corpus had core.fileMode=true
#      while some files were 0775 on disk but 100644 in the index, so the scope gate reported
#      phantom out-of-scope paths in EVERY mode and forced reward to 0. ----
RUN git config --global user.email "reviewer@fai-bench.local" \
 && git config --global user.name "fai-bench" \
 && git config --global init.defaultBranch main \
 && git config --global --add safe.directory /app/repo \
 && cd /app/repo \
 && git init -q \
 && git config core.fileMode false \
 && printf '__pycache__/\n*.pyc\n' >> .gitignore \
 && git add -A \
 && git commit -qm "carved baseline (graded starting point)" \
 && rm -rf .git/logs \
 && test "$(git rev-list --count HEAD)" = "1" \
 && test -z "$(git remote)" \
 && test "$(git config --get core.fileMode)" = "false" \
 && test -z "$(git status --porcelain)" \
 && echo "[build] /app/repo: 1 commit, no remote, fileMode=false, clean tree"

COPY environment/runtime/entrypoint.sh /app/entrypoint.sh
COPY environment/runtime/timer.sh /app/timer.sh
RUN chmod 0755 /app/entrypoint.sh /app/timer.sh && chown root:root /app/entrypoint.sh /app/timer.sh
COPY environment/runtime/run_dev_bench.sh /app/workspace/run_dev_bench.sh
RUN chmod 0755 /app/workspace/run_dev_bench.sh

# ---- OPEN grading suite is NOT baked: tests/ is MOUNTED FRESH at /tests when scoring. It ships
#   in this package — open-source, but outside the image so the graded harness cannot be edited
#   from inside the solving session.

# ---- 🔴 ANTI-LEAK ASSERTS ----
#      (The filename list below is an AUTHOR-SIDE ARTIFACT DENYLIST: names that only ever
#      exist in the authoring workspace. Asserting they are absent proves nothing leaked.)
RUN set -eu; rc=0; \
    hits="$(find / -xdev -type f \( -name 'oracle*.patch' -o -name 'negative*.patch' \
              -o -name 'scope_card.json' -o -name 'degradation_manifest.json' \) 2>/dev/null | head -10)"; \
    if [ -n "$hits" ]; then echo "BUILD_ASSERT_FAIL: a reference/authoring file is baked:"; echo "$hits"; rc=1; fi; \
    for p in /opt/verifier /opt/strong_baseline /opt/negative /opt/ceiling /tests; do \
      if [ -e "$p" ]; then echo "BUILD_ASSERT_FAIL: frozen-surface path baked: $p"; rc=1; fi; \
    done; \
    test "$rc" = 0; \
    echo "[build] ANTI-LEAK ASSERTS PASS"

ENV PYTHONPATH=/app/repo \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    LD_LIBRARY_PATH=/opt/kernelbench-venv/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64
WORKDIR /app/repo

RUN python - <<'PY'
import os, sys, subprocess
sys.path.insert(0, "/app/repo")
import torch
print("torch", torch.__version__)
import fla
print("fla imported from", fla.__file__)
assert fla.__file__.startswith("/app/repo/"), fla.__file__
for f in ("fla/ops/common/chunk_scaled_dot_kkt.py", "fla/ops/utils/solve_tril.py"):
    p = os.path.join("/app/repo", f)
    assert os.path.isfile(p), f"scope file missing: {p}"
    print("scope file present:", f, os.path.getsize(p), "B")
n = subprocess.run(["git","-C","/app/repo","rev-list","--count","HEAD"],capture_output=True,text=True).stdout.strip()
assert n == "1", n
print("OK: import origin, scope files and git base all in place")
PY

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["tail","-f","/dev/null"]
