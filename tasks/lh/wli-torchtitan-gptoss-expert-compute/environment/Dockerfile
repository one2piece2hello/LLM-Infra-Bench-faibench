# syntax=docker/dockerfile:1
# =============================================================================
# wli-torchtitan-gptoss-expert-compute — OPEN-SOURCE, self-contained task image
#
#   Task: implement the GPT-OSS expert-compute path in torchtitan's MoE layers so the producer/consumer contract holds across a hidden graded case set.
#         Editable scope (the ONLY files the scope gate allows to change):
##     - torchtitan/models/common/moe.py
#     - torchtitan/models/gpt_oss/moe.py
#
#   FULLY PUBLIC — no internal mirror, no private base image, no prebuilt image.
#   CPU ONLY — task.toml declares gpus = 0 and the harness runs genuine torch tensor ops on CPU.
#   Platform: linux/amd64 or arm64.
#   Build context = this task directory (the Dockerfile lives in environment/).
#
# -----------------------------------------------------------------------------
# BUILD  (needs outbound network for PyPI only — the working tree ships in this package)
#     cd <...>/wli-torchtitan-gptoss-expert-compute
#     docker build -f environment/Dockerfile -t fai/lh-torchtitan-gptoss-expert-compute:oss .
#
# RUN     docker run --rm -it fai/lh-torchtitan-gptoss-expert-compute:oss
# SCORE   docker run --rm -v "$PWD/tests:/tests:ro" fai/lh-torchtitan-gptoss-expert-compute:oss bash /tests/test.sh
#         # -> /logs/verifier/reward.json
#         # modes via -e KERNELBENCH_VERIFY_MODE=: candidate | noop | oracle | negative | baseline2
#
# REWARD: this is a CORRECTNESS task — primary_metric = graded_correctness, computed
#   entirely inside the container from references the harness itself derives. There is NO
#   hardware-calibrated constant, so results are hardware-independent and need no
#   re-calibration. Score the reference with:
#     docker run --rm -v "$PWD/tests:/tests:ro" -v "$PWD:/patches:ro" \
#       -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_PATCH=/patches/oracle.patch \
#       fai/lh-torchtitan-gptoss-expert-compute:oss bash /tests/test.sh
#   negative (KERNELBENCH_NEGATIVE_PATCH=...) MUST score 0.
#
# DEPENDENCIES: the harness imports ONLY torch and stubs the rest of torchtitan's stack
#   with unittest.mock, so torchtitan's own requirements are deliberately NOT installed.
#
# HOW THE GRADED WORKING TREE IS REPRODUCED
#   /app/repo is the torchtitan repository with the target implementation CARVED OUT (a slow-but-correct or
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
#   recovered from: <internal registry>/kernelbench/wli-torchtitan-gptoss-expert-compute:sa3fix3@sha256:22e76f3b11b9505652ace09b8ec38c06e19e58ed7c877389c9f7bcc0a5574c2f
#   method: `crane manifest` -> the small crane-appended layers -> `crane blob` + tar,
#           overlaid in manifest order; .git and any .wh..wh..opq marker dropped.
#   verification: `git apply --check -p1 oracle.patch` succeeds against these bytes,
#           and `git apply --check -p1 negative.patch` does too — i.e. this tree is the
#           exact carved baseline both reference patches were generated against.
# =============================================================================
FROM python:3.11-slim-bookworm

ARG TORCH_VERSION=2.7.0

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates curl procps coreutils tar gzip vim less file jq \
    && rm -rf /var/lib/apt/lists/*

# ---- venv at this EXACT path: tests/test.sh puts /opt/kernelbench-venv/bin at the head of PATH. ----
ENV VENV=/opt/kernelbench-venv
RUN python3 -m venv "$VENV"
ENV PATH=$VENV/bin:$PATH
RUN pip install --no-cache-dir -U pip setuptools wheel
# ---- CPU-ONLY torch on purpose: task.toml declares gpus = 0 and the harness comment says the
#      producer/consumer run "genuine tensor ops on CPU". The CPU wheel runs the SAME ATen CPU
#      kernels as the CUDA build, so numerics are unchanged, and the image is ~8 GB smaller. ----
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
 && python -c "import torch; print('torch', torch.__version__); assert torch.__version__.startswith('2.7.0'), torch.__version__"

# ---- 🔴 NO pip-installed twin of the graded package. ----
RUN python - <<'PY'
import importlib.util, sys
if importlib.util.find_spec("torchtitan") is not None:
    sys.exit("BUILD_ASSERT_FATAL: torchtitan is installed in site-packages - that reinstates the reference implementation")
print("OK: no pip-installed torchtitan twin; /app/repo will be the only copy")
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
    OPENBLAS_NUM_THREADS=1
WORKDIR /app/repo

RUN python - <<'PY'
import os, sys, subprocess
sys.path.insert(0, "/app/repo")
import torch
print("torch", torch.__version__)
for f in ("torchtitan/models/common/moe.py", "torchtitan/models/gpt_oss/moe.py"):
    p = os.path.join("/app/repo", f)
    assert os.path.isfile(p), f"scope file missing: {p}"
    print("scope file present:", f, os.path.getsize(p), "B")
n = subprocess.run(["git","-C","/app/repo","rev-list","--count","HEAD"],capture_output=True,text=True).stdout.strip()
assert n == "1", n
try:
    import importlib.util as u
    spec = u.spec_from_file_location("_moe_probe", "/app/repo/torchtitan/models/common/moe.py")
    m = u.module_from_spec(spec); spec.loader.exec_module(m)
    print("OK: the scope module imports standalone")
except Exception as e:
    print("WARN: standalone import of the scope module raised", repr(e),
          "- expected for a hollow start; the harness stubs torchtitan's stack with MagicMock")
print("OK: scope files and git base in place")
PY

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["tail","-f","/dev/null"]
