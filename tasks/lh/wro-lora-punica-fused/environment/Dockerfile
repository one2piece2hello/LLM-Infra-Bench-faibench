# syntax=docker/dockerfile:1
# =============================================================================
# wro-lora-punica-fused — OPEN-SOURCE, self-contained task image
#
#   Task: restore the Punica-style fused segmented multi-LoRA shrink/expand Triton GEMMs; the shipped tree loops over adapters eagerly.
#         Editable scope (the ONLY files the scope gate allows to change):
##     - vllm/lora/ops/triton_ops/kernel_utils.py
#     - vllm/lora/ops/triton_ops/lora_shrink_op.py
#     - vllm/lora/ops/triton_ops/lora_expand_op.py
#
#   FULLY PUBLIC — no internal mirror, no private base image, no prebuilt image.
#   Needs 1 NVIDIA GPU to RUN (task.toml: gpus = 1, anchor calibrated on H20).
#   Platform: linux/amd64 ONLY (the official vLLM wheel is manylinux x86_64). No GPU to BUILD.
#   Build context = this task directory (the Dockerfile lives in environment/).
#
# -----------------------------------------------------------------------------
# BUILD  (needs outbound network for PyPI)
#     cd <...>/wro-lora-punica-fused
#     docker build -f environment/Dockerfile -t fai/lh-lora-punica-fused:oss .
#
# RUN     docker run --rm --gpus all -it fai/lh-lora-punica-fused:oss
# SCORE   docker run --rm --gpus all -v "$PWD/tests:/tests:ro" fai/lh-lora-punica-fused:oss bash /tests/test.sh
#         # -> /logs/verifier/reward.json   (modes: candidate | noop | oracle | negative)
#
# 🔴 RE-CALIBRATE THE ANCHOR ON YOUR HARDWARE
#   reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; ref_speedup = 16.59 .
#   test.sh resolves it as: hardcoded fallback < /opt/verifier-correctness-manifest.json <
#   the MOUNTED tests/verifier-correctness-manifest.json (highest priority). Both of the latter
#   ship with this package and carry the same calibrated value, so the resolution is explicit.
#   Re-measure with the verifier's own mode dispatch (the reference patch is passed BY PATH and
#   is never baked into any layer):
#     docker run --rm --gpus all -v "$PWD/tests:/tests:ro" -v "$PWD:/patches:ro" \
#       -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_PATCH=/patches/oracle.patch \
#       fai/lh-lora-punica-fused:oss bash /tests/test.sh
#   Sanity: noop ≈ the no-op value; negative (KERNELBENCH_NEGATIVE_PATCH=...) MUST score 0.
#
# VERSION ANCHORS
#     CUDA 12.6 · Python 3.10 · vllm 0.10.1.1 (official public wheel) · torch 2.7.1 (cu126)
#   vLLM 0.10.1.1 HARD-REQUIRES torch==2.7.1, so torch is constraint-pinned and vLLM's own
#   resolver can never move it (its compiled _C*.so must match that ABI). The original image
#   installed torch 2.7.0 first and then `pip install vllm==0.10.1.1`, which upgraded torch to
#   2.7.1 — 2.7.1 is therefore the version the anchor was calibrated under.
#
# HOW THE GRADED WORKING TREE IS REPRODUCED  (sparse overlay + site-packages symlink)
#   vLLM itself comes from the OFFICIAL PUBLIC WHEEL (it carries prebuilt CUDA extensions, so no
#   GPU and no source build are needed). /app/repo is a SPARSE tree that contains ONLY the
#   task's scope files, in their CARVED (slow-but-correct) form; for each of them the copy inside
#   site-packages is replaced by a SYMLINK back to /app/repo. Consequences, all of them wanted:
#     * `import vllm` keeps resolving to the installed package, so every compiled extension and
#       relative RPATH keeps working (moving the package would break dlopen);
#     * the scope modules resolve THROUGH the symlink to /app/repo, which is what the verifier's
#       import-origin gate checks (it realpath()s the module file);
#     * edits under /app/repo take effect immediately, and `git -C /app/repo status --porcelain`
#       (the scope gate) sees exactly the scope files and nothing else.
#   The carved bytes were recovered from the shipped image / the authoring build and VERIFIED:
#   this package's oracle.patch applies cleanly to them, which proves they are the exact graded
#   starting point.
#   In addition, the build ASSERTS that the freshly installed wheel's pristine copies of the
#   scope files are byte-identical to the ones the carve was derived from (sha256 list in
#   environment/pristine_sha256.json) — so a re-published wheel can never silently change
#   the upstream the anchor was calibrated against.
#   environment/repo/ carries NO .git; the single baseline commit is created at build time.
#
#   oracle.patch / negative.patch are reviewer-only and excluded by the context-root .dockerignore.
#
# PROVENANCE of environment/repo/ (so anyone can re-derive it independently)
#   recovered from: <internal registry>/kernelbench/wro-lora-punica-fused-loop16:v1-ms1@sha256:da1e464dad443a5fecf7d125a78f67546d1eb03d82c3b361d5aa8ced5cbd02c8
#   method: `crane manifest` -> the small crane-appended layers -> `crane blob` + tar,
#           overlaid in manifest order; .git and any .wh..wh..opq marker dropped.
#   verification: `git apply --check -p1 oracle.patch` succeeds against these bytes,
#           and `git apply --check -p1 negative.patch` does too — i.e. this tree is the
#           exact carved baseline both reference patches were generated against.
# =============================================================================
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST="9.0"
ARG TORCH_VERSION=2.7.1
ARG VLLM_VERSION=0.10.1.1

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

# ---- torch first, then vLLM WITH all its declared dependencies under a constraint that pins
#      torch. Hand-listing vLLM's ~68 requirements is not viable; --no-deps would leave `import
#      vllm` broken. No `|| true` on purpose: a failed dependency install must fail the build. ----
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
        "torch==${TORCH_VERSION}"
RUN printf 'torch==%s\n' "${TORCH_VERSION}" > /tmp/vllm-constraints.txt \
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
        --constraint /tmp/vllm-constraints.txt "vllm==${VLLM_VERSION}" \
 && python -c "import torch; assert torch.__version__.startswith('2.7.1'), torch.__version__; print('torch pin held:', torch.__version__)" \
 && python -c "import importlib.metadata as m; v=m.version('vllm'); print('vllm', v); assert v=='0.10.1.1', v"

RUN mkdir -p /app/repo /app/stubs /app/.timer /app/workspace /logs/verifier /logs/agent /logs/artifacts
COPY environment/repo/ /app/repo/
COPY instruction.md /app/instruction.md
COPY environment/pristine_sha256.json /opt/pristine_sha256.json

# ---- 🔴 REPRODUCIBILITY GATE + the sparse overlay wiring, in one step.
#      1. (when applicable) assert the freshly installed wheel's PRISTINE scope files are
#         byte-identical to the ones this task's carve was derived from. If a future wheel
#         re-publish changed them, the build STOPS here instead of silently grading against a
#         different upstream than the anchor was calibrated on.
#      2. replace each scope file inside site-packages with a symlink to /app/repo, so edits
#         under /app/repo are what actually executes.
#      NOTE: this RUN is a BuildKit script heredoc (`RUN <<'SH'`). That form is REQUIRED here:
#      a nested `python - <<'PY'` inside a \-continued shell-form RUN makes the Dockerfile
#      parser end the instruction at `PY`, so the following `fi;` becomes an unknown
#      instruction and the build fails to parse. Inside a script heredoc, `#` comments are fine. ----
RUN <<'SH'
set -eu
SITE="$(python -c 'import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
echo "[build] site-packages = $SITE"
if [ -f /opt/pristine_sha256.json ]; then
  python - "$SITE" <<'PY'
import hashlib, json, os, sys
site = sys.argv[1]
want = json.load(open("/opt/pristine_sha256.json"))
bad = []
for rel, exp in want.items():
    p = os.path.join(site, rel)
    if not os.path.isfile(p):
        bad.append((rel, "MISSING", exp)); continue
    got = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print(f"  pristine {rel}: {'OK' if got == exp else 'MISMATCH'}")
    if got != exp:
        bad.append((rel, got, exp))
if bad:
    for rel, got, exp in bad:
        print(f"BUILD_ASSERT_FATAL: {rel}\n  expected {exp}\n  got      {got}")
    sys.exit("the installed vllm wheel does not match the upstream bytes this task's carve and "
             "anchor were derived from - re-calibrate before using this image")
print("OK: installed wheel matches the pristine bytes the carve was derived from")
PY
fi
for rel in "vllm/lora/ops/triton_ops/kernel_utils.py" "vllm/lora/ops/triton_ops/lora_shrink_op.py" "vllm/lora/ops/triton_ops/lora_expand_op.py"; do
  test -f "/app/repo/$rel" || { echo "BUILD_ASSERT_FATAL: carved scope file missing at /app/repo/$rel"; exit 1; }
  mkdir -p "$(dirname "$SITE/$rel")"
  rm -f "$SITE/$rel"
  ln -s "/app/repo/$rel" "$SITE/$rel"
  test -L "$SITE/$rel"
  echo "[build] wired $SITE/$rel -> /app/repo/$rel"
done
rm -f /opt/pristine_sha256.json
echo "[build] sparse overlay wired for all scope files"
SH

# ---- baseline commit; core.fileMode=false is LOAD-BEARING (a filemode skew makes
#      `git status --porcelain` report phantom out-of-scope paths in EVERY mode -> reward 0). ----
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
COPY environment/runtime/run_dev_bench.sh /app/workspace/run_dev_bench.sh
COPY environment/loop/ /opt/loop/
RUN chmod 0755 /app/entrypoint.sh /app/timer.sh /app/workspace/run_dev_bench.sh \
 && if [ -f /opt/loop/submit.sh ]; then chmod 0755 /opt/loop/submit.sh; fi \
 && chown -R root:root /app/entrypoint.sh /app/timer.sh /opt/loop \
 && if [ -f /opt/loop/score_engine.sh ]; then chmod 0700 /opt/loop/score_engine.sh; fi \
 && if [ -f /opt/loop/sanitize_feedback.py ]; then chmod 0700 /opt/loop/sanitize_feedback.py; fi \
 && if [ -d /opt/loop/private ]; then chmod -R go-rwx /opt/loop/private; fi

COPY tests/verifier-correctness-manifest.json /opt/verifier-correctness-manifest.json

# ---- 🔴 ANTI-LEAK ASSERTS ----
#      (The filename list below is an AUTHOR-SIDE ARTIFACT DENYLIST: names that only ever
#      exist in the authoring workspace. Asserting they are absent proves nothing leaked.)
RUN set -eu; rc=0; \
    hits="$(find / -xdev -type f \( -name 'oracle*.patch' -o -name 'negative*.patch' \
              -o -name 'scope_card.json' -o -name 'degradation_manifest.json' \
              -o -name 'pristine_sha256.json' \) 2>/dev/null | head -10)"; \
    if [ -n "$hits" ]; then echo "BUILD_ASSERT_FAIL: a reference/authoring file is baked:"; echo "$hits"; rc=1; fi; \
    for p in /opt/verifier /opt/strong_baseline /opt/negative /opt/ceiling /tests; do \
      if [ -e "$p" ]; then echo "BUILD_ASSERT_FAIL: frozen-surface path baked: $p"; rc=1; fi; \
    done; \
    test "$rc" = 0; \
    echo "[build] ANTI-LEAK ASSERTS PASS"

ENV PYTHONPATH=/app/repo:/app/stubs \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    LD_LIBRARY_PATH=/opt/kernelbench-venv/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64
WORKDIR /app/repo

# ---- build-time smoke.
#      HARD: the symlinks resolve into /app/repo, the git base is one clean commit, anchor sane.
#      SOFT: `import vllm` — on a GPU-less build host vLLM's platform probe can fail for reasons
#            unrelated to packaging; the authoritative check is at runtime.
RUN python - <<'PY'
import json, os, subprocess, sys
SCOPE = ('vllm/lora/ops/triton_ops/kernel_utils.py', 'vllm/lora/ops/triton_ops/lora_shrink_op.py', 'vllm/lora/ops/triton_ops/lora_expand_op.py')
site = None
try:
    import vllm
    site = os.path.dirname(os.path.dirname(vllm.__file__))
    print("vllm", vllm.__version__, "from", vllm.__file__)
except Exception as e:
    print("WARN: `import vllm` failed at BUILD time (often needs a GPU/driver):", repr(e))
    import sysconfig
    site = sysconfig.get_paths()["purelib"]
for rel in SCOPE:
    src = os.path.join("/app/repo", rel)
    dst = os.path.join(site, rel)
    assert os.path.isfile(src), f"carved scope file missing: {src}"
    assert os.path.islink(dst), f"site-packages entry is not a symlink: {dst}"
    assert os.path.realpath(dst) == os.path.realpath(src), (os.path.realpath(dst), src)
    print("wired:", dst, "->", os.readlink(dst))
m = json.load(open("/opt/verifier-correctness-manifest.json"))
print("baked anchor ref_speedup =", m.get("ref_speedup"))
assert isinstance(m.get("ref_speedup"), (int, float)) and m["ref_speedup"] > 1, m
n = subprocess.run(["git","-C","/app/repo","rev-list","--count","HEAD"],capture_output=True,text=True).stdout.strip()
assert n == "1", n
print("OK: sparse overlay, git base and anchor all in place")
PY

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["tail","-f","/dev/null"]
