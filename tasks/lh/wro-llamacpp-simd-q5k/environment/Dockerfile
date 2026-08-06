# syntax=docker/dockerfile:1
# =============================================================================
# wro-llamacpp-simd-q5k — OPEN-SOURCE, self-contained task image
#
#   Task: restore the AVX2/AVX SIMD implementation of the ggml Q5_K x Q8_K quantised dot product; the shipped tree has a scalar reference in its place.
#         Editable scope (the ONLY files the scope gate allows to change):
##     - ggml/src/ggml-cpu/arch/x86/quants.c
#
#   FULLY PUBLIC — no internal mirror, no private base image, no prebuilt image.
#   CPU ONLY — task.toml declares gpus = 0. Platform: linux/amd64 ONLY (the scope is x86 AVX2
#   intrinsics and the harness links with -mavx2 -mfma -mf16c).
#   Build context = this task directory (the Dockerfile lives in environment/).
#
# -----------------------------------------------------------------------------
# BUILD  (needs outbound network for PyPI only — the working tree ships in this package)
#     cd <...>/wro-llamacpp-simd-q5k
#     docker build -f environment/Dockerfile -t fai/lh-llamacpp-simd-q5k:oss .
#
# RUN     docker run --rm -it fai/lh-llamacpp-simd-q5k:oss
# SCORE   docker run --rm -v "$PWD/tests:/tests:ro" fai/lh-llamacpp-simd-q5k:oss bash /tests/test.sh
#         # -> /logs/verifier/reward.json
#         # modes via -e KERNELBENCH_VERIFY_MODE=: candidate | noop | oracle | negative
#
# 🔴 REWARD + RE-CALIBRATION
#   reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; ref_speedup = 11.5, read from
#   /opt/verifier-correctness-manifest.json (baked below from this package's
#   tests/verifier-correctness-manifest.json) and calibrated on the authoring CPU lane
#   (Intel Sapphire Rapids). ⚠️ This anchor is CPU-MICROARCHITECTURE sensitive: it is the
#   ratio of a scalar reference to hand-written AVX2 intrinsics, which differs between
#   CPU generations. The compiled ISA itself is FIXED, not host-tuned: tests/test.sh
#   configures ggml with -DGGML_AVX2=ON -DGGML_NATIVE=OFF, so the build does not follow
#   -march=native and is reproducible on any AVX2-capable x86-64 host.
#   Re-measure on your CPU with the verifier's own mode dispatch (the reference patch is
#   passed BY PATH and is never baked into any layer):
#     docker run --rm -v "$PWD/tests:/tests:ro" \
#       -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_PATCH=/tests/oracle.patch \
#       fai/lh-llamacpp-simd-q5k:oss bash /tests/test.sh
#   (this task ships its patches inside tests/, which is mounted at scoring time)
#   Sanity: noop -> speedup ~1.0 -> gated to 0; negative -> 0.
#
# CORRECTNESS SIGNATURE: the harness asserts a deterministic block dot == 2304.0 and calls
#   ggml_cpu_init() for the f16 lookup table, so a wrong SIMD path fails before timing.
#
# HOW THE GRADED WORKING TREE IS REPRODUCED
#   /app/repo is the llama.cpp / ggml source tree with the target implementation CARVED OUT (a slow-but-correct or
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
#   recovered from: <internal registry>/kernelbench/wro-llamacpp-simd-q5k-loop16:v1-ms1@sha256:2721d203d2ed366feaad473cdd3d5dc50b55d7430607d014587101cbe89ff468
#   method: `crane manifest` -> the small crane-appended layers -> `crane blob` + tar,
#           overlaid in manifest order; .git and any .wh..wh..opq marker dropped.
#   verification: `git apply --check -p1 oracle.patch` succeeds against these bytes,
#           and `git apply --check -p1 negative.patch` does too — i.e. this tree is the
#           exact carved baseline both reference patches were generated against.
# =============================================================================
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- the graded artefact is COMPILED C: tests/test.sh runs cmake on /app/repo/ggml, builds the
#      ggml-cpu target and links tests/harness.c against it. gcc / cmake / make are therefore part
#      of the graded contract, not conveniences. python3 is only used for JSON plumbing. ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ cmake make pkg-config \
        python3 python3-venv python3-pip \
        git ca-certificates curl procps coreutils tar gzip vim less file jq \
    && rm -rf /var/lib/apt/lists/* \
    && gcc --version | head -1 && cmake --version | head -1

# ---- a venv at this path exists for consistency with the rest of the bench (test.sh for other
#      tasks pins it); nothing here needs python packages. ----
ENV VENV=/opt/kernelbench-venv
RUN python3 -m venv "$VENV"
ENV PATH=$VENV/bin:$PATH

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
COPY environment/loop/ /opt/loop/
RUN if [ -f /opt/loop/submit.sh ]; then chmod 0755 /opt/loop/submit.sh; fi \
 && chown -R root:root /opt/loop \
 && if [ -f /opt/loop/score_engine.sh ]; then chmod 0700 /opt/loop/score_engine.sh; fi \
 && if [ -f /opt/loop/sanitize_feedback.py ]; then chmod 0700 /opt/loop/sanitize_feedback.py; fi \
 && if [ -d /opt/loop/private ]; then chmod -R go-rwx /opt/loop/private; fi

# ---- the FROZEN anchor. tests/test.sh reads ref_speedup from this exact path and falls back
#      to 1.0 (every reward 0) if it is missing, so it must be baked. Only the constant. ----
COPY tests/verifier-correctness-manifest.json /opt/verifier-correctness-manifest.json

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

ENV PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
WORKDIR /app/repo

RUN set -eu; \
    test -f /app/repo/ggml/src/ggml-cpu/arch/x86/quants.c || { echo "BUILD_ASSERT_FATAL: scope file missing"; exit 1; }; \
    test -f /app/repo/ggml/CMakeLists.txt || { echo "BUILD_ASSERT_FATAL: ggml CMakeLists.txt missing - cmake configure would fail at scoring"; exit 1; }; \
    test -d /app/repo/ggml/include || { echo "BUILD_ASSERT_FATAL: ggml/include missing - the harness link would fail"; exit 1; }; \
    n="$(grep -cE '_mm256_|__m256' /app/repo/ggml/src/ggml-cpu/arch/x86/quants.c || echo 0)"; \
    echo "[build] AVX2 intrinsics in the carved scope file: $n occurrences (the Q5_K path is the carved one; other quant types legitimately keep theirs)"; \
    echo "[build] file count under /app/repo: $(find /app/repo -type f -not -path '*/.git/*' | wc -l)"; \
    python3 -c "import json;m=json.load(open('/opt/verifier-correctness-manifest.json'));print('baked anchor ref_speedup =', m['ref_speedup']);assert m['ref_speedup']>1"; \
    test "$(git -C /app/repo rev-list --count HEAD)" = "1"; \
    echo "OK: scope file, ggml build inputs, anchor and git base all in place"

# ---- pre-warm the cmake configure so the FIRST scored run does not pay for it (and so a broken
#      ggml tree fails HERE instead of as cmake_configure_failed at scoring). Same flags
#      tests/test.sh uses. The build dir is left in place; test.sh reuses it when present. ----
RUN set -eu; \
    mkdir -p /app/repo/ggml/tests /app/repo/ggml/examples; \
    [ -f /app/repo/ggml/tests/CMakeLists.txt ]    || echo '# empty' > /app/repo/ggml/tests/CMakeLists.txt; \
    [ -f /app/repo/ggml/examples/CMakeLists.txt ] || echo '# empty' > /app/repo/ggml/examples/CMakeLists.txt; \
    [ -f /app/repo/ggml/ggml.pc.in ] || printf 'Name: ggml\nDescription: ggml\nVersion: @GGML_INSTALL_VERSION@\nLibs: -lggml\n' > /app/repo/ggml/ggml.pc.in; \
    cmake -S /app/repo/ggml -B /tmp/ggml_warm -DGGML_STANDALONE=OFF -DGGML_BUILD_TESTS=OFF \
      -DGGML_BUILD_EXAMPLES=OFF -DGGML_AVX2=ON -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF \
      -DGGML_OPENMP=OFF -DGGML_BACKEND_DL=OFF -DGGML_CCACHE=OFF -DCMAKE_BUILD_TYPE=Release > /tmp/cmake_warm.log 2>&1 \
      || { echo "BUILD_ASSERT_FATAL: cmake configure of the vendored ggml tree failed"; tail -30 /tmp/cmake_warm.log; exit 1; }; \
    cmake --build /tmp/ggml_warm --target ggml-cpu -j"$(nproc)" > /tmp/cmake_build.log 2>&1 \
      || { echo "BUILD_ASSERT_FATAL: the carved ggml tree does not compile"; tail -40 /tmp/cmake_build.log; exit 1; }; \
    echo "[build] ggml-cpu compiles from the carved tree ($(find /tmp/ggml_warm -name 'libggml*.a' | wc -l) static libs)"; \
    rm -rf /tmp/ggml_warm /tmp/cmake_warm.log /tmp/cmake_build.log; \
    git -C /app/repo add -A && git -C /app/repo commit -q --amend --no-edit \
    && test "$(git -C /app/repo rev-list --count HEAD)" = "1" \
    && test -z "$(git -C /app/repo status --porcelain)" \
    && echo "[build] cmake scaffolding folded into the baseline commit; tree still clean"

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["tail","-f","/dev/null"]
