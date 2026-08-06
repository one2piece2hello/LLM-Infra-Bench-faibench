# syntax=docker/dockerfile:1
# =============================================================================
# -loop16 overlay for e2e-g2-embed-compress-golf.
# FROM the open-source single-shot base image; add /opt/loop/ (submit + score_engine + sanitize +
# private dev proxy + baked pristine starter). Pure additive layer — NO dependency change, NO change
# to /tests, compute_reward.py, or ANY graded code path. The graded surface (tests/, held-out split,
# frozen manifest with the calibrated anchor + the qrels answer-key) is uploaded fresh at scoring and
# is byte-identical to the single-shot base. Preferably applied with `crane append` (~8s); this
# Dockerfile is the docker-build fallback and the authoritative description of the layer.
#
# BUILD (context = THIS directory, environment/loop_overlay/):
#     cd <...>/embed-compress-golf/task
#     docker build -f environment/Dockerfile             -t fai/e2e-g2-embed-compress-golf:oss .
#     docker build -f environment/loop_overlay/Dockerfile -t fai/e2e-g2-embed-compress-golf:oss-loop16 environment/loop_overlay
#
# 🔴 Structural red-line (modes MUST also be baked into the crane tar header, since `crane append`
# runs no RUN): submit.sh 0755 root; score_engine.sh + everything under /opt/loop/private/ +
# /opt/loop/baseline_submission/ is 0700/0600 root (unreadable by the dropped solver uid).
#
# 🔴 LEAK-FREE: the private dev proxy runs the task's OWN compute_reward.py on the PUBLIC, DISJOINT
# dev split baked at /data/retrieval/dev_* — it bakes NO held-out corpus/queries/qrels answer-key and
# NO calibrated anchor. The dev manifest carries ref_speedup=null + strong_baseline_ndcg=0, so the
# per-round signal is a RAW dev nDCG@10, never normalized against the graded anchor.
ARG BASE_IMAGE=fai/e2e-g2-embed-compress-golf:oss
FROM ${BASE_IMAGE}
SHELL ["/bin/bash","-c"]

# assert the base has NO /opt/loop (crane append is a layer union; overlaying an existing
# /opt/loop would let stale content show through).
RUN test ! -e /opt/loop || { echo "BUILD_ASSERT_FAIL: base already has /opt/loop"; exit 1; }

# bake the /opt/loop tree
COPY opt_loop/ /opt/loop/

# swap in the loop16 combined instruction (the base ships /app/instruction.md describing single-shot)
COPY instruction_loop16.md /app/instruction.md

# ownership + modes: root-owned; the solver runs unprivileged, so 0700/0600 are structurally
# unreadable to it. (Exact chmod/chown block from the a4 loop16 overlay.)
RUN chown -R root:root /opt/loop \
 && chmod 0755 /opt/loop \
 && chmod 0755 /opt/loop/submit.sh \
 && chmod 0700 /opt/loop/score_engine.sh \
 && chmod -R go-rwx /opt/loop/private /opt/loop/baseline_submission \
 && chmod 0700 /opt/loop/private /opt/loop/baseline_submission \
 && find /opt/loop/private /opt/loop/baseline_submission -type f -exec chmod 0600 {} \; \
 && find /opt/loop/private /opt/loop/baseline_submission -type d -exec chmod 0700 {} \; \
 && chmod 0644 /app/instruction.md

# /logs/loop is created world-writable at runtime by submit.sh; nothing to bake.

# 🔴 build-time assert: the overlay must carry NO calibrated anchor VALUE and NO held-out data.
# (Prose that NAMES ref_speedup/strong_baseline_ndcg to explain their ABSENCE is fine; a VALUE is
# not.) Fails the build if a regression re-introduces either.
RUN set -eu; rc=0; \
    if grep -rIEq "1\.4290238|0\.6561377|0\.459151" /opt/loop; then \
      echo "BUILD_ASSERT_FAIL: calibrated anchor VALUE present in /opt/loop"; rc=1; fi; \
    hits="$(find /opt/loop -type f \( -name 'heldout_*' -o -name 'verifier-correctness-manifest.json' \
              -o -name '*ceiling_ref*' \) 2>/dev/null | head -10)"; \
    if [ -n "$hits" ]; then echo "BUILD_ASSERT_FAIL: held-out/anchor/reference file baked into /opt/loop:"; echo "$hits"; rc=1; fi; \
    test -f /opt/loop/submit.sh && test -x /opt/loop/submit.sh || { echo "submit.sh missing/not exec"; rc=1; }; \
    test -f /opt/loop/private/tests/compute_reward.py || { echo "dev-proxy verifier missing"; rc=1; }; \
    test "$rc" = 0 && echo "[build] loop16 overlay asserts PASS"
