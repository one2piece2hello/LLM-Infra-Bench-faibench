# syntax=docker/dockerfile:1
# =============================================================================
# -loop16 overlay for e2e-h3-eval-harness-throughput-quality.
# FROM the open-source single-shot base image; add /opt/loop/ (submit + score_engine + sanitize +
# private dev proxy + baked pristine starter). Pure additive layer — NO dependency change, NO change
# to /tests, compute_reward.py, or ANY graded code path. The graded surface (tests/, held-out sample
# set, frozen manifest with the calibrated anchor, and the strong-baseline ABBA partner under
# tests/oracles/) is uploaded fresh at scoring and is byte-identical to the single-shot base.
# Preferably applied with `crane append` (~8s); this Dockerfile is the docker-build fallback.
#
# BUILD (context = THIS directory, environment/loop_overlay/):
#     cd <...>/eval-scoring-throughput/task
#     docker build -f environment/Dockerfile             -t fai/e2e-h3-eval-scoring-throughput:oss .
#     docker build -f environment/loop_overlay/Dockerfile -t fai/e2e-h3-eval-scoring-throughput:oss-loop16 environment/loop_overlay
#
# 🔴 Structural red-line (modes MUST also be baked into the crane tar header, since `crane append`
# runs no RUN): submit.sh 0755 root; score_engine.sh + everything under /opt/loop/private/ +
# /opt/loop/baseline_submission/ is 0700/0600 root (unreadable by the dropped solver uid).
#
# 🔴 LEAK-FREE: the private dev proxy runs the candidate on the PUBLIC, DISJOINT dev split baked at
# /data/eval_harness/dev_samples.jsonl, enforces the REAL consistency + anti-cache gates (recomputed
# from the PUBLICLY-disclosed reference semantics), and times ABBA against the PUBLIC naive template
# (dev_baseline_naive.py = the shipped starter). It bakes NO strong-baseline reference, NO held-out
# sample set and NO calibrated anchor (ref_speedup / strong_baseline_time_sec / the workload sha256).
ARG BASE_IMAGE=fai/e2e-h3-eval-scoring-throughput:oss
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

# 🔴 build-time assert: the overlay must carry NO calibrated anchor VALUE, NO held-out set, NO
# strong-baseline / ceiling reference. (Prose that NAMES ref_speedup to explain its ABSENCE is fine;
# a VALUE is not.) Fails the build if a regression re-introduces any.
RUN set -eu; rc=0; \
    if grep -rIEq "2\.24419|0\.174827|b387c1747" /opt/loop; then \
      echo "BUILD_ASSERT_FAIL: calibrated anchor VALUE / held-out digest present in /opt/loop"; rc=1; fi; \
    hits="$(find /opt/loop -type f \( -name 'heldout_*' -o -name 'strong_baseline*' \
              -o -name 'ceiling_*' -o -name 'verifier-correctness-manifest.json' \) 2>/dev/null | head -10)"; \
    if [ -n "$hits" ]; then echo "BUILD_ASSERT_FAIL: held-out/anchor/reference file baked into /opt/loop:"; echo "$hits"; rc=1; fi; \
    test -f /opt/loop/submit.sh && test -x /opt/loop/submit.sh || { echo "submit.sh missing/not exec"; rc=1; }; \
    test -f /opt/loop/private/tests/dev_eval.py || { echo "dev-proxy dev_eval.py missing"; rc=1; }; \
    test "$rc" = 0 && echo "[build] loop16 overlay asserts PASS"
