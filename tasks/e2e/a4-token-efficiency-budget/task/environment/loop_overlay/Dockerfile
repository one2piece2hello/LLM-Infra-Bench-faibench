# syntax=docker/dockerfile:1
# -loop16 overlay for e2e-a4-token-efficiency-budget.
# FROM the de-leaked single-shot base image; add /opt/loop/ (submit + score_engine + sanitize +
# private dev-training proxy + baked pristine starter) and swap /app/instruction.md for the
# loop16 combined statement. Pure additive layer — NO dependency change, NO change to /tests,
# compute_reward.py, or ANY graded code path. The overlay is preferably applied with
# `crane append` (~8s, zero GPU); this Dockerfile is the docker-build fallback and the
# authoritative description of what the layer contains.
#
# 🔴 Structural red-line enforcement (modes MUST also be baked into the crane tar header, since
# `crane append` runs no RUN): submit.sh 0755 root; score_engine.sh + sanitize_feedback.py +
# everything under /opt/loop/private/ + /opt/loop/baseline_submission/ is 0700/0600 root
# (unreadable by the solver's dropped uid). The private dir carries NO held-out corpus, NO
# reference recipe and NO calibrated anchor — only public dev-proxy constants.
ARG BASE_IMAGE=<internal registry>/kernelbench/e2e-a4-token-efficiency-budget:v3-deleak@sha256:28bbf1300c4cd9123faf5c66f4fa74c4347437f622d8094e0431c14121af3145
FROM ${BASE_IMAGE}
SHELL ["/bin/bash","-c"]

# assert the base has NO /opt/loop (crane append is a layer union; overlaying an existing
# /opt/loop would let stale content show through).
RUN test ! -e /opt/loop || { echo "BUILD_ASSERT_FAIL: base already has /opt/loop"; exit 1; }

# bake the /opt/loop tree
COPY opt_loop/ /opt/loop/

# swap in the loop16 combined instruction (the runner ships /app/instruction.md)
COPY instruction_loop16.md /app/instruction.md

# ownership + modes: root-owned; the solver runs unprivileged, so 0700/0600 are structurally
# unreadable to it.
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

# 🔴 build-time assert: the overlay must carry NO calibrated anchor VALUE and NO old-protocol
# token. (Prose that NAMES baseline_bpb/ref_speedup to explain their ABSENCE is fine; a VALUE is
# not.) Fails the build if a regression re-introduces either.
RUN set -eu; rc=0; \
    if grep -rIEq "1\.70214|1\.7034|1\.20452|1\.41311|1\.4094|1\.4131" /opt/loop; then \
      echo "BUILD_ASSERT_FAIL: calibrated anchor VALUE present in /opt/loop"; rc=1; fi; \
    if grep -rIEq "3-16|3 consecutive|min_submissions[ =]*3|MIN_SUB(MISSIONS)?=3" /opt/loop /app/instruction.md; then \
      echo "BUILD_ASSERT_FAIL: old-protocol token present"; rc=1; fi; \
    test -f /opt/loop/submit.sh && test -x /opt/loop/submit.sh || { echo "submit.sh missing/not exec"; rc=1; }; \
    test "$rc" = 0 && echo "[build] loop16 overlay asserts PASS"
