#!/usr/bin/env bash
# entrypoint.sh — vLLM serving E2E container entrypoint.
# Model weights are BAKED at /app/model at build time (no shared filesystem at run time).
# A pristine vLLM snapshot (/app/.vllm-baseline.tar) is baked at build so the verifier
# can restore un-modified vLLM before measuring the strong baseline (1.0 anchor) even
# after the agent edits vLLM source. Start the harness-owned timer daemon, then exec.
set -u

# Fallback: snapshot pristine vLLM if the build step didn't (defensive).
if [ ! -f /app/.vllm-baseline.tar ]; then
    SITE_PKG=$(python3 -c "import vllm,os; print(os.path.dirname(vllm.__path__[0]))" 2>/dev/null)
    if [ -n "$SITE_PKG" ] && [ -d "$SITE_PKG/vllm" ]; then
        tar cf /app/.vllm-baseline.tar -C "$SITE_PKG" vllm 2>/dev/null
        echo "$SITE_PKG" > /app/.vllm-site-packages-path
    fi
fi

FRONTIER_TIMER_BOOTSTRAP=1 env -u BASH_ENV -u ENV /app/timer.sh &
exec "$@"
