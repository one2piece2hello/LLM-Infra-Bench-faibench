#!/usr/bin/env bash
# REVIEWER-ONLY negative launcher: brings up the FAST-BUT-WRONG fake server so validation can
# prove the perf pre-gates bite. Never baked, never run at scoring.
set -euo pipefail
export PATH=/opt/kernelbench-venv/bin:$PATH
exec python3 /app/submission/negative_fake_server.py
