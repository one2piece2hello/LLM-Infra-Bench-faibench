#!/usr/bin/env bash
set -euo pipefail
export PATH=/opt/kernelbench-venv/bin:/root/.cargo/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
KERNELBENCH_VERIFY_MODE="${KERNELBENCH_VERIFY_MODE:-candidate}" bash /tests/test.sh
