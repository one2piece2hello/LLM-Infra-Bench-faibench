#!/usr/bin/env python3
"""wro-llamacpp-simd-q5k is a COMPILED-scope task: the workload (deterministic-block
correctness dot + random-block wall timing of q5_K_q8_K) is the C harness that
tests/test.sh compiles and runs against the freshly-built libggml-cpu.a. This stub exists only
to satisfy the loop16 packaging contract; test.sh does not import it."""
if __name__ == "__main__":
    print("WRO_NOTE workload is the embedded C harness in test.sh (compiled-scope task)")
