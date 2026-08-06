# Frozen baseline (pre-PR 982748aa parent 1b741cab): the Hopper grouped-GEMM
# persistent kernel did NOT exist.  This stub stands in for the absent product
# file so the verifier records hidden_missing_symbol (reward 0) in noop mode.
_KERNELBENCH_BASELINE_STUB = True

def run(*args, **kwargs):
    raise NotImplementedError(
        "grouped GEMM persistent kernel absent in pre-PR baseline (PR #3091 adds it)"
    )
