"""Stub for flash_attn_2_cuda C-extension (absent). Lets flash_attn/__init__ -> flash_attn_interface
import succeed so flash_attn.cute (pure CuTe-DSL, no C ext) is reachable. No fwd path uses these."""
def __getattr__(name):
    def _missing(*a, **k):
        raise RuntimeError(f"flash_attn_2_cuda.{name} is stubbed (CuTe path only)")
    return _missing
