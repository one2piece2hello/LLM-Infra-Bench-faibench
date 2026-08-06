"""Shared harness for the block-encoded integer dot-product task (CPU, pure Python).

Provides: candidate loader, an INDEPENDENT obviously-correct reference for
``blocked_dot`` (the ground truth — candidate/baseline/oracle are all scored against
this, never against each other), a small helper to pack signed codes into bytes,
deterministic block-corpus generators, and tolerant float comparators. Standard
library only (no tensor or numerics library) so the metric is a hardware-portable
instruction count.
"""

import hashlib
import importlib.util
import os
import random


def repo_dir():
    return os.environ.get("KB_REPO_DIR", "/app/repo")


def load_candidate():
    """Import the candidate module from the repo under test."""
    path = os.path.join(repo_dir(), "blocked_dot.py")
    spec = importlib.util.spec_from_file_location("candidate_blocked_dot", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "blocked_dot"):
        raise AttributeError(f"{path} does not define blocked_dot")
    return mod


def load_module(path):
    # deterministic module name derived from the path (no salted builtin hash()).
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location("kb_blockdot_mod_" + digest, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Test helper: pack 32 signed codes (each in [-8, 7]) into 16 bytes, low code of a
# byte first, high code second (the storage layout the contract defines).
# --------------------------------------------------------------------------- #
def pack_signed_codes(signed_codes):
    if len(signed_codes) != 32:
        raise ValueError("expected exactly 32 signed codes")
    packed = []
    for b in range(16):
        lo = signed_codes[2 * b] + 8
        hi = signed_codes[2 * b + 1] + 8
        if not (0 <= lo <= 15) or not (0 <= hi <= 15):
            raise ValueError("signed codes must lie in [-8, 7]")
        packed.append((hi << 4) | lo)
    return packed


# --------------------------------------------------------------------------- #
# Independent, obviously-correct reference (the ground truth).
# Per block: unpack both codes of each byte (low then high) with the fixed offset
# of 8, integer multiply-accumulate against the 32 companion codes, then apply the
# product of the two block scale factors once.
# --------------------------------------------------------------------------- #
def ref_blocked_dot(u_blocks, v_blocks):
    if len(u_blocks) != len(v_blocks):
        raise ValueError("length mismatch in reference")
    total = 0.0
    for (su, packed), (sv, codes) in zip(u_blocks, v_blocks):
        if len(packed) != 16:
            raise ValueError("bad packed block in reference")
        if len(codes) != 32:
            raise ValueError("bad code block in reference")
        acc = 0
        for b in range(16):
            byte = packed[b]
            lo = (byte & 0xF) - 8
            hi = ((byte >> 4) & 0xF) - 8
            acc += lo * codes[2 * b] + hi * codes[2 * b + 1]
        total += (su * sv) * acc
    return total


# --------------------------------------------------------------------------- #
# Tolerant comparators (the result is a single real number).
# --------------------------------------------------------------------------- #
def _close(a, b, rtol=1e-9, atol=1e-12):
    a = float(a)
    b = float(b)
    return abs(a - b) <= (atol + rtol * abs(b))


def assert_scalar_close(out, ref, msg=""):
    if isinstance(out, (list, tuple)):
        raise AssertionError(f"expected a scalar float result, got a sequence {msg}")
    if not _close(out, ref):
        raise AssertionError(f"value {out!r} != expected {ref!r} {msg}")


# --------------------------------------------------------------------------- #
# Deterministic block-corpus generator.
# Scale factors are dyadic (k / 256) so they are exactly representable in float;
# combined with integer lane products the whole accumulation is exact, so the
# result is identical regardless of accumulation order (naive vs per-block).
# --------------------------------------------------------------------------- #
def make_bench_corpus(num_vectors, blocks_per_vector, seed=12345):
    """A batch of paired block-encoded vectors. Each vector is (u_blocks, v_blocks)
    with ``blocks_per_vector`` blocks of 32 lanes. Aggregating the block dot over
    many blocks is the regime where the naive per-lane scale multiply and one-code-
    at-a-time unpack dominate."""
    rng = random.Random(seed)
    vectors = []
    for _ in range(num_vectors):
        u_blocks = []
        v_blocks = []
        for _ in range(blocks_per_vector):
            su = rng.randint(1, 255) / 256.0
            packed = [rng.randint(0, 255) for _ in range(16)]
            u_blocks.append((su, packed))
            sv = rng.randint(1, 255) / 256.0
            codes = [rng.randint(-127, 127) for _ in range(32)]
            v_blocks.append((sv, codes))
        vectors.append((u_blocks, v_blocks))
    return vectors
