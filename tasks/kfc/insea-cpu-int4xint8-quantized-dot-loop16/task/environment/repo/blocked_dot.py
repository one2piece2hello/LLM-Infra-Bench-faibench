"""Dot product of two block-encoded integer vectors, each block carrying a scale.

Public entry point:
    ``blocked_dot(u_blocks, v_blocks) -> float``

Two equal-length logical vectors ``U`` and ``V`` are each stored as a sequence of
blocks of 32 lanes. Block ``i`` of ``U`` pairs with block ``i`` of ``V`` (the same
32 lanes).

A block of ``U`` (a "packed block") is a pair ``(su, packed)``:

* ``su``     -- a real number, the scale factor of the block.
* ``packed`` -- 16 bytes (ints in ``[0, 255]``). The 32 signed codes are read two
  per byte: for byte index ``b`` in ``0..15`` the code at lane ``2*b`` is
  ``(packed[b] & 0xF) - 8`` and the code at lane ``2*b + 1`` is
  ``((packed[b] >> 4) & 0xF) - 8``. Every code is therefore an integer in ``[-8, 7]``.

A block of ``V`` (a "code block") is a pair ``(sv, codes)``:

* ``sv``    -- a real number, the scale factor of the block.
* ``codes`` -- 32 small signed integers (each in ``[-127, 127]``).

The result is a single real number::

    result = sum over blocks of ( su * sv * sum_{j in 0..31} code_U[j] * codes[j] )

where ``code_U[j]`` is the unpacked signed code of ``U`` at lane ``j`` in that block.

Contract / errors
-----------------
* ``u_blocks`` and ``v_blocks`` are equal-length sequences of blocks; zero blocks
  -> the result is ``0.0``.
* ``ValueError`` if the two sequences differ in length, if any packed byte list is
  not length 16, or if any code list is not length 32.

The scale factors are exactly representable and the per-lane products are integers,
so the result is well defined independently of the accumulation order.

Why the current implementation is slow
--------------------------------------
For every one of the 32 lanes it recomputes which byte holds that lane's code,
extracts that single code, multiplies the lane's integer product by the block's
scale factor separately, and accumulates in the scaled domain. The per-lane scale
multiply and the one-code-at-a-time unpack are redundant: the scale factor is
constant across a block's 32 lanes, and each byte already carries two adjacent
codes. Return identical values with fewer total operations -- for example
accumulate the integer lane products of a block first and apply the block's scale
factor once, and read both codes of a byte together. Do the unpack and the
accumulation yourself; do not delegate to an array/numerics library dot or
matrix-multiply helper.
"""


def _validate(u_blocks, v_blocks):
    if len(u_blocks) != len(v_blocks):
        raise ValueError("u_blocks and v_blocks must have the same number of blocks")
    for _su, packed in u_blocks:
        if len(packed) != 16:
            raise ValueError("each packed block must carry exactly 16 bytes")
    for _sv, codes in v_blocks:
        if len(codes) != 32:
            raise ValueError("each code block must carry exactly 32 codes")


def blocked_dot(u_blocks, v_blocks):
    _validate(u_blocks, v_blocks)
    result = 0.0
    for (su, packed), (sv, codes) in zip(u_blocks, v_blocks):
        combined = su * sv
        # one lane at a time: locate the byte holding this lane's code, pull out
        # that single code, scale the lane's integer product by the block factor,
        # and accumulate in the scaled domain (the redundant work this task removes).
        for j in range(32):
            byte = packed[j >> 1]
            if j & 1:
                code = ((byte >> 4) & 0xF) - 8
            else:
                code = (byte & 0xF) - 8
            result = result + (code * codes[j]) * combined
    return result
