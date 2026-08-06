"""Stable identity for a structured request signature.

Public entry point:
    ``identity_key(signature: dict) -> str``

A downstream stage keeps a table of already-processed *identities* and skips the
expensive work for a signature whose identity it has already seen. So the identity
must satisfy two properties:

* two signatures that are **equivalent** (they describe the same request, differing
  only in incidental spelling) should map to the **same** identity, and
* two signatures that are **genuinely different** must map to **different**
  identities -- a collision between two different signatures would let the wrong
  prior result be reused, which is never acceptable.

Signature shape
---------------
A signature is a mapping with:

* ``"op"``: the operation name (a string).
* ``"operands"``: a list of operands; each operand is a mapping with ``"shape"``
  (a list of integers) and ``"dtype"`` (a string).
* ``"flags"`` (optional): a mapping of configuration flag name -> value.
* ``"meta"`` (optional): a mapping of incidental annotations.

This starting implementation is **correct but conservative**: it derives the
identity from the signature almost verbatim (only the field order of a mapping is
neutralized). It never merges two different signatures, but signatures that are
merely spelled differently are treated as separate identities, so the downstream
table holds more entries than there are truly distinct requests and the expensive
work is repeated once per spelling. Reducing the number of distinct identities --
without ever merging two genuinely different signatures -- is the goal.
"""

import json


def _fnv1a_64(text):
    """Deterministic 64-bit FNV-1a digest of ``text`` (fixed, not process-salted)."""
    h = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return format(h, "016x")


def identity_key(signature):
    """Return a stable string identity for ``signature``.

    Conservative baseline: serialize the signature with a fixed field order and
    digest it. Only mapping-field order is neutralized; every other incidental
    difference (operand order, alternate spellings, redundant defaults, annotation
    fields) still yields a new identity.
    """
    text = json.dumps(signature, sort_keys=True, separators=(",", ":"), default=repr)
    return _fnv1a_64(text)
