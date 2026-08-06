"""Stable identity for a build-spec.

Public entry point:
    ``identity_key(spec: dict) -> str``

A downstream stage keeps a table of already-produced *identities* and skips the
expensive step for a build-spec whose identity it has already seen. So the identity
must satisfy two properties:

* two specs that are **equivalent** (they describe the same build, differing only in
  incidental spelling) should map to the **same** identity, and
* two specs that are **genuinely different** must map to **different** identities -- a
  collision between two different specs would let the wrong prior artifact be reused,
  which is never acceptable.

Build-spec shape
----------------
A spec is a mapping with:

* ``"source"``: the source text (a string).
* ``"target"``: the target-profile tag (a string) the artifact is produced for.
* ``"options"`` (optional): a list of build-option tokens (an unordered set).
* ``"toolchain"`` (optional): the toolchain-version tag (a string).
* ``"variants"`` (optional): a list of requested variant names (an unordered set).
* ``"build"`` (optional): a mapping of incidental build annotations.

This starting implementation is **correct but conservative**: it derives the identity
from the spec almost verbatim (only the field order of a mapping is neutralized). It
never merges two different specs, but specs that are merely spelled differently are
treated as separate identities, so the downstream table holds more entries than there
are truly distinct builds and the expensive step is repeated once per spelling.
Reducing the number of distinct identities -- without ever merging two genuinely
different specs -- is the goal.
"""

import json


def _fnv1a_64(text):
    """Deterministic 64-bit FNV-1a digest of ``text`` (fixed, not process-salted)."""
    h = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return format(h, "016x")


def identity_key(spec):
    """Return a stable string identity for ``spec``.

    Conservative baseline: serialize the spec with a fixed field order and digest it.
    Only mapping-field order is neutralized; every other incidental difference (source
    comments/whitespace, option order, redundant no-op options, annotation fields)
    still yields a new identity.
    """
    text = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=repr)
    return _fnv1a_64(text)
