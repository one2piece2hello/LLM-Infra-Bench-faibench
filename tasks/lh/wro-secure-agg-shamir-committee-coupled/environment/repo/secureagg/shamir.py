"""Shamir secret sharing over a prime field (secure-aggregation backend).

A privacy-preserving aggregation server splits each client's masked update into ``n`` shares
(threshold ``t``), distributes them to a committee, and later reconstructs per-coordinate sums
from any ``t`` shares. Reconstruction sits on the server's hot path: many coordinates, over many
rounds, are recovered from the SAME committee of shareholders, i.e. from the same set of share
x-coordinates.

Observable contract (must hold exactly, all values are residues mod ``field.p``):
  * ``split(secret, n, t, rng)`` -> list of ``n`` shares ``(x_i, y_i)`` with distinct x_i in 1..n,
    such that any ``t`` of them reconstruct ``secret``. Uses ``rng`` (a ``random.Random``) to draw the
    ``t-1`` random polynomial coefficients (so results are deterministic given the rng).
  * ``reconstruct(shares)`` -> the secret at x=0 via Lagrange interpolation over the provided shares
    (needs at least the threshold count; extra shares are consistent). Raises ``ValueError`` on
    duplicate x or empty input.
  * ``reconstruct_many(share_lists)`` -> ``[reconstruct(s) for s in share_lists]`` where every list in
    ``share_lists`` uses the SAME set of x-coordinates (the committee); the per-x Lagrange-at-zero
    weights depend only on those x-coordinates.
"""
from __future__ import annotations

from .field import DEFAULT_PRIME, eval_poly, lagrange_at_zero


class Shamir:
    def __init__(self, prime=DEFAULT_PRIME):
        self.p = prime

    def split(self, secret, n, t, rng):
        if not (1 <= t <= n):
            raise ValueError("require 1 <= t <= n")
        secret %= self.p
        coeffs = [secret] + [rng.randrange(self.p) for _ in range(t - 1)]
        return [(x, eval_poly(coeffs, x, self.p)) for x in range(1, n + 1)]

    def reconstruct(self, shares):
        if not shares:
            raise ValueError("no shares")
        # Interpolate the sharing polynomial at x=0 over the provided shares.
        return lagrange_at_zero([(x, y) for x, y in shares], self.p)

    def reconstruct_many(self, share_lists):
        # Recover one value per share-list; every list in ``share_lists`` uses
        # the same committee of x-coordinates.
        return [self.reconstruct(sl) for sl in share_lists]
