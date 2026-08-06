"""Prime-field arithmetic for the secure-aggregation subsystem.

This is a fixed building block (out of the editable scope): modular arithmetic over a prime
field ``p`` plus Lagrange interpolation at x=0, used by Shamir secret sharing. Everything in
the subsystem is defined in terms of residues in ``[0, p)``.

The default modulus is a fixed prime; callers may pass their own prime.
"""
from __future__ import annotations

# A fixed prime (2**61 - 1, a Mersenne prime) large enough for the masks used here.
DEFAULT_PRIME = (1 << 61) - 1


def inv_mod(a, p):
    """Modular inverse of ``a`` mod prime ``p`` (Fermat). Raises for a % p == 0."""
    a %= p
    if a == 0:
        raise ZeroDivisionError("no inverse for 0")
    return pow(a, p - 2, p)


def eval_poly(coeffs, x, p):
    """Evaluate polynomial (coeffs[0] + coeffs[1] x + ...) at x mod p (Horner)."""
    acc = 0
    for c in reversed(coeffs):
        acc = (acc * x + c) % p
    return acc


def lagrange_at_zero(points, p):
    """Interpolate the value at x=0 of the unique polynomial through ``points``.

    ``points`` = list of (x_i, y_i) with distinct non-zero x_i, all residues mod p.
    Returns f(0) mod p. Raises ValueError on duplicate x.
    """
    xs = [x % p for x, _ in points]
    if len(set(xs)) != len(xs):
        raise ValueError("duplicate share x-coordinates")
    total = 0
    k = len(points)
    for i in range(k):
        xi, yi = points[i][0] % p, points[i][1] % p
        num = 1
        den = 1
        for j in range(k):
            if j == i:
                continue
            xj = points[j][0] % p
            num = (num * (-xj)) % p
            den = (den * (xi - xj)) % p
        li0 = (num * inv_mod(den, p)) % p
        total = (total + yi * li0) % p
    return total % p
