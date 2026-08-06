"""Masked secure aggregation over a fixed shareholder committee (secure-aggregation backend).

Clients additively mask their length-``D`` update vectors; the server holds, per coordinate, a
list of Shamir shares from the SAME committee of shareholders (same x-coordinates for every
coordinate) and reconstructs the aggregated (summed) value per coordinate to recover the
plaintext aggregate. Aggregation runs on the server's hot path, once per aggregation round.

Observable contract (residues mod ``shamir.p``):
  * ``aggregate(coord_shares)`` -> list of reconstructed values, one per coordinate.
    ``coord_shares`` is a list (length ``D``) of share-lists; every share-list uses the same set of
    x-coordinates (the committee). Equivalent to ``[shamir.reconstruct(s) for s in coord_shares]``.
  * ``aggregate_rounds(rounds)`` -> ``[aggregate(r) for r in rounds]`` for several aggregation rounds
    that all reuse the same committee x-coordinates.
  * ``committee_of(coord_shares)`` -> the sorted tuple of x-coordinates used (the committee id).

Only the three methods listed above are observable, and only as residues mod ``shamir.p``. The
committee x-coordinates are identical for every coordinate of a round and for every round passed
to ``aggregate_rounds``. How this module and ``shamir`` divide the work between them is not part
of the contract.
"""
from __future__ import annotations

from .shamir import Shamir


class SecureAggregator:
    def __init__(self, shamir=None):
        self.shamir = shamir if shamir is not None else Shamir()

    def committee_of(self, coord_shares):
        if not coord_shares:
            return ()
        return tuple(sorted(x for x, _ in coord_shares[0]))

    def aggregate(self, coord_shares):
        # Recover one aggregated value per coordinate.
        return [self.shamir.reconstruct(shares) for shares in coord_shares]

    def aggregate_rounds(self, rounds):
        # Aggregate each round of the sequence.
        return [self.aggregate(r) for r in rounds]
