"""Batched guide-state advance for regex-constrained (guided) decoding.

At every decode step the guided-generation runtime holds one FSM/DFA *guide state* per
active sequence in the batch. After the sampler picks a token for each sequence, the guide
must ADVANCE every sequence's state along the transition its sampled token induces, and it
also needs the ADVANCED state's out-degree (how many distinct tokens are still valid) so it
can flag sequences that have reached a dead/terminal state. Doing this for the whole batch,
every step, is the "guide transition" hot path that sits between sampling and the next
allowed-token mask.

The DFA is provided as a transition EDGE LIST -- ``(state, token_key, next_state)`` triples,
the form interegular / outlines-style automata carry -- and the batch is a list of
``(current_state, token_key)`` queries, one per active sequence. ``batch_advance`` returns,
for the whole batch, the next state of each query (or ``-1`` if the token is not accepted
from that state, i.e. the sequence dies) and the out-degree of each resulting next state
(``0`` for a dead query).

(Grounded in the guide/FSM advance step of constrained decoding -- outlines' ``Guide`` /
``get_next_state`` and the sglang / xgrammar equivalents advance the automaton for every
sequence each step. Willard & Louf, "Efficient Guided Generation for Large Language
Models", 2023.)

SLOW-RESCAN-EDGES baseline
--------------------------
The shipped implementation is correct but slow: it never builds an index. For EACH query it
LINEAR-SCANS the entire edge list to find the matching ``(state, token_key)`` transition,
and then, for the resulting next state, LINEAR-SCANS the entire edge list AGAIN to count how
many edges leave it (the out-degree). Every query re-pays the full edge-list scan twice, so
the cost is O(#queries x #edges).

The editable scope is ``batch_advance`` (and its helpers) in this file.
"""

from typing import Dict, List, Sequence, Tuple

Edge = Tuple[int, int, int]        # (state, token_key, next_state)
Query = Tuple[int, int]            # (current_state, token_key)
Result = Dict[str, List[int]]      # {"next_states": [...], "out_degrees": [...]}


def batch_advance(
    num_states: int,
    edges: Sequence[Edge],
    queries: Sequence[Query],
) -> Result:
    """Advance a batch of guide states by their sampled tokens.

    :param num_states: number of DFA states (ids ``0..num_states-1``).
    :param edges: transition edge list of ``(state, token_key, next_state)`` triples
        (a deterministic automaton: at most one edge per ``(state, token_key)``).
    :param queries: batch of ``(current_state, token_key)`` pairs, one per active sequence.
    :returns: dict with
        ``next_states``  list[int]; the advanced state for each query, or ``-1`` if the
                         token is not accepted from the current state (sequence dies);
        ``out_degrees``  list[int]; the number of outgoing edges from each ``next_states``
                         entry (``0`` for a dead query).

    SLOW-RESCAN-EDGES: each query linear-scans the whole edge list (twice).
    """
    edges = list(edges)
    next_states: List[int] = []
    out_degrees: List[int] = []

    for (cur, tok) in queries:
        # (1) full edge-list scan to find this query's transition.
        nxt = -1
        for (s, k, ns) in edges:
            if s == cur and k == tok:
                nxt = ns
                break

        # (2) full edge-list scan AGAIN to count the out-degree of the landed state.
        if nxt == -1:
            deg = 0
        else:
            deg = 0
            for (s, k, ns) in edges:
                if s == nxt:
                    deg += 1

        next_states.append(nxt)
        out_degrees.append(deg)

    return {"next_states": next_states, "out_degrees": out_degrees}
