#!/usr/bin/env python3
"""Standalone verifier workload for the batched guide-state advance scope
(scope: /app/repo/guide_transition.py :: batch_advance).

Drives the scope STANDALONE on CPU (pure Python; no torch, no GPU): a deterministic DFA is
given as a transition edge list and a batch of ``(current_state, token_key)`` queries is
advanced -- returning each query's next state (or ``-1`` if the token dies) and the
out-degree of the landed state.

  correctness : compare the scope's ``next_states`` and ``out_degrees`` against an
                INDEPENDENT reference computed here (NOT part of the editable scope). Both
                lists must match exactly. The DFA has many states with MULTIPLE distinct
                token edges (and the query batch includes tokens that die from their state),
                so a "use the state's last edge, ignore the token" cheat diverges and scores
                0.
  timing      : warmup + timed repeats of batch_advance on a LARGE edge list and a LARGE
                query batch, so the per-query full-edge-list re-scans (O(#queries x #edges),
                twice) dominate and separate from the index-once implementation. The gap
                GROWS with #queries x #edges.

Emits one line ``WRO_GUIDEADV_RESULT {json}``. Timing uses ``time.process_time()`` (CPU
time, immune to OS descheduling on the contended fleet CPU lane).
"""
import gc
import json
import statistics
import sys
import time

import numpy as np

REPO = "/app/repo"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# correctness: small DFA, multiple token edges per state, mix of live/dead queries.
C_STATES = 12
C_TOKENS = 6
C_QUERIES = 80
C_SEED = 31337
# timing: many edges x many queries so the O(#queries x #edges) double re-scan dominates.
T_STATES = 140
T_TOKENS = 12
T_QUERIES = 1500
T_SEED = 55221
WARMUP = 2
ITERS = 5


def load_scope():
    import guide_transition
    return guide_transition


def _gen_dfa(num_states, num_tokens, seed):
    """Deterministic DFA edge list: each state gets edges on a random subset (~half) of the
    token keys, each to a random next state. At most one edge per (state, token) -> the
    automaton is deterministic and the next-state contract is well-defined. Returns
    (edges list, num_states, num_tokens)."""
    rng = np.random.default_rng(seed)
    trans = {}
    for s in range(num_states):
        n_valid = max(2, num_tokens // 2)  # >=2 edges per state so token matters
        keys = rng.choice(num_tokens, size=min(n_valid, num_tokens), replace=False)
        for k in sorted(int(x) for x in keys):
            trans[(s, k)] = int(rng.integers(0, num_states))
    # emit in a fixed (sorted) order; a token-ignoring "last edge wins" cheat picks the
    # highest token key's target for each state, which differs per real token.
    edges = [(s, k, ns) for (s, k), ns in sorted(trans.items())]
    return edges, num_states, num_tokens


def _gen_queries(edges, num_states, num_tokens, num_queries, seed):
    """Batch of (state, token) queries. ~70% are drawn from existing edges (live, and they
    share states heavily), ~30% are random pairs (often land on a state's non-edge token ->
    a DEAD -1 that a token-ignoring cheat wrongly reports as live)."""
    rng = np.random.default_rng(seed)
    edge_keys = [(s, k) for (s, k, _ns) in edges]
    out = []
    for _ in range(num_queries):
        if edge_keys and rng.random() < 0.7:
            s, k = edge_keys[int(rng.integers(0, len(edge_keys)))]
            out.append((int(s), int(k)))
        else:
            out.append((int(rng.integers(0, num_states)), int(rng.integers(0, num_tokens))))
    return out


def _reference(edges, num_states, queries):
    """Independent reference (distinct code path: dict + per-state out-degree list built once
    with plain loops, NOT the scope's double-scan). next_state = the unique (s,k) edge target
    or -1; out_degree = #edges leaving the landed state (0 if dead)."""
    trans = {}
    outdeg = [0] * num_states
    for (s, k, ns) in edges:
        trans[(s, k)] = ns
        outdeg[s] += 1
    ns_list = []
    od_list = []
    for (cur, tok) in queries:
        nxt = trans.get((cur, tok), -1)
        ns_list.append(nxt)
        od_list.append(outdeg[nxt] if nxt != -1 else 0)
    return {"next_states": ns_list, "out_degrees": od_list}


def _norm(res):
    return {"next_states": [int(x) for x in res["next_states"]],
            "out_degrees": [int(x) for x in res["out_degrees"]]}


def correctness():
    m = load_scope()
    ok = True
    detail = ""
    for seed in (C_SEED, C_SEED + 1, C_SEED + 2, C_SEED + 3):
        edges, ns, nt = _gen_dfa(C_STATES, C_TOKENS, seed)
        queries = _gen_queries(edges, ns, nt, C_QUERIES, seed + 9)
        got = _norm(m.batch_advance(ns, edges, queries))
        ref = _norm(_reference(edges, ns, queries))
        # sanity: some queries are live and some die, and some state has >1 token edge
        n_dead = sum(1 for x in ref["next_states"] if x == -1)
        n_live = sum(1 for x in ref["next_states"] if x != -1)
        multi = len(edges) > len({s for (s, _k, _ns) in edges})
        if got != ref or not (n_dead > 0 and n_live > 0 and multi):
            ok = False
            detail = (f"mismatch seed={seed}: match={got == ref} "
                      f"n_live={n_live} n_dead={n_dead} multi_token={multi}")
            break
    print("WRO_GUIDEADV_RESULT " + json.dumps({"correctness_ok": ok, "detail": detail}))


def timing():
    m = load_scope()
    edges, ns, nt = _gen_dfa(T_STATES, T_TOKENS, T_SEED)
    queries = _gen_queries(edges, ns, nt, T_QUERIES, T_SEED + 9)

    def run():
        return m.batch_advance(ns, edges, queries)

    for _ in range(WARMUP):
        run()
    gc.collect()
    gc_was = gc.isenabled()
    gc.disable()
    samples = []
    try:
        for _ in range(ITERS):
            t0 = time.process_time()
            r = run()
            t1 = time.process_time()
            samples.append((t1 - t0) * 1000.0)
    finally:
        if gc_was:
            gc.enable()
    ms = statistics.median(samples)
    print("WRO_GUIDEADV_RESULT " + json.dumps({
        "timing_ms": ms, "num_edges": len(edges), "num_queries": T_QUERIES,
        "num_states": ns}))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if mode == "correctness":
        correctness()
    elif mode == "timing":
        timing()
    else:
        print("WRO_GUIDEADV_RESULT " + json.dumps({"error": "bad_mode"}))
        sys.exit(2)
