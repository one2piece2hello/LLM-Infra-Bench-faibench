#!/usr/bin/env python3
"""Standalone verifier workload for the paraphrase miner
(scope: /app/repo/paraphrase_miner.py :: mine_paraphrases).

Drives the scope STANDALONE on CPU (pure Python / numpy, no torch, no GPU): a corpus of N
integer-quantized sentence embeddings. The scope must return the highest-similarity distinct
sentence pairs (similarity = integer dot product): each sentence's top_k neighbours, pooled,
canonicalised to i<j, de-duplicated, sorted by descending score (ties by ascending (i,j)),
capped at max_pairs.

  correctness : compare the scope's pair list against an INDEPENDENT reference computed here
                (not the editable scope) from the documented policy. The full list must match
                EXACTLY. Sanity checks confirm the workload is non-trivial: the pair list is
                non-empty and de-duplication genuinely removes directed duplicates (so an
                impl that skips the i<j canonicalization/dedup diverges).
  timing      : warmup + timed repeats of mine_paraphrases on a larger corpus, where the
                shipped baseline's triple-nested Python similarity loop and repeated per-row
                max-scan dominate and separate from the array-vectorised version. The gap
                GROWS with the corpus size N and embedding dim D.

Emits one line `WRO_RESULT {json}`.
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

C_N, C_D, C_TOPK, C_MAXP = 40, 24, 5, 100000
T_N, T_D, T_TOPK, T_MAXP = 150, 48, 8, 2000
WARMUP = 3
ITERS = 9


def load_scope():
    import paraphrase_miner as m
    return m


def _gen(n, d, seed):
    rng = np.random.default_rng(seed)
    e = rng.integers(-90, 91, size=(n, d), dtype=np.int64)
    return [[int(x) for x in row] for row in e]


def _reference(embeddings, top_k, max_pairs):
    """Independent correct miner from the documented policy (ties -> smaller neighbour)."""
    e = np.asarray(embeddings, dtype=np.int64)
    n = e.shape[0]
    if n < 2:
        return [], 0
    sims = e @ e.T
    masked = sims.copy()
    di = np.arange(n)
    masked[di, di] = -(10 ** 9)
    order = np.argsort(-masked, axis=1, kind="stable")
    k = min(top_k, n - 1)
    topk = order[:, :k]
    directed = 0
    seen = set()
    pool = []
    for i in range(n):
        for jj in topk[i]:
            jj = int(jj)
            directed += 1
            a, b = (i, jj) if i < jj else (jj, i)
            if (a, b) not in seen:
                seen.add((a, b))
                pool.append((int(sims[a, b]), a, b))
    pool.sort(key=lambda x: (-x[0], x[1], x[2]))
    pool = pool[:max_pairs]
    return [[int(s), int(a), int(b)] for (s, a, b) in pool], directed


def _correctness_case(m):
    emb = _gen(C_N, C_D, seed=17)
    got = m.mine_paraphrases([list(r) for r in emb], C_TOPK, C_MAXP)
    ref, directed = _reference(emb, C_TOPK, C_MAXP)

    pairs_exact = got == ref
    nonempty = len(ref) > 0
    canonical_invariant = all(p[1] < p[2] for p in ref)
    dedup_matters = directed > len(ref)  # directed duplicates were collapsed -> canon matters

    ok = bool(pairs_exact and nonempty and canonical_invariant and dedup_matters)
    return {"correctness_ok": ok, "pairs_exact": pairs_exact, "nonempty": nonempty,
            "canonical_invariant": canonical_invariant, "dedup_matters": dedup_matters,
            "n_pairs": len(ref), "directed": directed, "N": C_N, "D": C_D}


def _timing_case(m):
    emb = _gen(T_N, T_D, seed=5)
    e = [list(r) for r in emb]

    def once():
        m.mine_paraphrases(e, T_TOPK, T_MAXP)

    gc.collect()
    gc.disable()
    try:
        for _ in range(WARMUP):
            once()
        ts = []
        for _ in range(ITERS):
            t0 = time.process_time()
            once()
            ts.append((time.process_time() - t0) * 1000.0)
    finally:
        gc.enable()
    return statistics.median(ts)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    m = load_scope()
    if mode == "correctness":
        res = _correctness_case(m)
        res["mode"] = "correctness"
        res["module"] = m.__file__
        print("WRO_RESULT " + json.dumps(res))
        sys.exit(0 if res["correctness_ok"] else 3)
    elif mode == "timing":
        ms = _timing_case(m)
        print("WRO_RESULT " + json.dumps({"mode": "timing", "timing_ms": ms,
              "iters": ITERS, "N": T_N, "D": T_D, "module": m.__file__}))
        sys.exit(0)
    else:
        print("WRO_RESULT " + json.dumps({"error": "bad_mode"}))
        sys.exit(2)


if __name__ == "__main__":
    main()
