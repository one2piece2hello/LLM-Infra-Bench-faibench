"""All-pairs paraphrase mining over quantized sentence embeddings (sentence-transformers).

Given a corpus of ``N`` sentence embeddings, paraphrase mining finds the highest-similarity
DISTINCT sentence pairs -- the near-duplicate / paraphrase candidates. Similarity is the dot
product of the embeddings; the embeddings are 8-bit-quantized integer vectors (the
memory-efficient retrieval representation), so every pairwise score is an exact integer.

This mirrors sentence-transformers ``util/retrieval.py`` ::
``paraphrase_mining_embeddings``: for every sentence it keeps its ``top_k`` most-similar
other sentences, collects the resulting ``(score, i, j)`` candidate pairs into a single
pool, canonicalises each pair so ``i < j`` and de-duplicates it (a pair must be reported
once, not once per direction), then returns the pairs sorted by descending score (ties
broken by ascending ``(i, j)``), capped at ``max_pairs``.

Public entry point (signature is part of the contract; do not change it):

    mine_paraphrases(embeddings, top_k, max_pairs) -> list

The implementation below is functionally correct but slow. Three phases each run as an
explicit Python loop and dominate as the corpus grows:

1. Similarity. Every pairwise score is recomputed with a triple Python loop (for each
   ordered pair, sum the element-wise products over the embedding dimension) -- O(N^2 * D).
2. Per-row selection. For each sentence its ``top_k`` neighbours are found by repeatedly
   scanning its whole score row for the current maximum ``top_k`` times -- O(N^2 * top_k).
3. Pair assembly. Candidate pairs are canonicalised and de-duplicated through a Python set
   and then ordered with a final Python sort.

Make ``mine_paraphrases`` faster on the benchmark workload while producing exactly the same
observable output (see the behavioral contract in the task instructions).
"""

# DEGRADED-BASELINE-MARKER: triple-nested Python similarity loop + repeated-argmax per-row
# top-k selection (this slow full-scan baseline ships as the reference to beat).


def mine_paraphrases(embeddings, top_k, max_pairs):
    """Return the highest-similarity distinct sentence pairs.

    Args:
        embeddings: list of ``N`` equal-length lists of ints -- the quantized embedding of
            each sentence. Similarity between two sentences is the integer dot product of
            their embeddings.
        top_k: for each sentence, keep its ``top_k`` most-similar other sentences.
        max_pairs: cap the returned list to this many highest-scoring pairs.

    Returns:
        list of ``[score, i, j]`` (all ints, ``i < j``), sorted by descending ``score`` then
        ascending ``i`` then ascending ``j``, at most ``max_pairs`` entries.
    """
    n = len(embeddings)
    if n < 2:
        return []
    d = len(embeddings[0])

    # -------- POINT 1: pairwise similarity via a triple Python loop (O(N^2 * D)) --------
    sims = [[0] * n for _ in range(n)]
    for i in range(n):
        ei = embeddings[i]
        for j in range(n):
            ej = embeddings[j]
            s = 0
            for t in range(d):
                s += ei[t] * ej[t]
            sims[i][j] = s

    # -------- POINT 2 + 3: per-row top-k by repeated max-scan, then canonical dedup ------
    seen = set()
    pairs = []
    for i in range(n):
        # candidate neighbours (exclude self)
        cand = [(sims[i][j], j) for j in range(n) if j != i]
        used = [False] * len(cand)
        k = top_k if top_k < len(cand) else len(cand)
        for _ in range(k):
            best_idx = -1
            best_score = None
            best_j = None
            for idx in range(len(cand)):
                if used[idx]:
                    continue
                sc, jj = cand[idx]
                if best_score is None or sc > best_score or (sc == best_score and jj < best_j):
                    best_score = sc
                    best_j = jj
                    best_idx = idx
            used[best_idx] = True
            sc, jj = cand[best_idx]
            a = i if i < jj else jj
            b = jj if i < jj else i
            if (a, b) not in seen:
                seen.add((a, b))
                pairs.append((sc, a, b))

    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    pairs = pairs[:max_pairs]
    return [[int(s), int(a), int(b)] for (s, a, b) in pairs]
