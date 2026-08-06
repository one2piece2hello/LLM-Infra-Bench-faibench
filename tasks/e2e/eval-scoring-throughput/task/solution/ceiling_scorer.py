"""e2e-h3-eval-harness-throughput-quality CEILING scorer (reviewer-only; proves headroom >> 1.0).

NOT model-visible. NOT seeded by any VERIFIER_MODE — this file exists to answer DoD item 4
("does this task actually have headroom above the strong baseline?") with an EXECUTABLE artifact
instead of a README recipe. Score it by pointing the strong_baseline seed dir at this file
(see selftest_ceiling.py for the exact invocation).

Relationship to the other two artifacts in this directory:
  * negative_scorer.py       -> naive per-row loop      (< 1.0, proves the gradient floor)
  * scoring_pipeline_ref.py  -> competent optimised     (== 1.0 anchor; regex precompile + grouping)
  * ceiling_scorer.py (this) -> what the headroom is    (>> 1.0)

Every optimisation below is score-preserving BY CONSTRUCTION, and each one is justified against
the harness reference implementation rather than against the anchor:

  C1. MEMOISED normalise on the SMALL-DOMAIN strings only.
      The anchor calls _WS.sub() for every gold and every filtered candidate. After regex
      extraction the candidate domain collapses to {"A","B","C","D",""} and 48% of golds are a
      single letter, so a dict lookup replaces a regex substitution on the overwhelming majority
      of calls. normalise is a PURE function of its argument, so memoising it cannot change a
      score. Deliberately NOT applied to the free-form `contains`/`prefix_match` predictions:
      those are ~unique full responses, where a memo would be pure insert overhead and would grow
      the table with one entry per row.

  C2. HOIST `m.groups()` out of the per-row loop.
      Reference: `m.group(1) if (m and m.groups()) else (m.group(0) if m else "")`.
      `m.groups()` allocates a tuple PER ROW, and its truthiness is a property of the compiled
      pattern, not of the match: a pattern with >=1 group always returns a non-empty tuple (an
      unparticipating group yields `(None,)`, still truthy). So `m.groups()` is truthy
      <=> `rx.groups > 0`. Computing that ONCE per pattern is exactly equivalent and removes one
      tuple allocation per candidate.

  C3. C-level argmax for loglikelihood_acc (40% of the held-out rows).
      Reference: `max(range(len(lls)), key=lambda i: (lls[i], -i))` -- a Python-level lambda
      invoked per element, each call building a 2-tuple. `lls.index(max(lls))` is two C-level
      passes and picks the FIRST maximum, which is what `(lls[i], -i)` selects (maximising -i
      minimises i). Guarded to `float` elements only, and NaN-free: `float()` on an int above
      2**53 can reorder magnitudes, and with NaN present the two argmax formulations are not
      provably equivalent (tuple comparison short-circuits on `==`, plain `max` does not).
      Either case falls back to the reference expression verbatim.

  C4. Scalar-response fast path.
      Reference always materialises `cands = [str(raw)]`. When `raw` is already a str we skip both
      the list allocation and the str() call.

  C5. Single-pass majority vote (no full sort) + locals binding in every hot loop.
      Reference sorts `counts.items()` with a lambda key; highest-count-then-lexicographically-
      smallest is obtainable in one pass. (The anchor already does this -- kept so the remaining
      delta is attributable to C1-C4, not to re-winning something the anchor already had.)

What this file deliberately does NOT do: no numpy, no C extension, no multiprocessing, no
source changes under /app/repo. It is plain stdlib, so the measured speedup is a LOWER BOUND on
the real headroom -- a solver who vectorises with numpy or goes columnar should beat it.
"""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")

_KNOWN_METRICS = frozenset({"loglikelihood_acc", "exact_match", "contains", "prefix_match"})


def _normalise(text) -> str:
    """Byte-for-byte identical to the harness reference `_normalise`."""
    return _WS.sub(" ", str(text).strip().lower())


class MemoisedFastScorer:
    """Score-identical to the harness reference; faster via C1-C5 above."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._rx_cache: dict[str, tuple] = {}     # pattern -> (compiled, has_groups)
        self._norm_memo: dict[str, str] = {}      # C1: small-domain normalise cache

    # --- C1 -------------------------------------------------------------------
    def _norm_memo_get(self, text) -> str:
        key = text if type(text) is str else str(text)
        memo = self._norm_memo
        got = memo.get(key)
        if got is None:
            got = _WS.sub(" ", key.strip().lower())
            memo[key] = got
        return got

    # --- C2 -------------------------------------------------------------------
    def _rx(self, pattern: str) -> tuple:
        entry = self._rx_cache.get(pattern)
        if entry is None:
            compiled = re.compile(pattern)
            entry = (compiled, compiled.groups > 0)
            self._rx_cache[pattern] = entry
        return entry

    # --- C2 + C4 --------------------------------------------------------------
    def _filtered_pred(self, s: dict) -> str:
        raw = s.get("response")
        pattern = s.get("filter_pattern")
        majority = str(s.get("filter", "take_first")) == "majority_vote"

        if not pattern:
            if not majority:
                # C4: no extraction, take_first -> the first candidate verbatim
                if type(raw) is str:
                    return raw
                if isinstance(raw, list):
                    return str(raw[0]) if raw else ""
                return str(raw)
            cands = [str(x) for x in raw] if isinstance(raw, list) else [str(raw)]
        else:
            rx, has_groups = self._rx(pattern)
            search = rx.search
            if type(raw) is str:
                m = search(raw)
                if not majority:
                    if m is None:
                        return ""
                    return m.group(1) if has_groups else m.group(0)
                cands = ["" if m is None else (m.group(1) if has_groups else m.group(0))]
            else:
                src = [str(x) for x in raw] if isinstance(raw, list) else [str(raw)]
                cands = []
                append = cands.append
                if has_groups:
                    for c in src:
                        m = search(c)
                        append(m.group(1) if m else "")
                else:
                    for c in src:
                        m = search(c)
                        append(m.group(0) if m else "")
                if not majority:
                    return cands[0] if cands else ""

        # --- C5: single-pass majority vote, C1 on the (tiny-domain) candidates
        norm = self._norm_memo_get
        counts: dict[str, int] = {}
        for c in cands:
            key = norm(c)
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            return ""
        best_key = None
        best_cnt = -1
        for key, cnt in counts.items():
            if cnt > best_cnt or (cnt == best_cnt and (best_key is None or key < best_key)):
                best_cnt = cnt
                best_key = key
        return best_key or ""

    def score(self, samples: list[dict]) -> list[dict]:
        out: list[dict] = []
        append_out = out.append

        by_metric: dict[str, list[dict]] = {}
        for s in samples:
            m = s.get("metric", "exact_match")
            k = m if type(m) is str else str(m)
            bucket = by_metric.get(k)
            if bucket is None:
                by_metric[k] = [s]
            else:
                bucket.append(s)

        # --- C3: loglikelihood_acc ------------------------------------------
        for s in by_metric.get("loglikelihood_acc", ()):
            lls = s.get("choice_loglikelihoods", ())
            if not lls:
                append_out({"id": s["id"], "score": 0.0})
                continue
            for v in lls:
                if type(v) is not float or v != v:
                    # ints (float() can reorder magnitudes > 2**53) and NaN (where the two argmax
                    # formulations are not provably equivalent) take the reference path verbatim.
                    lls = [float(x) for x in lls]
                    pred = max(range(len(lls)), key=lambda i: (lls[i], -i))
                    break
            else:
                pred = lls.index(max(lls)) if type(lls) is list else list(lls).index(max(lls))
            append_out({"id": s["id"],
                        "score": 1.0 if pred == int(s.get("gold_index", -1)) else 0.0})

        norm = self._norm_memo_get
        filtered = self._filtered_pred

        for s in by_metric.get("exact_match", ()):
            append_out({"id": s["id"],
                        "score": 1.0 if norm(filtered(s)) == norm(s.get("gold", "")) else 0.0})

        # C1 is applied to gold only here: the prediction is a ~unique free-form response.
        for s in by_metric.get("contains", ()):
            append_out({"id": s["id"],
                        "score": 1.0 if norm(s.get("gold", "")) in _normalise(filtered(s)) else 0.0})

        for s in by_metric.get("prefix_match", ()):
            append_out({"id": s["id"],
                        "score": 1.0 if _normalise(filtered(s)).startswith(norm(s.get("gold", ""))) else 0.0})

        for m, rows in by_metric.items():
            if m not in _KNOWN_METRICS:
                for r in rows:
                    append_out({"id": r["id"], "score": 0.0})
        return out


def load_scoring_pipeline_for_verification(device: str = "cpu"):
    return MemoisedFastScorer(device=device)
