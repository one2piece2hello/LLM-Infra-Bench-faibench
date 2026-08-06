"""e2e-h3-eval-harness-throughput-quality HEADROOM PROBE #2 (reviewer-only).

WHY THIS FILE EXISTS
  ceiling_scorer.py (memoise + hoist + fast paths, all plain stdlib) measured 1.354x over the 1.0
  anchor. That is above the 1.05x "config flip" line but it is NOT obviously "large headroom", and
  the red line for a shippable optimisation task is a LARGE gap. 1.354x could mean
  either of two very different things:

    (i)  the task genuinely has little headroom  -> route_down; or
    (ii) my first ceiling was simply not aggressive enough.

  A README cannot tell those apart. This file settles it with two instruments:

  1. ColumnarBucketScorer  -- an AGGRESSIVE but still legally-submittable scorer. Everything a
     determined solver would actually reach for, short of numpy: bucket rows by the FULL shape
     (metric, filter, pattern) so each inner loop is a flat comprehension with zero per-row
     dispatch and zero per-row `.get()` for filter/pattern; drive regex extraction with C-level
     `map(search, col)`; count majority votes with `collections.Counter` (C `_count_elements`)
     instead of a Python dict-update loop; and normalise through a `dict.__missing__` memo, which
     is a C-level dict lookup that only enters Python on a miss. Rare bucket shapes fall through
     to a VERBATIM copy of the reference, so correctness on the edge corpus is by construction.

  2. ContractFloorScorer  -- NOT a submission and NOT a headroom claim. It returns the right ids
     with WRONG scores, i.e. it does the minimum work the `.score()` contract can possibly
     require: walk every input dict, read its id, allocate every output dict. Its time is a HARD
     LOWER BOUND for any correct implementation, so `anchor_time / floor_time` is the ABSOLUTE
     ceiling of achievable speedup on this workload. If that absolute bound is itself small, no
     solver -- numpy, Cython, anything -- can win big, and the task must be routed down regardless
     of how clever the reference solution is. This is the number that makes the ruling defensible
     instead of a guess.

  Read the pair together: ColumnarBucketScorer says "here is what a strong solver reaches", and
  ContractFloorScorer says "here is what nobody can beat". The headroom verdict lives between them.
"""
from __future__ import annotations

import re
from collections import Counter

_WS = re.compile(r"\s+")
_KNOWN_GENERATIVE = frozenset({"exact_match", "contains", "prefix_match"})


def _normalise(text) -> str:
    return _WS.sub(" ", str(text).strip().lower())


# --- verbatim reference, used ONLY for rare bucket shapes -------------------------------------
def _ref_apply_filter(raw, filt, pattern):
    cands = [str(x) for x in raw] if isinstance(raw, list) else [str(raw)]
    if pattern:
        rx = re.compile(pattern)
        extracted = []
        for c in cands:
            m = rx.search(c)
            extracted.append(m.group(1) if (m and m.groups()) else (m.group(0) if m else ""))
        cands = extracted
    if filt == "majority_vote":
        counts: dict[str, int] = {}
        for c in cands:
            key = _normalise(c)
            counts[key] = counts.get(key, 0) + 1
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return best[0][0] if best else ""
    return cands[0] if cands else ""


def _ref_score_one(sample) -> float:
    metric = str(sample.get("metric", "exact_match"))
    if metric == "loglikelihood_acc":
        lls = [float(x) for x in sample.get("choice_loglikelihoods", [])]
        if not lls:
            return 0.0
        pred = max(range(len(lls)), key=lambda i: (lls[i], -i))
        return 1.0 if pred == int(sample.get("gold_index", -1)) else 0.0
    pred = _ref_apply_filter(sample.get("response"), str(sample.get("filter", "take_first")),
                             sample.get("filter_pattern"))
    gold = str(sample.get("gold", ""))
    if metric == "exact_match":
        return 1.0 if _normalise(pred) == _normalise(gold) else 0.0
    if metric == "contains":
        return 1.0 if _normalise(gold) in _normalise(pred) else 0.0
    if metric == "prefix_match":
        return 1.0 if _normalise(pred).startswith(_normalise(gold)) else 0.0
    return 0.0


class _NormMemo(dict):
    """normalise() as a dict. `memo[s]` is a C-level lookup; Python runs only on a miss."""

    def __missing__(self, key):
        value = _WS.sub(" ", key.strip().lower())
        self[key] = value
        return value


class ColumnarBucketScorer:
    """Bucket by (metric, filter, pattern); each bucket is a flat comprehension. Score-identical."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._memo = _NormMemo()
        self._rx: dict[str, tuple] = {}

    def _search(self, pattern):
        entry = self._rx.get(pattern)
        if entry is None:
            compiled = re.compile(pattern)
            entry = (compiled.search, compiled.groups > 0)
            self._rx[pattern] = entry
        return entry

    # -- responses -> the ONE string take_first will look at ---------------------------------
    @staticmethod
    def _first_strings(rows):
        col = []
        append = col.append
        for s in rows:
            raw = s.get("response")
            if type(raw) is str:
                append(raw)
            elif isinstance(raw, list):
                if raw:
                    f = raw[0]
                    append(f if type(f) is str else str(f))
                else:
                    # empty list -> reference yields "". Extraction of "" is also "" (a match
                    # inside "" can only be empty), so "" is a faithful stand-in under a pattern.
                    append("")
            else:
                append(str(raw))
        return col

    def _extract_first(self, rows, pattern):
        col = self._first_strings(rows)
        if not pattern:
            return col
        search, has_groups = self._search(pattern)
        if has_groups:
            out = [m[1] if m is not None else "" for m in map(search, col)]
            if None in out:
                # unparticipating group: reference propagates None into normalise -> "none"
                out = ["None" if p is None else p for p in out]
            return out
        return [m[0] if m is not None else "" for m in map(search, col)]

    def _majority(self, rows, pattern):
        memo = self._memo
        out = []
        append = out.append
        if pattern:
            search, has_groups = self._search(pattern)
            gi = 1 if has_groups else 0
        for s in rows:
            raw = s.get("response")
            if isinstance(raw, list):
                src = [x if type(x) is str else str(x) for x in raw]
            else:
                src = [raw if type(raw) is str else str(raw)]
            if pattern:
                if has_groups:
                    src = [m[gi] if m is not None else "" for m in map(search, src)]
                    if None in src:
                        src = ["None" if p is None else p for p in src]
                else:
                    src = [m[gi] if m is not None else "" for m in map(search, src)]
            counts = Counter([memo[c] for c in src])
            if not counts:
                append("")
                continue
            best_key = None
            best_cnt = -1
            for key, cnt in counts.items():
                if cnt > best_cnt or (cnt == best_cnt and key < best_key):
                    best_cnt = cnt
                    best_key = key
            append(best_key)
        return out

    def score(self, samples: list[dict]) -> list[dict]:
        buckets: dict[tuple, list] = {}
        for s in samples:
            m = s.get("metric", "exact_match")
            if type(m) is not str:
                m = str(m)
            if m == "loglikelihood_acc":
                key = ("@L", None, None)
            elif m in _KNOWN_GENERATIVE:
                f = s.get("filter", "take_first")
                if type(f) is not str:
                    f = str(f)
                p = s.get("filter_pattern")
                key = (m, f, p) if (p is None or type(p) is str) else ("@REF", None, None)
            else:
                key = ("@ZERO", None, None)
            b = buckets.get(key)
            if b is None:
                buckets[key] = [s]
            else:
                b.append(s)

        memo = self._memo
        out: list[dict] = []
        extend = out.extend

        for (metric, filt, pattern), rows in buckets.items():
            if metric == "@ZERO":
                extend([{"id": s["id"], "score": 0.0} for s in rows])
                continue
            if metric == "@REF":
                extend([{"id": s["id"], "score": _ref_score_one(s)} for s in rows])
                continue
            if metric == "@L":
                extend(self._loglikelihood(rows))
                continue

            preds = self._majority(rows, pattern) if filt == "majority_vote" \
                else self._extract_first(rows, pattern)

            if metric == "exact_match":
                extend([{"id": s["id"],
                         "score": 1.0 if memo[p] == (memo[g] if type(g) is str else _normalise(g))
                         else 0.0}
                        for s, p in zip(rows, preds) for g in (s.get("gold", ""),)])
            elif metric == "contains":
                extend([{"id": s["id"],
                         "score": 1.0 if (memo[g] if type(g) is str else _normalise(g)) in memo[p]
                         else 0.0}
                        for s, p in zip(rows, preds) for g in (s.get("gold", ""),)])
            else:  # prefix_match
                extend([{"id": s["id"],
                         "score": 1.0 if memo[p].startswith(
                             memo[g] if type(g) is str else _normalise(g)) else 0.0}
                        for s, p in zip(rows, preds) for g in (s.get("gold", ""),)])
        return out

    @staticmethod
    def _loglikelihood(rows):
        out = []
        append = out.append
        for s in rows:
            lls = s.get("choice_loglikelihoods", ())
            if not lls:
                append({"id": s["id"], "score": 0.0})
                continue
            for v in lls:
                if type(v) is not float or v != v:
                    lls = [float(x) for x in lls]
                    pred = max(range(len(lls)), key=lambda i: (lls[i], -i))
                    break
            else:
                pred = lls.index(max(lls)) if type(lls) is list else list(lls).index(max(lls))
            append({"id": s["id"], "score": 1.0 if pred == int(s.get("gold_index", -1)) else 0.0})
        return out


class ContractFloorScorer:
    """INSTRUMENT ONLY -- returns WRONG scores. Measures the irreducible cost of the .score()
    contract (walk every input dict, read every id, allocate every output dict). No correct
    implementation can be faster than this, so it bounds the achievable speedup from above."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def score(self, samples: list[dict]) -> list[dict]:
        return [{"id": s["id"], "score": 0.0} for s in samples]


def load_scoring_pipeline_for_verification(device: str = "cpu"):
    return ColumnarBucketScorer(device=device)
