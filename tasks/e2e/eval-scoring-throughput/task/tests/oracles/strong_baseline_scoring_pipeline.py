"""FROZEN reviewer-owned STRONG-BASELINE implementation — the ABBA pairing partner.

Uploaded FRESH under the frozen surface at scoring; root-0700, NEVER model-visible, NEVER baked
into the task image (MOD_SPEC change 3). The reward driver re-measures THIS file in the same
process/host as the candidate, alternating baseline/candidate for >= 5 ABBA pairs, so the speedup
denominator is never a stale cross-host constant.

Byte-identical copy of the reviewer reference scorer under solution/ (2026-07-27).
"""

from __future__ import annotations

import re


_WS = re.compile(r"\s+")


def _normalise(text) -> str:
    return _WS.sub(" ", str(text).strip().lower())


class GroupedVectorisableScorer:
    """Precompiles regexes once and scores records grouped by metric (tight per-branch loops)."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._rx_cache: dict[str, re.Pattern] = {}

    def _rx(self, pattern: str) -> re.Pattern:
        rx = self._rx_cache.get(pattern)
        if rx is None:
            rx = re.compile(pattern)
            self._rx_cache[pattern] = rx
        return rx

    def _extract(self, cands: list[str], pattern) -> list[str]:
        if not pattern:
            return cands
        rx = self._rx(pattern)
        out = []
        for c in cands:
            m = rx.search(c)
            out.append(m.group(1) if (m and m.groups()) else (m.group(0) if m else ""))
        return out

    def _filtered_pred(self, s: dict) -> str:
        raw = s.get("response")
        cands = [str(x) for x in raw] if isinstance(raw, list) else [str(raw)]
        cands = self._extract(cands, s.get("filter_pattern"))
        if str(s.get("filter", "take_first")) == "majority_vote":
            counts: dict[str, int] = {}
            for c in cands:
                key = _normalise(c)
                counts[key] = counts.get(key, 0) + 1
            if not counts:
                return ""
            # single-pass argmax with lexicographic tie-break (faster than full sort)
            best_key = None
            best_cnt = -1
            for key, cnt in counts.items():
                if cnt > best_cnt or (cnt == best_cnt and (best_key is None or key < best_key)):
                    best_cnt = cnt
                    best_key = key
            return best_key or ""
        return cands[0] if cands else ""

    def score(self, samples: list[dict]) -> list[dict]:
        out: list[dict] = []
        # group by metric so each branch is a tight typed loop (no per-row dispatch)
        by_metric: dict[str, list[dict]] = {}
        for s in samples:
            by_metric.setdefault(str(s.get("metric", "exact_match")), []).append(s)

        for s in by_metric.get("loglikelihood_acc", []):
            lls = s.get("choice_loglikelihoods", [])
            if not lls:
                out.append({"id": s["id"], "score": 0.0})
                continue
            best_i = 0
            best_v = float(lls[0])
            for i in range(1, len(lls)):
                v = float(lls[i])
                if v > best_v:
                    best_v = v
                    best_i = i
            out.append({"id": s["id"], "score": 1.0 if best_i == int(s.get("gold_index", -1)) else 0.0})

        for s in by_metric.get("exact_match", []):
            pred = _normalise(self._filtered_pred(s))
            out.append({"id": s["id"], "score": 1.0 if pred == _normalise(s.get("gold", "")) else 0.0})

        for s in by_metric.get("contains", []):
            pred = _normalise(self._filtered_pred(s))
            out.append({"id": s["id"], "score": 1.0 if _normalise(s.get("gold", "")) in pred else 0.0})

        for s in by_metric.get("prefix_match", []):
            pred = _normalise(self._filtered_pred(s))
            out.append({"id": s["id"], "score": 1.0 if pred.startswith(_normalise(s.get("gold", ""))) else 0.0})

        # any unknown metric -> 0 (mirrors reference)
        known = {"loglikelihood_acc", "exact_match", "contains", "prefix_match"}
        for m, rows in by_metric.items():
            if m not in known:
                out.extend({"id": r["id"], "score": 0.0} for r in rows)
        return out


def load_scoring_pipeline_for_verification(device: str = "cpu"):
    return GroupedVectorisableScorer(device=device)
