"""e2e-h3-eval-harness-throughput-quality HEADROOM PROBE #4 (reviewer-only): leave the interpreter.

WHAT PROBES #1-#3 ESTABLISHED
  #1 memoise/hoist            1.32x
  #2 bucket/columnar          1.41x
  #3 regex -> str.find        1.01x  (verified CORRECT, and SLOWER)
  ContractFloorScorer        11.4x   (absolute wall: the cost of the .score() contract alone)

  #3 is the informative failure. Replacing `rx.search` (ONE C call) with a hand-rolled matcher (a
  dozen Python bytecodes) LOST time even though the matcher itself is exactly equivalent. So the
  ~1.4x plateau is not the regex engine and not algorithmic waste -- it is CPython interpreter
  overhead spread thinly across every row. No amount of pure-Python cleverness removes that, which
  is why two independent competent designs converged on the same number.

  The corollary is the whole point of the task: to claim the remaining ~68% you must move the
  per-row work OUT of the interpreter. That is precisely what the workspace template advertises
  ("vectorise the regex extraction and metric compute, batch over records"), and the tools are
  already in the image -- lm-evaluation-harness depends on `datasets`, which depends on pyarrow, so
  `pyarrow.compute` (RE2-backed, C++-vectorised over a whole column) and numpy are both importable
  without installing anything.

WHAT THIS FILE DOES
  A1. loglikelihood_acc (40% of rows) -> numpy. Ragged per-row choice lists are flattened once with
      `chain.from_iterable`, padded into a 2-D array by C-level index arithmetic, and reduced with
      `argmax(axis=1)`. numpy's argmax returns the FIRST maximum, which is exactly the reference's
      `(lls[i], -i)` tie-break. float() conversion is part of the reference semantics too, so the
      only real hazard is NaN -- caught with one C-level `isnan().any()` and routed to the verbatim
      reference.

  A2. regex extraction -> `pyarrow.compute.extract_regex`, one C++ call per COLUMN instead of one
      Python call per row. Arrow needs RE2 named groups, so group 1 is mechanically rewritten to
      `(?P<g1>...)`; that rewrite is only attempted for patterns the restricted recogniser in
      ceiling_fastmatch.py already understands, and it is verified against Python's own `re` on a
      generated probe set before use.

  A3. normalise -> `utf8_lower` + `replace_substring_regex(r"\\s+", " ")` + `utf8_trim_whitespace`,
      again one C++ call per column. Reference order is sub(strip(lower(x))); doing the trim LAST is
      equivalent because strip only ever removes leading/trailing whitespace and collapsing a
      leading run to a single space still leaves it strippable.

  THE CORRECTNESS TRAP, AND THE GUARD THAT DISARMS IT
      Python's `\\s` and RE2's `\\s` are NOT the same set, and `utf8_lower` is not `str.lower()`:
        * RE2 `\\s` == [ \\t\\n\\f\\r] exactly. Python `\\s` also matches VT (\\x0b), the four
          separators \\x1c-\\x1f, NEL (\\x85), NBSP, thin space, and the rest of Unicode Zs.
        * `str.strip()` strips every `str.isspace()` char, which likewise includes \\x0b and
          \\x1c-\\x1f.
        * `utf8_lower` differs from `str.lower()` on context-sensitive cases (Greek final sigma) and
          on multi-codepoint expansions (U+0130 lowers to TWO codepoints in Python).
      Any of those in a column and the C++ path would silently produce a different score -- fatal
      under min_consistency_fraction = 1.0.

      NOTE: "is the column ASCII?" is NOT a sufficient guard, and the selftest's guard probe caught
      exactly that -- \\x0b and \\x1c-\\x1f are ASCII yet sit precisely on the Python/RE2 whitespace
      disagreement. So the guard is a WHITELIST, not a blacklist: every character must be in
      `\\t\\n\\f\\r` + printable ASCII (\\x20-\\x7e). Inside that set the three primitives are
      provably identical -- the only whitespace present is { space \\t \\n \\f \\r }, which Python
      `\\s`, RE2 `\\s`, `str.strip()` and `utf8_trim_whitespace` all agree on, and case mapping is
      pure ASCII so `utf8_lower` == `str.lower()` character-for-character. Two C++ predicates
      (`string_is_ascii`, then one negated-class `match_substring_regex`) decide it per column; a
      column with anything else falls back to the verbatim reference. The real held-out records are
      printable ASCII, so the fast path covers them all, while the adversarial edge corpus exercises
      the fallback.

  Everything not fully understood -- unrecognised pattern, non-ASCII column, NaN loglikelihoods,
  unknown metric, unhashable filter_pattern -- routes to a verbatim copy of the harness reference.
  That is what makes the speedup safe to claim rather than a gamble on the input distribution.
"""
from __future__ import annotations

import re
from collections import Counter
from itertools import chain

try:
    import numpy as _np
    _HAVE_NP = True
except Exception:                                   # pragma: no cover
    _HAVE_NP = False

try:
    import pyarrow as _pa
    import pyarrow.compute as _pc
    _HAVE_ARROW = True
except Exception:                                   # pragma: no cover
    _HAVE_ARROW = False

_WS = re.compile(r"\s+")
_KNOWN_GENERATIVE = frozenset({"exact_match", "contains", "prefix_match"})

# RE2 class matching any char OUTSIDE the range where python-str and arrow-utf8 semantics provably
# coincide. One hit anywhere in a column and that column goes to the reference implementation.
_UNSAFE_CHAR = r"[^\t\n\f\r\x20-\x7e]"


def _normalise(text) -> str:
    return _WS.sub(" ", str(text).strip().lower())


# --- verbatim reference, for every shape the vectorised paths decline ---------------------------
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
    def __missing__(self, key):
        value = _WS.sub(" ", key.strip().lower())
        self[key] = value
        return value


# ----------------------------------------------------------------------------------------------
# Pattern -> RE2 named-group rewrite, restricted to shapes we can verify
# ----------------------------------------------------------------------------------------------
_SAFE_PAT = re.compile(r"^[A-Za-z0-9 _:,;'\"/<>=@#%&~`-]*"      # leading literal
                       r"(?:\\s[*+][A-Za-z0-9 _:,;'\"/<>=@#%&~`-]*)*"   # \s* + literal, repeated
                       r"(?:\((\[[^]\\^]+\])\)|(\[[^]\\^]+\]))$")       # one trailing char class


def arrow_pattern(pattern):
    """Return (re2_pattern, has_group) if `pattern` is a shape we can hand to RE2 safely.

    Restricted to: literals + `\\s*`/`\\s+` + exactly ONE trailing single-character class,
    optionally captured. Such a pattern has no alternation, no backtracking subtleties and no
    Unicode-class dependence beyond `\\s`, which the caller neutralises with an all-ASCII guard.
    Returns (None, False) for anything else so the caller falls back to Python `re`.
    """
    m = _SAFE_PAT.match(pattern)
    if not m:
        return None, False
    grouped, bare = m.group(1), m.group(2)
    cls = grouped or bare
    head = pattern[:m.start(1) if grouped else m.start(2)]
    if grouped:
        head = head[:-1]                                    # drop the "(" we are replacing
    return head + "(?P<g1>" + cls + ")", bool(grouped)


def verify_arrow_pattern(pattern, re2_pattern, has_group, probes):
    """Cross-check the RE2 rewrite against Python's `re` on `probes`. True iff identical."""
    if not _HAVE_ARROW:
        return False
    rx = re.compile(pattern)
    try:
        got = _pc.extract_regex(_pa.array(probes, type=_pa.string()), re2_pattern).to_pylist()
    except Exception:
        return False
    for probe, row in zip(probes, got):
        m = rx.search(probe)
        if m is None:
            if row is not None:
                return False
            continue
        want = m.group(1) if has_group else m.group(0)
        if row is None or row.get("g1") != want:
            return False
    return True


_PROBES = ["", "x", "answer", "answer is", "answer is A", "answer  is   B", "answer\tis\nC",
           "answer is Z", "answerisD", "the answer is D.", "noise answer answer is B tail",
           "answer is answer is C", "aanswer is A", "answer\n\n\nis\n\n\nD", "ANSWER IS A"]


class ArrowVectorisedScorer:
    """Vectorised scorer: numpy for the choice argmax, pyarrow for regex + normalise.

    Every fast path is guarded so that anything it cannot prove identical to the harness reference
    is routed to `_ref_score_one`. Score-exact by construction, not by hoping about the inputs.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._memo = _NormMemo()
        self._pat_cache: dict[str, tuple] = {}

    # -- pattern plumbing -------------------------------------------------------------------
    def _arrow_pat(self, pattern):
        entry = self._pat_cache.get(pattern)
        if entry is None:
            re2, has_group = arrow_pattern(pattern)
            if re2 is not None and verify_arrow_pattern(pattern, re2, has_group, _PROBES):
                entry = (re2, has_group)
            else:
                entry = (None, False)
            self._pat_cache[pattern] = entry
        return entry

    @staticmethod
    def _ascii_ok(arr):
        """True only if every char is in `\\t\\n\\f\\r` + printable ASCII.

        A whitelist, deliberately: \\x0b and \\x1c-\\x1f ARE ascii but are whitespace to Python and
        not to RE2, so `string_is_ascii` alone would let a divergent column onto the C++ path.
        """
        try:
            if not _pc.all(_pc.string_is_ascii(arr)).as_py():
                return False
            return not bool(_pc.any(_pc.match_substring_regex(arr, _UNSAFE_CHAR)).as_py())
        except Exception:
            return False

    @staticmethod
    def _arrow_normalise(arr):
        """utf8 normalise == reference normalise, given an all-ASCII column."""
        return _pc.utf8_trim_whitespace(
            _pc.replace_substring_regex(_pc.utf8_lower(arr), r"\s+", " "))

    def _extract_arrow(self, arr, re2, has_group):
        """One C++ call for a whole column; nulls (no match) become ''."""
        st = _pc.extract_regex(arr, re2)
        if has_group:
            vals = _pc.struct_field(st, "g1")
        else:
            # zero-group pattern: reference returns group(0). RE2 has no group(0) accessor here, so
            # the recogniser only wraps the class -- group(0) == the whole match, which for these
            # shapes is literal+ws+class. Rebuild it is not worth it: decline instead.
            return None
        return _pc.fill_null(vals, "")

    # -- blocks -----------------------------------------------------------------------------
    def _loglikelihood(self, rows):
        n = len(rows)
        cols = [s.get("choice_loglikelihoods", ()) for s in rows]
        if not _HAVE_NP:
            return self._loglikelihood_py(rows, cols)
        lens = [len(c) for c in cols]
        total = 0
        for k in lens:
            total += k
        if total == 0:
            return [{"id": s["id"], "score": 0.0} for s in rows]
        try:
            flat = _np.fromiter(chain.from_iterable(cols), dtype=_np.float64, count=total)
        except (TypeError, ValueError):
            return self._loglikelihood_py(rows, cols)
        if _np.isnan(flat).any():
            # with NaN present the tuple-keyed argmax and a plain argmax are not provably equal
            return [{"id": s["id"], "score": _ref_score_one(s)} for s in rows]
        lens_a = _np.asarray(lens, dtype=_np.int64)
        width = int(lens_a.max())
        offs = _np.zeros(n, dtype=_np.int64)
        _np.cumsum(lens_a[:-1], out=offs[1:])
        pad = _np.full((n, width), -_np.inf, dtype=_np.float64)
        rowix = _np.repeat(_np.arange(n, dtype=_np.int64), lens_a)
        colix = _np.arange(total, dtype=_np.int64) - offs[rowix]
        pad[rowix, colix] = flat
        pred = pad.argmax(axis=1)                       # argmax -> FIRST max == reference tie-break
        golds = _np.fromiter((int(s.get("gold_index", -1)) for s in rows), dtype=_np.int64, count=n)
        hit = (pred == golds) & (lens_a > 0)            # empty choice list always scores 0.0
        return [{"id": s["id"], "score": 1.0 if h else 0.0} for s, h in zip(rows, hit.tolist())]

    @staticmethod
    def _loglikelihood_py(rows, cols):
        out = []
        for s, lls in zip(rows, cols):
            out.append({"id": s["id"], "score": _ref_score_one(s)} if lls
                       else {"id": s["id"], "score": 0.0})
        return out

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
                    append("")
            else:
                append(str(raw))
        return col

    def _majority_arrow(self, rows, pattern):
        """Flatten every candidate of every row into ONE column, extract once, regroup."""
        re2, has_group = self._arrow_pat(pattern)
        if re2 is None or not has_group:
            return None
        cand_lists = []
        append = cand_lists.append
        for s in rows:
            raw = s.get("response")
            if isinstance(raw, list):
                append([x if type(x) is str else str(x) for x in raw])
            else:
                append([raw if type(raw) is str else str(raw)])
        flat = list(chain.from_iterable(cand_lists))
        if not flat:
            return ["" for _ in rows]
        arr = _pa.array(flat, type=_pa.string())
        if not self._ascii_ok(arr):
            return None
        vals = self._extract_arrow(arr, re2, has_group)
        if vals is None:
            return None
        ext = self._arrow_normalise(vals).to_pylist()
        out = []
        push = out.append
        pos = 0
        for lst in cand_lists:
            k = len(lst)
            if k == 0:
                push("")
                continue
            counts = Counter(ext[pos:pos + k])
            pos += k
            best_key = None
            best_cnt = -1
            for key, cnt in counts.items():
                if cnt > best_cnt or (cnt == best_cnt and key < best_key):
                    best_cnt = cnt
                    best_key = key
            push(best_key)
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

            done = False
            if _HAVE_ARROW:
                done = self._generative_arrow(rows, metric, filt, pattern, extend)
            if not done:
                extend([{"id": s["id"], "score": _ref_score_one(s)} for s in rows])
        return out

    def _generative_arrow(self, rows, metric, filt, pattern, extend) -> bool:
        """Try the vectorised path for one bucket. Return False to request the reference fallback."""
        try:
            if filt == "majority_vote":
                if not pattern:
                    return False
                preds = self._majority_arrow(rows, pattern)
                if preds is None:
                    return False
                pred_arr = _pa.array(preds, type=_pa.string())     # already normalised
            else:
                col = self._first_strings(rows)
                arr = _pa.array(col, type=_pa.string())
                if not self._ascii_ok(arr):
                    return False
                if pattern:
                    re2, has_group = self._arrow_pat(pattern)
                    if re2 is None:
                        return False
                    vals = self._extract_arrow(arr, re2, has_group)
                    if vals is None:
                        return False
                    pred_arr = self._arrow_normalise(vals)
                else:
                    pred_arr = self._arrow_normalise(arr)

            golds = [g if type(g) is str else str(g) for g in (s.get("gold", "") for s in rows)]
            gold_arr = _pa.array(golds, type=_pa.string())
            if not self._ascii_ok(gold_arr):
                return False
            gold_arr = self._arrow_normalise(gold_arr)

            if metric == "exact_match":
                hit = _pc.equal(pred_arr, gold_arr).to_pylist()
                extend([{"id": s["id"], "score": 1.0 if h else 0.0}
                        for s, h in zip(rows, hit)])
                return True

            # arrow has no elementwise substring-containment kernel, so the compare stays in
            # Python -- the win here is that normalise ran in C++ over the whole column
            preds_py = pred_arr.to_pylist()
            golds_py = gold_arr.to_pylist()
            if metric == "contains":
                extend([{"id": s["id"], "score": 1.0 if g in p else 0.0}
                        for s, p, g in zip(rows, preds_py, golds_py)])
            else:                                                   # prefix_match
                extend([{"id": s["id"], "score": 1.0 if p.startswith(g) else 0.0}
                        for s, p, g in zip(rows, preds_py, golds_py)])
            return True
        except Exception:
            return False


def load_scoring_pipeline_for_verification(device: str = "cpu"):
    return ArrowVectorisedScorer(device=device)
