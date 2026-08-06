"""e2e-h3-eval-harness-throughput-quality HEADROOM PROBE #3 (reviewer-only): the regex wall.

WHY A THIRD PROBE
  Probes #1 (memoise/hoist) and #2 (bucket/columnar) both plateaued at ~1.3-1.5x while the
  ContractFloorScorer instrument in ceiling_columnar.py says the absolute wall is ~11x. That gap
  had to be explained before this task could be called ready OR routed down, because the two
  possible explanations demand opposite verdicts:

    (i)  the remaining ~69% is irreducible in practice -> the real headroom is ~1.5x -> route_down;
    (ii) the remaining ~69% sits in ONE hot primitive that a solver can legitimately replace
         -> headroom is large -> the task is sound.

  Arithmetic on the probe-#2 numbers points hard at (ii): the held-out mix issues ~1 regex search
  per take_first row plus ~5 per majority_vote row, i.e. ~1.38 searches per row, and the entire
  gap between the best ceiling and the contract floor works out to ~1us per search. The workload
  is REGEX-BOUND, not Python-overhead-bound. So the decisive question is whether `rx.search` can be
  legally beaten -- and it can, because the hot pattern is not a general regex.

WHAT THIS DOES
  `answer\\s*is\\s*([A-D])` is a literal, some whitespace, and one single-character class. That
  shape is matchable with `str.find` (a C memchr scan) plus a few index tests, with NO regex engine
  involved. This file recognises that restricted grammar, compiles a specialised matcher for it,
  and falls back to `re` for anything it does not fully understand.

  THREE THINGS MAKE THIS SAFE RATHER THAN CLEVER:
    1. Conservative recogniser. Only `LIT`, `\\s*`, `\\s+` and one trailing single-char class are
       accepted, the class must be the last token, and every `\\s*`/`\\s+` must be followed by
       something that cannot itself start with whitespace -- that last condition is what makes a
       greedy maximal whitespace skip equivalent to the regex engine's backtracking one. Anything
       else (alternation, `.`, quantified groups, backreferences, an unparticipating group, ...)
       returns None and the caller uses `re`.
    2. Leftmost-first semantics are preserved explicitly. `search` must find the LEFTMOST position
       where the WHOLE pattern matches, so a failed attempt retries at the next occurrence of the
       leading literal (`find(lit, start + 1)`) instead of giving up.
    3. SELF-VERIFICATION AT COMPILE TIME. Before any specialised matcher is used, it is checked
       against the real `re` module on a generated probe set built from the pattern's own literals
       and character class (empty string, no-match, tight/loose/absent whitespace, every class
       member, shadowing earlier partial hits, trailing truncation). One disagreement and the
       specialisation is discarded. This runs ONCE per pattern, not per row.

  A solver can reach every line of this: it is plain stdlib, it is score-preserving by
  construction, and it is exactly the kind of "look at what the hot primitive actually is and
  replace it" work an optimisation task is supposed to reward.

NOT A SUBMISSION REQUIREMENT -- A HEADROOM MEASUREMENT
  This file exists to answer DoD item 4 with a number. It is standalone on purpose (no import from
  ceiling_columnar.py) so it can be seeded as-is and timed by the real verifier.
"""
from __future__ import annotations

import re
from collections import Counter

_WS = re.compile(r"\s+")
_KNOWN_GENERATIVE = frozenset({"exact_match", "contains", "prefix_match"})

# characters allowed inside a recognised LITERAL run: no regex metacharacter, no backslash
_LIT_OK = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-:,;'\"/<>=@#%&~`"
)

_T_LIT, _T_WS, _T_CLS = 0, 1, 2


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


# ----------------------------------------------------------------------------------------------
# The restricted-pattern recogniser
# ----------------------------------------------------------------------------------------------
def _parse_charclass(pattern, i):
    """Parse `[abcA-D]` at pattern[i]. Return (frozenset, next_i) or (None, i)."""
    if i >= len(pattern) or pattern[i] != "[":
        return None, i
    j = i + 1
    if j < len(pattern) and pattern[j] in "^]":
        return None, i                      # negation / immediate ] -> not supported
    chars = set()
    while j < len(pattern) and pattern[j] != "]":
        c = pattern[j]
        if c == "\\":
            return None, i                  # escapes inside a class -> not supported
        if pattern[j + 1:j + 2] == "-" and pattern[j + 2:j + 3] not in ("", "]"):
            hi = pattern[j + 2]
            if hi == "\\" or ord(hi) < ord(c):
                return None, i
            chars.update(chr(k) for k in range(ord(c), ord(hi) + 1))
            j += 3
        else:
            chars.add(c)
            j += 1
    if j >= len(pattern) or not chars:
        return None, i
    return frozenset(chars), j + 1


def _tokenise(pattern):
    """Return (tokens, has_group) for a recognised pattern, else (None, False)."""
    toks = []
    has_group = False
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "\\":
            if pattern[i:i + 3] in (r"\s*", r"\s+"):
                toks.append((_T_WS, pattern[i + 2] == "+"))
                i += 3
                continue
            return None, False
        if c == "(":
            # only `([...])` -- one capturing group wrapping a single char class
            if pattern[i + 1:i + 2] != "[" or has_group:
                return None, False
            chars, j = _parse_charclass(pattern, i + 1)
            if chars is None or pattern[j:j + 1] != ")":
                return None, False
            toks.append((_T_CLS, chars))
            has_group = True
            i = j + 1
            continue
        if c == "[":
            chars, j = _parse_charclass(pattern, i)
            if chars is None:
                return None, False
            toks.append((_T_CLS, chars))
            i = j
            continue
        if c in _LIT_OK:
            j = i
            while j < n and pattern[j] in _LIT_OK:
                j += 1
            toks.append((_T_LIT, pattern[i:j]))
            i = j
            continue
        return None, False                  # any other metacharacter

    # structural constraints that make the fast matcher exactly equivalent
    if len(toks) < 2 or toks[0][0] != _T_LIT:
        return None, False
    if sum(1 for k, _ in toks if k == _T_CLS) != 1 or toks[-1][0] != _T_CLS:
        return None, False
    for idx, (kind, _) in enumerate(toks[:-1]):
        if kind != _T_WS:
            continue
        nk, nv = toks[idx + 1]
        # `\s*\s+` needs the first skip to give a char back to the second one; a maximal skip
        # cannot do that, so adjacent whitespace tokens are not accepted.
        if nk == _T_WS:
            return None, False
        # a greedy maximal whitespace skip == the engine's backtracking one only if what follows
        # cannot itself begin with whitespace
        if nk == _T_LIT and nv[:1].isspace():
            return None, False
        if nk == _T_CLS and any(ch.isspace() for ch in nv):
            return None, False
    return toks, has_group


def _build_matcher(toks):
    """Return search(s) -> (whole_match, class_char) or None. Leftmost-first, like re.search."""
    lead = toks[0][1]
    lead_len = len(lead)
    rest = tuple(toks[1:])

    def search(s):
        find = s.find
        n = len(s)
        start = find(lead)
        while start >= 0:
            pos = start + lead_len
            ok = True
            for kind, val in rest:
                if kind == _T_LIT:
                    if not s.startswith(val, pos):
                        ok = False
                        break
                    pos += len(val)
                elif kind == _T_WS:
                    j = pos
                    while j < n and s[j].isspace():
                        j += 1
                    if val and j == pos:            # `\s+` needs at least one
                        ok = False
                        break
                    pos = j
                else:                               # _T_CLS, always last, consumes exactly 1 char
                    if pos >= n or s[pos] not in val:
                        ok = False
                        break
                    pos += 1
            if ok:
                return s[start:pos], s[pos - 1]
            start = find(lead, start + 1)
        return None

    return search


def _probe_strings(toks):
    """Generate adversarial probes from the pattern's own pieces."""
    lits = [v for k, v in toks if k == _T_LIT]
    cls = sorted(next(v for k, v in toks if k == _T_CLS))
    joiners = ["", " ", "  ", "\t", "\n", " \t\n ", " "]
    probes = ["", "x", "no match here at all"]
    for ch in cls[:6]:
        for j in joiners:
            body = j.join(lits) + j + ch
            probes.append(body)
            probes.append("prefix noise " + body + " trailing noise")
            probes.append(lits[0] + " DECOY " + body)        # earlier partial hit must not shadow
            probes.append(body[:-1])                          # truncated: class char missing
            probes.append(body + body)                        # two hits -> leftmost wins
    for j in joiners:
        probes.append(j.join(lits) + j + "%")                 # char outside the class
        probes.append(j.join(lits))
    probes.extend([lits[0], lits[0] * 3, " ".join(lits) + " "])
    return probes


def compile_fast(pattern):
    """Return (search_fn, has_group) if `pattern` is provably fast-matchable, else (None, False).

    search_fn(s) -> (whole, class_char) | None. Verified against `re` before being handed back.
    """
    toks, has_group = _tokenise(pattern)
    if toks is None:
        return None, False
    fast = _build_matcher(toks)
    rx = re.compile(pattern)
    if (rx.groups > 0) != has_group:
        return None, False
    for probe in _probe_strings(toks):
        m = rx.search(probe)
        got = fast(probe)
        if m is None:
            if got is not None:
                return None, False
            continue
        if got is None:
            return None, False
        if got[0] != m.group(0):
            return None, False
        if has_group and got[1] != m.group(1):
            return None, False
    return fast, has_group


# ----------------------------------------------------------------------------------------------
class FastMatchColumnarScorer:
    """Bucketed/columnar scorer whose regex extraction uses the verified specialised matcher."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._memo = _NormMemo()
        self._cache: dict[str, tuple] = {}

    def _matcher(self, pattern):
        """-> (fn, has_groups, is_fast). `fn` is either the fast matcher or Pattern.search."""
        entry = self._cache.get(pattern)
        if entry is None:
            fast, has_group = compile_fast(pattern)
            if fast is not None:
                entry = (fast, has_group, True)
            else:
                compiled = re.compile(pattern)
                entry = (compiled.search, compiled.groups > 0, False)
            self._cache[pattern] = entry
        return entry

    def _extract_col(self, col, pattern):
        fn, has_groups, is_fast = self._matcher(pattern)
        if is_fast:
            if has_groups:
                return [r[1] if r is not None else "" for r in map(fn, col)]
            return [r[0] if r is not None else "" for r in map(fn, col)]
        if has_groups:
            out = [m[1] if m is not None else "" for m in map(fn, col)]
            if None in out:
                out = ["None" if p is None else p for p in out]
            return out
        return [m[0] if m is not None else "" for m in map(fn, col)]

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

    def _majority(self, rows, pattern):
        memo = self._memo
        out = []
        append = out.append
        extract = self._extract_col
        for s in rows:
            raw = s.get("response")
            if isinstance(raw, list):
                src = [x if type(x) is str else str(x) for x in raw]
            else:
                src = [raw if type(raw) is str else str(raw)]
            if pattern:
                src = extract(src, pattern)
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

            if filt == "majority_vote":
                preds = self._majority(rows, pattern)
            else:
                col = self._first_strings(rows)
                preds = self._extract_col(col, pattern) if pattern else col

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


def load_scoring_pipeline_for_verification(device: str = "cpu"):
    return FastMatchColumnarScorer(device=device)
