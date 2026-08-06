"""Reviewer-only selftest for the e2e-h3-eval-harness-throughput-quality CEILING artifact.

PURPOSE
  Answer DoD item 4 with numbers instead of a README recipe:
    (a) EQUIVALENCE -- ceiling_scorer must reproduce the harness reference scores EXACTLY on the
        real held-out distribution AND on an adversarial edge-case corpus. The scoring gate is
        min_consistency_fraction = 1.0 (score_atol 1e-9), so a single divergent row means 0.
    (b) HEADROOM -- how much faster than the 1.0 anchor is it, under the harness's own timing
        protocol (median of `timing_repeats` full-set passes)?

  This file NEVER enters an image and is NEVER seeded into the submission dir. It is a
  reviewer tool only.

WHERE TO RUN IT
  On a CPU worker with the image available -- NOT on a shared front-end host.

      python3 selftest_ceiling.py --samples 20000 --repeats 5

  Give it a dedicated multi-core CPU box (>= 8 cores, >= 32 GiB) and keep it off a shared
  front-end host: the timing rows are medians of full-set passes and are sensitive to
  neighbours. If your environment injects an HTTP proxy, unset it first — this selftest is
  fully offline and a proxy only adds latency to nothing.
  It reaches sideways for ../environment/workspace/scoring_pipeline_template.py; if you stage the
  files elsewhere, keep that relative layout or the template row is reported as SKIP.

THE SCORER ROSTER AND WHAT EACH ONE MUST DO
  Scorers are addressed BY CLASS, not through load_scoring_pipeline_for_verification(), because
  negative_scorer.py deliberately ships two negatives with OPPOSITE expected outcomes:
    NaiveScoringPipeline      (template)   must MATCH, must be SLOWER than the anchor
    SlowButCorrectScorer      (negative)   must MATCH, must be SLOWER than the anchor
    GroupedVectorisableScorer (anchor)     must MATCH, is 1.000x by definition
    MemoisedFastScorer        (CEILING 1)  must MATCH, memoise/hoist/fast-path stdlib ceiling
    ColumnarBucketScorer      (CEILING 2)  must MATCH, aggressive bucketed/columnar stdlib ceiling
    FastMatchColumnarScorer   (CEILING 3)  must MATCH, str.find specialisation instead of `re`
    ArrowVectorisedScorer     (CEILING 4)  must MATCH, numpy argmax + pyarrow C++ regex/normalise
    SkippingScorer            (negative)   must DIVERGE -- it is the consistency-gate control, and
                                           a match there means the control has died
    ContractFloorScorer       (instrument) NOT scored for equivalence: it returns wrong scores on
                                           purpose and only measures what the .score() contract
                                           costs, i.e. the absolute wall no correct solver can beat
  Section (a0) additionally asserts each file's default entry point returns the class timed here,
  so "what this selftest measured" == "what the seeding actually runs".

WHAT IT PROVES / DOES NOT PROVE
  It measures the scorers against each other under the harness timing protocol. It does NOT
  replace a real scoring run: the reward number, the anti-cache probe and the welded consistency
  gate still have to come from the real verifier. Treat the speedup printed here as the headroom
  ESTIMATE that justifies spending the session, and the equivalence result as a hard
  precondition -- if equivalence fails, do not spend the session at all.

EXIT CODE
  0 only if every must-match scorer is exactly equivalent to the reference on every corpus AND the
  skipping negative diverges. Non-zero otherwise, with the first divergent row printed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# 1. The reference semantics, inlined VERBATIM from the harness reference scorer.
#    Kept byte-for-byte so this selftest cannot drift from what actually grades.
# -----------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def _ref_normalise(text) -> str:
    return _WS.sub(" ", str(text).strip().lower())


def _ref_apply_filter(raw, filt: str, pattern):
    if isinstance(raw, list):
        cands = [str(x) for x in raw]
    else:
        cands = [str(raw)]
    if pattern:
        rx = re.compile(pattern)
        extracted = []
        for c in cands:
            m = rx.search(c)
            extracted.append(m.group(1) if (m and m.groups()) else (m.group(0) if m else ""))
        cands = extracted
    if filt == "take_first":
        return cands[0] if cands else ""
    if filt == "majority_vote":
        counts: dict[str, int] = {}
        for c in cands:
            key = _ref_normalise(c)
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
        return 1.0 if _ref_normalise(pred) == _ref_normalise(gold) else 0.0
    if metric == "contains":
        return 1.0 if _ref_normalise(gold) in _ref_normalise(pred) else 0.0
    if metric == "prefix_match":
        return 1.0 if _ref_normalise(pred).startswith(_ref_normalise(gold)) else 0.0
    return 0.0


def reference_scores(samples):
    return {str(s["id"]): _ref_score_one(s) for s in samples}


# -----------------------------------------------------------------------------
# 2. Corpus A: the REAL held-out distribution (same generator as the env builder).
# -----------------------------------------------------------------------------
_ANSWER_RX = r"answer\s*is\s*([A-D])"
_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
          "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa"]


def _rand_text(rng, n):
    return " ".join(rng.choice(_WORDS) for _ in range(n))


# Realistic response-length mix for a chain-of-thought eval run. HEAVY-TAILED on purpose: most
# responses are short, but the minority of long ones carry most of the BYTES, and bytes are what the
# scoring work is proportional to. This shape is what makes the task's headroom real -- see the ws
# sweep recorded in runs/: with 80-byte stub responses the per-row cost is ~1us of thinly-spread
# interpreter overhead that nothing can remove, while at CoT lengths the cost is linear scanning,
# which memchr/split-join/early-exit all attack.
def _cot_words(rng):
    r = rng.random()
    if r < 0.55:
        return rng.randint(8, 30)          # terse answer
    if r < 0.85:
        return rng.randint(60, 200)        # a few reasoning steps
    return rng.randint(300, 900)           # long deliberation


def _make_record(prefix, i, rng, mix="stub"):
    """`mix="stub"` reproduces the originally-built dataset; `"cot"` is the realistic length mix."""
    cot = mix == "cot"
    kind = rng.random()
    rid = f"{prefix}_s{i}"
    if kind < 0.4:
        n_choices = rng.choice([2, 4, 4, 5])
        lls = [round(rng.uniform(-30.0, -1.0), 6) for _ in range(n_choices)]
        return {"id": rid, "metric": "loglikelihood_acc",
                "choice_loglikelihoods": lls, "gold_index": rng.randrange(n_choices)}
    if kind < 0.7:
        letter = rng.choice(list("ABCD"))
        nw = _cot_words(rng) if cot else rng.randint(3, 12)
        resp = f"{_rand_text(rng, nw)}. Therefore the answer is {letter}."
        gold = letter if rng.random() < 0.6 else rng.choice(list("ABCD"))
        return {"id": rid, "metric": "exact_match", "filter": "take_first",
                "filter_pattern": _ANSWER_RX, "response": resp, "gold": gold}
    if kind < 0.88:
        letter = rng.choice(list("ABCD"))
        comps = []
        for _ in range(rng.choice([3, 5, 5, 7])):
            l = letter if rng.random() < 0.65 else rng.choice(list("ABCD"))
            nw = _cot_words(rng) if cot else rng.randint(2, 8)
            comps.append(f"{_rand_text(rng, nw)} the answer is {l}")
        gold = letter if rng.random() < 0.6 else rng.choice(list("ABCD"))
        return {"id": rid, "metric": "exact_match", "filter": "majority_vote",
                "filter_pattern": _ANSWER_RX, "response": comps, "gold": gold}
    gold = _rand_text(rng, rng.randint(1, 3))
    if rng.random() < 0.5:
        # NOTE: draw order mirrors environment/build_dataset.py exactly, so this corpus is
        # distributionally identical to the one that actually gets built.
        if cot:
            resp = (f"{_rand_text(rng, _cot_words(rng))} {gold} "
                    f"{_rand_text(rng, _cot_words(rng))}").strip()
        else:
            resp = (f"{_rand_text(rng, rng.randint(0, 5))} {gold} "
                    f"{_rand_text(rng, rng.randint(0, 5))}").strip()
        metric = "contains"
    else:
        if rng.random() < 0.7:
            nw = _cot_words(rng) if cot else rng.randint(0, 6)
            resp = f"{gold} {_rand_text(rng, nw)}".strip()
        else:
            nw = _cot_words(rng) if cot else rng.randint(2, 6)
            resp = _rand_text(rng, nw)
        metric = "prefix_match"
    return {"id": rid, "metric": metric, "filter": "take_first",
            "filter_pattern": None, "response": resp, "gold": gold}


def corpus_real(n, seed=20240725, mix="stub"):
    rng = random.Random(seed)
    return [_make_record("held", i, rng, mix) for i in range(n)]


def corpus_from_jsonl(path, n=None):
    """Load the corpus from a jsonl file that environment/build_dataset.py actually emitted.

    corpus_real() is only a distributional PROXY of the built set: it re-implements the same
    generator, so a divergence between the two (a reordered rng draw, a changed branch weight)
    would silently make every headroom number here describe a corpus that never ships. Pointing
    this at the real heldout_samples.jsonl removes the proxy from the loop entirely.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                if n is not None and len(rows) >= n:
                    break
    return rows


# -----------------------------------------------------------------------------
# 3. Corpus B: adversarial edge cases -- one per optimisation that could break.
#    Each entry targets a specific place where a "faster" scorer usually diverges.
# -----------------------------------------------------------------------------
def corpus_edge():
    c = []
    a = c.append
    # -- C4 / filter plumbing: odd response containers
    a({"id": "e01", "metric": "exact_match", "filter": "take_first", "filter_pattern": None,
       "response": "", "gold": ""})                                    # empty pred == empty gold
    a({"id": "e02", "metric": "exact_match", "filter": "take_first", "filter_pattern": None,
       "response": [], "gold": ""})                                    # EMPTY list -> ""
    a({"id": "e03", "metric": "exact_match", "filter": "take_first", "filter_pattern": None,
       "response": None, "gold": "none"})                              # str(None) == "None"
    a({"id": "e04", "metric": "exact_match", "filter": "take_first", "filter_pattern": None,
       "response": 42, "gold": "42"})                                  # non-str scalar
    a({"id": "e05", "metric": "exact_match", "filter": "take_first", "filter_pattern": None,
       "response": [7, 8], "gold": "7"})                               # non-str list -> str(x)
    a({"id": "e06", "metric": "exact_match", "filter": "take_first", "filter_pattern": None,
       "response": ("t1", "t2"), "gold": "('t1', 't2')"})              # TUPLE is not a list!
    # -- C2 / regex extraction: no match, zero-group pattern, unparticipating group
    a({"id": "e07", "metric": "exact_match", "filter": "take_first",
       "filter_pattern": _ANSWER_RX, "response": "no answer here", "gold": ""})   # no match -> ""
    a({"id": "e08", "metric": "exact_match", "filter": "take_first",
       "filter_pattern": r"answer\s*is\s*[A-D]", "response": "the answer is C",
       "gold": "answer is c"})                                         # 0 groups -> group(0)
    a({"id": "e09", "metric": "exact_match", "filter": "take_first",
       "filter_pattern": r"(?:x)(A)?", "response": "x", "gold": "none"})  # (None,) is still truthy
    a({"id": "e10", "metric": "exact_match", "filter": "take_first",
       "filter_pattern": _ANSWER_RX, "response": [], "gold": ""})      # pattern + empty list
    # -- C5 / majority vote: exact ties must break lexicographically smallest
    a({"id": "e11", "metric": "exact_match", "filter": "majority_vote", "filter_pattern": None,
       "response": ["b", "a"], "gold": "a"})                           # 1-1 tie -> "a"
    a({"id": "e12", "metric": "exact_match", "filter": "majority_vote", "filter_pattern": None,
       "response": ["d", "c", "b", "a"], "gold": "a"})                 # 4-way tie -> "a"
    a({"id": "e13", "metric": "exact_match", "filter": "majority_vote", "filter_pattern": None,
       "response": [], "gold": ""})                                    # empty -> ""
    a({"id": "e14", "metric": "exact_match", "filter": "majority_vote",
       "filter_pattern": _ANSWER_RX, "response": ["nope", "nope"], "gold": ""})  # all no-match
    a({"id": "e15", "metric": "exact_match", "filter": "majority_vote", "filter_pattern": None,
       "response": "solo", "gold": "solo"})                            # scalar + majority
    # -- C1 / normalise: whitespace runs, tabs/newlines, unicode ws, case
    a({"id": "e16", "metric": "exact_match", "filter": "take_first", "filter_pattern": None,
       "response": "  MiXeD   \t Case \n Here  ", "gold": "mixed case here"})
    a({"id": "e17", "metric": "exact_match", "filter": "take_first", "filter_pattern": None,
       "response": "a b", "gold": "a b"})                         # NBSP: \s matches it
    a({"id": "e18", "metric": "exact_match", "filter": "take_first", "filter_pattern": None,
       "response": "a  b", "gold": "a b"})                   # thin space run
    a({"id": "e19", "metric": "exact_match", "filter": "take_first", "filter_pattern": None,
       "response": "a\x1cb", "gold": "a b"})                            # \x1c: \s matches, split too
    a({"id": "e20", "metric": "exact_match", "filter": "take_first", "filter_pattern": None,
       "response": "İSTANBUL", "gold": "i̇stanbul"})         # dotted-I lowercases to 2 cp
    # -- C3 / loglikelihood argmax: ties, empties, non-floats, NaN, big ints
    a({"id": "e21", "metric": "loglikelihood_acc", "choice_loglikelihoods": [], "gold_index": 0})
    a({"id": "e22", "metric": "loglikelihood_acc",
       "choice_loglikelihoods": [-1.0, -1.0, -2.0], "gold_index": 0})  # tie -> FIRST index
    a({"id": "e23", "metric": "loglikelihood_acc",
       "choice_loglikelihoods": [-1.0, -1.0, -2.0], "gold_index": 1})  # ...so this must be 0.0
    a({"id": "e24", "metric": "loglikelihood_acc",
       "choice_loglikelihoods": ("-1.0", "-3.0"), "gold_index": 0})    # STRINGS -> float() path
    a({"id": "e25", "metric": "loglikelihood_acc",
       "choice_loglikelihoods": [-3, -1, -2], "gold_index": 1})        # ints
    a({"id": "e26", "metric": "loglikelihood_acc",
       "choice_loglikelihoods": [True, False], "gold_index": 0})       # bool is not float
    a({"id": "e27", "metric": "loglikelihood_acc",
       "choice_loglikelihoods": (-1.5, -0.5), "gold_index": 1})        # tuple of floats
    a({"id": "e28", "metric": "loglikelihood_acc",
       "choice_loglikelihoods": [float("nan"), -1.0], "gold_index": 1})   # NaN -> ref path
    a({"id": "e29", "metric": "loglikelihood_acc",
       "choice_loglikelihoods": [2**53, 2**53 + 1], "gold_index": 1})  # float() collapses these
    a({"id": "e30", "metric": "loglikelihood_acc",
       "choice_loglikelihoods": [-1.0], "gold_index": -1})             # gold_index sentinel
    # -- metric dispatch: unknown, missing, non-str
    a({"id": "e31", "metric": "rouge_l", "response": "x", "gold": "x"})   # unknown -> 0.0
    a({"id": "e32", "response": "x", "gold": "x"})                        # missing -> exact_match
    a({"id": "e33", "metric": None, "response": "x", "gold": "x"})        # str(None) unknown -> 0.0
    a({"id": "e34", "metric": "contains", "filter": "take_first", "filter_pattern": None,
       "response": "abc", "gold": ""})                                    # "" in anything
    a({"id": "e35", "metric": "prefix_match", "filter": "take_first", "filter_pattern": None,
       "response": "", "gold": "x"})                                      # empty pred
    a({"id": "e36", "metric": "contains", "filter": "take_first", "filter_pattern": None,
       "response": "  PADDED  gold  ", "gold": "padded gold"})            # ws-collapse in contains
    # -- unknown filter name falls through to take-first
    a({"id": "e37", "metric": "exact_match", "filter": "weird_filter", "filter_pattern": None,
       "response": ["first", "second"], "gold": "first"})
    # -- ASCII-but-not-portable whitespace, ISOLATED per code path.
    #    A vectorised scorer decides per COLUMN, so a hazard sharing a bucket with another hazard is
    #    masked: the bucket falls back for the wrong reason and the guard is never really tested.
    #    e16-e20 are all in the (exact_match, take_first, None) bucket, so \x1c there proves nothing.
    #    These four put ONE hazard into a bucket whose other members are plain printable ASCII, one
    #    per path: normalise-only, contains, regex extraction, majority-vote extraction.
    a({"id": "e38", "metric": "prefix_match", "filter": "take_first", "filter_pattern": None,
       "response": "a\x1cb", "gold": "a b"})          # python \s splits \x1c, RE2 \s does not
    a({"id": "e39", "metric": "contains", "filter": "take_first", "filter_pattern": None,
       "response": "a\x0bb", "gold": "a b"})          # VT: python ws, RE2 not
    a({"id": "e40", "metric": "exact_match", "filter": "take_first", "filter_pattern": _ANSWER_RX,
       "response": "answer\x1cis\x1cB", "gold": "b"})  # python re MATCHES here, RE2 does not
    a({"id": "e41", "metric": "exact_match", "filter": "majority_vote",
       "filter_pattern": _ANSWER_RX, "response": ["answer\x1cis\x1cA", "answer is A"],
       "gold": "a"})                                  # same, on the flattened majority column
    return c


# -----------------------------------------------------------------------------
# 4. Harness-protocol timing + equivalence driver
# -----------------------------------------------------------------------------
def load_module(filename):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_scorer(filename, cls_name=None, device="cpu"):
    """Instantiate a scorer.

    `cls_name=None` uses the file's own `load_scoring_pipeline_for_verification`, i.e. exactly what
    the verifier will get. A class name pins one specific scorer inside a multi-scorer file --
    negative_scorer.py deliberately ships TWO negatives (a slow-but-correct one that proves the
    SPEED gradient, and a row-skipping one that proves the consistency gate bites), and its loader
    returns the skipping one. Timing the skipping scorer would be meaningless and asserting
    equivalence on it would be wrong, so both are addressed by class here.
    """
    mod = load_module(filename)
    if cls_name is None:
        return mod.load_scoring_pipeline_for_verification(device)
    return getattr(mod, cls_name)(device=device)


def as_map(out):
    m = {}
    for row in out:
        m[str(row["id"])] = float(row["score"])
    return m


def check_equiv(name, pipe, samples, ref, atol=1e-9):
    got = as_map(pipe.score([dict(s) for s in samples]))
    bad = []
    for rid, rv in ref.items():
        if rid not in got:
            bad.append((rid, rv, "<MISSING>"))
        elif abs(got[rid] - rv) > atol:
            bad.append((rid, rv, got[rid]))
    extra = [k for k in got if k not in ref]
    return bad, extra


def time_scoring(pipe, samples, repeats):
    times = []
    for _ in range(repeats):
        payload = [dict(s) for s in samples]
        t0 = time.perf_counter()
        pipe.score(payload)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2], times


def fuzz_fastmatch():
    """Fuzz ceiling_fastmatch's specialised matcher against `re` itself.

    Two outcomes are acceptable per pattern: it is REJECTED (the scorer then uses `re`, which is
    correct by definition), or it is ACCEPTED and then must agree with `re` on EVERY probe string.
    Anything else is a silent wrong-answer bug, which under min_consistency_fraction=1.0 means the
    ceiling is worthless -- so this is checked directly rather than inferred from corpus scores.
    """
    fm = load_module("ceiling_fastmatch.py")
    patterns = [
        r"answer\s*is\s*([A-D])",            # the hot held-out pattern
        r"answer\s*is\s*[A-D]",              # same, zero groups -> group(0)
        r"answer is ([A-D])",                # no \s at all
        r"the\s+answer\s*is\s*([a-dA-D0-9])",
        r"x\s*([AB])",
        r"answer\s*\s+is\s*([A-D])",         # adjacent \s tokens -> must be rejected
        r"(?:x)(A)?",                        # optional group -> rejected
        r"ans.*?([A-D])",                    # dot/quantifier -> rejected
        r"(a|b)([A-D])",                     # alternation + 2 groups -> rejected
        r"[^A-D]",                           # negated class -> rejected
        r"answer\s*is\s*([A-D])x",           # class not last -> rejected
        r"answer\s*is\s*([A-D\s])",          # class contains whitespace -> rejected
        r"answer\s*is\s*(\w)",               # escape in group -> rejected
    ]
    probes = [
        "", "x", "answer", "answer is", "answer is A", "answer  is   B",
        "answer\tis\nC", "answer is Z", "answerisD", "answer is a",
        "the answer is D.", "noise answer answer is B tail",
        "answer is answer is C", "answer is  ", "answer is A",
        "answer is B", "answer is\x1cC", "ANSWER IS A", "the answer is 3",
        "answer is A answer is B", "aanswer is A", "answer\n\n\nis\n\n\nD",
    ]
    accepted, rejected, bad = [], [], []
    for pat in patterns:
        fast, has_group = fm.compile_fast(pat)
        if fast is None:
            rejected.append(pat)
            continue
        accepted.append(pat)
        rx = re.compile(pat)
        for p in probes:
            m = rx.search(p)
            got = fast(p)
            want = None if m is None else (m.group(0), m.group(1) if has_group else None)
            gotn = None if got is None else (got[0], got[1] if has_group else None)
            if want != gotn:
                bad.append((pat, p, want, gotn))
    return accepted, rejected, bad


def fuzz_arrow():
    """Fuzz ceiling_arrow's RE2 rewrite + the all-ASCII guard that makes the C++ path legal.

    Two independent things have to hold, and they fail in different ways:

      1. The RE2 REWRITE. `answer\\s*is\\s*([A-D])` becomes `answer\\s*is\\s*(?P<g1>[A-D])` so that
         pyarrow's `extract_regex` (which only returns NAMED groups) can be used. A botched rewrite
         is a silent wrong-answer bug, so every accepted pattern is compared against Python's own
         `re` on every probe -- exactly as ceiling_arrow does internally before it trusts a pattern.

      2. The ASCII GUARD. This is the one that actually protects the scores. RE2's `\\s` is ASCII
         only, and `utf8_lower` is not `str.lower()` (Greek final sigma; U+0130 lowercases to TWO
         codepoints). On a non-ASCII column the C++ path would diverge -- so `_ascii_ok` must return
         False for exactly those columns. We assert both directions here rather than hoping the edge
         corpus happens to cover it.
    """
    ar = load_module("ceiling_arrow.py")
    if not getattr(ar, "_HAVE_ARROW", False):
        return None
    patterns = [
        r"answer\s*is\s*([A-D])",            # the hot held-out pattern
        r"the\s+answer\s*is\s*([a-dA-D0-9])",
        r"x\s*([AB])",
        r"answer\s*is\s*[A-D]",              # zero groups -> declined (no RE2 group(0) accessor)
        r"(?:x)(A)?",                        # optional group -> rejected
        r"ans.*?([A-D])",                    # dot/quantifier -> rejected
        r"(a|b)([A-D])",                     # alternation -> rejected
        r"[^A-D]",                           # negated class -> rejected
        r"answer\s*is\s*([A-D])x",           # class not last -> rejected
        r"answer\s*is\s*(\w)",               # escape in class -> rejected
    ]
    probes = [
        "", "x", "answer", "answer is", "answer is A", "answer  is   B", "answer\tis\nC",
        "answer is Z", "answerisD", "the answer is D.", "noise answer answer is B tail",
        "answer is answer is C", "aanswer is A", "answer\n\n\nis\n\n\nD", "ANSWER IS A",
        "the answer is 3", "answer is A answer is B", "x A", "xA", "x  B",
    ]
    accepted, rejected, bad = [], [], []
    for pat in patterns:
        re2, has_group = ar.arrow_pattern(pat)
        if re2 is None or not ar.verify_arrow_pattern(pat, re2, has_group, probes):
            rejected.append(pat)
            continue
        accepted.append((pat, re2))
        rx = re.compile(pat)
        got = ar._pc.extract_regex(ar._pa.array(probes, type=ar._pa.string()), re2).to_pylist()
        for probe, row in zip(probes, got):
            m = rx.search(probe)
            want = None if m is None else (m.group(1) if has_group else m.group(0))
            gotv = None if row is None else row.get("g1")
            if want != gotv:
                bad.append((pat, probe, want, gotv))

    # The guard MUST refuse every column where python-str and arrow-utf8 semantics can differ, or
    # the C++ normalise silently changes scores. The ASCII CONTROLS are the interesting half:
    # `string_is_ascii` says yes to all of them, yet python `\s`/`str.strip()` and RE2 `\s` disagree
    # about which ones are whitespace -- so an ascii-only guard is NOT sufficient.
    guard = []
    for label, col, expect_ascii in (
        ("printable ascii", ["answer is A", "the answer is B"], True),
        ("ascii + tab/nl/ff/cr", ["a\tb\nc\fd\re"], True),   # the 5 both engines agree on
        ("VT \\x0b", ["a\x0bb"], False),                     # python \s yes, RE2 \s NO
        ("NUL \\x00", ["a\x00b"], False),                    # conservative: neither calls it ws
        ("DEL \\x7f", ["a\x7fb"], False),                    # conservative decline
        ("one bad row in 65", ["ok"] * 64 + ["a\x1cb"], False),   # must scan the WHOLE column
        ("NBSP", ["a b"], False),                       # python \s matches it, RE2 does not
        ("thin space", ["a b"], False),
        ("\\x1c group sep", ["a\x1cb"], False),              # matched by python \s AND by str.split
        ("\\x1f unit sep", ["a\x1fb"], False),
        ("NEL \\x85", ["a\x85b"], False),
        ("dotted capital I", ["İSTANBUL"], False),      # lowercases to TWO codepoints
        ("greek final sigma", ["ΣOΣ"], False),     # str.lower() is context sensitive
    ):
        arr = ar._pa.array(col, type=ar._pa.string())
        got_ok = ar.ArrowVectorisedScorer._ascii_ok(arr)
        guard.append((label, expect_ascii, got_ok, got_ok == expect_ascii))
    return accepted, rejected, bad, guard


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20000, help="held-out size (env builder uses 20000)")
    ap.add_argument("--repeats", type=int, default=5, help="timing_repeats from the manifest")
    ap.add_argument("--mix", choices=("stub", "cot"), default="stub",
                    help="response length mix: stub = the originally-built 80-byte responses, "
                         "cot = the realistic heavy-tailed chain-of-thought mix")
    ap.add_argument("--from-jsonl", default=None, metavar="PATH",
                    help="score the records in PATH (e.g. the heldout_samples.jsonl that "
                         "build_dataset.py actually emitted) instead of the synthetic proxy corpus")
    args = ap.parse_args()

    # (file, class, label). Class is pinned explicitly because negative_scorer.py ships TWO
    # negatives with OPPOSITE expected outcomes and its loader returns the skipping one.
    must_match = [
        ("../environment/workspace/scoring_pipeline_template.py", "NaiveScoringPipeline",
         "template (naive)   "),
        ("negative_scorer.py", "SlowButCorrectScorer", "negative (slow)    "),
        ("scoring_pipeline_ref.py", "GroupedVectorisableScorer", "anchor  (1.0)      "),
        ("ceiling_scorer.py", "MemoisedFastScorer", "CEILING memoised   "),
        ("ceiling_columnar.py", "ColumnarBucketScorer", "CEILING columnar   "),
        ("ceiling_fastmatch.py", "FastMatchColumnarScorer", "CEILING fastmatch  "),
        ("ceiling_arrow.py", "ArrowVectorisedScorer", "CEILING arrow      "),
    ]
    # This one MUST diverge. If it ever matches, the negative has stopped being a negative and the
    # the consistency-gate control is dead -- that is a FAILURE of this selftest, not a pass.
    must_diverge = [("negative_scorer.py", "SkippingScorer", "negative (skipping)")]

    missing = [t for t in must_match if not (HERE / t[0]).exists()]
    for fname, _, label in missing:
        print(f"  [SKIP] {label} -- {fname} not reachable from {HERE}")
    must_match = [t for t in must_match if (HERE / t[0]).exists()]

    if args.from_jsonl:
        real = corpus_from_jsonl(args.from_jsonl, args.samples)
        src = f"jsonl {args.from_jsonl}"
    else:
        real = corpus_real(args.samples, mix=args.mix)
        src = f"synthetic mix={args.mix}"
    edge = corpus_edge()
    resp_chars = [len(r["response"]) if isinstance(r.get("response"), str)
                  else sum(len(x) for x in r["response"])
                  for r in real if r.get("response") is not None]
    print(f"corpus REAL: {len(real)} rows   corpus EDGE: {len(edge)} rows   repeats: {args.repeats}")
    print(f"response bytes per row ({src}): "
          f"mean {statistics.mean(resp_chars):.0f}  max {max(resp_chars)}")
    # the arrow/numpy ceiling is only a headroom number if those libraries are actually here; print
    # the versions so a reader can tell a real measurement from a silent fallback
    env = [f"python {sys.version.split()[0]}"]
    for modname in ("numpy", "pyarrow"):
        try:
            env.append(f"{modname} {__import__(modname).__version__}")
        except Exception:
            env.append(f"{modname} ABSENT")
    print("env: " + "   ".join(env))

    # ---- (a0) the default entry point must hand back the class we are about to time -----------
    print("\n=== (a0) default loader identity (what the harness actually seeds) ===")
    for fname, cls_name in (("scoring_pipeline_ref.py", "GroupedVectorisableScorer"),
                            ("ceiling_scorer.py", "MemoisedFastScorer"),
                            ("ceiling_columnar.py", "ColumnarBucketScorer"),
                            ("ceiling_fastmatch.py", "FastMatchColumnarScorer"),
                            ("ceiling_arrow.py", "ArrowVectorisedScorer"),
                            ("../environment/workspace/scoring_pipeline_template.py", "NaiveScoringPipeline"),
                            ("negative_scorer.py", "SkippingScorer")):
        if not (HERE / fname).exists():
            print(f"  [SKIP] {fname} not reachable from {HERE}")
            continue
        got = type(load_scorer(fname, None)).__name__
        ok = got == cls_name
        print(f"  [{'OK' if ok else 'FAIL'}] {fname}: loader -> {got} (expected {cls_name})")
        if not ok:
            print("\n🔴 loader/class drift -- the timed object is not the one the harness seeds.")
            return 1

    # ---- (a) equivalence, on BOTH corpora -----------------------------------
    print("\n=== (a) EQUIVALENCE vs harness reference (gate needs 1.0 exact) ===")
    failed = False
    for fname, cls_name, label in must_match:
        pipe = load_scorer(fname, cls_name)
        for cname, corpus in (("REAL", real), ("EDGE", edge)):
            ref = reference_scores(corpus)
            bad, extra = check_equiv(label, pipe, corpus, ref)
            status = "OK" if (not bad and not extra) else "FAIL"
            if bad or extra:
                failed = True
            print(f"  [{status}] {label} {cname}: {len(ref) - len(bad)}/{len(ref)} match, "
                  f"{len(bad)} divergent, {len(extra)} extra ids")
            for rid, want, got in bad[:5]:
                row = next(r for r in corpus if str(r["id"]) == rid)
                print(f"        id={rid} want={want} got={got}  row={row}")
            for k in extra[:5]:
                print(f"        EXTRA id={k}")

    # ---- (a2) the consistency-gate negative must NOT be equivalent ----------
    print("\n=== (a2) consistency-gate negative must DIVERGE (else the control is dead) ===")
    for fname, cls_name, label in must_diverge:
        pipe = load_scorer(fname, cls_name)
        ref = reference_scores(real)
        bad, extra = check_equiv(label, pipe, real, ref)
        diverged = bool(bad or extra)
        print(f"  [{'OK' if diverged else 'FAIL'}] {label} REAL: {len(bad)} divergent / "
              f"{len(extra)} extra -> {'gate will fire (reward 0)' if diverged else 'GATE WOULD PASS'}")
        if not diverged:
            failed = True

    if failed:
        print("\n🔴 EQUIVALENCE FAILED -- fix the scorer before running a full scoring pass.")
        return 1
    print("  -> every must-match scorer exactly reproduces the reference; the skipping negative"
          " diverges as designed.")

    # ---- (a3) fuzz the specialised pattern matcher against `re` ---------------
    if (HERE / "ceiling_fastmatch.py").exists():
        print("\n=== (a3) FUZZ ceiling_fastmatch's specialised matcher vs the `re` module ===")
        accepted, rejected, bad = fuzz_fastmatch()
        print(f"  specialised {len(accepted)} pattern(s), fell back on {len(rejected)}, "
              f"{len(bad)} disagreement(s)")
        for pat in accepted:
            print(f"        FAST  {pat!r}")
        for pat in rejected:
            print(f"        re    {pat!r}")
        for pat, probe, want, got in bad[:8]:
            print(f"        🔴 {pat!r} on {probe!r}: re={want!r} fast={got!r}")
        if bad or not accepted:
            print("\n🔴 the specialised matcher is wrong (or specialises nothing) -- probe #3 void.")
            return 1

    # ---- (a4) fuzz the RE2 rewrite AND the all-ASCII guard --------------------
    if (HERE / "ceiling_arrow.py").exists():
        print("\n=== (a4) ceiling_arrow: RE2 rewrite vs `re`, and the all-ASCII guard ===")
        res = fuzz_arrow()
        if res is None:
            print("  [SKIP] pyarrow not importable here -- ArrowVectorisedScorer degrades to the")
            print("         verbatim reference, so its timing row below is NOT a headroom number.")
        else:
            accepted, rejected, bad, guard = res
            print(f"  vectorised {len(accepted)} pattern(s), declined {len(rejected)}, "
                  f"{len(bad)} disagreement(s) vs `re`")
            for pat, re2 in accepted:
                print(f"        RE2   {pat!r} -> {re2!r}")
            for pat in rejected:
                print(f"        re    {pat!r}")
            for pat, probe, want, got in bad[:8]:
                print(f"        🔴 {pat!r} on {probe!r}: re={want!r} arrow={got!r}")
            print("  the guard that makes utf8_lower/RE2-\\s legal (must refuse every non-ASCII col):")
            for label, expect, got, ok in guard:
                print(f"        [{'OK' if ok else 'FAIL'}] {label:<20} ascii_ok={got} "
                      f"(expected {expect})")
            if bad or not accepted or not all(g[3] for g in guard):
                print("\n🔴 arrow rewrite wrong, nothing vectorised, or the ASCII guard leaks --"
                      " probe #4 void.")
                return 1

    print(f"\n=== (b) HEADROOM: median of {args.repeats} full-set passes over {len(real)} rows ===")
    med = {}
    for fname, cls_name, label in must_match:
        pipe = load_scorer(fname, cls_name)
        m, allt = time_scoring(pipe, real, args.repeats)
        med[label.strip()] = m
        print(f"  {label} median {m*1000:9.2f} ms   (min {min(allt)*1000:.2f} / max {max(allt)*1000:.2f})")

    anchor = med["anchor  (1.0)"]
    print("\n  speedup vs the 1.0 anchor (this is the DoD item-4 number):")
    for _, _, label in must_match:
        print(f"    {label} {anchor / med[label.strip()]:6.3f}x")

    # ---- (c) the absolute wall: what the .score() contract costs on its own ---
    floor_pipe = load_scorer("ceiling_columnar.py", "ContractFloorScorer")
    fm, fallt = time_scoring(floor_pipe, real, args.repeats)
    print(f"\n=== (c) ABSOLUTE WALL (instrument, returns WRONG scores -- not a submission) ===")
    print(f"  contract floor      median {fm*1000:9.2f} ms   (min {min(fallt)*1000:.2f})"
          f"  -> {anchor / fm:6.3f}x")
    print("  That row is `[{'id': s['id'], 'score': 0.0} for s in samples]`: walk every input dict,")
    print("  read every id, allocate every output dict. NO correct implementation -- numpy, Cython,")
    print("  a rewritten /app/repo -- can go faster, because the .score() contract itself costs")
    print("  that much. So it is a HARD upper bound on the achievable speedup, and the honest")
    print("  headroom question is: how much of the gap between the anchor and that wall is left")
    print("  unclaimed by the best ceiling above?")
    best_label = min(must_match, key=lambda t: med[t[2].strip()])[2].strip()
    best = med[best_label]
    gap_total = anchor - fm
    gap_left = best - fm
    print(f"\n  anchor {anchor*1000:.2f} ms -> wall {fm*1000:.2f} ms  = {gap_total*1000:.2f} ms of"
          f" reducible work")
    print(f"  best ceiling ({best_label}) {best*1000:.2f} ms claims"
          f" {(gap_total - gap_left)/gap_total*100:5.1f}% of it,"
          f" leaving {gap_left*1000:.2f} ms ({gap_left/gap_total*100:.1f}%) on the table")
    print(f"  => solver-reachable speedup band: {anchor/best:.3f}x (measured, stdlib) .."
          f" {anchor/fm:.3f}x (absolute wall)")

    print("\n  READING THIS: the anchor is 1.000x by definition. `template` and `negative (slow)`")
    print("  must be < 1.0 (that is the gradient floor a solver starts from) and a CEILING must be")
    print("  >> 1.0 (that is the headroom). Judge the task on the BAND, not on one row: if the")
    print("  absolute wall itself is close to 1.0, no amount of solver cleverness can win and the")
    print("  task belongs in the route_down pile no matter how good the reference solution is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
