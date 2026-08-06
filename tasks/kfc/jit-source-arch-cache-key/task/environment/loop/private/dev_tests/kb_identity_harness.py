"""Shared harness for the build-spec identity task (CPU, pure Python).

Provides: the candidate loader, the equivalence-rule tables (source normalization,
no-op option tokens), a deterministic *labeled* workload of build-specs grouped into
true equivalence classes, and helpers that turn a candidate ``identity_key`` into a
partition of that workload so the verifier can count distinct identities and detect a
false merge (two genuinely-different specs sharing an identity). Standard library
only, so the metric -- the number of distinct identities produced over the workload --
is a deterministic count that is identical on any machine.

The ground truth is the class LABEL attached to every spec when the workload is built:
two specs share a class iff they were generated as equivalent spellings of the same
base. candidate / baseline / oracle are all judged against these labels, never against
each other.
"""

import hashlib
import importlib.util
import os
import re


# --------------------------------------------------------------------------- #
# Equivalence rules (also disclosed to the solver in instruction.md).
# --------------------------------------------------------------------------- #
# Source text: comments and surrounding whitespace do not change the built artifact,
# so they are incidental. Everything else in the source text is meaningful.
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")

# Option tokens equal to a build default; each is equivalent to the token being absent.
NOOP_OPTIONS = {"o0", "dbgoff", "std0"}


def canon_source(text):
    """Canonical form of a source text: strip comments, collapse whitespace runs."""
    text = _BLOCK_COMMENT.sub(" ", text)
    text = _LINE_COMMENT.sub(" ", text)
    return " ".join(text.split())


def canon_options(options):
    """Canonical option set: drop no-op defaults, de-duplicate, sort (order-free)."""
    kept = {o for o in (options or []) if o not in NOOP_OPTIONS}
    return sorted(kept)


def _with_line_comment(source):
    return source + "   // tuned build\n"


def _with_block_comment(source):
    return "/* rev2 */\n" + source + "\n\n  "


def repo_dir():
    return os.environ.get("KB_REPO_DIR", "/app/repo")


def load_candidate():
    """Import the candidate module from the repo under test."""
    path = os.path.join(repo_dir(), "identity_key.py")
    spec = importlib.util.spec_from_file_location("candidate_identity_key", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "identity_key"):
        raise AttributeError(f"{path} does not define identity_key")
    return mod


def load_module(path):
    # deterministic module name derived from the path (no salted builtin hash()).
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location("kb_identity_mod_" + digest, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Labeled workload: 12 true classes, each emitted as 4 equivalent spellings.
# Distinct classes differ in at least one meaning-bearing field (source text /
# target profile / effective option / toolchain tag / requested variant set).
# classes 0 and 1 differ ONLY in target profile -- a normalizer that drops the target
# merges them, which is a false merge (a wrong-target artifact would be reused).
# --------------------------------------------------------------------------- #
SRC_A = "o[i] = a[i] * b[i];"
SRC_B = "o[i] = a[i] + b[i];"
SRC_C = "o[i] = a[i] - b[i];"
SRC_D = "acc = acc + a[k] * b[k];"
SRC_E = "o[i] = a[i] > 0 ? a[i] : 0;"
SRC_F = "o[i] = a[i] / b[i];"

# base = (source, target, effective_options, toolchain, variants)
CLASS_BASES = [
    (SRC_A, "p80", (),             "v1", ()),           # 0  (target-trap A)
    (SRC_A, "p90", (),             "v1", ()),           # 1  (target-trap B: only target differs from 0)
    (SRC_B, "p80", (),             "v1", ()),           # 2  distinct source
    (SRC_C, "p80", (),             "v1", ()),           # 3
    (SRC_C, "p80", ("fast",),      "v1", ()),           # 4  effective option vs 3
    (SRC_D, "p80", ("o3",),        "v1", ()),           # 5
    (SRC_D, "p80", ("o2",),        "v1", ()),           # 6  different effective option value vs 5
    (SRC_D, "p80", ("o3",),        "v2", ()),           # 7  different toolchain tag vs 5
    (SRC_E, "p80", (),             "v1", ()),           # 8
    (SRC_E, "p80", (),             "v1", ("kA",)),       # 9  variant present vs 8
    (SRC_E, "p80", (),             "v1", ("kB", "kC")),  # 10 different variant set vs 9
    (SRC_F, "p86", ("fast", "o3"), "v1", ()),           # 11 third target + two options
]


def _spellings(base):
    """Return four equivalent-but-differently-spelled specs for one class.

    All four normalize to the same spec (they differ only in source comments/whitespace,
    option order, redundant no-op options, incidental build annotations, and requested
    variant order), yet all four are byte-distinct so a conservative identity produces
    four identities.
    """
    source, target, options, toolchain, variants = base
    opts = list(options)
    vs = list(variants)

    def build_spec(src, opt_list, build=None, var_list=None):
        d = {"source": src, "target": target, "options": list(opt_list), "toolchain": toolchain}
        if var_list:
            d["variants"] = list(var_list)
        if build is not None:
            d["build"] = build
        return d

    # S0: canonical source, options sorted, variants sorted, no build annotations.
    s0 = build_spec(source, sorted(opts), var_list=sorted(vs))
    # S1: source carries a trailing line comment + reflowed spacing (ignored).
    s1 = build_spec(_with_line_comment(source), sorted(opts), var_list=sorted(vs))
    # S2: options in a different ORDER + a redundant no-op token; a non-semantic build
    #     annotation block; requested variants in a different order.
    s2 = build_spec(source, list(reversed(opts)) + ["o0"],
                    build={"tmpdir": "/tmp/b_" + target, "ts": 1700000000},
                    var_list=list(reversed(vs)))
    # S3: source with a block comment + extra newlines; options reversed with a DUPLICATE
    #     of the first option (when present) + a different no-op token; a different build.
    dup = [opts[0]] if opts else []
    s3 = build_spec(_with_block_comment(source), list(reversed(opts)) + dup + ["dbgoff"],
                    build={"tmpdir": "/tmp/x", "ts": 1700000001, "host": "node7"},
                    var_list=list(reversed(vs)))
    return [s0, s1, s2, s3]


def build_labeled_workload():
    """Deterministic list of ``(class_id, spec)`` pairs (48 = 12 classes x 4)."""
    workload = []
    for class_id, base in enumerate(CLASS_BASES):
        for spec in _spellings(base):
            workload.append((class_id, spec))
    return workload


def true_class_count(workload):
    return len({class_id for class_id, _ in workload})


# --------------------------------------------------------------------------- #
# Turn a candidate identity function into a partition of the workload.
# --------------------------------------------------------------------------- #
def key_to_classes(module, workload):
    """Map each produced identity -> the set of true class ids that landed on it."""
    mapping = {}
    for class_id, spec in workload:
        key = module.identity_key(spec)
        mapping.setdefault(key, set()).add(class_id)
    return mapping


def find_false_merges(module, workload):
    """Return {identity: {class_ids}} for identities shared by >1 true class.

    A non-empty result is a correctness failure: two genuinely different specs were
    given the same identity.
    """
    return {k: v for k, v in key_to_classes(module, workload).items() if len(v) > 1}


def count_distinct_keys(module, workload):
    """Number of distinct identities the candidate produces over the workload."""
    return len({module.identity_key(spec) for _, spec in workload})
