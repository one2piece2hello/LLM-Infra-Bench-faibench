"""Shared harness for the request-signature identity task (CPU, pure Python).

Provides: the candidate loader, the equivalence-rule tables (dtype-spelling
aliases, order-insensitive ops, flag defaults), a deterministic *labeled* workload
of signatures grouped into true equivalence classes, and helpers that turn a
candidate ``identity_key`` into a partition of that workload so the verifier can
count distinct identities and detect a false merge (two genuinely-different
signatures sharing an identity). Standard library only, so the metric -- the number
of distinct identities produced over the workload -- is a deterministic count that
is identical on any machine.

The ground truth is the class LABEL attached to every signature when the workload
is built: two signatures share a class iff they were generated as equivalent
spellings of the same base. candidate / baseline / oracle are all judged against
these labels, never against each other.
"""

import hashlib
import importlib.util
import os


# --------------------------------------------------------------------------- #
# Equivalence rules (also disclosed to the solver in instruction.md).
# --------------------------------------------------------------------------- #
# Alternate spellings of one logical element type; the first entry is the
# normalized form. Any spelling in a group is equivalent to any other.
DTYPE_ALIASES = {
    "f32": ["f32", "float32", "single"],
    "f16": ["f16", "float16", "half"],
    "i32": ["i32", "int32", "int"],
    "bf16": ["bf16", "bfloat16"],
}
# spelling -> normalized element type
DTYPE_CANON = {sp: canon for canon, group in DTYPE_ALIASES.items() for sp in group}

# operations whose operand ORDER carries no meaning (order-insensitive); every
# other op is order-sensitive and its operand order must be preserved.
COMMUTATIVE = {"add", "mul", "max", "min"}

# flags whose value equals the default are equivalent to the flag being absent.
FLAG_DEFAULTS = {"fastmath": False, "layout": "row_major", "precision": "default"}


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
# Distinct classes differ in at least one meaning-bearing field (op / operand
# shape / normalized dtype / order for order-sensitive ops / a non-default flag).
# classes 0 and 1 differ ONLY in operand dtype (f32 vs f16) -- a normalizer that
# drops dtype merges them, which is a false merge (correctness failure).
# --------------------------------------------------------------------------- #
# base = (op, [ (shape, normalized_dtype), ... ], {non_default_flags})
CLASS_BASES = [
    ("add",    [([256], "f32"), ([256], "f32")], {}),                       # 0  (dtype-trap A)
    ("add",    [([256], "f16"), ([256], "f16")], {}),                       # 1  (dtype-trap B)
    ("mul",    [([128], "f32"), ([128], "f32")], {}),                       # 2
    ("matmul", [([128, 256], "f32"), ([256, 512], "f32")], {}),             # 3
    ("matmul", [([64, 64], "f16"), ([64, 64], "f16")], {"precision": "high"}),  # 4
    ("conv",   [([1, 3, 224, 224], "f32"), ([64, 3, 7, 7], "f32")], {"layout": "channels_last"}),  # 5
    ("sub",    [([512], "f32"), ([512], "f32")], {}),                       # 6
    ("max",    [([100], "i32"), ([100], "i32")], {}),                       # 7
    ("add",    [([256], "f32"), ([256], "f32")], {"fastmath": True}),       # 8  (non-default flag vs 0)
    ("matmul", [([256, 512], "f32"), ([128, 256], "f32")], {}),             # 9  (operands of 3 reversed)
    ("mul",    [([128], "f32"), ([256], "f32")], {}),                       # 10
    ("div",    [([32], "bf16"), ([32], "bf16")], {}),                       # 11
]


def _alias(dtype, i):
    group = DTYPE_ALIASES[dtype]
    return group[i % len(group)]


def _spellings(base):
    """Return four equivalent-but-differently-spelled signatures for one class.

    All four normalize to the same signature (they differ only in dtype spelling,
    redundant default flags, incidental annotations, and mapping-field order), yet
    all four are byte-distinct so a conservative identity produces four identities.
    """
    op, operands, flags = base

    def op_list(alt_index_by_pos=None):
        alt_index_by_pos = alt_index_by_pos or {}
        out = []
        for pos, (shape, dtype) in enumerate(operands):
            out.append({"shape": list(shape), "dtype": _alias(dtype, alt_index_by_pos.get(pos, 0))})
        return out

    # S0: normalized dtype spellings, only the non-default flags, no annotations.
    s0 = {"op": op, "operands": op_list(), "flags": dict(flags)}
    # S1: an alternate spelling of the first operand's dtype.
    s1 = {"op": op, "operands": op_list({0: 1}), "flags": dict(flags)}
    # S2: default-valued flags written out explicitly + an annotation block.
    f2 = dict(FLAG_DEFAULTS)
    f2.update(flags)
    s2 = {"op": op, "operands": op_list(), "flags": f2, "meta": {"note": "a"}}
    # S3: an alternate spelling of the last operand's dtype + a different annotation.
    s3 = {"op": op, "operands": op_list({len(operands) - 1: 1}), "flags": dict(flags),
          "meta": {"note": "b", "seq": 3}}
    return [s0, s1, s2, s3]


def build_labeled_workload():
    """Deterministic list of ``(class_id, signature)`` pairs (48 = 12 classes x 4)."""
    workload = []
    for class_id, base in enumerate(CLASS_BASES):
        for sig in _spellings(base):
            workload.append((class_id, sig))
    return workload


def true_class_count(workload):
    return len({class_id for class_id, _ in workload})


# --------------------------------------------------------------------------- #
# Turn a candidate identity function into a partition of the workload.
# --------------------------------------------------------------------------- #
def key_to_classes(module, workload):
    """Map each produced identity -> the set of true class ids that landed on it."""
    mapping = {}
    for class_id, sig in workload:
        key = module.identity_key(sig)
        mapping.setdefault(key, set()).add(class_id)
    return mapping


def find_false_merges(module, workload):
    """Return {identity: {class_ids}} for identities shared by >1 true class.

    A non-empty result is a correctness failure: two genuinely different signatures
    were given the same identity.
    """
    return {k: v for k, v in key_to_classes(module, workload).items() if len(v) > 1}


def count_distinct_keys(module, workload):
    """Number of distinct identities the candidate produces over the workload."""
    return len({module.identity_key(sig) for _, sig in workload})
