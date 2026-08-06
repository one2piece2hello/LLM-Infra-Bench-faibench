"""Distinct-identity count for the build-spec identity task.

The metric is a deterministic count: over a fixed labeled workload of build-specs,
how many DISTINCT identities does a module's ``identity_key`` produce? A conservative
identity treats every incidental spelling as different and emits many identities; a
normalizing identity collapses the equivalent spellings of each class down toward the
number of true classes. The score is the ratio baseline_count / candidate_count, so
fewer identities (for the same workload) is better -- but only while no two
genuinely-different specs share an identity.

Invoked one MODE per process. ``test.sh`` runs candidate and the frozen baseline and
takes the ratio. No profiler / no numerics library -- it is a pure count.

Modes:
  candidate  count distinct identities from /app/repo/identity_key.py; print COUNT=<n>
  baseline   same, using the frozen baseline module (KB_BASELINE_MODULE)
  verify     (safety) confirm the candidate NEVER gives two different-class specs
             the same identity; print VERIFY_OK / VERIFY_FAIL
"""

import os
import sys

from kb_identity_harness import (
    build_labeled_workload,
    count_distinct_keys,
    find_false_merges,
    load_candidate,
    load_module,
    true_class_count,
)


def _baseline_module():
    path = os.environ.get("KB_BASELINE_MODULE", "/opt/verifier-baseline/identity_key.py")
    return load_module(path)


def mode_candidate():
    workload = build_labeled_workload()
    print(f"COUNT={count_distinct_keys(load_candidate(), workload)}")
    return 0


def mode_baseline():
    workload = build_labeled_workload()
    print(f"COUNT={count_distinct_keys(_baseline_module(), workload)}")
    return 0


def mode_verify():
    workload = build_labeled_workload()
    classes = true_class_count(workload)
    cand = load_candidate()
    base = _baseline_module()

    cand_bad = find_false_merges(cand, workload)
    if cand_bad:
        sample = sorted((sorted(v), k) for k, v in cand_bad.items())[:3]
        print(f"VERIFY_FAIL candidate false-merges distinct classes: {sample}")
        return 1
    # sanity: the frozen baseline must itself be false-merge free.
    base_bad = find_false_merges(base, workload)
    if base_bad:
        print(f"VERIFY_FAIL baseline unexpectedly false-merges: {sorted(base_bad.values())[:3]}")
        return 1

    cand_count = count_distinct_keys(cand, workload)
    base_count = count_distinct_keys(base, workload)
    # work-evidence: the baseline emits (base_count - classes) identities more than
    # there are true classes; a full normalizer removes those redundant identities.
    redundant = base_count - classes
    print(f"VERIFY_OK signatures={len(workload)} true_classes={classes} "
          f"candidate_identities={cand_count} baseline_identities={base_count} "
          f"baseline_redundant_identities={redundant}")
    return 0


MODES = {"candidate": mode_candidate, "baseline": mode_baseline, "verify": mode_verify}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "candidate"
    fn = MODES.get(mode)
    if fn is None:
        print(f"unknown mode {mode!r}; expected one of {sorted(MODES)}")
        sys.exit(2)
    sys.exit(fn())


if __name__ == "__main__":
    main()
