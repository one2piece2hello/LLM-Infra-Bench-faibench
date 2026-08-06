"""Correctness suite for the build-spec identity contract.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>". The runner counts
CASE_PASS lines. Pure standard library.

Every case tests a *safety* / determinism property that any acceptable identity must
satisfy: it returns a stable string, and it NEVER gives two genuinely-different specs
the same identity. (Collapsing equivalent spellings is rewarded by the distinct-identity
count, not gated here -- the conservative baseline, which collapses nothing, is still
correct.)
"""

import sys
import traceback

from kb_identity_harness import (
    build_labeled_workload,
    find_false_merges,
    load_candidate,
)

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def spec(source, target, options=None, toolchain="v1", variants=None, build=None):
    d = {"source": source, "target": target, "toolchain": toolchain}
    if options is not None:
        d["options"] = list(options)
    if variants is not None:
        d["variants"] = list(variants)
    if build is not None:
        d["build"] = dict(build)
    return d


def _key(mod, s):
    k = mod.identity_key(s)
    if not isinstance(k, str) or not k:
        raise AssertionError(f"identity_key must return a non-empty str, got {k!r}")
    return k


@case
def returns_string_for_various(mod):
    samples = [
        spec("o[i]=a[i]*b[i];", "p80"),
        spec("o[i]=a[i]+b[i];", "p90", options=["fast"], variants=["kA"]),
        spec("acc+=a[k]*b[k];", "p80", options=["o3"], toolchain="v2", build={"tmpdir": "/tmp/x"}),
        {"source": "x=y;", "target": "p80"},  # minimal spec (no options/toolchain/variants/build)
    ]
    for s in samples:
        _key(mod, s)


@case
def deterministic_repeat_same_object(mod):
    s = spec("o[i]=a[i]*b[i];", "p80", options=["o3"])
    if _key(mod, s) != _key(mod, s):
        raise AssertionError("identity not stable across repeated calls on the same object")


@case
def deterministic_fresh_build(mod):
    a = spec("acc+=a[k]*b[k];", "p80", options=["o3"], variants=["kA"])
    b = spec("acc+=a[k]*b[k];", "p80", options=["o3"], variants=["kA"])
    if _key(mod, a) != _key(mod, b):
        raise AssertionError("two independently built identical specs got different identities")


@case
def distinct_source_distinct_keys(mod):
    a = spec("o[i]=a[i]*b[i];", "p80")
    b = spec("o[i]=a[i]+b[i];", "p80")
    if _key(mod, a) == _key(mod, b):
        raise AssertionError("different source text collided")


@case
def distinct_target_distinct_keys(mod):
    # same source + options + toolchain, different target profile -> a different artifact.
    a = spec("o[i]=a[i]*b[i];", "p80", options=["o3"])
    b = spec("o[i]=a[i]*b[i];", "p90", options=["o3"])
    if _key(mod, a) == _key(mod, b):
        raise AssertionError("specs for different target profiles collided (target dropped?)")


@case
def distinct_effective_option_distinct_keys(mod):
    a = spec("o[i]=a[i]*b[i];", "p80", options=["fast"])
    b = spec("o[i]=a[i]*b[i];", "p80", options=[])
    if _key(mod, a) == _key(mod, b):
        raise AssertionError("an effective (non-default) option was ignored")


@case
def distinct_option_value_distinct_keys(mod):
    a = spec("acc+=a[k]*b[k];", "p80", options=["o3"])
    b = spec("acc+=a[k]*b[k];", "p80", options=["o2"])
    if _key(mod, a) == _key(mod, b):
        raise AssertionError("two different option values were merged")


@case
def distinct_toolchain_distinct_keys(mod):
    a = spec("acc+=a[k]*b[k];", "p80", options=["o3"], toolchain="v1")
    b = spec("acc+=a[k]*b[k];", "p80", options=["o3"], toolchain="v2")
    if _key(mod, a) == _key(mod, b):
        raise AssertionError("different toolchain tags were merged (a stale artifact would be reused)")


@case
def distinct_variant_set_distinct_keys(mod):
    a = spec("o[i]=a[i]>0?a[i]:0;", "p80")                       # no variants
    b = spec("o[i]=a[i]>0?a[i]:0;", "p80", variants=["kA"])      # one variant
    c = spec("o[i]=a[i]>0?a[i]:0;", "p80", variants=["kB", "kC"])  # a different variant set
    keys = {_key(mod, a), _key(mod, b), _key(mod, c)}
    if len(keys) != 3:
        raise AssertionError("the requested variant set was ignored")


@case
def no_false_merge_full_workload(mod):
    workload = build_labeled_workload()
    bad = find_false_merges(mod, workload)
    if bad:
        sample = sorted((sorted(v), k) for k, v in bad.items())[:3]
        raise AssertionError(f"identity shared by >1 true class (false merge): {sample}")


@case
def handles_edge_schema(mod):
    empty_opts = spec("x=y;", "p80", options=[])           # explicit empty option set
    no_opts = {"source": "x=y;", "target": "p80"}          # no options/toolchain/variants keys
    with_build = spec("x=y;", "p80", build={"ts": 1})      # only an incidental build annotation
    for s in (empty_opts, no_opts, with_build):
        _key(mod, s)
    # even at the degenerate schema, two specs for different targets must stay distinct.
    other_target = {"source": "x=y;", "target": "p90"}
    if _key(mod, no_opts) == _key(mod, other_target):
        raise AssertionError("degenerate specs for different targets collided")


@case
def many_distinct_specs_pairwise_distinct(mod):
    specs = [
        spec("o[i]=a[i]*b[i];", "p80"),
        spec("o[i]=a[i]+b[i];", "p80"),
        spec("o[i]=a[i]*b[i];", "p90"),                             # target differs from #0
        spec("acc+=a[k]*b[k];", "p80", options=["o3"]),
        spec("acc+=a[k]*b[k];", "p80", options=["o2"]),             # option value differs
        spec("acc+=a[k]*b[k];", "p80", options=["o3"], toolchain="v2"),  # toolchain differs
        spec("o[i]=a[i]/b[i];", "p86", options=["fast", "o3"]),
        spec("o[i]=a[i]>0?a[i]:0;", "p80", variants=["kA"]),
    ]
    keys = [_key(mod, s) for s in specs]
    if len(set(keys)) != len(specs):
        raise AssertionError(f"expected {len(specs)} distinct identities, got {len(set(keys))}")


def main():
    mod = load_candidate()
    passed = 0
    for fn_case in CASES:
        name = fn_case.__name__
        try:
            fn_case(mod)
            passed += 1
            print(f"CASE_PASS {name}")
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            print(f"CASE_FAIL {name}: {reason.splitlines()[0][:300]}")
            traceback.print_exc(file=sys.stderr)
    total = len(CASES)
    print(f"CASES_PASSED={passed}/{total}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
