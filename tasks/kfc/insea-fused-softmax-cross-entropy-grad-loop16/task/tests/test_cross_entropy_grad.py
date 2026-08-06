"""Correctness suite for the cross-entropy loss+gradient contract — 12 cases.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>".
The runner greps the number of CASE_PASS lines (expects 12).

The candidate may overwrite its ``logits`` argument in place, so every case
passes the candidate a fresh clone and computes the reference from the pristine
tensor first.
"""

import sys
import traceback

import torch

from kb_ce_harness import (
    assert_grad_close,
    assert_loss_close,
    count_nonzero_rows,
    forbidden_ce_guard,
    load_candidate,
    make_labels,
    make_logits,
    ref_ce,
)

IGN = -100
CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def _run(fn, N, V, seed, dtype=torch.float32, ls=0.0, ignore_frac=0.0):
    logits = make_logits(N, V, seed, dtype=dtype)
    labels = make_labels(N, V, seed + 1, ignore_frac=ignore_frac, ignore_index=IGN)
    ref_loss, ref_grad = ref_ce(logits, labels, ignore_index=IGN, label_smoothing=ls)
    with forbidden_ce_guard():
        c_loss, c_grad = fn(logits.clone(), labels, IGN, ls)
    bf16 = dtype == torch.bfloat16
    tag = f"[N={N} V={V} dtype={dtype} ls={ls} ign={ignore_frac}]"
    assert_loss_close(c_loss, ref_loss, bf16=bf16, msg=tag)
    assert_grad_close(c_grad, ref_grad, bf16=bf16, msg=tag)


@case
def normal_fp32(fn):
    _run(fn, 256, 4096, seed=100)


@case
def normal_bf16(fn):
    _run(fn, 256, 4096, seed=150, dtype=torch.bfloat16)


@case
def mean_with_ignore(fn):
    _run(fn, 320, 3072, seed=200, ignore_frac=0.35)


@case
def all_ignore(fn):
    # every row ignored -> loss 0, grad all zero
    N, V = 128, 2048
    logits = make_logits(N, V, 300)
    labels = torch.full((N,), IGN, dtype=torch.int64, device="cuda")
    with forbidden_ce_guard():
        c_loss, c_grad = fn(logits.clone(), labels, IGN, 0.0)
    if abs(float(c_loss)) > 1e-5:
        raise AssertionError(f"all-ignore loss should be 0, got {float(c_loss)}")
    if c_grad.shape != (N, V):
        raise AssertionError(f"bad grad shape {tuple(c_grad.shape)}")
    if not (c_grad.to(torch.float32) == 0).all():
        raise AssertionError("all-ignore grad should be exactly zero")


@case
def single_class(fn):
    # V == 1: softmax is 1, loss is 0, grad is 0
    _run(fn, 64, 1, seed=400)


@case
def single_row(fn):
    _run(fn, 1, 2048, seed=500)


@case
def nontile_vocab(fn):
    # V not a multiple of any common chunk size
    _run(fn, 128, 4099, seed=600)


@case
def label_smoothing(fn):
    _run(fn, 256, 2048, seed=700, ls=0.1)


@case
def error_contract(fn):
    N, V = 32, 512
    logits = make_logits(N, V, 800)
    labels = make_labels(N, V, 801)
    # integer logits -> TypeError
    try:
        fn(logits.to(torch.int32), labels, IGN, 0.0)
        raise AssertionError("integer logits did not raise TypeError")
    except TypeError:
        pass
    # float labels -> TypeError
    try:
        fn(logits.clone(), labels.to(torch.float32), IGN, 0.0)
        raise AssertionError("float labels did not raise TypeError")
    except TypeError:
        pass
    # 1-D logits -> ValueError
    try:
        fn(logits.reshape(-1).clone(), labels, IGN, 0.0)
        raise AssertionError("1-D logits did not raise ValueError")
    except ValueError:
        pass
    # labels wrong length -> ValueError
    try:
        fn(logits.clone(), labels[:-1], IGN, 0.0)
        raise AssertionError("wrong-length labels did not raise ValueError")
    except ValueError:
        pass
    # negative label_smoothing -> ValueError
    try:
        fn(logits.clone(), labels, IGN, -0.1)
        raise AssertionError("negative label_smoothing did not raise ValueError")
    except ValueError:
        pass
    # a non-ignored label out of range [0, V) -> ValueError
    bad = labels.clone()
    bad[0] = V  # out of range, not the ignore sentinel
    try:
        fn(logits.clone(), bad, IGN, 0.0)
        raise AssertionError("out-of-range label did not raise ValueError")
    except ValueError:
        pass


@case
def metamorphic_const_shift(fn):
    """Adding a per-row constant to all logits leaves loss AND grad unchanged."""
    N, V = 192, 2048
    logits = make_logits(N, V, 900)
    labels = make_labels(N, V, 901)
    with forbidden_ce_guard():
        loss_a, grad_a = fn(logits.clone(), labels, IGN, 0.0)
    g = torch.Generator(device="cpu").manual_seed(902)
    # a large positive per-row shift: a numerically stable softmax is invariant,
    # while a non-stable one (no max subtraction) overflows -> NaN -> fails here.
    shift = (torch.rand(N, generator=g) * 40.0 + 80.0).to(logits.device).to(logits.dtype)
    logits_b = logits + shift[:, None]
    with forbidden_ce_guard():
        loss_b, grad_b = fn(logits_b.clone(), labels, IGN, 0.0)
    assert_loss_close(loss_b, loss_a, msg="[const-shift loss]")
    assert_grad_close(grad_b, grad_a, msg="[const-shift grad]")


@case
def metamorphic_permute_rows(fn):
    """Permuting the rows permutes the gradient rows and leaves the mean loss unchanged."""
    N, V = 256, 2048
    logits = make_logits(N, V, 1000)
    labels = make_labels(N, V, 1001, ignore_frac=0.2, ignore_index=IGN)
    with forbidden_ce_guard():
        loss_a, grad_a = fn(logits.clone(), labels, IGN, 0.0)
    g = torch.Generator(device="cpu").manual_seed(1002)
    perm = torch.randperm(N, generator=g).to(logits.device)
    with forbidden_ce_guard():
        loss_b, grad_b = fn(logits[perm].clone(), labels[perm], IGN, 0.0)
    assert_loss_close(loss_b, loss_a, msg="[permute loss]")
    assert_grad_close(grad_b, grad_a[perm], msg="[permute grad]")


@case
def work_evidence(fn):
    """Exactly the non-ignored rows carry a nonzero gradient; ignored rows are zero."""
    N, V = 384, 2048
    logits = make_logits(N, V, 1100)
    labels = make_labels(N, V, 1101, ignore_frac=0.5, ignore_index=IGN)
    n_valid = int((labels != IGN).sum().item())
    with forbidden_ce_guard():
        _, c_grad = fn(logits.clone(), labels, IGN, 0.0)
    nz = count_nonzero_rows(c_grad)
    if nz != n_valid:
        raise AssertionError(f"nonzero-grad rows {nz} != valid labels {n_valid}")
    ignored = (labels == IGN)
    if ignored.any() and not (c_grad[ignored].to(torch.float32) == 0).all():
        raise AssertionError("ignored rows have nonzero gradient")


def main():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        sys.exit(2)
    mod = load_candidate()
    fn = mod.cross_entropy_loss_grad
    passed = 0
    for fn_case in CASES:
        name = fn_case.__name__
        try:
            fn_case(fn)
            torch.cuda.synchronize()
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
