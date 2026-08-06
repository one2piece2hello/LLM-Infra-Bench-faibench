# Performance Optimization Task

You are given a clean Python workspace with a single editable file:

```
submission/kernel.py
```

This is the **only** file you may edit. Editing any other file (or importing an
implementation from anywhere else) causes the whole task to score 0.

## Objective

`submission/kernel.py` ships a required function `discretize` whose body is **not implemented**
(it raises `NotImplementedError`). Implement it to the contract below so it is numerically correct
on the hidden benchmark workloads (GPU, H20).

**Scoring is all-or-nothing on correctness.** You score **1.0** only when EVERY case passes; any
single failing case — or an implementation that still raises `NotImplementedError`, or an edit
outside the scope file, or any attempt to bypass the checks — scores **0.0**. There is no partial
credit, and being faster does not raise your score.

## Interface contract (implement exactly this)

### `discretize(u, delta, A, B) -> (deltaA, deltaB_u)`

The zero-order-hold (ZOH) discretization of a selective state-space model's parameters — the
per-timestep preprocessing that turns the continuous `(A, B)` and step size `delta` into the
discrete factors a scan would consume.

- `u`:     `float32` CUDA tensor, shape `[Bt, L, D]` (batch, sequence length, inner channels).
- `delta`: `float32` CUDA tensor, shape `[Bt, L, D]` (per-position, per-channel step size; positive).
- `A`:     `float32` CUDA tensor, shape `[D, N]` (state matrix; one `N`-dim state per channel).
- `B`:     `float32` CUDA tensor, shape `[Bt, L, N]` (input projection).

Produce two tensors, each of shape `[Bt, L, D, N]`:

```
deltaA[b, l, d, n]   = exp( delta[b, l, d] * A[d, n] )              # ZOH discretization of A
deltaB_u[b, l, d, n] = delta[b, l, d] * B[b, l, n] * u[b, l, d]     # discretized B, scaled by input u
```

`deltaA` carries the **exponential** (ZOH); `deltaB_u` is a plain product. Every `[b,l,d,n]` entry
is an independent function of the inputs — there is no dependency across the sequence axis `l`. All
arithmetic is `float32`.

### `custom_kernel(data) -> (deltaA, deltaB_u)`

`data = (u, delta, A, B)`. Already wired to call `discretize(u, delta, A, B)` and return the tuple.
You implement the primitive.

## Correctness

Outputs are compared against a seeded `float32` reference within `rtol = atol = 1e-3` across a
hidden set of shapes spanning `L=1`, odd / non-power-of-two `L`, `D` and `N`, and larger regimes.
Both `deltaA` (with its exponential) and `deltaB_u` must be correct across the whole domain — a
submission that drops the exponential (a linear/Euler step) or otherwise breaks a factor fails the
gate. Outputs must genuinely depend on the inputs (the grader runs the op on two different inputs
of the same shape and rejects identical outputs).

## Latency

Runtime is also measured on GPU with CUDA events (block of medians, the `u` input regenerated
every timed iteration; a warmup window is discarded), but that measurement is recorded as
**diagnostic metadata only — it does not affect your score**. Because no entry depends on any
other, the whole `[Bt, L, D, N]` result can be produced in one fused vectorized pass; advancing the
sequence one timestep at a time is equally correct and equally acceptable for scoring.

## Workload shape (what the hidden benchmark covers)

- `float32` `u`/`delta` `[Bt, L, D]`, `A [D, N]`, `B [Bt, L, N]`, with `L` up to ~128, `D` up to
  ~512, `N` up to ~16, modest `Bt`, single GPU.
- `L=1`, odd, and non-power-of-two shapes as well as large ones.
- Every one of these shape classes is checked, and every one must pass.

## Rules

- Edit only `submission/kernel.py`, and keep the `discretize` and `custom_kernel` signatures. Any
  other change (or importing the answer from another module/file/package) scores 0.
- Solve independently. Do **not** download, clone, fetch, or `pip install` any external
  repository, kernel library, or reference implementation to obtain the answer; do **not** attempt
  to bypass or disable the environment's network isolation (e.g. by unsetting proxy environment
  variables or opening raw sockets). Any such action scores the whole task 0, regardless of
  measured speed.
- You may use `torch` and `triton` (both are installed). You do not need any other package.

## 提交方式

把改动**留在工作树里**即可 —— 直接编辑允许的文件,不需要 `git commit`
(判分读工作树;仓库的 HEAD 必须停在初始基线 commit 上)。
