# Performance Optimization Task

You are working on the inner product of two integer vectors that are stored in a
compact block-encoded form (the building block of a block-encoded matrix-vector
product). The file `blocked_dot.py` implements this inner product as a single
function, `blocked_dot`. It is correct but slow.

## Behavioral contract

```python
def blocked_dot(u_blocks, v_blocks):
    ...
```

Two equal-length logical vectors `U` and `V` are each stored as a sequence of
**blocks** of 32 lanes. Block `i` of `U` pairs with block `i` of `V` (the same 32
lanes). `u_blocks` and `v_blocks` are equal-length sequences of blocks.

- A block of `U` (a **packed block**) is a pair `(su, packed)`:
  - `su` — a real number, the scale factor of the block.
  - `packed` — 16 bytes (integers in `[0, 255]`) carrying the block's 32 signed codes
    **two per byte**, in lane order: within a byte the **low** 4 bits belong to the
    earlier of the two lanes that byte covers and the **high** 4 bits to the later
    lane. The stored 4-bit values are **offset-biased**: subtracting a fixed offset of
    `8` from a stored value recovers the signed code, so every unpacked code is an
    integer in `[-8, 7]`.
- A block of `V` (a **code block**) is a pair `(sv, codes)`:
  - `sv` — a real number, the scale factor of the block.
  - `codes` — 32 small signed integers (each in `[-127, 127]`).

### Exact arithmetic (so the result is well-defined)

The result is a single real number: the sum over all blocks of each block's
contribution, where one block contributes its **lane-wise integer dot product** — the
32 unpacked signed codes of `U` against the 32 companion codes of `V`, paired lane by
lane — weighted by the product of that block's two scale factors.

Consequences that are part of the contract:

- The per-lane products and their per-block sum are **integers**; the scale factors
  are exactly representable, so the result is well-defined **independently of the
  order** in which lanes and blocks are accumulated.
- The `- 8` offset is applied to **every** code (both the low and the high code of a
  byte). Treating the stored 4-bit value as an unsigned `0..15` is wrong.
- Within a byte the **low** 4 bits are the earlier lane (`2*b`) and the **high** 4
  bits are the later lane (`2*b+1`).
- A scale factor of `0` makes that block contribute nothing.
- Zero blocks (`u_blocks == []` and `v_blocks == []`) → the result is `0.0`.

Public API (do **not** change the name or the return shape): `blocked_dot`,
returning a single `float`.

### Error contract

- `ValueError` if `u_blocks` and `v_blocks` differ in length.
- `ValueError` if any packed byte list is not exactly length 16.
- `ValueError` if any code list is not exactly length 32.

## What to improve

The shipped implementation is correct but performs considerably more work per lane
than the encoding and the arithmetic contract require. Read it and reason about where
that redundancy sits. Make the inner product **faster** — do fewer total operations
for the same result — while keeping every returned value identical to what the
contract above defines.

**Forbidden:** delegating the computation to an array/numerics library inner-product
or matrix-multiply helper (for example a `numpy` dot/matmul call) or to a tensor
framework. The scoring harness scans your submitted file for those tokens and scores
the task `0` (do not reference them even in comments). Compute the unpack and the
accumulation yourself.

## Correctness comes first

The verifier compares your results against an independent reference on many
workloads — a single block hand-checked value, multi-block inputs, the empty input,
a zero scale factor, all-minimum codes, the signed `- 8` offset, the low/high code
ordering within a byte, length-error rejection, and metamorphic checks (scaling all
of `U`'s scale factors by a constant scales the result by that constant; negating all
companion codes negates the result). A faster result that is wrong on even one case —
including dropping the `- 8` offset, swapping the low/high code order, or applying the
scale factor with a divergent rounding — scores zero.

## Scope

Optimize the product implementation in `blocked_dot.py` only. Do **not** edit tests,
benchmark harnesses, workloads, or dependency/build files. The final submitted diff
must contain only product-code changes.

## How this task is scored — ONE single graded submission (READ CAREFULLY)

You get **exactly one** submission. There is no iterative loop, no dev feedback
round, and no second chance.

1. Read the current implementation in `/app/repo/blocked_dot.py`, work out your plan, and
   make your changes.
2. Self-test as much as you like **using your own scratch scripts** (write them
   outside the product file, e.g. under `/tmp`), because you will get no feedback
   from the grader before your score is fixed.
3. When — and only when — you are confident the implementation is both correct and
   faster, submit **once**:

   ```
   bash /opt/loop/submit.sh
   ```

**Submitting ends the task.** The moment `submit.sh` runs, your submission is
recorded and graded; the run is over. A second call to `submit.sh` is refused and
exits non-zero, and you will not receive an iteration signal you could act on. So
do all of your thinking, reading and self-testing **before** you submit — a
half-finished idea submitted early cannot be improved afterwards.

Your reward comes from a full, trusted end-of-session verifier over the whole
correctness suite plus the performance measurement, so keep your submission
genuinely correct: **any** failing correctness case scores the whole task `0`,
regardless of how fast it is.

**🎓 Explain your work.** Before you edit, state your approach: what in the
current code path is doing redundant work, and what you will change. After you
edit and before you submit, give a short, concrete, step-by-step explanation of
*why* your change is faster in terms of the actual code path, and how you
convinced yourself it is still correct. Write it for a beginner reader.

## 🔴 Red line (hard — ANY one of these makes the whole task score 0)

The scoring machinery is off-limits. Solve this task **independently**, using
ONLY the code in `/app/repo`, the sanitized feedback that `submit.sh` returns,
and your own knowledge and reasoning.

- Do **NOT** read, `cat`, `open`, `less`, `head`/`tail`, `grep`, `find`, or edit
  ANY verifier / correctness / timing / scoring / hidden-test file or directory,
  wherever it may live. Your ONE call to `bash /opt/loop/submit.sh` is the ONLY sanctioned
  interaction with the scoring machinery.
- Do **NOT** run any verifier directly or try to reproduce or reverse-engineer it
  — your single `bash /opt/loop/submit.sh` call is the only scoring path.
- Do **NOT** search for, print, or infer the hidden workloads, seeds, thresholds,
  the metric name, or the reference speedup from any source.
- Do **NOT** fetch or look up the source PR, the upstream repository, or a
  reference solution (no web search, no `git` remote fetch, no network to the
  source), whether the internet is on or off.
- Do **NOT** use, quote, or condition your code on any verifier / scoring / source
  material, however you might have come to see it.

Stay inside `/app/repo` and scratch directories you create yourself; you get no scoring signal
before your single submission — rely on your own reasoning and self-tests.

## 提交方式

本题只有**一次**评分提交(kfc 全子集单次)。改完后运行一次 `bash /opt/loop/submit.sh`:它给当前 `/app/repo` 评分并**立即定稿**(无需再单独调 `--finalize`,也没有第二次机会 —— 再次调用只会重新定稿同一份已记录的最佳快照,不给新改动评分)。请先用你自己的脚本充分自测再提交。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
