# Performance Optimization Task

You are tuning the tile-size planner for a blocked matrix-multiply pipeline. Each job
is one multiply `C[M,N] += A[M,K] * B[K,N]`, run as a loop nest over tiles: a
`Tm x Tn` block of the output is kept in a small on-chip buffer and accumulated across
the `K` dimension, reading a `Tm x Tk` block of `A` and a `Tk x Tn` block of `B` each
step. The file `tile_planner.py` implements `plan_tiling`, which chooses `[Tm, Tn, Tk]`
for every job. It works but is not optimal.

## Behavioral contract

```python
def plan_tiling(problems, cap):
    """-> a list of [Tm, Tn, Tk], one per problem, in order"""
```

Each entry of `problems` is a mapping with:

- **`M`**, **`N`**, **`K`**: positive int dimensions of the multiply.
- **`esz`**: positive int element size in bytes.
- **`tm_choices`**, **`tn_choices`**, **`tk_choices`**: non-empty lists of the allowed
  tile sizes for that axis. Every `Tm` choice divides `M` exactly, every `Tn` divides
  `N`, every `Tk` divides `K`.

`cap` is the on-chip buffer capacity in bytes (a positive int), shared by all jobs.

For one job the buffer must hold one block of each of A, B and C at once:

```
footprint = (Tm*Tk + Tk*Tn + Tm*Tn) * esz     must be <= cap
```

**Rules a returned plan must satisfy.** For every job the chosen `[Tm, Tn, Tk]` must be
drawn from that job's respective choice lists **and** fit the buffer
(`footprint <= cap`). A plan that picks a tile outside the choices, or one whose
footprint exceeds `cap`, is invalid (the block would not fit on chip) and is rejected.
A valid choice always exists — the smallest tile of each axis fits.

### Error contract

Raise `ValueError` for a malformed problem: a non-positive `M`/`N`/`K`/`esz`, an empty
choice list, a choice that is not a positive divisor of its dimension, or a
non-positive `cap`. Raise `TypeError` for a non-int where an int is required (a `bool`
is rejected). (Do **not** change the name of `plan_tiling` or the return shape.)

## Correctness comes first

The verifier checks your plan for **validity** on many problem sets — square and
tall-skinny multiplies, several element sizes and capacities, single-choice and
single-problem degenerate cases, and malformed problems that must raise. A plan that is
invalid on even one problem — most importantly one whose tile does not fit the
buffer — scores **zero**, no matter how little data it appears to move.

## Scope

Optimize the product implementation in `tile_planner.py` only. Do **not** edit tests,
harnesses, workloads, or dependency/build files. The final submitted diff must contain
only product-code changes.

## How this task is scored — ONE single graded submission (READ CAREFULLY)

You get **exactly one** graded submission. There is no iteration loop, no budget
of retries, and no feedback round you can learn from: **submitting ends the
task.**

1. Read the current implementation in `/app/repo/tile_planner.py` and work out, before you touch
   anything, what will actually make it faster.
2. Make your changes to `/app/repo/tile_planner.py`. Test them yourself as thoroughly as you can
   with scratch scripts you write in your own scratch directory — that self-testing
   is your only safety net.
3. When — and only when — you are confident the implementation is both **correct**
   and **as good as you can make it**, submit it exactly once:

   ```
   bash /opt/loop/submit.sh
   ```

**Submitting is final and irreversible.**

- The moment `submit.sh` returns, this task is **over**. Stop editing and stop
  working on it.
- A second call to `submit.sh` is **refused** and exits non-zero. There is no
  second chance, no "best of several attempts", and no separate finalization step
  to run afterwards — the single submission *is* the finalization.
- You will **not** receive iterative feedback to improve against. The receipt
  `submit.sh` prints is a record that your one submission was taken, not a signal
  to try again.
- Whatever state `/app/repo/tile_planner.py` is in at the moment you submit is exactly the state
  that gets graded. Nothing is restored, re-selected or rolled back for you.

Because you only get one shot: think the design through first, re-read the
behavioural contract above, convince yourself every rule and error case is
handled, and satisfy yourself about performance — **then** submit.

Your score comes from a full, trusted end-of-session verifier (more workloads
than any self-test you can write), so correctness is not optional: a fast but
incorrect submission scores **zero**.

## 🔴 Red line (hard — ANY one of these makes the whole task score 0)

The scoring machinery is off-limits. Solve this task **independently**, using
ONLY the code in `/app/repo` and your own knowledge and reasoning. `submit.sh`
is a one-shot submission channel, not a scoring oracle you may probe.

- Do **NOT** read, `cat`, `open`, `less`, `head`/`tail`, `grep`, `find`, or edit
  ANY verifier / correctness / timing / scoring / hidden-test file or directory,
  wherever it may live. Running `bash /opt/loop/submit.sh` is the ONLY sanctioned
  interaction with the scoring machinery.
- Do **NOT** run any verifier directly or try to reproduce or reverse-engineer it
  — the ONLY sanctioned scoring action is your single `bash /opt/loop/submit.sh`.
- Do **NOT** search for, print, or infer the hidden workloads, seeds, thresholds,
  the metric name, or the reference speedup from any source.
- Do **NOT** fetch or look up the source PR, the upstream repository, or a
  reference solution (no web search, no `git` remote fetch, no network to the
  source), whether the internet is on or off.
- Do **NOT** use, quote, or condition your code on any verifier / scoring / source
  material, however you might have come to see it.

Stay inside `/app/repo` and scratch directories you create yourself. `submit.sh`
may be called exactly once, and only to submit your finished answer.

## 提交方式

本题只有**一次**评分提交(kfc 全子集单次)。改完后运行一次 `bash /opt/loop/submit.sh`:它给当前 `/app/repo` 评分并**立即定稿**(无需再单独调 `--finalize`,也没有第二次机会 —— 再次调用只会重新定稿同一份已记录的最佳快照,不给新改动评分)。请先用你自己的脚本充分自测再提交。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
