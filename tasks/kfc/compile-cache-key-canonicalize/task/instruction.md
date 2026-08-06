# Performance Optimization Task

You are working on the work-reuse layer of a batch execution service. The service
receives a stream of structured **request signatures**. Before running an expensive
step for a signature, it computes a short string **identity** for that signature and
looks the identity up in a table of already-processed identities; a hit means the
prior result can be reused and the expensive step is skipped. The file
`identity_key.py` implements this identity as a single function, `identity_key`. It
is correct but produces far more distinct identities than necessary.

## Behavioral contract

```python
def identity_key(signature: dict) -> str:
    ...
```

A **signature** is a mapping describing one request:

- `"op"`: the operation name — a string.
- `"operands"`: a list of operands. Each operand is a mapping with `"shape"` (a list
  of integers) and `"dtype"` (a string).
- `"flags"` (optional): a mapping of configuration flag name → value.
- `"meta"` (optional): a mapping of incidental annotations.

`identity_key(signature)` returns a **string**. It must be **deterministic and
reproducible**: the same signature (and any signature equivalent to it, see below)
always yields the same string, on every run and every machine — so derive it from the
signature's content with a fixed rule, not from anything process- or run-dependent.

### When are two signatures equivalent?

Two signatures are **equivalent** (they describe the same request and must receive
the **same** identity) when they are equal after applying **all** of the following
normalizations. Any difference that survives all of them makes the signatures
**distinct**, and distinct signatures must receive **different** identities.

1. **Incidental annotations are ignored.** The `"meta"` mapping never affects the
   identity — two signatures that differ only in `"meta"` (or in whether `"meta"` is
   present at all) are equivalent.
2. **Field order is irrelevant.** The order in which the keys of any mapping
   (`signature` itself, an operand, `"flags"`, `"meta"`) are written does not matter.
3. **Equivalent dtype spellings are equal.** Each of these spelling groups denotes
   one element type; any spelling in a group equals any other:
   - `"f32"`, `"float32"`, `"single"`
   - `"f16"`, `"float16"`, `"half"`
   - `"i32"`, `"int32"`, `"int"`
   - `"bf16"`, `"bfloat16"`
4. **Default-valued flags are equivalent to absent flags.** A flag set to its default
   value is the same as omitting it. The defaults are:
   `fastmath = false`, `layout = "row_major"`, `precision = "default"`.
   A flag set to any **other** value is meaningful and must be kept.
5. **Order-insensitive operations.** For the operations `add`, `mul`, `max`, `min`
   the **order of the operands carries no meaning** (the same operands in any order
   describe the same request). For **every other operation** the operand order is
   meaningful and two signatures that differ only in operand order are **distinct**.

Everything not listed above is **meaningful**: the operation name, each operand's
shape, each operand's normalized dtype, the operand order for order-sensitive
operations, the number of operands, and every non-default flag value. Signatures that
differ in any of these are **distinct** and must not share an identity.

## Why the current implementation is wasteful

The current code derives the identity from the signature almost verbatim: it only
neutralizes the order of a mapping's keys. So two signatures that are equivalent but
merely spelled differently — an alternate dtype spelling, a default flag written out
in full, a stray annotation, order-insensitive operands in a different order — are
treated as **different** identities. The reuse table then holds one entry per spelling
instead of one entry per genuinely distinct request, and the expensive step is repeated
for every spelling of the same request.

Make the identity **collapse the equivalent spellings** so that equivalent signatures
map to one identity and the reuse table holds as few entries as there are genuinely
distinct requests — while **never** giving two genuinely different signatures the same
identity.

## Correctness comes first

A single **collision between two genuinely different signatures** — for example
treating two different element types, two different shapes, a different operand count,
a swapped pair of order-sensitive operands, or two different non-default flag values as
the same — is a hard failure and scores **zero**, no matter how few identities you
produce. Merging spellings that are only *incidentally* different is the goal; merging
requests that are *actually* different is never allowed. The verifier checks both a
suite of correctness cases and, over the whole workload, that no identity is shared by
two genuinely different requests.

## Forbidden

Do **not** import a tensor / graph / array framework, and do **not** import or read
the test harness or its workload. Compute the identity yourself from the signature
content using the standard library only. The scoring harness scans your submitted file
for those references and scores the task `0` (do not reference them even in comments).

## Scope

Optimize the product implementation in `identity_key.py` only. Do **not** edit tests,
harnesses, workloads, or dependency/build files. The final submitted diff must contain
only product-code changes. Public/dev feedback, when present, is representative only;
final verification runs additional trusted and hidden workloads.

## How this task is scored — ONE single graded submission (READ CAREFULLY)

You get **exactly one** graded submission. There is no iteration loop, no budget
of retries, and no feedback round you can act on: **submitting ends the task.**

1. Work on `/app/repo/identity_key.py` until you believe it is both **correct** and as
   fast as you can make it. Test it yourself as much as you like —
   your own scratch scripts, your own timing harnesses, your own reasoning about
   the code path. None of that costs you anything.
2. When — and only when — you are done, submit **once**:

   ```
   bash /opt/loop/submit.sh
   ```

3. **That call is final.** It scores the current state of `/app/repo`, records it,
   and closes the task. A second call to `submit.sh` is refused and exits non-zero.
   You will not get another attempt, and you will not get iterative feedback you
   can use to improve — whatever `/app/repo` contains at that moment is what is
   graded.

Because you cannot iterate, **think the design through before you submit** and
self-test thoroughly: read the current implementation, decide on the change,
convince yourself the contract still holds on the normal, boundary, degenerate and
error-path cases described above, and only then submit. A submission that is fast
but wrong on even one case scores zero, and you cannot repair it afterwards.

The grade is produced by a full, trusted end-of-session verifier (more workloads
than anything you can see), so the submitted state must be genuinely correct — not
just correct on the cases you happened to try.

## 🔴 Red line (hard — ANY one of these makes the whole task score 0)

The scoring machinery is off-limits. Solve this task **independently**, using
ONLY the code in `/app/repo`, the sanitized feedback that `submit.sh` returns,
and your own knowledge and reasoning.

- Do **NOT** read, `cat`, `open`, `less`, `head`/`tail`, `grep`, `find`, or edit
  ANY verifier / correctness / timing / scoring / hidden-test file or directory,
  wherever it may live. Running `bash /opt/loop/submit.sh` is the ONLY sanctioned
  interaction with the scoring machinery.
- Do **NOT** run any verifier directly or try to reproduce or reverse-engineer it
  — score ONLY by calling `bash /opt/loop/submit.sh`.
- Do **NOT** search for, print, or infer the hidden workloads, seeds, thresholds,
  the metric name, or the reference speedup from any source.
- Do **NOT** fetch or look up the source PR, the upstream repository, or a
  reference solution (no web search, no `git` remote fetch, no network to the
  source), whether the internet is on or off.
- Do **NOT** use, quote, or condition your code on any verifier / scoring / source
  material, however you might have come to see it.

Stay inside `/app/repo` and scratch directories you create yourself; use only
what `submit.sh` returns for scoring signal.

## 提交方式

本题只有**一次**评分提交(kfc 全子集单次)。改完后运行一次 `bash /opt/loop/submit.sh`:它给当前 `/app/repo` 评分并**立即定稿**(无需再单独调 `--finalize`,也没有第二次机会 —— 再次调用只会重新定稿同一份已记录的最佳快照,不给新改动评分)。请先用你自己的脚本充分自测再提交。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
