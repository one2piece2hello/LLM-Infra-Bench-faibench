# Oracle / strong-baseline notes (reviewer-only)

- `solution/tiered_store.py` is the ORACLE: a complete, correct reference implementation of the frozen
  contract in `../instruction.md`. It is NOT baked into the image (the Dockerfile must never COPY
  `solution/`).
- It is applied FRESH at score time via the verifier MODE mechanism: `tests/test.sh` with
  `KERNELBENCH_VERIFY_MODE=oracle` and `KERNELBENCH_ORACLE_PATCH` set to a patch that turns the
  shipped starter stub into this oracle. Produce that patch with:

  ```
  diff -u ../environment/workspace/submission/tiered_store.py  ./tiered_store.py  > oracle.patch
  # then apply -p1 onto /app/submission/tiered_store.py
  ```

  (or `git diff` between the two trees). The patch replaces `/app/submission/tiered_store.py` in place.
- Expected calibration on a CPU lane (gpus=0): `MODE=oracle` -> correctness_frac == 1.0;
  `MODE=candidate` on the shipped stub -> near 0 (stub has no eviction, no resumable transfer, no
  coalescing, no integrity re-fetch); each negative-control patch -> < 1.0 or hard-fail 0.0.
- The oracle uses the SAME public contract exposed to the candidate; it never reads `tests/` or the
  harness reference. Scoring is answer-free: the harness recomputes the expected result itself with an
  independent OrderedDict-LRU reference model.

## Negative controls (reviewer should stage these as KERNELBENCH_NEGATIVE_PATCH, expect < 1.0)

1. Remove eviction (keep everything hot): fails `hot_size()` invariant on eviction/atomic_eviction axes.
2. `transfer` ignores `resume_from` (re-sends all): fails resumable_transfer (resent_already_received).
3. `transfer` swallows the bad-segment signal without re-fetch: fails transfer_integrity (checksum).
4. `get` fabricates a value instead of calling `cold_fetch_fn`: fails coalesce_fetch (fetches < 1).
5. Non-thread-safe hot dict (no lock): fails concurrent_rw (torn/phantom) or coalesce_fetch (over-fetch).
6. `get` returns cold value without touching LRU: fails eviction ordering (hot_key_must_stay) cases.

## Key contract subtleties encoded in the harness

- Hot capacity is measured in VALUE BYTES (sum of len(value)), not entry count.
- The cold tier is authoritative: an evicted key must remain readable byte-identical via `get`.
- `transfer` distinguishes an interruption (propagate) from a corruption signal (retriable): the
  oracle tells them apart by exception CLASS NAME (it cannot import the harness). The harness raises a
  class named `_BadSegment` for corruption and `_Interrupted` for interruption.
