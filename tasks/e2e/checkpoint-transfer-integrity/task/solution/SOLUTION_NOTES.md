# Oracle / strong-baseline notes (reviewer-only)

- `solution/ckpt_transfer.py` is the ORACLE: a complete, correct reference implementation of the frozen
  contract in `../instruction.md`. It is NOT baked into the image (the Dockerfile must never COPY
  `solution/`).
- Applied FRESH at score time via the verifier MODE mechanism: `tests/test.sh` with
  `KERNELBENCH_VERIFY_MODE=oracle` and `KERNELBENCH_ORACLE_PATCH` set to a patch that turns the shipped
  starter stub into this oracle. Produce that patch with:

  ```
  diff -u ../environment/workspace/submission/ckpt_transfer.py  ./ckpt_transfer.py  > oracle.patch
  # then apply -p1 onto /app/submission/ckpt_transfer.py
  ```

- Expected calibration on a CPU lane (gpus=0): `MODE=oracle` -> correctness_frac == 1.0;
  `MODE=candidate` on the shipped stub -> low (stub has no crcs, no parity, no rollback, no resume, no
  path safety); each negative-control patch -> < 1.0 or hard-fail 0.0.
- Scoring is answer-free: the harness recomputes chunking / parity / normalization with its own
  independent reference model and never trusts self-reported numbers.

## Negative controls (reviewer should stage these as KERNELBENCH_NEGATIVE_PATCH, expect < 1.0)

1. `upload` does not roll back on store failure: fails atomic_upload (num_committed != 0).
2. `download` skips crc verification: fails tamper_detect (returns tampered bytes silently).
3. `download` fabricates bytes when >1 chunk missing: fails parity_exceeded.
4. `download` ignores resume_from (re-fetches): fails resumable_download (refetched_existing).
5. `normalize_path` does not reject '..' escapes: fails path_escape.
6. No parity reconstruction: fails parity_reconstruct (blob not recovered from one missing chunk).

## Key contract subtleties encoded in the harness

- Data chunk keys must NOT contain 'parity'; the parity chunk key MUST contain 'parity'. The harness
  drop() maps an ordinal to the ordinal-th DATA chunk by sorting keys and excluding 'parity' keys.
- Single XOR parity recovers exactly ONE missing/corrupt chunk. The LAST chunk may be short: after XOR
  reconstruction the oracle trims to `total_size - m*chunk_size` and re-verifies the per-chunk crc.
- The oracle stashes the source bytes on the manifest under `_blob` (and parity under `_parity`) so
  `upload` can derive chunk payloads; candidates may keep chunk payloads however they like as long as
  the store keys are stable across pack/upload/download.
- `namespace` isolates concurrent transfers (chunk keys are prefixed) so parallel checkpoints never
  cross-contaminate.
