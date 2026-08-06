#!/usr/bin/env python3
"""Regenerate tests/verifier-correctness-manifest.json for e2e-b1-kv-traffic-sol.

Pure stdlib (hashlib/json) — text bookkeeping, nothing functional. Run after ANY edit to a
frozen-surface file, otherwise test.sh's sha256 gate will reject the surface it ships with.
"""
import hashlib
import json
import os
import sys

TASK = os.environ.get("TASK_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FROZEN = ["compute_reward.py", "test.sh", "harness/bench_kvtraffic.py",
          "harness/baseline_kv_traffic.py", "harness/hidden_suite.json"]


def sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    tests = os.path.join(TASK, "tests")
    suite = json.load(open(os.path.join(tests, "harness", "hidden_suite.json")))
    man = {
        "task_id": "e2e-b1-kv-traffic-sol",
        "lane": "e2e_task",
        "family": "B_serving",
        "task_kind": "performance",
        "metric": "log_speedup_vs_ref_speedup",
        # 🔴 2026-07-27: the reward migrated to the reward.md log form. Regenerating this
        #    manifest MUST NOT revert it -- keep reward_model/metric/ref_speedup/abba_pairs
        #    in sync with tests/verifier-correctness-manifest.json.
        "reward_model": "reward_md_log_speedup_v2_oracle_zero",
        "reward_formula": "reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; range [0,1]",
        "abba_pairs": 5,
        "abba_max_seconds": 9000,
        "base_image": ("<internal registry>/kernelbench/wro-vllm-consistent-repo-base:v1"
                       "@sha256:64a9efc82e1003574ee10461382ae088f12e2fa3432b080dbd72d3e73630c259"),
        "source_commit": "1da94e673c257373280026f75ceb4effac80e892",
        "baked_repo_commit_subject": "baseline",
        "expected_case_count": len(suite["timed_cases"]),
        "expected_correctness_case_count": len(suite["correctness_cases"]),
        "expected_case_ids": [c["case_id"] for c in suite["timed_cases"]],
        "expected_correctness_case_ids": [c["case_id"] for c in suite["correctness_cases"]],
        "measured_h20_peak": {
            "peak_hbm_gbps": 3687.3,
            "peak_hbm_gbps_reproduced": 3683.0,
            "peak_hbm_gbps_median": 3671.4,
            "read_bound_gbps": 3671.8,
            "copy_bound_contiguous_gbps": 3463.7,
            "copy_bound_page_run_gbps": {
                "page16_row2048B": 3207.0, "page16_row128B": 2124.9,
                "page64_row256B": 3220.8, "page64_row1152B": 3245.7,
                "page256_row1024B": 3316.3},
            "device": "NVIDIA H20 sm90 95GiB, driver 550.144.03",
            "measured_on": "NVIDIA H20 2026-07-26",
            "harness_denominator_gbps": 3687.3,
            "method": ("peak = 3-stream fp32 elementwise add over 512 MiB buffers, best of 30 "
                       "(measured twice in two independent sessions: 3687.3 and 3683.0, a 0.1% spread; the harness denominator is 3687.3 and every calibrated number in this manifest was produced with it); read-bound = vectorised Triton "
                       "grid-stride reduce over 2 GiB; copy-bound = vectorised Triton "
                       "grid-stride copy over 1 GiB (torch copy_ reaches only 2082.9); "
                       "copy-bound page-run = the same copy kernel restricted to shuffled "
                       "page-sized runs, i.e. the honest achievable ceiling for THIS task's "
                       "traffic at each page/row geometry")},
        "strong_baseline": {
            "impl": "tests/harness/baseline_kv_traffic.py",
            "geomean_sol_fraction": 0.13014,
            "calibrated_on": "NVIDIA H20 2026-07-26 (device-synced timing harness; mean of 4 fresh sub-process measurements measured 0.13253 / 0.13163 / 0.12877 / 0.12761, spread 3.9%)",
            "calibrated_on_event_timing_harness": 0.13657,
            "per_case": {"t1_gather_gqa8_ragged": 0.1008, "t2_gather_tinyrow_h1d64": 0.0796,
                         "t3_gather_page64_h2d64": 0.0865, "t4_gather_wide_h1d576": 0.0928,
                         "t5_scatter_gqa8_ragged": 0.1800, "t6_scatter_tinyrow_h1d64": 0.1258,
                         "t7_scatter_page256_h4d128": 0.1909,
                         "t8_copypages_page16_gqa8": 0.2335,
                         "t9_copypages_wide_h1d576": 0.2546},
            "reward_noise_floor": "baseline-vs-itself measured 0.9965 and 1.0362 in the same session (+-3.6% on the REWARD, which is what matters: the reward divides by a baseline re-measured in the same run, so absolute drift largely cancels); per-case step-time spread within a run 1-5%",
            "noise_floor_pct_per_case": "1-5 (device-synced harness); 0.09-1.55 (event harness)"},
        "gates": {
            "bit_exact_roundtrip": "torch.equal on every gathered/copied element",
            "poison_full_write": "verifier-owned outputs pre-filled with -12345.0",
            "no_alias": "source buffers overwritten after every scatter, then read back",
            "current_plan": "two begin_step calls back to back; the second must win",
            "page_addressing": "gather request b through request (b+1)'s PAGES: storage must be addressed by the physical page the block table names, not by (request, position)",
            "pool_footprint": "DUAL: live bytes <= 1.10x nominal pool after allocate() AND again after the write phase (a lazily built shadow layout cannot slip past the first check)",
            "peak_alloc": "pool budget + working allowance + the harness's own buffers",
            "plausibility_hard_fail_sol": 1.02,
            "timing": "harness-owned time.perf_counter bracketed by full torch.cuda.synchronize() on both sides of the step (side-stream work cannot escape the window); a CUDA-event pair is reported alongside as a cross-check"},
        "frozen_surface_sha256": {rel: sha(os.path.join(tests, rel)) for rel in FROZEN},
        "note": ("Uploaded FRESH at scoring; the baked /opt/verifier copy is a root-0700 "
                 "fallback only. Regenerate with solution/gen_manifest.py after any edit."),
    }
    out = os.path.join(tests, "verifier-correctness-manifest.json")
    with open(out, "w") as fh:
        json.dump(man, fh, indent=1)
    print("wrote", out)
    for k, v in man["frozen_surface_sha256"].items():
        print("  %-34s %s" % (k, v[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
