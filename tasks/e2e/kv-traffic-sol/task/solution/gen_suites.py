#!/usr/bin/env python3
"""Emit the HIDDEN scored suite and the PUBLIC dev suite for e2e-b1-kv-traffic-sol.

Pure stdlib text/JSON authoring (no torch, nothing functional) — safe on the login host.
The hidden suite's shapes/regimes/seeds are reviewer-only; the public dev suite is deliberately
disjoint (uniform contexts, different head/page geometry, no ragged mixes, no page-sharing, no
wide-row or tiny-row regime) so it cannot be used to reverse-engineer the scored workload.
"""
import json
import sys

TIMED = [
    # ---- gather: paged pool -> packed varlen buffer -------------------------------------
    dict(case_id="t1_gather_gqa8_ragged", op="gather", seed=110411, batch=8, page_size=16,
         num_kv_heads=8, head_size=128, num_layers=8, num_pages=6000,
         ctx_lens=[0, 512, 1000, 33, 2048, 7, 900, 4096],
         new_lens=[4096, 2048, 1024, 512, 2048, 1024, 512, 2048]),
    dict(case_id="t2_gather_tinyrow_h1d64", op="gather", seed=220717, batch=16, page_size=16,
         num_kv_heads=1, head_size=64, num_layers=8, num_pages=40000,
         ctx_lens=[0, 17, 33, 7, 129, 5, 63, 1, 255, 11, 47, 3, 511, 9, 31, 2],
         new_lens=[4096] * 16),
    dict(case_id="t3_gather_page64_h2d64", op="gather", seed=330913, batch=4, page_size=64,
         num_kv_heads=2, head_size=64, num_layers=8, num_pages=8000,
         ctx_lens=[0, 100, 65, 3], new_lens=[16384, 8192, 8192, 4096]),
    dict(case_id="t4_gather_wide_h1d576", op="gather", seed=441229, batch=6, page_size=64,
         num_kv_heads=1, head_size=576, num_layers=4, num_pages=3000,
         ctx_lens=[0, 64, 33, 200, 1, 500],
         new_lens=[8192, 4096, 4096, 2048, 2048, 1024]),
    # ---- scatter: packed varlen buffer -> paged pool ------------------------------------
    dict(case_id="t5_scatter_gqa8_ragged", op="scatter", seed=551607, batch=8, page_size=16,
         num_kv_heads=8, head_size=128, num_layers=8, num_pages=6000,
         ctx_lens=[64, 0, 777, 128, 1024, 15, 300, 2048],
         new_lens=[2048, 4096, 1024, 2048, 1024, 512, 1024, 2048]),
    dict(case_id="t6_scatter_tinyrow_h1d64", op="scatter", seed=661811, batch=16, page_size=16,
         num_kv_heads=1, head_size=64, num_layers=8, num_pages=40000,
         ctx_lens=[3, 0, 19, 65, 7, 33, 1, 127, 5, 15, 257, 9, 41, 2, 513, 23],
         new_lens=[4096] * 16),
    dict(case_id="t7_scatter_page256_h4d128", op="scatter", seed=772011, batch=4, page_size=256,
         num_kv_heads=4, head_size=128, num_layers=8, num_pages=800,
         ctx_lens=[0, 300, 777, 100], new_lens=[16384, 8192, 8192, 8192]),
    # ---- copy_pages: intra-pool page duplication (copy-on-write / dedup / defrag) -------
    dict(case_id="t8_copypages_page16_gqa8", op="copy_pages", seed=882203, batch=8,
         page_size=16, num_kv_heads=8, head_size=128, num_layers=8, num_pages=6000,
         ctx_lens=[0] * 8, new_lens=[2048] * 8, n_copy_pages=1024),
    dict(case_id="t9_copypages_wide_h1d576", op="copy_pages", seed=992407, batch=4,
         page_size=64, num_kv_heads=1, head_size=576, num_layers=4, num_pages=2400,
         ctx_lens=[0] * 4, new_lens=[12800] * 4, n_copy_pages=800),
]

CORRECT = [
    dict(case_id="c01_single_partial_page", axes="single page + partial tail + batch of 1",
         seed=90101, batch=1, page_size=16, num_kv_heads=2, head_size=64, num_layers=2,
         num_pages=64, seq_lens=[7], write_chunks=[3, 1], read_ranges=[[[0], [7]], [[2], [4]]]),
    dict(case_id="c02_exact_page_multiple", axes="page-aligned lengths",
         seed=90202, batch=2, page_size=16, num_kv_heads=2, head_size=64, num_layers=2,
         num_pages=64, seq_lens=[32, 16], write_chunks=[16],
         read_ranges=[[[0, 0], [32, 16]], [[16, 0], [16, 16]]]),
    dict(case_id="c03_partial_tail_mixed", axes="partial tail pages + ragged lengths",
         seed=90303, batch=3, page_size=32, num_kv_heads=2, head_size=64, num_layers=2,
         num_pages=64, seq_lens=[33, 65, 1], write_chunks=[33, 7, 1],
         read_ranges=[[[0, 0, 0], [33, 65, 1]], [[31, 63, 0], [2, 2, 1]]]),
    dict(case_id="c04_unaligned_subranges", axes="unaligned mid-page read/write offsets",
         seed=90404, batch=2, page_size=16, num_kv_heads=4, head_size=64, num_layers=2,
         num_pages=64, seq_lens=[100, 50], write_chunks=[17, 5, 1],
         read_ranges=[[[3, 17], [7, 3]], [[15, 0], [20, 50]], [[99, 49], [1, 1]]]),
    dict(case_id="c05_shuffled_pages", axes="non-contiguous shuffled physical pages",
         seed=90505, batch=4, page_size=16, num_kv_heads=2, head_size=64, num_layers=2,
         num_pages=256, seq_lens=[64, 48, 17, 1], write_chunks=[9, 33],
         read_ranges=[[[0, 0, 0, 0], [64, 48, 17, 1]]]),
    dict(case_id="c06_shared_pages_prefix", axes="shared/duplicated pages across requests",
         seed=90606, batch=4, page_size=16, num_kv_heads=2, head_size=64, num_layers=2,
         num_pages=256, seq_lens=[64, 64, 32, 16], share_pages=True, write_chunks=[16],
         read_ranges=[[[0, 0, 0, 0], [64, 64, 32, 16]], [[8, 8, 8, 8], [24, 24, 24, 8]]]),
    dict(case_id="c07_max_length_single", axes="max-length request",
         seed=90707, batch=1, page_size=16, num_kv_heads=2, head_size=64, num_layers=2,
         num_pages=300, seq_lens=[4096], write_chunks=[1024, 7],
         read_ranges=[[[0], [4096]], [[4095], [1]]]),
    dict(case_id="c08_zero_length_ranges", axes="zero-length ranges / idle requests",
         seed=90808, batch=3, page_size=16, num_kv_heads=2, head_size=64, num_layers=2,
         num_pages=64, seq_lens=[16, 32, 16], write_chunks=[8],
         read_ranges=[[[0, 0, 0], [0, 8, 0]], [[0, 24, 0], [16, 8, 16]],
                      [[0, 0, 0], [0, 0, 0]]]),
    dict(case_id="c09_page64_single_big", axes="large page size, one request",
         seed=90909, batch=1, page_size=64, num_kv_heads=2, head_size=64, num_layers=2,
         num_pages=64, seq_lens=[1000], write_chunks=[65, 1],
         read_ranges=[[[0], [1000]], [[999], [1]]]),
    dict(case_id="c10_page_size_one", axes="degenerate page_size == 1",
         seed=91010, batch=2, page_size=1, num_kv_heads=2, head_size=64, num_layers=2,
         num_pages=64, seq_lens=[8, 5], write_chunks=[1, 3],
         read_ranges=[[[0, 0], [8, 5]], [[7, 4], [1, 1]]]),
    dict(case_id="c11_wide_rows_h1d576", axes="wide rows (MLA-like head_size)",
         seed=91111, batch=2, page_size=64, num_kv_heads=1, head_size=576, num_layers=2,
         num_pages=32, seq_lens=[128, 70], write_chunks=[70, 3],
         read_ranges=[[[0, 0], [128, 70]], [[63, 65], [65, 5]]]),
    dict(case_id="c12_six_layers_independent", axes="per-layer independence",
         seed=91212, batch=2, page_size=16, num_kv_heads=2, head_size=64, num_layers=6,
         num_pages=64, seq_lens=[48, 33], write_chunks=[17],
         read_ranges=[[[0, 0], [48, 33]], [[32, 16], [16, 17]]]),
    dict(case_id="c13_copy_pages_roundtrip", axes="copy_pages page duplication",
         seed=91313, batch=2, page_size=16, num_kv_heads=2, head_size=64, num_layers=2,
         num_pages=256, seq_lens=[64, 48], write_chunks=[16], n_copy_pages=4,
         read_ranges=[[[0, 0], [64, 48]]]),
    dict(case_id="c14_float16_dtype", axes="fp16 dtype round-trip",
         seed=91414, batch=2, page_size=32, num_kv_heads=2, head_size=64, num_layers=2,
         num_pages=64, seq_lens=[70, 33], dtype="float16", write_chunks=[33, 5],
         read_ranges=[[[0, 0], [70, 33]], [[31, 31], [8, 2]]]),
    dict(case_id="c15_single_token_writes", axes="one-token-at-a-time writes (decode)",
         seed=91515, batch=2, page_size=16, num_kv_heads=4, head_size=64, num_layers=2,
         num_pages=64, seq_lens=[20, 9], write_chunks=[1],
         read_ranges=[[[0, 0], [20, 9]], [[19, 8], [1, 1]]]),
    dict(case_id="c16_oversized_chunk_writes", axes="whole-context single write",
         seed=91616, batch=2, page_size=16, num_kv_heads=2, head_size=64, num_layers=2,
         num_pages=64, seq_lens=[300, 177], write_chunks=[10000],
         read_ranges=[[[0, 0], [300, 177]], [[299, 176], [1, 1]]]),
]

DEV_TIMED = [
    dict(case_id="d1_gather_public", op="gather", seed=1001, batch=2, page_size=32,
         num_kv_heads=4, head_size=128, num_layers=4, num_pages=1200,
         ctx_lens=[0, 0], new_lens=[8192, 8192], timed_blocks=6, warmup_blocks=2),
    dict(case_id="d2_scatter_public", op="scatter", seed=1002, batch=2, page_size=32,
         num_kv_heads=4, head_size=128, num_layers=4, num_pages=1200,
         ctx_lens=[0, 0], new_lens=[8192, 8192], timed_blocks=6, warmup_blocks=2),
    dict(case_id="d3_copypages_public", op="copy_pages", seed=1003, batch=2, page_size=32,
         num_kv_heads=4, head_size=128, num_layers=4, num_pages=1200,
         ctx_lens=[0, 0], new_lens=[4096, 4096], n_copy_pages=256,
         timed_blocks=6, warmup_blocks=2),
]

DEV_CORRECT = [
    dict(case_id="p01_public_basic", axes="page-aligned + partial tail",
         seed=1101, batch=2, page_size=32, num_kv_heads=4, head_size=128, num_layers=2,
         num_pages=64, seq_lens=[64, 33], write_chunks=[33, 5],
         read_ranges=[[[0, 0], [64, 33]], [[31, 16], [8, 8]]]),
    dict(case_id="p02_public_copy_pages", axes="copy_pages round-trip",
         seed=1102, batch=2, page_size=32, num_kv_heads=4, head_size=128, num_layers=2,
         num_pages=128, seq_lens=[64, 64], write_chunks=[32], n_copy_pages=2,
         read_ranges=[[[0, 0], [64, 64]]]),
]


def finish(cases, blocks, warm):
    for c in cases:
        c.setdefault("dtype", "bfloat16")
        if "op" in c:
            c.setdefault("timed_blocks", blocks)
            c.setdefault("warmup_blocks", warm)
    return cases


def main():
    hidden = {"name": "kv_traffic_hidden_v1",
              "timed_cases": finish(TIMED, 10, 3),
              "correctness_cases": finish(CORRECT, 10, 3)}
    dev = {"name": "kv_traffic_public_dev_v1",
           "timed_cases": finish(DEV_TIMED, 6, 2),
           "correctness_cases": finish(DEV_CORRECT, 6, 2)}
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    with open("%s/hidden_suite.json" % out, "w") as fh:
        json.dump(hidden, fh, indent=1)
    with open("%s/dev_suite.json" % out, "w") as fh:
        json.dump(dev, fh, indent=1)
    print("timed=%d correctness=%d dev_timed=%d dev_correct=%d"
          % (len(TIMED), len(CORRECT), len(DEV_TIMED), len(DEV_CORRECT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
