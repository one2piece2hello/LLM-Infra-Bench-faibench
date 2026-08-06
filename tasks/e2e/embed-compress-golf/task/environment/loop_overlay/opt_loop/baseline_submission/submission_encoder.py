"""Starter template for e2e-g2-embed-compress-golf (family C, aggressive-compression golf).

Copy this to /app/submission/submission_encoder.py and make it your own. TWO hooks:

  * REQUIRED `load_encoder_for_verification(device)` -> object with `.encode(texts, ...) ->
    np.ndarray[n, dim]`. The grader RE-MEASURES per-vector bytes = dim*itemsize and rejects you
    if it exceeds a SMALL budget. A 384-float32 (1536 B) AND a 384-int8 (384 B) vector are BOTH
    too big — you must compress hard: binary quantization (pack bits into uint8), product
    quantization (PQ codes), or aggressive int8 with few dims.

  * OPTIONAL `load_refiner_for_verification(device)` -> object with
    `.rescore(query_text: str, doc_texts: list[str]) -> list[float]`. After the grader's
    first-stage cosine search over your COMPRESSED vectors produces a top-N shortlist, it calls
    your refiner to re-rank that shortlist. The refiner runs on a BOUNDED shortlist the grader
    chooses (so it can't become a second full-corpus pass). A good refiner (e.g. full-precision
    re-scoring of the shortlist, or a cross-encoder) recovers quality the compression lost.

The grader owns the corpus, queries, relevance labels, the search, the refinement orchestration,
the metric, and the byte measurement. It scores held-out nDCG@10 AFTER refinement on a BOUNDED
0.0-1.0 scale against a strong compression baseline at the same byte budget: matching that
baseline or doing worse scores 0, and the score rises as you close the remaining quality gap.
🔴 HARD BUDGET: at most 64 bytes per vector (dim x dtype.itemsize), re-measured by the grader from
the array you actually return; one byte over scores 0. You never see the held-out text or labels.

THE BINARY-QUANTIZATION MATH (example): pack `dim` bits (sign of each float dim) into `dim/8`
uint8 bytes. 512 float dims -> 512 bits -> 64 uint8 bytes. Hamming distance on packed bits ~
cosine on signs. Consult sentence_transformers.util.quantization (quantize_embeddings supports
'binary'/'ubinary') and the public docs.
"""
from __future__ import annotations

import os
import numpy as np


DEFAULT_MODEL_DIR = os.environ.get("BASE_EMBED_MODEL", "/opt/models/all-MiniLM-L6-v2")


class BinaryPackedEncoder:
    """MiniLM (384 dims) binarized by sign, then bit-packed to uint8: 384 bits -> 48 bytes.

    This is only a scaffold — plain sign-binarization is a weak compression baseline. Improve it
    (PQ / learned rotation / more informative bit allocation)."""

    def __init__(self, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(DEFAULT_MODEL_DIR, device=device)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 256,
        is_query: bool = False,
        normalize: bool = False,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        emb = self.model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        emb = np.asarray(emb, dtype=np.float32)
        bits = (emb > 0).astype(np.uint8)          # sign bits, shape [n, 384]
        packed = np.packbits(bits, axis=1)          # -> [n, 48] uint8 (384/8)
        # NOTE: cosine on packed uint8 bytes is NOT Hamming — this naive packing scores poorly
        # under cosine search. A stronger submission unpacks to +/-1 floats within the byte
        # budget, or uses PQ codes whose cosine geometry is meaningful. Left as an exercise.
        return np.ascontiguousarray(packed)


def load_encoder_for_verification(device: str = "cpu"):
    """REQUIRED entry point."""
    return BinaryPackedEncoder(device=device)


# Optional: define load_refiner_for_verification(device) to enable the two-stage refine step.
# Example skeleton (disabled by default; a real refiner re-embeds the shortlist at full precision
# or uses a cross-encoder):
#
# class ShortlistRefiner:
#     def __init__(self, device="cpu"):
#         from sentence_transformers import SentenceTransformer
#         self.model = SentenceTransformer(DEFAULT_MODEL_DIR, device=device)
#     def rescore(self, query_text, doc_texts):
#         q = self.model.encode([query_text], normalize_embeddings=True)[0]
#         d = self.model.encode(doc_texts, normalize_embeddings=True)
#         return (d @ q).tolist()
#
# def load_refiner_for_verification(device="cpu"):
#     return ShortlistRefiner(device=device)
