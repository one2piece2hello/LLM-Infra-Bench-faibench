"""e2e-g2-embed-compress-golf STRONG-BASELINE reference (reviewer-only; the 1.0 anchor).

NOT model-visible. Seeded by the strong_baseline VERIFIER_MODE from /opt/strong_baseline at calibration time.

Strong recipe (two-stage, deliberately above the naive packbits scaffold):
  * FIRST STAGE encoder: reduce MiniLM 384-dim to a small subspace (PCA/random-rotation-free here:
    take the top-`keep_dims` dims after normalisation) and store as SIGN-based int8 in {-1,+1}, so
    cosine over these int8 vectors == cosine over the retained-dim signs. keep_dims chosen so
    keep_dims * 1 byte <= the 64-byte budget (candidate keep_dims=64 -> 64 bytes). This gives a
    meaningful-geometry compressed index (unlike naive bit-packing whose bytes break cosine).
  * REFINE STAGE: a full-precision shortlist refiner re-scores the first-stage top-N by cosine on
    the full 384-dim float embeddings. This recovers most of the quality the 1-bit-per-dim
    compression loses, on a BOUNDED shortlist — which is exactly the intended two-stage pattern.

Why this beats the naive scaffold: the scaffold packs 384 sign bits into 48 uint8 bytes and lets
the grader run cosine on those PACKED bytes, whose numeric values do NOT reflect sign geometry, so
first-stage recall is poor and there is no refiner. This baseline keeps cosine-meaningful int8
signs AND adds full-precision refinement.

🔴 ANCHOR RE-CALIBRATION RECIPE (on an H20):
  * confirm MiniLM dim (384); set max_bytes_per_vector=64 so 384-int8 (384 B) is OVER budget and a
    64-dim sign-int8 index (64 B) fits; confirm the naive packbits scaffold scores well below this;
  * run this baseline >=5x through the verifier -> strong_baseline_ndcg = median (AFTER refine);
  * set quality_floor / min_plausible bands; confirm oracle passes its own gates 5/5; confirm a
    random/constant control fails the anti-degenerate probes; confirm refine_shortlist_n gives the
    refiner a meaningful lift.
"""
from __future__ import annotations

import os
import numpy as np

DEFAULT_MODEL_DIR = os.environ.get("BASE_EMBED_MODEL", "/opt/models/all-MiniLM-L6-v2")


class SignInt8Encoder:
    """First-stage compressed index: top-`keep_dims` normalised dims stored as int8 signs in {-1,+1}.

    dim=keep_dims, itemsize=1 -> keep_dims bytes/vec (fits the 64-byte budget at keep_dims=64)."""

    def __init__(self, device: str = "cpu", keep_dims: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(DEFAULT_MODEL_DIR, device=device)
        self.keep_dims = int(keep_dims)

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
        emb = np.asarray(emb, dtype=np.float32)[:, : self.keep_dims]
        signs = np.where(emb >= 0, 1, -1).astype(np.int8)
        return np.ascontiguousarray(signs)


class FullPrecisionRefiner:
    """Re-scores a shortlist by cosine on the FULL 384-dim float embeddings (harness-bounded)."""

    def __init__(self, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(DEFAULT_MODEL_DIR, device=device)

    def rescore(self, query_text: str, doc_texts: list[str]) -> list[float]:
        q = self.model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        d = self.model.encode(doc_texts, convert_to_numpy=True, normalize_embeddings=True)
        return (np.asarray(d, dtype=np.float32) @ np.asarray(q, dtype=np.float32)).tolist()


def load_encoder_for_verification(device: str = "cpu"):
    return SignInt8Encoder(device=device, keep_dims=64)


def load_refiner_for_verification(device: str = "cpu"):
    return FullPrecisionRefiner(device=device)
