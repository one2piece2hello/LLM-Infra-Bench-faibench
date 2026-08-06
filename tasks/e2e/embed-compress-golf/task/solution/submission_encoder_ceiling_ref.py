"""e2e-g2-embed-compress-golf — IN-BUDGET CEILING reference (reviewer-only).

WHY THIS FILE EXISTS
--------------------
`ref_speedup` for this task is the frozen authoring-time constant **1.4290238072817099** — the
quality ratio (`candidate_nDCG@10 / strong_baseline_nDCG@10`) reached by the best IN-BUDGET
variant the authoring session measured (nDCG@10 = 0.6561377101372043, 32 dims ×
float16 = 64 B/vector, refine_calls = 1500; archived at
the authoring calibration run).

That authoring session's *implementation* was never archived — only its measurement was — so
reward.md's "re-measure the reference ≥5×" could not be satisfied and `ref_speedup` sat at
`null`, hard-failing every run. This file re-archives an implementation with the SAME
fingerprint (32 dims, float16, 64 B/vector, full-precision shortlist refiner) so the constant can
be re-measured on demand.

🔴 NOT model-visible. NEVER COPYed into the image (see environment/Dockerfile build assert).
Seeded at review time by uploading it to a scratch dir and pointing `STRONG_BASELINE_DIR` /
`VERIFIER_MODE=strong_baseline` at it, exactly like `submission_encoder_ref.py`.

RECIPE
------
  * first stage: L2-normalised MiniLM 384-d embeddings -> PCA to 32 dims (eigendecomposition of
    the corpus covariance, deterministic sign convention) -> row-L2-normalise -> float16.
    32 * 2 B = 64 B/vector, exactly on the budget. Unlike sign-int8 or naive bit-packing, a PCA
    projection keeps *graded* cosine geometry, so first-stage recall@50 is much higher.
  * refine stage: full-precision (384-d fp32) cosine re-scoring of the harness-chosen shortlist —
    identical to the strong baseline's refiner, so the delta measured against the strong baseline
    is attributable to the FIRST STAGE alone.
  * PCA is fitted on the first `.encode` call (the harness always encodes the corpus first) and
    reused for the query pass, so corpus and queries share one compressed space, as the contract
    requires.

Determinism: MiniLM CPU inference + `numpy.linalg.eigh` on a fixed 384x384 covariance are both
deterministic, and the component signs are canonicalised, so repeated runs agree (the authoring
measurement had sigma = 0).
"""
from __future__ import annotations

import os

import numpy as np

DEFAULT_MODEL_DIR = os.environ.get("BASE_EMBED_MODEL", "/opt/models/all-MiniLM-L6-v2")
OUT_DIM = 32  # 32 * float16(2 B) = 64 B/vector == max_bytes_per_vector


class PCAFp16Encoder:
    """First-stage compressed index: PCA-32 of the normalised MiniLM space, stored float16."""

    def __init__(self, device: str = "cpu", out_dim: int = OUT_DIM):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(DEFAULT_MODEL_DIR, device=device)
        self.out_dim = int(out_dim)
        self._mean: np.ndarray | None = None
        self._components: np.ndarray | None = None  # (out_dim, 384)

    # -- deterministic PCA fit -------------------------------------------------------------
    def _fit(self, emb: np.ndarray) -> None:
        x = emb.astype(np.float64, copy=False)
        mean = x.mean(axis=0)
        xc = x - mean
        cov = (xc.T @ xc) / max(1, xc.shape[0] - 1)
        cov = 0.5 * (cov + cov.T)                      # exact symmetry for eigh
        vals, vecs = np.linalg.eigh(cov)               # ascending eigenvalues
        order = np.argsort(vals)[::-1][: self.out_dim]
        comp = np.ascontiguousarray(vecs[:, order].T)  # (out_dim, 384)
        # canonical sign: largest-magnitude loading of each component is positive
        for i in range(comp.shape[0]):
            j = int(np.argmax(np.abs(comp[i])))
            if comp[i, j] < 0:
                comp[i] = -comp[i]
        self._mean = mean.astype(np.float32)
        self._components = comp.astype(np.float32)

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
        if self._components is None:
            self._fit(emb)                             # fitted on the corpus pass
        z = (emb - self._mean) @ self._components.T    # (n, out_dim)
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        z = z / norms                                  # unit rows -> full fp16 precision in [-1,1]
        return np.ascontiguousarray(z.astype(np.float16))


class FullPrecisionRefiner:
    """Re-scores a harness-bounded shortlist by cosine on the FULL 384-d float embeddings."""

    def __init__(self, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(DEFAULT_MODEL_DIR, device=device)

    def rescore(self, query_text: str, doc_texts: list[str]) -> list[float]:
        q = self.model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        d = self.model.encode(doc_texts, convert_to_numpy=True, normalize_embeddings=True)
        return (np.asarray(d, dtype=np.float32) @ np.asarray(q, dtype=np.float32)).tolist()


def load_encoder_for_verification(device: str = "cpu"):
    return PCAFp16Encoder(device=device, out_dim=OUT_DIM)


def load_refiner_for_verification(device: str = "cpu"):
    return FullPrecisionRefiner(device=device)
