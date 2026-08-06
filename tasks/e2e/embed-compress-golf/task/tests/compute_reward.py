"""e2e-g2-embed-compress-golf verifier (family C, quality-under-budget; G2 大规模 Embedding 系统).

Distinct angle from e2e-g2-embed-quality-budget: an AGGRESSIVE fixed per-vector byte budget
(small enough to force binary / product-quantization compression, not mere dimension choice),
plus an OPTIONAL harness-orchestrated two-stage RETRIEVE-then-REFINE pipeline. The candidate
submits a compressed-index encoder AND (optionally) a lightweight refinement scorer; the harness
measures nDCG@10 AFTER refinement.

Single-shot, eval-only. The harness:
  * loads the encoder via `load_encoder_for_verification(device)` in a CLEAN -I process;
  * encodes OUR held-out corpus + queries; RE-MEASURES per-vector bytes = dim*itemsize;
  * first stage: harness-owned cosine search over the COMPRESSED vectors -> top-N shortlist;
  * refine stage (optional): if the submission exposes `load_refiner_for_verification`, the harness
    calls refiner.rescore(query_text, [shortlist_doc_texts]) -> scores, and re-ranks the shortlist;
    the refiner runs on a BOUNDED shortlist WE choose, so it cannot become a second full-corpus pass;
  * computes held-out nDCG@10 on the final ranking;
  * reward = min(1.0, ln(ratio/ref_speedup)/ln(ref_speedup)) if ratio > ref_speedup else 0.0 in [0,1] per the bench reward spec, where
    ratio = candidate_ndcg / strong_baseline_ndcg and ref_speedup is a frozen AUTHORING-TIME
    constant (the oracle is never run at scoring). 0 on any hard-fail, on ratio <= 1, and on a
    missing/invalid ref_speedup.  ref_speedup = 1.4290238072817099 since 2026-07-27 — the
    the authoring measurement of the best in-budget variant (32 dims x float16 = 64 B/vector,
    nDCG@10 = 0.6561377101372043).  🔴 DISCLOSED SEMANTIC MISMATCH: `ratio` is a QUALITY ratio
    (nDCG@10 under a fixed byte budget), NOT a wall-clock speedup — nothing here is timed. The
    arithmetic is legal (1.42902 >= reward.md's 1.15 legality floor, sigma = 0) and the bounded log
    form is KEPT (owner ruling 2026-07-27: the ceiling clears the floor with margin, is well
    conditioned, and binarising would destroy a genuine gradient -- naive 0.319 -> 0.0, baseline
    1.0000011 -> 1.5e-06, oracle 1.42902 -> 0.5, cap only at ratio 2.042). The naming, not the
    reward, is what was fixed: `speedup`/`ref_speedup` keep their reward.md names for schema
    compatibility, and the result JSON now also carries `metric_kind: "quality_ratio"`,
    `metric_name: "ndcg@10"`, `metric_direction: "higher_is_better"` and `timing_measured: false`
    so no downstream consumer can read 1.429 as a latency ratio.
    [superseded form: candidate_ndcg / strong_baseline_ndcg, 1.0 anchor, no cap]

Everything (corpus, queries, labels, search, refinement orchestration, metric, byte measurement)
is HARNESS-OWNED. The refiner sees only the shortlist doc TEXT WE hand it for the held-out queries,
never the labels; a cached query->doc table is useless.

Thresholds are read from the FROZEN manifest; all PLACEHOLDERS to be CALIBRATED on NVIDIA H20.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


# -----------------------------------------------------------------------------
# Frozen paths + manifest-driven thresholds
# -----------------------------------------------------------------------------
WORKSPACE = Path(os.environ.get("SUBMISSION_DIR", "/app/submission"))


def _frozen(name: str, env: str | None = None) -> Path:
    if env and os.environ.get(env):
        return Path(os.environ[env])
    for base in ("/tests", "/opt/verifier"):
        p = Path(base) / name
        if p.exists():
            return p
    return Path("/opt/verifier") / name


MANIFEST_PATH = _frozen("verifier-correctness-manifest.json", "E2E_MANIFEST_PATH")
HELDOUT_CORPUS_PATH = _frozen("heldout_corpus.jsonl", "E2E_CORPUS_PATH")
HELDOUT_QUERIES_PATH = _frozen("heldout_queries.jsonl", "E2E_QUERIES_PATH")
HELDOUT_QRELS_PATH = _frozen("heldout_qrels.json", "E2E_QRELS_PATH")

_DEFAULTS = {
    # AGGRESSIVE budget: small enough to force binary / PQ compression.
    "max_bytes_per_vector": 64,           # e.g. 512 binary dims (=64 B) or 64 int8 dims, etc.
    "ndcg_k": 10,                         # scored cutoff (after refinement)
    "refine_shortlist_n": 50,             # harness-chosen shortlist size for the refine stage
    "quality_floor_ndcg": 0.08,
    "min_plausible_ndcg": 0.02,
    "strong_baseline_ndcg": 0.40,         # the ratio denominator (calibrated in the manifest)
    "ref_speedup": None,                  # reward.md reward denominator (frozen const)
    "embedding_variance_floor": 1e-9,     # binary vectors have small per-dim variance; loose floor
    "min_distinct_fraction": 0.80,
    "max_encode_batch": 256,
    "refine_max_calls": 6000,             # cap total refiner query invocations (bounded work)
    "probe_seed": 20240725,
    "max_corpus_docs": 60000,
    "max_query_count": 6000,
    "encode_timeout_sec": 1800,
}


def load_manifest() -> dict[str, Any]:
    cfg = dict(_DEFAULTS)
    try:
        if MANIFEST_PATH.exists():
            m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            th = m.get("thresholds", m)
            for k in _DEFAULTS:
                if k in th and th[k] is not None:
                    cfg[k] = th[k]
    except Exception:
        pass
    return cfg


# -----------------------------------------------------------------------------
# Check / result helpers (shared shape with the sibling verifier)
# -----------------------------------------------------------------------------
class Check:
    def __init__(self, name: str, passed: bool, message: str, details: dict[str, Any] | None = None, hard: bool = True):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.hard = hard


def result(name: str, passed: bool, message: str, hard: bool = True, **details: Any) -> Check:
    return Check(name, passed, message, details, hard)


def sanitize_python_path(workspace: Path) -> None:
    blocked = {workspace.resolve(), Path.cwd().resolve()}
    clean: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            resolved = Path(entry).resolve()
        except OSError:
            continue
        if resolved in blocked:
            continue
        clean.append(entry)
    sys.path[:] = clean


def is_regular_workspace_file(path: Path, workspace: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.is_symlink():
        return False, "symlinks are not accepted"
    if not path.is_file():
        return False, "not a regular file"
    try:
        path.resolve().relative_to(workspace.resolve())
    except ValueError:
        return False, "file resolves outside the submission dir"
    return True, "present"


def import_solution(path: Path):
    spec = importlib.util.spec_from_file_location("submitted_encoder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["submitted_encoder"] = module
    workspace_str = str(path.parent.resolve())
    inserted = False
    if workspace_str not in sys.path:
        sys.path.insert(0, workspace_str)
        inserted = True
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            try:
                sys.path.remove(workspace_str)
            except ValueError:
                pass
    return module


# -----------------------------------------------------------------------------
# Held-out data loading (harness-owned)
# -----------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("# provenance-marker"):
            continue
        rows.append(json.loads(line))
    return rows


def load_heldout(cfg: dict[str, Any]):
    for p in (HELDOUT_CORPUS_PATH, HELDOUT_QUERIES_PATH, HELDOUT_QRELS_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Held-out asset not found at {p}")
    corpus_rows = _read_jsonl(HELDOUT_CORPUS_PATH)[: int(cfg["max_corpus_docs"])]
    query_rows = _read_jsonl(HELDOUT_QUERIES_PATH)[: int(cfg["max_query_count"])]
    qrels = json.loads(HELDOUT_QRELS_PATH.read_text(encoding="utf-8"))
    corpus_ids = [str(r["id"]) for r in corpus_rows]
    corpus_texts = [str(r["text"]) for r in corpus_rows]
    query_ids = [str(r["id"]) for r in query_rows]
    query_texts = [str(r["text"]) for r in query_rows]
    return corpus_ids, corpus_texts, query_ids, query_texts, qrels


# -----------------------------------------------------------------------------
# Encoder + optional refiner invocation
# -----------------------------------------------------------------------------
def call_encoder_loader(module: Any, device: str):
    loader = getattr(module, "load_encoder_for_verification", None)
    if loader is None:
        raise RuntimeError(
            "submission must define load_encoder_for_verification(device) returning an object with "
            ".encode(list[str], batch_size, is_query, normalize) -> np.ndarray[n, dim]."
        )
    sig = inspect.signature(loader)
    kwargs: dict[str, Any] = {}
    positional: list[Any] = []
    for name, parameter in sig.parameters.items():
        if name == "device":
            kwargs[name] = device
        elif parameter.default is inspect._empty and not positional:
            positional.append(device)
    encoder = loader(*positional, **kwargs)
    if encoder is None:
        raise RuntimeError("load_encoder_for_verification returned None")
    return encoder


def call_refiner_loader(module: Any, device: str):
    """Optional. Returns a refiner exposing .rescore(query_text, list[doc_text]) -> list[float],
    or None if the submission does not provide one (first-stage ranking is then final)."""
    loader = getattr(module, "load_refiner_for_verification", None)
    if loader is None:
        return None
    sig = inspect.signature(loader)
    kwargs: dict[str, Any] = {}
    positional: list[Any] = []
    for name, parameter in sig.parameters.items():
        if name == "device":
            kwargs[name] = device
        elif parameter.default is inspect._empty and not positional:
            positional.append(device)
    return loader(*positional, **kwargs)


def _encode(encoder: Any, texts: list[str], cfg: dict[str, Any], is_query: bool) -> np.ndarray:
    batch_size = int(cfg["max_encode_batch"])
    enc = getattr(encoder, "encode", None)
    if enc is None or not callable(enc):
        raise RuntimeError("encoder has no callable .encode method")
    sig = inspect.signature(enc)
    kwargs: dict[str, Any] = {}
    if "batch_size" in sig.parameters:
        kwargs["batch_size"] = batch_size
    if "is_query" in sig.parameters:
        kwargs["is_query"] = is_query
    if "normalize" in sig.parameters:
        kwargs["normalize"] = False
    if "convert_to_numpy" in sig.parameters:
        kwargs["convert_to_numpy"] = True
    if "show_progress_bar" in sig.parameters:
        kwargs["show_progress_bar"] = False
    out = enc(texts, **kwargs)
    arr = np.asarray(out)
    if arr.ndim != 2 or arr.shape[0] != len(texts):
        raise RuntimeError(f"encode returned shape {arr.shape}, expected ({len(texts)}, dim)")
    return arr


def _measure_vector_bytes(arr: np.ndarray) -> tuple[int, int, str]:
    dim = int(arr.shape[1])
    itemsize = int(arr.dtype.itemsize)
    return dim * itemsize, dim, str(arr.dtype)


# -----------------------------------------------------------------------------
# Harness-owned retrieval, refinement orchestration, nDCG@k
# -----------------------------------------------------------------------------
def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    m = mat.astype(np.float32, copy=False)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return m / norms


def _first_stage_shortlist(query_emb: np.ndarray, corpus_emb: np.ndarray, shortlist_n: int) -> np.ndarray:
    """Top-N corpus indices per query by cosine over the COMPRESSED vectors (harness search)."""
    q = _l2_normalise(query_emb)
    c = _l2_normalise(corpus_emb)
    n_q = q.shape[0]
    shortlist_n = min(shortlist_n, c.shape[0])
    scores = q @ c.T
    idx = np.argpartition(-scores, kth=shortlist_n - 1, axis=1)[:, :shortlist_n]
    out = np.empty((n_q, shortlist_n), dtype=np.int64)
    for i in range(n_q):
        cols = idx[i]
        order = np.lexsort((cols, -scores[i, cols]))
        out[i] = cols[order]
    return out


def _apply_refiner(refiner: Any, query_texts: list[str], corpus_texts: list[str],
                   shortlist: np.ndarray, cfg: dict[str, Any]) -> tuple[list[list[int]], int]:
    """Re-rank each query's shortlist via refiner.rescore(query_text, [doc_texts]) -> scores.

    The refiner is called on a BOUNDED shortlist (harness-chosen size) for at most
    refine_max_calls queries, so it cannot degenerate into a full-corpus second pass."""
    rescore = getattr(refiner, "rescore", None)
    if rescore is None or not callable(rescore):
        raise RuntimeError("refiner has no callable .rescore(query_text, list[doc_text]) method")
    max_calls = int(cfg["refine_max_calls"])
    reranked: list[list[int]] = []
    calls = 0
    for qi in range(len(query_texts)):
        row = shortlist[qi].tolist()
        if calls >= max_calls:
            reranked.append(row)  # leave first-stage order once the call budget is spent
            continue
        docs = [corpus_texts[j] for j in row]
        scores = rescore(query_texts[qi], docs)
        calls += 1
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.shape[0] != len(row) or not np.isfinite(scores).all():
            reranked.append(row)  # malformed refiner output -> fall back to first stage
            continue
        order = np.lexsort((np.arange(len(row)), -scores))  # score desc, tie: shortlist order
        reranked.append([row[j] for j in order])
    return reranked, calls


def _ndcg_at_k(ranked_corpus_ids: list[list[str]], query_ids: list[str],
               qrels: dict[str, dict[str, int]], k: int) -> float:
    total = 0.0
    counted = 0
    for qi, qid in enumerate(query_ids):
        rels = qrels.get(qid, {})
        if not rels:
            continue
        ranked = ranked_corpus_ids[qi][:k]
        dcg = 0.0
        for rank, cid in enumerate(ranked):
            rel = int(rels.get(cid, 0))
            if rel > 0:
                dcg += (2.0 ** rel - 1.0) / math.log2(rank + 2.0)
        ideal_rels = sorted((int(v) for v in rels.values()), reverse=True)[:k]
        idcg = sum((2.0 ** rel - 1.0) / math.log2(rank + 2.0)
                   for rank, rel in enumerate(ideal_rels) if rel > 0)
        if idcg > 0:
            total += dcg / idcg
            counted += 1
    return (total / counted) if counted else 0.0


# -----------------------------------------------------------------------------
# Anti-degenerate / anti-spoof probes
# -----------------------------------------------------------------------------
def check_embeddings_nondegenerate(corpus_emb: np.ndarray, cfg: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    finite = bool(np.isfinite(corpus_emb).all())
    checks.append(result("Corpus embeddings are finite", finite,
                         "all finite" if finite else "contains NaN/Inf (spoof/degenerate)"))
    if not finite:
        return checks
    per_dim_var = corpus_emb.astype(np.float64).var(axis=0)
    mean_var = float(np.mean(per_dim_var))
    var_floor = float(cfg["embedding_variance_floor"])
    ok_var = math.isfinite(mean_var) and mean_var > var_floor
    checks.append(result("Embedding variance floor (anti-degenerate)", ok_var,
                         f"mean per-dim var={mean_var:.3e} > floor {var_floor:.1e}" if ok_var
                         else f"mean per-dim var={mean_var:.3e} <= floor {var_floor:.1e} (collapsed/constant)",
                         mean_per_dim_variance=mean_var, variance_floor=var_floor))
    rng = np.random.default_rng(int(cfg["probe_seed"]))
    n = corpus_emb.shape[0]
    sample_n = min(n, 4000)
    sample_idx = rng.choice(n, size=sample_n, replace=False) if n > sample_n else np.arange(n)
    sample = corpus_emb[sample_idx].astype(np.float64)
    # for compressed (possibly binary/int) vectors, hash exact rows.
    distinct = len({row.tobytes() for row in np.ascontiguousarray(sample)})
    frac = distinct / sample_n
    min_frac = float(cfg["min_distinct_fraction"])
    ok_distinct = frac >= min_frac
    checks.append(result("Distinct-vector fraction (anti-degenerate)", ok_distinct,
                         f"distinct fraction={frac:.3f} >= {min_frac:.2f}" if ok_distinct
                         else f"distinct fraction={frac:.3f} < {min_frac:.2f} (collapsed to few vectors)",
                         distinct_fraction=frac, min_distinct_fraction=min_frac))
    return checks


# -----------------------------------------------------------------------------
# The validation gate
# -----------------------------------------------------------------------------
def check_required_files(workspace: Path) -> list[Check]:
    path = workspace / "submission_encoder.py"
    passed, message = is_regular_workspace_file(path, workspace)
    return [result("Required file: submission_encoder.py", passed, message)]


def check_validation(workspace: Path, cfg: dict[str, Any]) -> list[Check]:
    try:
        import torch  # noqa: F401
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    module = import_solution(workspace / "submission_encoder.py")
    encoder = call_encoder_loader(module, device)
    checks: list[Check] = []

    corpus_ids, corpus_texts, query_ids, query_texts, qrels = load_heldout(cfg)

    corpus_emb = _encode(encoder, corpus_texts, cfg, is_query=False)
    bytes_per_vec, dim, dtype = _measure_vector_bytes(corpus_emb)
    cap = int(cfg["max_bytes_per_vector"])
    ok_budget = bytes_per_vec <= cap
    checks.append(result("Per-vector byte budget (harness-measured)", ok_budget,
                         f"{bytes_per_vec} bytes/vec (dim={dim} dtype={dtype}) <= {cap}" if ok_budget
                         else f"{bytes_per_vec} bytes/vec (dim={dim} dtype={dtype}) > {cap} (budget exceeded)",
                         bytes_per_vector=bytes_per_vec, max_bytes_per_vector=cap,
                         embedding_dim=dim, embedding_dtype=dtype))
    if not ok_budget:
        return checks

    checks.extend(check_embeddings_nondegenerate(corpus_emb, cfg))
    if not all(c.passed for c in checks if c.hard):
        return checks

    query_emb = _encode(encoder, query_texts, cfg, is_query=True)
    if query_emb.shape[1] != corpus_emb.shape[1]:
        checks.append(result("Query/corpus dimension match", False,
                             f"query dim {query_emb.shape[1]} != corpus dim {corpus_emb.shape[1]}"))
        return checks
    checks.append(result("Query/corpus dimension match", True, f"shared dim={corpus_emb.shape[1]}"))

    # First stage: harness cosine search over the COMPRESSED vectors -> shortlist.
    k = int(cfg["ndcg_k"])
    shortlist_n = max(int(cfg["refine_shortlist_n"]), k)
    shortlist = _first_stage_shortlist(query_emb, corpus_emb, shortlist_n)

    # Optional refine stage (harness-orchestrated, bounded).
    refiner = call_refiner_loader(module, device)
    refine_calls = 0
    if refiner is not None:
        try:
            reranked, refine_calls = _apply_refiner(refiner, query_texts, corpus_texts, shortlist, cfg)
        except Exception as exc:
            checks.append(result("Refiner rescore", False, f"{type(exc).__name__}: {exc}"))
            return checks
        ranked_idx = reranked
        checks.append(result("Refiner rescore", True,
                             f"refined {refine_calls} queries over shortlist_n={shortlist_n}",
                             refine_calls=refine_calls, shortlist_n=shortlist_n, hard=False))
    else:
        ranked_idx = shortlist.tolist()

    ranked_ids = [[corpus_ids[j] for j in row] for row in ranked_idx]
    ndcg = _ndcg_at_k(ranked_ids, query_ids, qrels, k)

    floor = float(cfg["quality_floor_ndcg"])
    min_plausible = float(cfg["min_plausible_ndcg"])
    checks.append(result(f"Held-out nDCG@{k} above quality floor",
                         math.isfinite(ndcg) and ndcg >= floor,
                         f"nDCG@{k}={ndcg:.4f} >= floor {floor:.4f}" if ndcg >= floor
                         else f"nDCG@{k}={ndcg:.4f} < floor {floor:.4f}",
                         hard=False,
                         ndcg=ndcg, ndcg_k=k, quality_floor=floor, refine_calls=refine_calls))
    checks.append(result(f"Held-out nDCG@{k} is plausible (anti-spoof floor)",
                         math.isfinite(ndcg) and ndcg >= min_plausible,
                         f"nDCG@{k}={ndcg:.4f} >= floor {min_plausible:.4f}" if ndcg >= min_plausible
                         else f"nDCG@{k}={ndcg:.4f} below floor {min_plausible:.4f} (likely degenerate/spoof)",
                         ndcg=ndcg, min_plausible=min_plausible))
    return checks


def run_all(workspace: Path, cfg: dict[str, Any]) -> dict[str, list[Check]]:
    checks: dict[str, list[Check]] = {"required_files": check_required_files(workspace)}
    if not all(c.passed for c in checks["required_files"]):
        return checks
    checks["validation"] = check_validation(workspace, cfg)
    return checks


# -----------------------------------------------------------------------------
# Reporting + OPEN-ENDED reward
# -----------------------------------------------------------------------------
def markdown_report(checks: dict[str, list[Check]]) -> str:
    total = sum(len(group) for group in checks.values())
    passed = sum(1 for group in checks.values() for c in group if c.passed)
    lines = ["# Verification Report\n\n", "## Summary\n\n",
             f"- Total checks: {total}\n", f"- Passed: {passed}\n", f"- Failed: {total - passed}\n\n"]
    for category, group in checks.items():
        lines.append(f"## {category.replace('_', ' ').title()}\n\n")
        for c in group:
            status = "PASS" if c.passed else "FAIL"
            details = f" Details: {json.dumps(c.details, sort_keys=True, default=str)}" if c.details else ""
            lines.append(f"- {status}: {c.name} - {c.message}.{details}\n")
        lines.append("\n")
    return "".join(lines)


def leaderboard_metrics(checks: dict[str, list[Check]]) -> dict[str, Any]:
    out: dict[str, Any] = {"score_metric": "ndcg"}
    for group in checks.values():
        for c in group:
            for key in ("ndcg", "ndcg_k", "bytes_per_vector", "max_bytes_per_vector",
                        "embedding_dim", "embedding_dtype", "mean_per_dim_variance",
                        "distinct_fraction", "refine_calls", "shortlist_n"):
                if key in c.details:
                    out[key] = c.details[key]
    return out


def quality_ratio(all_hard_pass: bool, ndcg: float | None, cfg: dict[str, Any]) -> float | None:
    """The RAW improvement ratio over the strong baseline: candidate_ndcg / strong_baseline_ndcg.

    🔴 This is a QUALITY ratio (nDCG@10 under a fixed 64 B/vector budget), NOT a wall-clock
    speedup. It is fed to reward.md's log formula as `speedup` because that formula is the bench's
    only bounded-[0,1] performance envelope; the substitution is deliberate and is documented in
    the frozen manifest under `ref_speedup.metric_kind_caveat`."""
    base = float(cfg["strong_baseline_ndcg"])
    if (not all_hard_pass or not isinstance(ndcg, (int, float)) or not math.isfinite(ndcg)
            or ndcg < 0 or base <= 0):
        return None
    return float(ndcg) / base


def compute_log_reward(all_hard_pass: bool, ndcg: float | None,
                       cfg: dict[str, Any]) -> tuple[float, list[str], float | None]:
    """reward.md 性能类: reward = min(1.0, ln(ratio/ref_speedup)/ln(ref_speedup)) if ratio > ref_speedup else 0.0, range [0, 1].

    Returns (reward, hard_fail_reasons, measured_ratio).

    🔴 `ref_speedup` is an AUTHORING-TIME calibrated CONSTANT read from the frozen manifest; the
    oracle is NOT in the image and is NEVER run at scoring time. When the manifest carries no
    valid constant the run is a HARD FAIL with an explicit reason — it is NEVER silently treated
    as 1.0 (MOD_SPEC 改动 2A)."""
    ratio = quality_ratio(all_hard_pass, ndcg, cfg)
    if not all_hard_pass:
        return 0.0, ["correctness_failed"], ratio
    if ratio is None:
        return 0.0, ["build_or_entry_contract_failed"], None
    ref = cfg.get("ref_speedup")
    try:
        ref = None if ref is None else float(ref)
    except (TypeError, ValueError):
        ref = None
    if ref is None or not math.isfinite(ref) or ref <= 1.0:
        return 0.0, ["ref_speedup_invalid_or_missing"], ratio
    if ratio <= 1.0:
        return 0.0, ["speedup_not_above_baseline"], ratio
    return max(0.0, min(1.0, max(0.0, min(1.0, math.log(ratio) / math.log(ref) - 1.0)))), [], ratio


def write_outputs(checks: dict[str, list[Check]], workspace: Path, cfg: dict[str, Any]) -> bool:
    total = sum(len(group) for group in checks.values())
    passed = sum(1 for group in checks.values() for c in group if c.passed)
    all_hard_pass = all(c.passed for group in checks.values() for c in group if c.hard)
    all_pass = all(c.passed for group in checks.values() for c in group)
    lb = leaderboard_metrics(checks)
    ndcg = lb.get("ndcg")
    reward, hard_fail_reasons, measured_ratio = compute_log_reward(all_hard_pass, ndcg, cfg)
    binary_floor = 1 if all_pass else 0
    lb["quality_ratio_vs_strong_baseline"] = measured_ratio

    metrics = {
        "task_type": "performance",
        "reward": reward,
        "partial_score": reward,
        "binary_pass": binary_floor,
        "all_hard_gates_pass": all_hard_pass,
        "hard_fail_reasons": hard_fail_reasons,
        "speedup": measured_ratio,          # 🔴 a QUALITY ratio (nDCG@10), not a wall-clock speedup
        "ref_speedup": cfg.get("ref_speedup"),
        "cv": {"baseline": 0.0, "candidate": 0.0},   # eval-only + deterministic (sigma = 0 over 5)
        # --- metric self-description (2026-07-27). The reward MATHS is unchanged and validated;
        #     these fields exist so nobody downstream reads `speedup`/`ref_speedup` as latency.
        #     `speedup` is KEPT under that name for reward.md schema compatibility.
        "metric_kind": "quality_ratio",
        "metric_name": "ndcg@10",
        "metric_direction": "higher_is_better",
        "timing_measured": False,
        "speedup_semantics": ("the `speedup` field carries candidate_ndcg / strong_baseline_ndcg at a "
                              "fixed 64 B/vector budget -- a QUALITY ratio, not a wall-clock ratio. "
                              "Nothing in this task is timed; there is no baseline_ms. reward.md's "
                              "bounded log form is applied to that ratio (it is ratio-agnostic; only "
                              "the variable's name says latency)."),
        "reward_form": "reward.md 性能类: min(1.0, ln(ratio/ref_speedup)/ln(ref_speedup)) if ratio > ref_speedup else 0.0 in [0,1]; ratio = candidate_ndcg / strong_baseline_ndcg; ref_speedup = 1.4290238072817099 (frozen authoring-time constant = the best measured IN-BUDGET variant, nDCG@10 0.6561377 vs the 0.459151 strong-baseline anchor; the oracle is never run at scoring). 🔴 DISCLOSED SEMANTIC MISMATCH: ratio is a QUALITY ratio (nDCG@10 at 64 B/vector), not a wall-clock speedup — nothing in this task is timed. 0 at or below the strong baseline, 0.5 at the demonstrated in-budget ceiling, 1.0 only at ratio 2.0421 (unreachable).",
        "strong_baseline_ndcg": cfg["strong_baseline_ndcg"],
        "passed": passed, "total": total, "pass_rate": passed / total if total else 0.0,
        "leaderboard": lb,
        "failed_checks": [
            {"category": cat, "name": c.name, "message": c.message, "hard": c.hard, "details": c.details}
            for cat, group in checks.items() for c in group if not c.passed
        ],
    }

    (workspace / "verification_report.md").write_text(markdown_report(checks), encoding="utf-8")
    vdir = Path("/logs/verifier")
    if vdir.exists():
        (vdir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        (vdir / "benchmark_results.json").write_text(json.dumps(lb, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        (vdir / "correctness_results.json").write_text(json.dumps({"binary_pass": binary_floor, "all_hard_gates_pass": all_hard_pass, "passed": passed, "total": total, "failed_checks": metrics["failed_checks"]}, indent=2, default=str) + "\n", encoding="utf-8")
        (vdir / "verifier_state.json").write_text(json.dumps({"task_id": "e2e-g2-embed-compress-golf", "task_type": "performance", "reward": reward, "hard_fail_reasons": hard_fail_reasons, "speedup": measured_ratio, "ref_speedup": cfg.get("ref_speedup"), "passed": passed, "total": total}, indent=2, default=str) + "\n", encoding="utf-8")
        # 🔴 the authoritative result JSON shape (reward.md §结果 JSON)
        (vdir / "reward.json").write_text(json.dumps({"task_type": "performance", "reward": reward, "hard_fail_reasons": hard_fail_reasons, "speedup": measured_ratio, "ref_speedup": cfg.get("ref_speedup"), "cv": {"baseline": 0.0, "candidate": 0.0}, "metric_kind": "quality_ratio", "metric_name": "ndcg@10", "metric_direction": "higher_is_better", "timing_measured": False, "binary_pass": binary_floor}, indent=2, default=str) + "\n", encoding="utf-8")
        # 🔴 reward.txt carries the FINAL NUMERIC reward (was a 1/0 binary_pass flag before
        #    2026-07-27 — that contradicted the 5-file contract and the reward spec).
        (vdir / "reward.txt").write_text(f"{reward:.6f}\n", encoding="utf-8")
    adir = Path("/logs/artifacts")
    if adir.exists():
        for name in ("submission_encoder.py", "encoder_config.json", "verification_report.md", "action.log"):
            p = workspace / name
            if p.exists():
                try:
                    shutil.copy2(p, adir / name)
                except Exception:
                    pass
    return all_hard_pass


def main() -> int:
    cfg = load_manifest()
    workspace = WORKSPACE if WORKSPACE.exists() else Path.cwd()
    sanitize_python_path(workspace)
    try:
        checks = run_all(workspace, cfg)
    except Exception as exc:
        checks = {"verifier_error": [result("Verifier error", False, f"{type(exc).__name__}: {exc}")]}
    all_hard_pass = write_outputs(checks, workspace, cfg)
    print(markdown_report(checks))
    return 0 if all_hard_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
