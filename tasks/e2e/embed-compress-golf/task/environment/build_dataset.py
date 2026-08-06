"""Build the e2e-g2-embed-quality-budget retrieval dataset at Docker BUILD time.

Runs on a CPU-only build worker with proxy internet (allow_internet=false at scoring).
It downloads WikiText-103 (via the HF/ModelScope proxy, mirroring subquad's build_dataset),
constructs a self-contained retrieval benchmark, and writes:

  Agent-visible DEV split (under /data — the agent's progress monitor):
    /data/retrieval/dev_corpus.jsonl     {"id","text"}   passages
    /data/retrieval/dev_queries.jsonl    {"id","text"}   queries
    /data/retrieval/dev_qrels.json       {qid: {cid: rel}}

  HELD-OUT TEST split (Dockerfile moves these under /opt/verifier root-0700; the harness
  re-encodes them at score time; the agent NEVER sees them):
    /tmp/heldout_corpus.jsonl
    /tmp/heldout_queries.jsonl
    /tmp/heldout_qrels.json

Construction (deterministic; seeded): split WikiText-103 paragraphs into DEV and TEST
paragraph pools (disjoint). Each pool's paragraphs (len-filtered) form the corpus. For a
random subset of corpus paragraphs we draw ONE sentence as a query whose single graded-
relevant document (rel=1) is the source paragraph. Distractor paragraphs make retrieval
non-trivial. DEV and TEST are same-distribution but disjoint, so the agent's dev nDCG is a
faithful proxy for the scored held-out nDCG without leaking the scored labels.
"""
from __future__ import annotations

import json
import random
import re
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

DATA_DIR = Path("/data/retrieval")
TMP_DIR = Path("/tmp")

WIKITEXT_BASE = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1"
)
# Two parquet shards give plenty of paragraphs; we cap what we use.
WIKITEXT_TRAIN_URLS = (
    f"{WIKITEXT_BASE}/train-00000-of-00002.parquet",
)
WIKITEXT_PARQUETS = tuple(
    Path(f"/tmp/g2_embed_wikitext_{i:02d}.parquet") for i in range(len(WIKITEXT_TRAIN_URLS))
)

SEED = 20240725
MIN_PARA_CHARS = 400          # paragraphs long enough to host a distinctive query sentence
MAX_PARA_CHARS = 2000
MIN_SENT_CHARS = 40
CORPUS_PER_SPLIT = 8000       # passages in each of dev / test corpora
QUERIES_PER_SPLIT = 1500      # queried subset of each corpus


def _read_paragraphs(path: Path) -> list[str]:
    table = pq.read_table(str(path), columns=["text"])
    paras: list[str] = []
    buf: list[str] = []
    for row in table.column("text").to_pylist():
        if row is None:
            continue
        line = row.rstrip("\n")
        stripped = line.strip()
        # WikiText headings look like " = = Title = = "; treat as separators.
        if not stripped or re.match(r"^=+ .* =+$", stripped):
            if buf:
                paras.append(" ".join(buf).strip())
                buf = []
            continue
        buf.append(stripped)
    if buf:
        paras.append(" ".join(buf).strip())
    # length filter
    out = [p for p in paras if MIN_PARA_CHARS <= len(p) <= MAX_PARA_CHARS]
    return out


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(paragraph) if len(s.strip()) >= MIN_SENT_CHARS]


MIN_BODY_CHARS = 200          # 摘掉一句之后段落正文仍需保留的最小长度


def _collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _build_split(paras: list[str], prefix: str, rng: random.Random):
    """构造语料 + 查询。

    ⚠️ 2026-07-26 反捷径修复（原版有致命缺陷）：
    原版把**完整**段落放进语料，同时从该段落里抽一句当查询 —— 于是 gold 文档
    逐字包含 query。这让一个 5 行的「子串包含 / 精确短语匹配」打分器就能拿到
    nDCG@10≈1.0，题目从「检索/重排质量」退化成「有没有发现 query 是子串」，
    强基线（MaxSim / 混合 RRF / 压缩向量）反而被这种平凡技巧碾压。

    修复：对**每一个**段落都摘掉它最长的那句，只把剩余正文写进语料；被摘出的
    句子只在采样到的子集里充当查询。两个要点：
      1) gold 文档不再包含 query 的任何逐字片段 —— 词汇重叠退回到「同段落其余
         句子的自然共享词汇」这个正常水平，语义匹配才是主信号；
      2) 「被摘一句」对全语料一致，不会留下「gold 段落系统性更短」的长度捷径
         （如果只摘被查询的那些段落就会有这个漏洞）。
    """
    held: list[str | None] = []
    bodies: list[str] = []
    for p in paras:
        sents = _sentences(p)
        picked = None
        if len(sents) >= 2:                      # 摘掉一句后必须还剩正文
            q = max(sents, key=len)              # 最长句最distinctive
            body = _collapse(p.replace(q, " ", 1))
            if len(body) >= MIN_BODY_CHARS:
                picked = q
                bodies.append(body)
        if picked is None:
            bodies.append(_collapse(p))
        held.append(picked)

    corpus_ids = [f"{prefix}_d{i}" for i in range(len(bodies))]
    corpus_rows = [{"id": cid, "text": t} for cid, t in zip(corpus_ids, bodies)]

    # 查询子集只从「成功摘出句子」的段落里采（确定性：同 seed 同结果）
    candidate_idx = [i for i, q in enumerate(held) if q is not None]
    rng.shuffle(candidate_idx)
    query_rows = []
    qrels: dict[str, dict[str, int]] = {}
    for n_taken, di in enumerate(candidate_idx[:QUERIES_PER_SPLIT]):
        qid = f"{prefix}_q{n_taken}"
        query_rows.append({"id": qid, "text": held[di]})
        qrels[qid] = {corpus_ids[di]: 1}

    # 构建期自证：query 不得再是其 gold 文档的子串（必须 0/N）
    body_by_id = {cid: t for cid, t in zip(corpus_ids, bodies)}
    leaked = sum(1 for r in query_rows
                 if r["text"] in body_by_id[next(iter(qrels[r["id"]]))])
    print(f"[{prefix}] verbatim-substring leakage: {leaked}/{len(query_rows)} (must be 0)", flush=True)
    if leaked:
        raise RuntimeError(f"{prefix}: {leaked} queries still appear verbatim in their gold passage")
    return corpus_rows, query_rows, qrels


def _write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for url, path in zip(WIKITEXT_TRAIN_URLS, WIKITEXT_PARQUETS, strict=True):
        if not path.exists():
            urllib.request.urlretrieve(url, str(path))
    paras: list[str] = []
    for path in WIKITEXT_PARQUETS:
        paras.extend(_read_paragraphs(path))
    if len(paras) < 2 * CORPUS_PER_SPLIT + 100:
        raise RuntimeError(
            f"Only {len(paras)} usable paragraphs; need >= {2 * CORPUS_PER_SPLIT + 100}. Parquet truncated?"
        )

    rng = random.Random(SEED)
    rng.shuffle(paras)
    dev_paras = paras[:CORPUS_PER_SPLIT]
    test_paras = paras[CORPUS_PER_SPLIT:2 * CORPUS_PER_SPLIT]

    dev_corpus, dev_queries, dev_qrels = _build_split(dev_paras, "dev", random.Random(SEED + 1))
    test_corpus, test_queries, test_qrels = _build_split(test_paras, "test", random.Random(SEED + 2))

    # agent-visible dev split
    _write_jsonl(DATA_DIR / "dev_corpus.jsonl", dev_corpus)
    _write_jsonl(DATA_DIR / "dev_queries.jsonl", dev_queries)
    (DATA_DIR / "dev_qrels.json").write_text(json.dumps(dev_qrels, ensure_ascii=False), encoding="utf-8")

    # held-out test split -> /tmp (Dockerfile relocates to /opt/verifier root-0700)
    _write_jsonl(TMP_DIR / "heldout_corpus.jsonl", test_corpus)
    _write_jsonl(TMP_DIR / "heldout_queries.jsonl", test_queries)
    (TMP_DIR / "heldout_qrels.json").write_text(json.dumps(test_qrels, ensure_ascii=False), encoding="utf-8")

    print(
        f"dev: {len(dev_corpus)} docs / {len(dev_queries)} queries; "
        f"test(held-out): {len(test_corpus)} docs / {len(test_queries)} queries",
        flush=True,
    )

    for p in WIKITEXT_PARQUETS:
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
