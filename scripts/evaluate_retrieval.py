#!/usr/bin/env python3
"""Offline Retrieval Benchmark Runner for Phase 4 Knowledge-Base Evaluation.

Evaluates semantic retrieval quality (Recall@k, Hit Rate@k, MRR) using the frozen
sentence-transformers/all-MiniLM-L6-v2 ONNX adapter against a curated synthetic corpus
and labeled financial support questions.

Guarantees:
- Fully offline: zero network access, zero database access, CPU-only.
- Production fidelity: reuses production chunker, production ticket text normalization,
  and exact cosine similarity with production tie-breaking (article_id ASC, chunk_index ASC).
- Deterministic, frozen evaluation: reports dataset SHA-256 hashes.
- Immutable run artifacts: writes full metrics and rankings to a newly created run directory
  without overwriting previous runs.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

# Ensure project root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ml.chunking import chunk_article_text
from app.ml.embeddings import EmbeddingModelAdapter, normalize_and_format_text
from app.ml.manifest import (
    DEFAULT_MODEL_DIR_REL,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
)


@dataclass(frozen=True)
class BenchmarkChunk:
    """Represents a chunk passage in the evaluation corpus."""
    article_id: str
    chunk_index: int
    article_title: str
    content: str
    char_count: int
    token_count: int


@dataclass(frozen=True)
class QueryEvaluationResult:
    """Per-query retrieval evaluation results."""
    query_id: str
    subject: str
    relevant_chunks: list[dict[str, Any]]
    relevant_count: int
    retrieved_top_5: list[dict[str, Any]]
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float
    reciprocal_rank: float
    first_relevant_rank: int | None


@dataclass(frozen=True)
class BenchmarkSummaryMetrics:
    """Macro-averaged metrics across all queries."""
    query_count: int
    corpus_article_count: int
    corpus_chunk_count: int
    macro_recall_at_1: float
    macro_recall_at_3: float
    macro_recall_at_5: float
    macro_hit_rate_at_1: float
    macro_hit_rate_at_3: float
    macro_hit_rate_at_5: float
    mean_reciprocal_rank: float


def compute_recall_at_k(relevant_set: set[tuple[str, int]], ranked_list: list[tuple[str, int]], k: int) -> float:
    """Compute Recall@k for a query: |R intersect Top-k| / |R|.

    Raises ValueError if relevant_set is empty or k < 1.
    """
    if not relevant_set:
        raise ValueError("Relevant set must not be empty.")
    if k < 1:
        raise ValueError("k must be >= 1.")
    top_k = set(ranked_list[:k])
    return len(relevant_set.intersection(top_k)) / len(relevant_set)


def compute_hit_rate_at_k(relevant_set: set[tuple[str, int]], ranked_list: list[tuple[str, int]], k: int) -> float:
    """Compute Hit Rate@k for a query: 1.0 if |R intersect Top-k| > 0 else 0.0.

    Raises ValueError if relevant_set is empty or k < 1.
    """
    if not relevant_set:
        raise ValueError("Relevant set must not be empty.")
    if k < 1:
        raise ValueError("k must be >= 1.")
    top_k = set(ranked_list[:k])
    return 1.0 if len(relevant_set.intersection(top_k)) > 0 else 0.0


def compute_reciprocal_rank(relevant_set: set[tuple[str, int]], ranked_list: list[tuple[str, int]]) -> tuple[float, int | None]:
    """Compute Reciprocal Rank (1/rank) of the first relevant chunk in ranked_list.

    Returns (0.0, None) if no relevant chunk is retrieved.
    """
    if not relevant_set:
        raise ValueError("Relevant set must not be empty.")
    for idx, item in enumerate(ranked_list, start=1):
        if item in relevant_set:
            return 1.0 / idx, idx
    return 0.0, None


def exact_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute exact cosine similarity between two unit-normalized vectors.

    For unit vectors, cosine similarity equals the dot product.
    Clamps result to [-1.0, 1.0] and enforces mathematical finiteness.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimensions must match ({len(vec_a)} vs {len(vec_b)})")
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    if not math.isfinite(dot):
        raise ValueError(f"Non-finite dot product encountered: {dot}")
    return max(-1.0, min(1.0, float(dot)))


def validate_benchmark_data(articles: list[dict[str, Any]], queries: list[dict[str, Any]]) -> dict[str, list[BenchmarkChunk]]:
    """Validate articles and queries, chunk articles, and verify all relevance references.

    Returns dict mapping article_id -> list of BenchmarkChunk.
    """
    if not articles:
        raise ValueError("Corpus articles list cannot be empty.")
    if not queries:
        raise ValueError("Evaluation queries list cannot be empty.")

    # 1. Validate articles and chunk deterministically
    seen_article_ids: set[str] = set()
    chunk_catalog: dict[tuple[str, int], BenchmarkChunk] = {}
    articles_by_id: dict[str, list[BenchmarkChunk]] = {}

    for art in articles:
        aid = art.get("id")
        title = art.get("title", "")
        body = art.get("body", "")

        if not aid or not isinstance(aid, str):
            raise ValueError(f"Invalid article ID: {aid}")
        if aid in seen_article_ids:
            raise ValueError(f"Duplicate article ID: {aid}")
        seen_article_ids.add(aid)

        chunks = chunk_article_text(title=title, body=body)
        if not chunks:
            raise ValueError(f"Article {aid} produced 0 chunks.")

        art_chunks: list[BenchmarkChunk] = []
        for c in chunks:
            bench_chunk = BenchmarkChunk(
                article_id=aid,
                chunk_index=c.chunk_index,
                article_title=title,
                content=c.content,
                char_count=c.char_count,
                token_count=c.token_count,
            )
            art_chunks.append(bench_chunk)
            chunk_catalog[(aid, c.chunk_index)] = bench_chunk

        articles_by_id[aid] = art_chunks

    # 2. Validate queries and relevance references
    seen_query_ids: set[str] = set()
    for q in queries:
        qid = q.get("id")
        subject = q.get("subject", "")
        body = q.get("body", "")
        rel = q.get("relevant_chunks", [])

        if not qid or not isinstance(qid, str):
            raise ValueError(f"Invalid query ID: {qid}")
        if qid in seen_query_ids:
            raise ValueError(f"Duplicate query ID: {qid}")
        seen_query_ids.add(qid)

        if not subject and not body:
            raise ValueError(f"Query {qid} has empty subject and body.")

        if not rel:
            raise ValueError(f"Query {qid} has an empty relevance set; every query must have at least one ground-truth target.")

        seen_chunks_for_query: set[tuple[str, int]] = set()
        for r in rel:
            aid = r.get("article_id")
            cidx = r.get("chunk_index")
            if (aid, cidx) not in chunk_catalog:
                raise ValueError(
                    f"Query {qid} references non-existent chunk (article_id='{aid}', chunk_index={cidx})"
                )
            if (aid, cidx) in seen_chunks_for_query:
                raise ValueError(f"Query {qid} contains duplicate relevance reference for ({aid}, {cidx})")
            seen_chunks_for_query.add((aid, cidx))

    return articles_by_id


def rank_corpus_for_query(
    query_vector: list[float],
    corpus_chunks: list[BenchmarkChunk],
    chunk_vectors: dict[tuple[str, int], list[float]],
) -> list[tuple[tuple[str, int], float, BenchmarkChunk]]:
    """Compute exact cosine similarity against all corpus chunks and sort deterministically.

    Tie-breaking matches production pgvector query:
    ORDER BY similarity_score DESC, article_id ASC, chunk_index ASC.
    """
    scored: list[tuple[tuple[str, int], float, BenchmarkChunk]] = []
    for chunk in corpus_chunks:
        ref = (chunk.article_id, chunk.chunk_index)
        vec = chunk_vectors[ref]
        sim = exact_cosine_similarity(query_vector, vec)
        scored.append((ref, sim, chunk))

    # Deterministic sort: similarity descending, then article_id ascending, chunk_index ascending
    scored.sort(key=lambda x: (-x[1], x[0][0], x[0][1]))
    return scored


def evaluate_benchmark(
    articles: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    adapter: EmbeddingModelAdapter,
) -> tuple[BenchmarkSummaryMetrics, list[QueryEvaluationResult], dict[str, float]]:
    """Run full offline retrieval evaluation and return metrics, query results, and timings."""
    timings: dict[str, float] = {}

    # 1. Validation & Chunking
    t0 = time.perf_counter()
    articles_by_id = validate_benchmark_data(articles, queries)
    all_chunks: list[BenchmarkChunk] = [c for chunks in articles_by_id.values() for c in chunks]
    timings["validation_and_chunking_seconds"] = time.perf_counter() - t0

    # 2. Corpus Chunk Embeddings
    t0 = time.perf_counter()
    chunk_vectors: dict[tuple[str, int], list[float]] = {}
    for chunk in all_chunks:
        ref = (chunk.article_id, chunk.chunk_index)
        res = adapter.embed_text(chunk.content)
        chunk_vectors[ref] = res.vector
    timings["corpus_embedding_seconds"] = time.perf_counter() - t0

    # 3. Query Ticket Embeddings (Using production embed_ticket formatting)
    t0 = time.perf_counter()
    query_vectors: dict[str, list[float]] = {}
    for q in queries:
        qid = q["id"]
        # Production ticket embedding path:
        res = adapter.embed_ticket(subject=q.get("subject"), body=q.get("body"))
        query_vectors[qid] = res.vector
    timings["query_embedding_seconds"] = time.perf_counter() - t0

    # 4. Ranking & Metric Evaluation
    t0 = time.perf_counter()
    query_results: list[QueryEvaluationResult] = []
    for q in queries:
        qid = q["id"]
        q_vec = query_vectors[qid]
        relevant_set = {(r["article_id"], r["chunk_index"]) for r in q["relevant_chunks"]}

        ranked = rank_corpus_for_query(
            query_vector=q_vec,
            corpus_chunks=all_chunks,
            chunk_vectors=chunk_vectors,
        )

        ranked_refs = [r[0] for r in ranked]

        rec_1 = compute_recall_at_k(relevant_set, ranked_refs, 1)
        rec_3 = compute_recall_at_k(relevant_set, ranked_refs, 3)
        rec_5 = compute_recall_at_k(relevant_set, ranked_refs, 5)

        hit_1 = compute_hit_rate_at_k(relevant_set, ranked_refs, 1)
        hit_3 = compute_hit_rate_at_k(relevant_set, ranked_refs, 3)
        hit_5 = compute_hit_rate_at_k(relevant_set, ranked_refs, 5)

        rr, first_rank = compute_reciprocal_rank(relevant_set, ranked_refs)

        top_5_display = [
            {
                "rank": idx,
                "article_id": ref[0],
                "chunk_index": ref[1],
                "article_title": chunk.article_title,
                "similarity_score": round(sim, 4),
                "is_relevant": ref in relevant_set,
                "excerpt_snippet": chunk.content[:120] + "...",
            }
            for idx, (ref, sim, chunk) in enumerate(ranked[:5], start=1)
        ]

        query_results.append(
            QueryEvaluationResult(
                query_id=qid,
                subject=q.get("subject", ""),
                relevant_chunks=q["relevant_chunks"],
                relevant_count=len(relevant_set),
                retrieved_top_5=top_5_display,
                recall_at_1=rec_1,
                recall_at_3=rec_3,
                recall_at_5=rec_5,
                hit_rate_at_1=hit_1,
                hit_rate_at_3=hit_3,
                hit_rate_at_5=hit_5,
                reciprocal_rank=rr,
                first_relevant_rank=first_rank,
            )
        )

    timings["ranking_evaluation_seconds"] = time.perf_counter() - t0

    # 5. Macro Metrics Computation
    n = len(query_results)
    macro_metrics = BenchmarkSummaryMetrics(
        query_count=n,
        corpus_article_count=len(articles),
        corpus_chunk_count=len(all_chunks),
        macro_recall_at_1=sum(r.recall_at_1 for r in query_results) / n,
        macro_recall_at_3=sum(r.recall_at_3 for r in query_results) / n,
        macro_recall_at_5=sum(r.recall_at_5 for r in query_results) / n,
        macro_hit_rate_at_1=sum(r.hit_rate_at_1 for r in query_results) / n,
        macro_hit_rate_at_3=sum(r.hit_rate_at_3 for r in query_results) / n,
        macro_hit_rate_at_5=sum(r.hit_rate_at_5 for r in query_results) / n,
        mean_reciprocal_rank=sum(r.reciprocal_rank for r in query_results) / n,
    )

    return macro_metrics, query_results, timings


def get_next_run_dir(base_runs_dir: Path) -> Path:
    """Find the next sequential run directory and create it exclusively."""
    base_runs_dir.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in base_runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]

    max_num = 0
    for name in existing:
        parts = name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            max_num = max(max_num, int(parts[1]))

    next_num = max_num + 1
    run_dir = base_runs_dir / f"run_{next_num:03d}_retrieval_benchmark"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> None:
    print("=" * 70)
    print("PHASE 4 OFFLINE KNOWLEDGE-BASE RETRIEVAL BENCHMARK")
    print("=" * 70)

    # 1. Check local model files
    model_dir = REPO_ROOT / DEFAULT_MODEL_DIR_REL
    onnx_file = model_dir / "onnx" / "model.onnx"
    tok_file = model_dir / "tokenizer.json"
    if not onnx_file.exists() or not tok_file.exists():
        print(f"\nERROR: Required local embedding model artifacts missing under {model_dir}")
        print("Please provision the verified weights via scripts/download_embedding_model.py first.")
        sys.exit(1)

    # 2. Load dataset files
    data_dir = REPO_ROOT / "data" / "retrieval_benchmark"
    art_file = data_dir / "articles.json"
    q_file = data_dir / "queries.json"

    if not art_file.exists() or not q_file.exists():
        print(f"\nERROR: Benchmark dataset files not found under {data_dir}")
        sys.exit(1)

    art_bytes = art_file.read_bytes()
    q_bytes = q_file.read_bytes()

    art_sha = hashlib.sha256(art_bytes).hexdigest()
    q_sha = hashlib.sha256(q_bytes).hexdigest()

    articles = json.loads(art_bytes.decode("utf-8"))
    queries = json.loads(q_bytes.decode("utf-8"))

    print(f"\nDataset Provenance:")
    print(f"  Articles File: {art_file.name} (SHA-256: {art_sha})")
    print(f"  Queries File:  {q_file.name} (SHA-256: {q_sha})")
    print(f"  Total Articles: {len(articles)}")
    print(f"  Total Queries:  {len(queries)}")

    # 3. Initialize Model Adapter
    t0 = time.perf_counter()
    print(f"\nInitializing EmbeddingModelAdapter ({EMBEDDING_MODEL_NAME})...")
    adapter = EmbeddingModelAdapter()
    init_time = time.perf_counter() - t0
    print(f"  Model Revision: {adapter.model_revision}")
    print(f"  Initialization Time: {init_time:.3f}s")

    # 4. Execute Benchmark
    print("\nRunning offline retrieval evaluation...")
    metrics, query_results, timings = evaluate_benchmark(articles, queries, adapter)
    timings["model_initialization_seconds"] = init_time

    # 5. Create Exclusive Run Directory
    runs_dir = REPO_ROOT / "experiments" / "runs"
    run_dir = get_next_run_dir(runs_dir)
    print(f"\nCreated immutable run directory: {run_dir.relative_to(REPO_ROOT)}")

    # 6. Save Run Artifacts
    run_meta = {
        "benchmark": "Phase 4 Knowledge-Base Retrieval Quality Benchmark",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": adapter.model_name,
        "model_revision": adapter.model_revision,
        "dataset": {
            "articles_filename": art_file.name,
            "articles_sha256": art_sha,
            "queries_filename": q_file.name,
            "queries_sha256": q_sha,
            "article_count": len(articles),
            "query_count": len(queries),
            "chunk_count": metrics.corpus_chunk_count,
        },
        "chunking": {
            "max_title_chars": 300,
            "max_article_chars": 20000,
            "target_chunk_tokens": 160,
            "overlap_tokens": 30,
        },
        "timings_seconds": {k: round(v, 4) for k, v in timings.items()},
        "timing_notice": (
            "Timings reflect single-threaded local CPU execution on synthetic fixtures. "
            "They do NOT represent production database latency, connection pooling, or network roundtrips."
        ),
    }

    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2)

    with open(run_dir / "query_results.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in query_results], f, indent=2)

    # 7. Print Console Summary
    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 70)
    print(f"Queries Evaluated: {metrics.query_count}")
    print(f"Corpus Chunks:     {metrics.corpus_chunk_count} (across {metrics.corpus_article_count} articles)")
    print("-" * 70)
    print(f"Macro Recall@1:    {metrics.macro_recall_at_1:.4f}  ({metrics.macro_recall_at_1 * 100:.1f}%)")
    print(f"Macro Recall@3:    {metrics.macro_recall_at_3:.4f}  ({metrics.macro_recall_at_3 * 100:.1f}%)")
    print(f"Macro Recall@5:    {metrics.macro_recall_at_5:.4f}  ({metrics.macro_recall_at_5 * 100:.1f}%)")
    print(f"Macro Hit Rate@1:  {metrics.macro_hit_rate_at_1:.4f}  ({metrics.macro_hit_rate_at_1 * 100:.1f}%)")
    print(f"Macro Hit Rate@3:  {metrics.macro_hit_rate_at_3:.4f}  ({metrics.macro_hit_rate_at_3 * 100:.1f}%)")
    print(f"Macro Hit Rate@5:  {metrics.macro_hit_rate_at_5:.4f}  ({metrics.macro_hit_rate_at_5 * 100:.1f}%)")
    print(f"Mean Reciprocal Rank (MRR): {metrics.mean_reciprocal_rank:.4f}")
    print("-" * 70)
    print("Exploratory Targets Reference (suggested, non-SPEC):")
    print(f"  Recall@5 Target (>= 0.85): {'MET' if metrics.macro_recall_at_5 >= 0.85 else 'UNMET'} ({metrics.macro_recall_at_5:.4f})")
    print(f"  MRR Target      (>= 0.65): {'MET' if metrics.mean_reciprocal_rank >= 0.65 else 'UNMET'} ({metrics.mean_reciprocal_rank:.4f})")
    print("=" * 70)

    # Print any query misses (where HitRate@5 == 0 or rank > 3)
    misses = [r for r in query_results if r.hit_rate_at_5 < 1.0]
    rank_drops = [r for r in query_results if r.first_relevant_rank and r.first_relevant_rank > 1]
    if misses:
        print(f"\nComplete Misses at k=5 ({len(misses)} queries):")
        for m in misses:
            print(f"  [{m.query_id}] '{m.subject}' - Target: {m.relevant_chunks}")
    else:
        print("\nComplete Misses at k=5: 0 queries (100% Hit Rate@5)")

    if rank_drops:
        print(f"\nSub-Ranked Queries (first relevant rank > 1, {len(rank_drops)} queries):")
        for d in rank_drops:
            print(f"  [{d.query_id}] '{d.subject}' -> First relevant at Rank #{d.first_relevant_rank} (RR={d.reciprocal_rank:.4f})")


if __name__ == "__main__":
    main()
