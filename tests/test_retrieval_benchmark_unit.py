"""Artifact-free unit tests for Phase 4 retrieval benchmark ranking and metrics.

Verifies:
1. Exact hand-computed metric cases:
   - Single relevant chunk at ranks 1, 2, 4, 7, and unretrieved.
   - Multiple relevant chunks at distributed ranks.
   - Missing relevant results.
   - k larger than corpus size.
2. Deterministic ranking tie-breaking:
   - Ties broken by article_id ASC, then chunk_index ASC.
3. Mathematical metric bounds and monotonicity:
   - 0.0 <= Recall@k <= 1.0 and 0.0 <= HitRate@k <= 1.0.
   - Recall@1 <= Recall@3 <= Recall@5.
   - HitRate@1 <= HitRate@3 <= HitRate@5.
   - For single-relevant queries, Recall@k == HitRate@k.
4. Cosine similarity mathematical precision and boundary clamping:
   - Identity (1.0), orthogonal (0.0), opposite (-1.0).
   - Rejection of dimension mismatches and non-finite values.
5. Dataset validation contracts:
   - Rejection of duplicate article/query IDs.
   - Rejection of empty relevance sets.
   - Rejection of non-existent chunk references.
"""

import json
import math
from pathlib import Path
import pytest

from scripts.evaluate_retrieval import (
    BenchmarkChunk,
    compute_hit_rate_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    exact_cosine_similarity,
    rank_corpus_for_query,
    validate_benchmark_data,
)


# ---------------------------------------------------------------------------
# 1. Hand-Computed Metric Cases
# ---------------------------------------------------------------------------

def test_metrics_single_relevant_at_rank_1():
    rel = {("art_1", 0)}
    ranked = [("art_1", 0), ("art_2", 0), ("art_3", 0), ("art_4", 0), ("art_5", 0)]

    assert compute_recall_at_k(rel, ranked, 1) == 1.0
    assert compute_recall_at_k(rel, ranked, 3) == 1.0
    assert compute_recall_at_k(rel, ranked, 5) == 1.0

    assert compute_hit_rate_at_k(rel, ranked, 1) == 1.0
    assert compute_hit_rate_at_k(rel, ranked, 3) == 1.0
    assert compute_hit_rate_at_k(rel, ranked, 5) == 1.0

    rr, first_rank = compute_reciprocal_rank(rel, ranked)
    assert rr == 1.0
    assert first_rank == 1


def test_metrics_single_relevant_at_rank_2():
    rel = {("art_2", 0)}
    ranked = [("art_1", 0), ("art_2", 0), ("art_3", 0), ("art_4", 0), ("art_5", 0)]

    assert compute_recall_at_k(rel, ranked, 1) == 0.0
    assert compute_recall_at_k(rel, ranked, 3) == 1.0
    assert compute_recall_at_k(rel, ranked, 5) == 1.0

    assert compute_hit_rate_at_k(rel, ranked, 1) == 0.0
    assert compute_hit_rate_at_k(rel, ranked, 3) == 1.0
    assert compute_hit_rate_at_k(rel, ranked, 5) == 1.0

    rr, first_rank = compute_reciprocal_rank(rel, ranked)
    assert rr == 0.5
    assert first_rank == 2


def test_metrics_single_relevant_at_rank_4():
    rel = {("art_4", 0)}
    ranked = [("art_1", 0), ("art_2", 0), ("art_3", 0), ("art_4", 0), ("art_5", 0)]

    assert compute_recall_at_k(rel, ranked, 1) == 0.0
    assert compute_recall_at_k(rel, ranked, 3) == 0.0
    assert compute_recall_at_k(rel, ranked, 5) == 1.0

    assert compute_hit_rate_at_k(rel, ranked, 1) == 0.0
    assert compute_hit_rate_at_k(rel, ranked, 3) == 0.0
    assert compute_hit_rate_at_k(rel, ranked, 5) == 1.0

    rr, first_rank = compute_reciprocal_rank(rel, ranked)
    assert rr == 0.25
    assert first_rank == 4


def test_metrics_single_relevant_outside_top_5():
    rel = {("art_7", 0)}
    ranked = [
        ("art_1", 0), ("art_2", 0), ("art_3", 0),
        ("art_4", 0), ("art_5", 0), ("art_6", 0),
        ("art_7", 0), ("art_8", 0)
    ]

    assert compute_recall_at_k(rel, ranked, 1) == 0.0
    assert compute_recall_at_k(rel, ranked, 3) == 0.0
    assert compute_recall_at_k(rel, ranked, 5) == 0.0

    assert compute_hit_rate_at_k(rel, ranked, 1) == 0.0
    assert compute_hit_rate_at_k(rel, ranked, 3) == 0.0
    assert compute_hit_rate_at_k(rel, ranked, 5) == 0.0

    rr, first_rank = compute_reciprocal_rank(rel, ranked)
    assert rr == pytest.approx(1.0 / 7.0, rel=1e-5)
    assert first_rank == 7


def test_metrics_missing_relevant_chunk():
    rel = {("art_missing", 0)}
    ranked = [("art_1", 0), ("art_2", 0), ("art_3", 0)]

    assert compute_recall_at_k(rel, ranked, 1) == 0.0
    assert compute_recall_at_k(rel, ranked, 3) == 0.0
    assert compute_recall_at_k(rel, ranked, 5) == 0.0

    assert compute_hit_rate_at_k(rel, ranked, 1) == 0.0
    assert compute_hit_rate_at_k(rel, ranked, 3) == 0.0
    assert compute_hit_rate_at_k(rel, ranked, 5) == 0.0

    rr, first_rank = compute_reciprocal_rank(rel, ranked)
    assert rr == 0.0
    assert first_rank is None


def test_metrics_multiple_relevant_chunks():
    """Two relevant chunks: one at rank 1, one at rank 4."""
    rel = {("art_1", 0), ("art_4", 0)}
    ranked = [("art_1", 0), ("art_2", 0), ("art_3", 0), ("art_4", 0), ("art_5", 0)]

    # At k=1: only art_1 is found (1 of 2 = 0.5 recall; hit rate = 1.0)
    assert compute_recall_at_k(rel, ranked, 1) == 0.5
    assert compute_hit_rate_at_k(rel, ranked, 1) == 1.0

    # At k=3: only art_1 is found (1 of 2 = 0.5 recall; hit rate = 1.0)
    assert compute_recall_at_k(rel, ranked, 3) == 0.5
    assert compute_hit_rate_at_k(rel, ranked, 3) == 1.0

    # At k=5: both art_1 and art_4 are found (2 of 2 = 1.0 recall; hit rate = 1.0)
    assert compute_recall_at_k(rel, ranked, 5) == 1.0
    assert compute_hit_rate_at_k(rel, ranked, 5) == 1.0

    rr, first_rank = compute_reciprocal_rank(rel, ranked)
    assert rr == 1.0
    assert first_rank == 1


def test_metrics_k_larger_than_corpus():
    """Corpus has only 2 items; requesting k=5 should not raise and should clamp correctly."""
    rel = {("art_2", 0)}
    ranked = [("art_1", 0), ("art_2", 0)]

    assert compute_recall_at_k(rel, ranked, 5) == 1.0
    assert compute_hit_rate_at_k(rel, ranked, 5) == 1.0
    rr, first_rank = compute_reciprocal_rank(rel, ranked)
    assert rr == 0.5
    assert first_rank == 2


def test_empty_relevant_set_raises():
    with pytest.raises(ValueError, match="Relevant set must not be empty"):
        compute_recall_at_k(set(), [("art_1", 0)], 1)
    with pytest.raises(ValueError, match="Relevant set must not be empty"):
        compute_hit_rate_at_k(set(), [("art_1", 0)], 1)
    with pytest.raises(ValueError, match="Relevant set must not be empty"):
        compute_reciprocal_rank(set(), [("art_1", 0)])


# ---------------------------------------------------------------------------
# 2. Deterministic Ranking & Tie-Breaking
# ---------------------------------------------------------------------------

def test_deterministic_ranking_tie_breaking():
    """When similarity scores tie, tie-breaker must sort by article_id ASC, then chunk_index ASC."""
    corpus_chunks = [
        BenchmarkChunk(article_id="art_b", chunk_index=0, article_title="B", content="b0", char_count=2, token_count=2),
        BenchmarkChunk(article_id="art_a", chunk_index=1, article_title="A", content="a1", char_count=2, token_count=2),
        BenchmarkChunk(article_id="art_a", chunk_index=0, article_title="A", content="a0", char_count=2, token_count=2),
        BenchmarkChunk(article_id="art_c", chunk_index=0, article_title="C", content="c0", char_count=2, token_count=2),
    ]

    # Assign identical vector [1.0, 0.0] to all chunks
    chunk_vectors = {
        ("art_b", 0): [1.0, 0.0],
        ("art_a", 1): [1.0, 0.0],
        ("art_a", 0): [1.0, 0.0],
        ("art_c", 0): [1.0, 0.0],
    }
    q_vec = [1.0, 0.0]

    ranked = rank_corpus_for_query(q_vec, corpus_chunks, chunk_vectors)
    ranked_refs = [r[0] for r in ranked]

    # Scores all tie at 1.0; order must strictly be:
    # 1. ('art_a', 0)
    # 2. ('art_a', 1)
    # 3. ('art_b', 0)
    # 4. ('art_c', 0)
    assert ranked_refs == [
        ("art_a", 0),
        ("art_a", 1),
        ("art_b", 0),
        ("art_c", 0),
    ]


# ---------------------------------------------------------------------------
# 3. Mathematical Monotonicity and Bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ranked_order", [
    [("art_1", 0), ("art_2", 0), ("art_3", 0), ("art_4", 0), ("art_5", 0)],
    [("art_5", 0), ("art_4", 0), ("art_3", 0), ("art_2", 0), ("art_1", 0)],
    [("art_3", 0), ("art_1", 0), ("art_5", 0), ("art_2", 0), ("art_4", 0)],
])
@pytest.mark.parametrize("relevant_targets", [
    {("art_1", 0)},
    {("art_3", 0)},
    {("art_1", 0), ("art_2", 0)},
    {("art_1", 0), ("art_5", 0)},
    {("art_missing", 0)},
])
def test_metric_monotonicity_and_bounds(ranked_order, relevant_targets):
    r1 = compute_recall_at_k(relevant_targets, ranked_order, 1)
    r3 = compute_recall_at_k(relevant_targets, ranked_order, 3)
    r5 = compute_recall_at_k(relevant_targets, ranked_order, 5)

    h1 = compute_hit_rate_at_k(relevant_targets, ranked_order, 1)
    h3 = compute_hit_rate_at_k(relevant_targets, ranked_order, 3)
    h5 = compute_hit_rate_at_k(relevant_targets, ranked_order, 5)

    rr, _ = compute_reciprocal_rank(relevant_targets, ranked_order)

    # 1. Bounds
    assert 0.0 <= r1 <= 1.0
    assert 0.0 <= r3 <= 1.0
    assert 0.0 <= r5 <= 1.0

    assert h1 in (0.0, 1.0)
    assert h3 in (0.0, 1.0)
    assert h5 in (0.0, 1.0)

    assert 0.0 <= rr <= 1.0

    # 2. Monotonicity
    assert r1 <= r3 <= r5
    assert h1 <= h3 <= h5

    # 3. Recall is always <= Hit Rate
    assert r1 <= h1
    assert r3 <= h3
    assert r5 <= h5

    # 4. For single-relevant items, Recall@k == HitRate@k
    if len(relevant_targets) == 1:
        assert r1 == h1
        assert r3 == h3
        assert r5 == h5


# ---------------------------------------------------------------------------
# 4. Exact Cosine Similarity Precision
# ---------------------------------------------------------------------------

def test_cosine_similarity_identity_and_orthogonality():
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    v_opp = [-1.0, 0.0, 0.0]

    assert exact_cosine_similarity(v1, v1) == pytest.approx(1.0, rel=1e-6)
    assert exact_cosine_similarity(v1, v2) == pytest.approx(0.0, abs=1e-6)
    assert exact_cosine_similarity(v1, v_opp) == pytest.approx(-1.0, rel=1e-6)


def test_cosine_similarity_dimension_mismatch():
    with pytest.raises(ValueError, match="dimensions must match"):
        exact_cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


def test_cosine_similarity_non_finite_rejection():
    with pytest.raises(ValueError, match="Non-finite dot product"):
        exact_cosine_similarity([float("nan"), 0.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="Non-finite dot product"):
        exact_cosine_similarity([float("inf"), 0.0], [1.0, 0.0])


# ---------------------------------------------------------------------------
# 5. Dataset Validation Contracts
# ---------------------------------------------------------------------------

def test_validate_benchmark_data_happy_path():
    articles = [
        {"id": "art_1", "title": "Title 1", "body": "Paragraph 1 text content."}
    ]
    queries = [
        {"id": "q1", "subject": "Subj", "body": "Body", "relevant_chunks": [{"article_id": "art_1", "chunk_index": 0}]}
    ]

    res = validate_benchmark_data(articles, queries)
    assert "art_1" in res
    assert len(res["art_1"]) == 1
    assert res["art_1"][0].chunk_index == 0


def test_validate_benchmark_data_duplicate_article_id():
    articles = [
        {"id": "art_1", "title": "Title 1", "body": "Body 1"},
        {"id": "art_1", "title": "Title 2", "body": "Body 2"},
    ]
    queries = [{"id": "q1", "subject": "S", "body": "B", "relevant_chunks": [{"article_id": "art_1", "chunk_index": 0}]}]

    with pytest.raises(ValueError, match="Duplicate article ID"):
        validate_benchmark_data(articles, queries)


def test_validate_benchmark_data_duplicate_query_id():
    articles = [{"id": "art_1", "title": "Title 1", "body": "Body 1"}]
    queries = [
        {"id": "q1", "subject": "S1", "body": "B1", "relevant_chunks": [{"article_id": "art_1", "chunk_index": 0}]},
        {"id": "q1", "subject": "S2", "body": "B2", "relevant_chunks": [{"article_id": "art_1", "chunk_index": 0}]},
    ]

    with pytest.raises(ValueError, match="Duplicate query ID"):
        validate_benchmark_data(articles, queries)


def test_validate_benchmark_data_empty_relevance_set():
    articles = [{"id": "art_1", "title": "Title 1", "body": "Body 1"}]
    queries = [{"id": "q1", "subject": "S1", "body": "B1", "relevant_chunks": []}]

    with pytest.raises(ValueError, match="empty relevance set"):
        validate_benchmark_data(articles, queries)


def test_validate_benchmark_data_unknown_chunk_reference():
    articles = [{"id": "art_1", "title": "Title 1", "body": "Short body"}]
    # art_1 only has chunk 0; referencing chunk 5 must raise
    queries = [{"id": "q1", "subject": "S1", "body": "B1", "relevant_chunks": [{"article_id": "art_1", "chunk_index": 5}]}]

    with pytest.raises(ValueError, match="references non-existent chunk"):
        validate_benchmark_data(articles, queries)


def test_frozen_benchmark_dataset_integrity():
    """Verify that the frozen benchmark dataset files exist, parse, and validate cleanly."""
    articles_path = Path("data/retrieval_benchmark/articles.json")
    queries_path = Path("data/retrieval_benchmark/queries.json")

    assert articles_path.exists(), "articles.json must exist"
    assert queries_path.exists(), "queries.json must exist"

    with open(articles_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    catalog = validate_benchmark_data(articles, queries)
    assert len(catalog) == 10, "Expected exactly 10 articles in benchmark catalog"
    total_chunks = sum(len(chunks) for chunks in catalog.values())
    assert len(queries) == 30, "Expected exactly 30 queries in benchmark dataset"

    # In clean CI checkouts where models/ tokenizer files are not downloaded,
    # chunking uses heuristic word estimation. When the real tokenizer is present,
    # it must produce exactly 12 chunks.
    from app.ml.chunking import get_local_tokenizer
    if get_local_tokenizer() is not None:
        assert total_chunks == 12, "Expected exactly 12 chunks with provisioned tokenizer"
    else:
        assert total_chunks >= 10, "Expected at least 10 chunks from 10 articles with heuristic token estimation"
