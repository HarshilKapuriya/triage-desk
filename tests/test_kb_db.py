"""Database integration tests for tenant-scoped knowledge base articles, chunks, and exact cosine retrieval.

Strict safety constraints:
- Database tests run ONLY against an already established disposable local PostgreSQL
  instance (localhost or 127.0.0.1).
- Hosted databases (e.g. Supabase poolers) are strictly prohibited.
- If no established disposable local test database exists, tests are cleanly skipped.
"""

import uuid
import psycopg
import pytest

import app.repositories.kb as kb_repo
import app.repositories.duplicates as dup_repo
from tests.test_embedding_jobs_db import (
    LOCAL_SAFETY_STATUS,
    LOCAL_TEST_DB_URL,
    _create_test_tenant,
    _create_test_ticket,
)

pytestmark = pytest.mark.skipif(
    LOCAL_TEST_DB_URL is None,
    reason=f"Database integration tests require an established disposable local PostgreSQL instance: {LOCAL_SAFETY_STATUS}",
)


def _make_unit_vector(dim: int = 384, active_idx: int = 0) -> list[float]:
    """Helper returning an L2-normalized unit vector with a 1.0 at active_idx."""
    v = [0.0] * dim
    v[active_idx % dim] = 1.0
    return v


def test_kb_db_article_and_chunks_composite_fk_cascade():
    """Verify composite foreign key cascade on kb_chunks when article is deleted."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_id = _create_test_tenant("KB Cascade Co")

    from app.ml.chunking import ChunkData

    chunks = [
        ChunkData(chunk_index=0, content="First passage", char_count=13, token_count=3),
        ChunkData(chunk_index=1, content="Second passage", char_count=14, token_count=3),
    ]

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        with conn.transaction():
            article = kb_repo.create_kb_article(conn, tenant_id, "Policy Document", "Body text")
            kb_repo.create_kb_chunks(conn, tenant_id, article.id, chunks)

        stored_chunks = kb_repo.get_kb_chunks_for_article(conn, tenant_id, article.id)
        assert len(stored_chunks) == 2

        # Delete article and verify chunks cascade
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("DELETE FROM kb_articles WHERE tenant_id = %s AND id = %s;", (tenant_id, article.id))

        remaining_chunks = kb_repo.get_kb_chunks_for_article(conn, tenant_id, article.id)
        assert len(remaining_chunks) == 0


def test_kb_db_multi_tenant_isolation():
    """Verify Tenant A cannot access or retrieve Tenant B's articles or chunks."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_a = _create_test_tenant("Tenant A KB")
    tenant_b = _create_test_tenant("Tenant B KB")

    from app.ml.chunking import ChunkData

    chunks_a = [ChunkData(chunk_index=0, content="Tenant A policy", char_count=15, token_count=4)]
    chunks_b = [ChunkData(chunk_index=0, content="Tenant B secret", char_count=15, token_count=4)]

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        with conn.transaction():
            art_a = kb_repo.create_kb_article(conn, tenant_a, "Article A", "Body A")
            kb_repo.create_kb_chunks(conn, tenant_a, art_a.id, chunks_a)

            art_b = kb_repo.create_kb_article(conn, tenant_b, "Article B", "Body B")
            kb_repo.create_kb_chunks(conn, tenant_b, art_b.id, chunks_b)

        # Cross-tenant direct query returns None
        assert kb_repo.get_kb_article_by_id(conn, tenant_a, art_b.id) is None
        assert kb_repo.get_kb_article_by_id(conn, tenant_b, art_a.id) is None

        # Cross-tenant list returns only tenant's own articles
        list_a = kb_repo.list_kb_articles(conn, tenant_a)
        assert all(a.tenant_id == tenant_a for a in list_a)
        assert any(a.id == art_a.id for a in list_a)
        assert not any(a.id == art_b.id for a in list_a)


def test_kb_db_exact_cosine_retrieval_ranking():
    """Verify ticket-based chunk retrieval ranks by exact cosine similarity."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_id = _create_test_tenant("KB Ranking Co")
    ticket_id = _create_test_ticket(tenant_id, "Disputed ATM withdrawal complaint")

    from app.ml.chunking import ChunkData

    chunks = [
        ChunkData(chunk_index=0, content="Unrelated topic passage", char_count=23, token_count=5),
        ChunkData(chunk_index=1, content="Exact match ATM dispute rules", char_count=30, token_count=6),
    ]

    # Source ticket vector with 1.0 at index 5
    v_source = _make_unit_vector(384, active_idx=5)
    # Chunk 0 has 1.0 at index 10 (orthogonal -> cosine sim = 0.0)
    v_chunk_0 = _make_unit_vector(384, active_idx=10)
    # Chunk 1 has 1.0 at index 5 (identical -> cosine sim = 1.0)
    v_chunk_1 = _make_unit_vector(384, active_idx=5)

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        with conn.transaction():
            dup_repo.save_ticket_embedding(conn, tenant_id, ticket_id, v_source, "all-MiniLM-L6-v2", "rev1")

            art = kb_repo.create_kb_article(conn, tenant_id, "Dispute Rules", "Full body text")
            stored = kb_repo.create_kb_chunks(conn, tenant_id, art.id, chunks)

            # Persist embeddings on chunks
            kb_repo.upsert_kb_chunk_embedding(conn, tenant_id, stored[0].id, v_chunk_0, "all-MiniLM-L6-v2", "rev1")
            kb_repo.upsert_kb_chunk_embedding(conn, tenant_id, stored[1].id, v_chunk_1, "all-MiniLM-L6-v2", "rev1")

        state, model, rev, candidates = kb_repo.find_relevant_chunks_for_ticket(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id=ticket_id,
            limit=5,
        )

        assert state == "ready"
        assert len(candidates) == 2
        # Highest similarity (Chunk 1 with sim=1.0) must rank first
        assert candidates[0].chunk_index == 1
        assert pytest.approx(candidates[0].similarity_score, 1e-4) == 1.0
        assert candidates[1].chunk_index == 0
        assert pytest.approx(candidates[1].similarity_score, 1e-4) == 0.0


def test_kb_db_article_jobs_claim_and_completion():
    """Verify claim, fencing, and completion on the dedicated article_jobs table."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_id = _create_test_tenant("KB Job Claim Co")

    from app.ml.chunking import ChunkData

    chunks = [
        ChunkData(chunk_index=0, content="First passage", char_count=13, token_count=3),
        ChunkData(chunk_index=1, content="Second passage", char_count=14, token_count=3),
    ]

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        with conn.transaction():
            art = kb_repo.create_kb_article(conn, tenant_id, "Policy Doc", "Body content")
            created_chunks = kb_repo.create_kb_chunks(conn, tenant_id, art.id, chunks)
            job = kb_repo.enqueue_article_job(conn, tenant_id, art.id, "article_embedding")

        assert job.status == "queued"

        # Claim the job
        with conn.transaction():
            claimed = kb_repo.claim_next_article_job(conn, "worker_kb_db", lease_seconds=300, tenant_id=tenant_id)
        assert claimed is not None
        assert claimed.id == job.id
        assert claimed.status == "running"
        assert claimed.attempts == 1

        v0 = _make_unit_vector(384, active_idx=1)
        v1 = _make_unit_vector(384, active_idx=2)
        chunk_embeddings = [
            (created_chunks[0].id, v0, "all-MiniLM-L6-v2", "rev1"),
            (created_chunks[1].id, v1, "all-MiniLM-L6-v2", "rev1"),
        ]

        # Complete the job atomically
        with conn.transaction():
            kb_repo.complete_article_job(
                conn=conn,
                tenant_id=tenant_id,
                job_id=claimed.id,
                article_id=art.id,
                worker_id="worker_kb_db",
                generation=claimed.attempts,
                chunk_embeddings=chunk_embeddings,
            )

        # Verify article is now ready
        updated_art = kb_repo.get_kb_article_by_id(conn, tenant_id, art.id)
        assert updated_art is not None
        assert updated_art.status == "ready"

        # Verify chunks have embeddings
        chunks_after = kb_repo.get_kb_chunks_for_article(conn, tenant_id, art.id)
        assert len(chunks_after) == 2
        assert all(c.embedding is not None for c in chunks_after)


def test_kb_db_article_jobs_failure_and_retry_backoff():
    """Verify exponential backoff retry and terminal failure handling on article_jobs."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_id = _create_test_tenant("KB Retry Co")

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        with conn.transaction():
            art = kb_repo.create_kb_article(conn, tenant_id, "Retry Policy", "Body content")
            job = kb_repo.enqueue_article_job(conn, tenant_id, art.id, "article_embedding")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE article_jobs
                    SET max_attempts = 2
                    WHERE tenant_id = %s AND id = %s;
                    """,
                    (tenant_id, job.id),
                )

        # Claim attempt 1
        with conn.transaction():
            c1 = kb_repo.claim_next_article_job(conn, "worker_retry", tenant_id=tenant_id)
        assert c1 is not None

        # Fail attempt 1 -> requeued
        with conn.transaction():
            res1 = kb_repo.fail_or_retry_article_job(
                conn=conn,
                tenant_id=tenant_id,
                job_id=c1.id,
                worker_id="worker_retry",
                generation=c1.attempts,
                error_message="Simulated temporary failure",
                article_id=art.id,
            )
        assert res1 == "requeued"

        # Article remains in processing
        art_mid = kb_repo.get_kb_article_by_id(conn, tenant_id, art.id)
        assert art_mid is not None
        assert art_mid.status == "processing"

        # Force run_after to the past to claim attempt 2
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE article_jobs
                    SET run_after = NOW() - INTERVAL '10 seconds'
                    WHERE tenant_id = %s AND id = %s;
                    """,
                    (tenant_id, c1.id),
                )

        # Claim attempt 2
        with conn.transaction():
            c2 = kb_repo.claim_next_article_job(conn, "worker_retry", tenant_id=tenant_id)
        assert c2 is not None
        assert c2.attempts == 2

        # Fail attempt 2 -> terminal failure
        with conn.transaction():
            res2 = kb_repo.fail_or_retry_article_job(
                conn=conn,
                tenant_id=tenant_id,
                job_id=c2.id,
                worker_id="worker_retry",
                generation=c2.attempts,
                error_message="Terminal failure",
                article_id=art.id,
            )
        assert res2 == "failed"

        # Both job and article must now be marked 'failed'
        art_final = kb_repo.get_kb_article_by_id(conn, tenant_id, art.id)
        assert art_final is not None
        assert art_final.status == "failed"


def test_kb_db_article_jobs_stale_lease_recovery():
    """Verify expired running leases on article_jobs are safely recovered."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_id = _create_test_tenant("KB Recovery Co")

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        with conn.transaction():
            art = kb_repo.create_kb_article(conn, tenant_id, "Recovery Policy", "Body content")
            job = kb_repo.enqueue_article_job(conn, tenant_id, art.id, "article_embedding")

        # Claim the job
        with conn.transaction():
            claimed = kb_repo.claim_next_article_job(conn, "worker_crashing", tenant_id=tenant_id)
        assert claimed is not None

        # Simulate crashed worker by rolling locked_at back 500 seconds
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE article_jobs
                    SET locked_at = NOW() - INTERVAL '500 seconds'
                    WHERE tenant_id = %s AND id = %s;
                    """,
                    (tenant_id, claimed.id),
                )

        # Run recovery
        with conn.transaction():
            recovered = kb_repo.recover_stale_article_jobs(conn, lease_seconds=300, tenant_id=tenant_id)

        assert recovered >= 1

        # Job must be back in 'queued' state with cleared lock
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                """
                SELECT status, locked_by, locked_at
                FROM article_jobs
                WHERE tenant_id = %s AND id = %s;
                """,
                (tenant_id, claimed.id),
            )
            row = cur.fetchone()
            assert row is not None
            assert row["status"] == "queued"
            assert row["locked_by"] is None
            assert row["locked_at"] is None
