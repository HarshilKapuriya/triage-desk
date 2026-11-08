"""Knowledge Base Repository: Handles database operations for kb_articles, kb_chunks, and article_jobs.

Per SPEC.md Section 6:
- All tenant-owned repository functions take tenant_id as a required argument and filter by it.
- All SQL statements live strictly in the repository layer.
- Tenant isolation is enforced in every query; composite foreign keys prevent cross-tenant links.
"""

from dataclasses import dataclass
from datetime import datetime
import json
import math
from typing import Any
import uuid

import psycopg
from psycopg.rows import dict_row

from app.ml.chunking import ChunkData

ALLOWED_ARTICLE_STATUSES = {"processing", "ready", "failed"}


class StaleClaimError(Exception):
    """Raised when worker attempts operation on an article job whose lease expired or generation changed."""


@dataclass(frozen=True)
class KBArticleRecord:
    """Represents a row from kb_articles."""
    id: uuid.UUID
    tenant_id: uuid.UUID
    title: str
    body: str
    status: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class KBArticleSummaryRecord:
    """Summary of an article with chunk count for listing."""
    id: uuid.UUID
    tenant_id: uuid.UUID
    title: str
    status: str
    chunk_count: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class KBChunkRecord:
    """Represents a row from kb_chunks."""
    id: uuid.UUID
    tenant_id: uuid.UUID
    article_id: uuid.UUID
    chunk_index: int
    content: str
    char_count: int
    token_count: int
    embedding: list[float] | None
    model_name: str | None
    model_version: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ArticleJobRecord:
    """Represents a row from the article_jobs queue table."""
    id: uuid.UUID
    tenant_id: uuid.UUID
    article_id: uuid.UUID
    type: str
    status: str
    attempts: int
    max_attempts: int
    run_after: datetime
    locked_at: datetime | None
    locked_by: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    failed_at: datetime | None
    error: str | None
    payload: dict[str, Any]


@dataclass(frozen=True)
class RelevantKnowledgeCandidate:
    """Ranked article chunk candidate matching a ticket via exact cosine similarity."""
    chunk_id: uuid.UUID
    article_id: uuid.UUID
    article_title: str
    chunk_index: int
    excerpt: str
    char_count: int
    token_count: int
    similarity_score: float


def _format_vector(embedding: list[float] | str) -> str:
    """Format float list into postgres vector literal string '[0.1,0.2,...]'."""
    if isinstance(embedding, str):
        return embedding
    return f"[{','.join(str(x) for x in embedding)}]"


def _parse_vector(embedding_val: Any) -> list[float] | None:
    """Parse postgres vector representation into a float list."""
    if embedding_val is None:
        return None
    if isinstance(embedding_val, list):
        return [float(x) for x in embedding_val]
    if isinstance(embedding_val, str):
        stripped = embedding_val.strip("[]() \t\n\r")
        if not stripped:
            return []
        return [float(x.strip()) for x in stripped.split(",") if x.strip()]
    return None


def _row_to_article_job(row: dict[str, Any]) -> ArticleJobRecord:
    payload_val = row.get("payload", {})
    if isinstance(payload_val, str):
        try:
            payload_dict = json.loads(payload_val)
        except Exception:
            payload_dict = {}
    elif isinstance(payload_val, dict):
        payload_dict = payload_val
    else:
        payload_dict = {}

    return ArticleJobRecord(
        id=uuid.UUID(str(row["id"])),
        tenant_id=uuid.UUID(str(row["tenant_id"])),
        article_id=uuid.UUID(str(row["article_id"])),
        type=row["type"],
        status=row["status"],
        attempts=int(row["attempts"]),
        max_attempts=int(row["max_attempts"]),
        run_after=row["run_after"],
        locked_at=row["locked_at"],
        locked_by=row["locked_by"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
        failed_at=row["failed_at"],
        error=row["error"],
        payload=payload_dict,
    )


# ==============================================================================
# Article & Chunk Operations
# ==============================================================================

def create_kb_article(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    title: str,
    body: str,
    status: str = "processing",
) -> KBArticleRecord:
    """Create a new tenant-owned knowledge-base article."""
    if status not in ALLOWED_ARTICLE_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of: {sorted(ALLOWED_ARTICLE_STATUSES)}")

    query = """
        INSERT INTO kb_articles (tenant_id, title, body, status)
        VALUES (%s, %s, %s, %s)
        RETURNING id, tenant_id, title, body, status, created_at, updated_at;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (tenant_id, title, body, status))
        row = cur.fetchone()
        assert row is not None
        return KBArticleRecord(
            id=uuid.UUID(str(row["id"])),
            tenant_id=uuid.UUID(str(row["tenant_id"])),
            title=row["title"],
            body=row["body"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def create_kb_chunks(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    article_id: uuid.UUID,
    chunks: list[ChunkData],
) -> list[KBChunkRecord]:
    """Bulk insert chunks for an article enforcing composite tenant-article foreign key."""
    if not chunks:
        return []

    query = """
        INSERT INTO kb_chunks (
            tenant_id, article_id, chunk_index, content, char_count, token_count
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id, tenant_id, article_id, chunk_index, content, char_count, token_count,
                  embedding::text, model_name, model_version, created_at, updated_at;
    """
    results: list[KBChunkRecord] = []
    with conn.cursor(row_factory=dict_row) as cur:
        for c in chunks:
            cur.execute(query, (tenant_id, article_id, c.chunk_index, c.content, c.char_count, c.token_count))
            row = cur.fetchone()
            assert row is not None
            results.append(
                KBChunkRecord(
                    id=uuid.UUID(str(row["id"])),
                    tenant_id=uuid.UUID(str(row["tenant_id"])),
                    article_id=uuid.UUID(str(row["article_id"])),
                    chunk_index=row["chunk_index"],
                    content=row["content"],
                    char_count=row["char_count"],
                    token_count=row["token_count"],
                    embedding=_parse_vector(row["embedding"]),
                    model_name=row["model_name"],
                    model_version=row["model_version"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
    return results


def get_kb_article_by_id(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    article_id: uuid.UUID,
) -> KBArticleRecord | None:
    """Retrieve an article by ID strictly scoped to tenant_id."""
    query = """
        SELECT id, tenant_id, title, body, status, created_at, updated_at
        FROM kb_articles
        WHERE tenant_id = %s AND id = %s;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (tenant_id, article_id))
        row = cur.fetchone()
        if not row:
            return None
        return KBArticleRecord(
            id=uuid.UUID(str(row["id"])),
            tenant_id=uuid.UUID(str(row["tenant_id"])),
            title=row["title"],
            body=row["body"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def list_kb_articles(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    limit: int = 50,
    offset: int = 0,
) -> list[KBArticleSummaryRecord]:
    """List tenant articles with chunk counts, newest first."""
    clamped_limit = max(1, min(100, limit))
    clamped_offset = max(0, offset)

    query = """
        SELECT ka.id, ka.tenant_id, ka.title, ka.status,
               ka.created_at, ka.updated_at,
               COUNT(kc.id)::int AS chunk_count
        FROM kb_articles ka
        LEFT JOIN kb_chunks kc
          ON kc.tenant_id = ka.tenant_id AND kc.article_id = ka.id
        WHERE ka.tenant_id = %s
        GROUP BY ka.id, ka.tenant_id, ka.title, ka.status, ka.created_at, ka.updated_at
        ORDER BY ka.created_at DESC
        LIMIT %s OFFSET %s;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (tenant_id, clamped_limit, clamped_offset))
        rows = cur.fetchall()
        return [
            KBArticleSummaryRecord(
                id=uuid.UUID(str(r["id"])),
                tenant_id=uuid.UUID(str(r["tenant_id"])),
                title=r["title"],
                status=r["status"],
                chunk_count=r["chunk_count"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]


def get_kb_chunks_for_article(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    article_id: uuid.UUID,
) -> list[KBChunkRecord]:
    """Retrieve all chunks for an article ordered by chunk_index ASC."""
    query = """
        SELECT id, tenant_id, article_id, chunk_index, content, char_count, token_count,
               embedding::text, model_name, model_version, created_at, updated_at
        FROM kb_chunks
        WHERE tenant_id = %s AND article_id = %s
        ORDER BY chunk_index ASC;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (tenant_id, article_id))
        rows = cur.fetchall()
        return [
            KBChunkRecord(
                id=uuid.UUID(str(r["id"])),
                tenant_id=uuid.UUID(str(r["tenant_id"])),
                article_id=uuid.UUID(str(r["article_id"])),
                chunk_index=r["chunk_index"],
                content=r["content"],
                char_count=r["char_count"],
                token_count=r["token_count"],
                embedding=_parse_vector(r["embedding"]),
                model_name=r["model_name"],
                model_version=r["model_version"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]


def upsert_kb_chunk_embedding(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    chunk_id: uuid.UUID,
    embedding: list[float] | str,
    model_name: str,
    model_version: str,
) -> KBChunkRecord | None:
    """Persist embedding vector for a chunk scoped strictly to tenant_id."""
    vec_str = _format_vector(embedding)
    query = """
        UPDATE kb_chunks
        SET embedding = %s::vector,
            model_name = %s,
            model_version = %s,
            updated_at = NOW()
        WHERE tenant_id = %s AND id = %s
        RETURNING id, tenant_id, article_id, chunk_index, content, char_count, token_count,
                  embedding::text, model_name, model_version, created_at, updated_at;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (vec_str, model_name, model_version, tenant_id, chunk_id))
        row = cur.fetchone()
        if not row:
            return None
        return KBChunkRecord(
            id=uuid.UUID(str(row["id"])),
            tenant_id=uuid.UUID(str(row["tenant_id"])),
            article_id=uuid.UUID(str(row["article_id"])),
            chunk_index=row["chunk_index"],
            content=row["content"],
            char_count=row["char_count"],
            token_count=row["token_count"],
            embedding=_parse_vector(row["embedding"]),
            model_name=row["model_name"],
            model_version=row["model_version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def update_kb_article_status(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    article_id: uuid.UUID,
    status: str,
) -> KBArticleRecord | None:
    """Update status of a knowledge-base article."""
    if status not in ALLOWED_ARTICLE_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of: {sorted(ALLOWED_ARTICLE_STATUSES)}")

    query = """
        UPDATE kb_articles
        SET status = %s, updated_at = NOW()
        WHERE tenant_id = %s AND id = %s
        RETURNING id, tenant_id, title, body, status, created_at, updated_at;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (status, tenant_id, article_id))
        row = cur.fetchone()
        if not row:
            return None
        return KBArticleRecord(
            id=uuid.UUID(str(row["id"])),
            tenant_id=uuid.UUID(str(row["tenant_id"])),
            title=row["title"],
            body=row["body"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


# ==============================================================================
# Article Jobs Queue Operations (Dedicated Queue)
# ==============================================================================

def enqueue_article_job(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    article_id: uuid.UUID,
    job_type: str = "article_embedding",
    payload: dict[str, Any] | None = None,
) -> ArticleJobRecord:
    """Enqueue an article embedding job atomically in the caller transaction."""
    payload_json = json.dumps(payload or {})
    query = """
        INSERT INTO article_jobs (
            tenant_id, article_id, type, status, payload
        )
        VALUES (%s, %s, %s, 'queued', %s::jsonb)
        ON CONFLICT (tenant_id, article_id, type)
        DO UPDATE SET
            status = CASE WHEN article_jobs.status = 'failed' THEN 'queued' ELSE article_jobs.status END,
            run_after = CASE WHEN article_jobs.status = 'failed' THEN NOW() ELSE article_jobs.run_after END,
            error = CASE WHEN article_jobs.status = 'failed' THEN NULL ELSE article_jobs.error END,
            updated_at = NOW()
        RETURNING
            id, tenant_id, article_id, type, status,
            attempts, max_attempts, run_after, locked_at, locked_by,
            created_at, updated_at, completed_at, failed_at, error, payload;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (tenant_id, article_id, job_type, payload_json))
        row = cur.fetchone()
        assert row is not None
        return _row_to_article_job(row)


def claim_next_article_job(
    conn: psycopg.Connection[Any],
    worker_id: str,
    lease_seconds: int = 300,
    tenant_id: uuid.UUID | None = None,
) -> ArticleJobRecord | None:
    """Claim the next eligible queued article job atomically via FOR UPDATE SKIP LOCKED.

    Fences ownership with worker_id lease and increments attempts generation.
    Deterministic ordering: run_after ASC, created_at ASC, id ASC.
    """
    query = """
        WITH next_job AS (
            SELECT id
            FROM article_jobs
            WHERE status = 'queued'
              AND run_after <= NOW()
              AND attempts < max_attempts
              AND (%s::uuid IS NULL OR tenant_id = %s)
              AND type = 'article_embedding'
            ORDER BY run_after ASC, created_at ASC, id ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        UPDATE article_jobs aj
        SET status = 'running',
            locked_at = NOW(),
            locked_by = %s,
            attempts = attempts + 1,
            updated_at = NOW()
        FROM next_job
        WHERE aj.id = next_job.id
        RETURNING
            aj.id, aj.tenant_id, aj.article_id, aj.type, aj.status,
            aj.attempts, aj.max_attempts, aj.run_after, aj.locked_at, aj.locked_by,
            aj.created_at, aj.updated_at, aj.completed_at, aj.failed_at, aj.error, aj.payload;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (tenant_id, tenant_id, worker_id))
        row = cur.fetchone()
        if not row:
            return None
        return _row_to_article_job(row)


def renew_article_job_lease(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    job_id: uuid.UUID,
    worker_id: str,
    generation: int,
    lease_seconds: int = 300,
) -> bool:
    """Renew active lease on an in-progress article job."""
    query = """
        UPDATE article_jobs
        SET locked_at = NOW(),
            updated_at = NOW()
        WHERE id = %s
          AND tenant_id = %s
          AND status = 'running'
          AND locked_by = %s
          AND attempts = %s
          AND locked_at >= NOW() - (%s || ' seconds')::interval;
    """
    with conn.cursor() as cur:
        cur.execute(query, (job_id, tenant_id, worker_id, generation, lease_seconds))
        return cur.rowcount > 0


def release_claimed_article_job_to_queued(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    job_id: uuid.UUID,
    worker_id: str,
    generation: int,
) -> bool:
    """Release claimed article job back to queued when embeddings are disabled."""
    query = """
        UPDATE article_jobs
        SET status = 'queued',
            locked_at = NULL,
            locked_by = NULL,
            updated_at = NOW()
        WHERE id = %s
          AND tenant_id = %s
          AND status = 'running'
          AND locked_by = %s
          AND attempts = %s;
    """
    with conn.cursor() as cur:
        cur.execute(query, (job_id, tenant_id, worker_id, generation))
        return cur.rowcount > 0


def complete_article_job(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    job_id: uuid.UUID,
    article_id: uuid.UUID,
    worker_id: str,
    generation: int,
    chunk_embeddings: list[tuple[uuid.UUID, list[float], str, str]],
    lease_seconds: int = 300,
) -> bool:
    """Atomically persist chunk embeddings, set article ready, and complete job claim.

    Guarantees:
    - Verifies active claim generation before mutations; raises StaleClaimError on mismatch.
    - Strictly asserts that every expected chunk for the article is embedded; does not mark
      article ready until all chunks are embedded.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        # 1. Verify expected chunk count
        cur.execute(
            "SELECT COUNT(*)::int AS count FROM kb_chunks WHERE tenant_id = %s AND article_id = %s;",
            (tenant_id, article_id),
        )
        c_row = cur.fetchone()
        expected_chunk_count = c_row["count"] if c_row else 0
        if len(chunk_embeddings) != expected_chunk_count:
            raise ValueError(
                f"Cannot complete article job: received {len(chunk_embeddings)} embeddings but article has {expected_chunk_count} chunks."
            )

        # 2. Complete job claim with fencing
        cur.execute(
            """
            UPDATE article_jobs
            SET status = 'completed',
                completed_at = NOW(),
                locked_at = NULL,
                locked_by = NULL,
                updated_at = NOW()
            WHERE id = %s
              AND tenant_id = %s
              AND status = 'running'
              AND locked_by = %s
              AND attempts = %s
              AND locked_at >= NOW() - (%s || ' seconds')::interval;
            """,
            (job_id, tenant_id, worker_id, generation, lease_seconds),
        )
        if cur.rowcount == 0:
            raise StaleClaimError("Cannot complete article job: lease expired or generation mismatch.")

        # 3. Persist embeddings for all chunks
        for chunk_id, vector, model_name, model_version in chunk_embeddings:
            upsert_kb_chunk_embedding(
                conn=conn,
                tenant_id=tenant_id,
                chunk_id=chunk_id,
                embedding=vector,
                model_name=model_name,
                model_version=model_version,
            )

        # 4. Mark article ready
        cur.execute(
            """
            UPDATE kb_articles
            SET status = 'ready',
                updated_at = NOW()
            WHERE tenant_id = %s AND id = %s;
            """,
            (tenant_id, article_id),
        )

    return True


def fail_or_retry_article_job(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    job_id: uuid.UUID,
    worker_id: str,
    generation: int,
    error_message: str,
    article_id: uuid.UUID | None = None,
) -> str:
    """Record article job failure or schedule exponential backoff retry.

    When retries are exhausted, sets both article_jobs and kb_articles to 'failed'.
    Returns: 'requeued', 'failed', or 'stale_claim'.
    """
    sanitized_error = error_message.strip()[:255] if error_message else "unknown_error"

    query_check = """
        SELECT attempts, max_attempts, article_id
        FROM article_jobs
        WHERE id = %s
          AND tenant_id = %s
          AND status = 'running'
          AND locked_by = %s
          AND attempts = %s;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query_check, (job_id, tenant_id, worker_id, generation))
        row = cur.fetchone()
        if not row:
            return "stale_claim"

        attempts = row["attempts"]
        max_attempts = row["max_attempts"]
        resolved_article_id = article_id or uuid.UUID(str(row["article_id"]))

        if attempts < max_attempts:
            delay_seconds = 2 * (2 ** max(0, attempts - 1))
            cur.execute(
                """
                UPDATE article_jobs
                SET status = 'queued',
                    run_after = NOW() + (%s || ' seconds')::interval,
                    locked_at = NULL,
                    locked_by = NULL,
                    error = %s,
                    updated_at = NOW()
                WHERE id = %s
                  AND tenant_id = %s
                  AND status = 'running'
                  AND locked_by = %s
                  AND attempts = %s;
                """,
                (delay_seconds, sanitized_error, job_id, tenant_id, worker_id, generation),
            )
            if cur.rowcount == 0:
                return "stale_claim"
            return "requeued"
        else:
            cur.execute(
                """
                UPDATE article_jobs
                SET status = 'failed',
                    failed_at = NOW(),
                    locked_at = NULL,
                    locked_by = NULL,
                    error = %s,
                    updated_at = NOW()
                WHERE id = %s
                  AND tenant_id = %s
                  AND status = 'running'
                  AND locked_by = %s
                  AND attempts = %s;
                """,
                (sanitized_error, job_id, tenant_id, worker_id, generation),
            )
            if cur.rowcount == 0:
                return "stale_claim"

            cur.execute(
                """
                UPDATE kb_articles
                SET status = 'failed',
                    updated_at = NOW()
                WHERE tenant_id = %s AND id = %s;
                """,
                (tenant_id, resolved_article_id),
            )
            return "failed"


def recover_stale_article_jobs(
    conn: psycopg.Connection[Any],
    lease_seconds: int = 300,
    tenant_id: uuid.UUID | None = None,
) -> int:
    """Recover expired running leases on article_jobs."""
    query = """
        SELECT id, tenant_id, article_id, attempts, max_attempts
        FROM article_jobs
        WHERE status = 'running'
          AND locked_at < NOW() - (%s || ' seconds')::interval
          AND (%s::uuid IS NULL OR tenant_id = %s)
        FOR UPDATE SKIP LOCKED;
    """
    recovered_count = 0
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (lease_seconds, tenant_id, tenant_id))
        stale_rows = cur.fetchall()

        for row in stale_rows:
            job_id = row["id"]
            row_tenant_id = row["tenant_id"]
            art_id = row["article_id"]
            attempts = row["attempts"]
            max_attempts = row["max_attempts"]

            if attempts >= max_attempts:
                cur.execute(
                    """
                    UPDATE article_jobs
                    SET status = 'failed',
                        failed_at = NOW(),
                        locked_at = NULL,
                        locked_by = NULL,
                        error = 'lease_timeout_exhausted',
                        updated_at = NOW()
                    WHERE id = %s
                      AND tenant_id = %s
                      AND status = 'running'
                      AND attempts = %s;
                    """,
                    (job_id, row_tenant_id, attempts),
                )
                if cur.rowcount > 0:
                    cur.execute(
                        """
                        UPDATE kb_articles
                        SET status = 'failed',
                            updated_at = NOW()
                        WHERE tenant_id = %s AND id = %s;
                        """,
                        (row_tenant_id, art_id),
                    )
                    recovered_count += 1
            else:
                cur.execute(
                    """
                    UPDATE article_jobs
                    SET status = 'queued',
                        locked_at = NULL,
                        locked_by = NULL,
                        run_after = NOW(),
                        updated_at = NOW()
                    WHERE id = %s
                      AND tenant_id = %s
                      AND status = 'running'
                      AND attempts = %s;
                    """,
                    (job_id, row_tenant_id, attempts),
                )
                if cur.rowcount > 0:
                    recovered_count += 1

    return recovered_count


# ==============================================================================
# Exact Cosine Retrieval
# ==============================================================================

def find_relevant_chunks_for_ticket(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    ticket_id: uuid.UUID,
    limit: int = 5,
    model_name: str | None = None,
    model_version: str | None = None,
) -> tuple[str, str | None, str | None, list[RelevantKnowledgeCandidate]]:
    """Retrieve top-k relevant article chunks for a ticket using exact cosine similarity.

    Distinct States:
    - 'missing_ticket_embedding': Ticket has no embedding vector in ticket_embeddings.
    - 'no_chunks_available': Tenant has no articles or chunks.
    - 'articles_failed': All tenant articles failed processing.
    - 'model_mismatch': Chunk embeddings exist, but for a different model version.
    - 'articles_processing': Articles exist with pending embeddings.
    - 'ready': Retrieval succeeded (ranked candidates returned).
    """
    bounded_limit = max(1, min(20, limit))

    with conn.cursor(row_factory=dict_row) as cur:
        # 1. Fetch source ticket embedding
        if model_name is not None and model_version is not None:
            source_emb_query = """
                SELECT embedding::text, model_name, model_version
                FROM ticket_embeddings
                WHERE tenant_id = %s AND ticket_id = %s AND model_name = %s AND model_version = %s;
            """
            cur.execute(source_emb_query, (tenant_id, ticket_id, model_name, model_version))
        else:
            source_emb_query = """
                SELECT embedding::text, model_name, model_version
                FROM ticket_embeddings
                WHERE tenant_id = %s AND ticket_id = %s
                ORDER BY updated_at DESC, id DESC
                LIMIT 1;
            """
            cur.execute(source_emb_query, (tenant_id, ticket_id))

        source_row = cur.fetchone()
        if not source_row or not source_row.get("embedding"):
            return "missing_ticket_embedding", None, None, []

        source_vec_str = str(source_row["embedding"])
        resolved_model_name = str(source_row["model_name"])
        resolved_model_version = str(source_row["model_version"])

        # 2. Check tenant article inventory & processing status
        cur.execute(
            """
            SELECT COUNT(ka.id)::int AS article_count,
                   COUNT(CASE WHEN ka.status = 'failed' THEN 1 END)::int AS failed_article_count,
                   COUNT(CASE WHEN ka.status = 'processing' THEN 1 END)::int AS processing_article_count,
                   COUNT(kc.id)::int AS chunk_count,
                   COUNT(CASE WHEN kc.embedding IS NOT NULL THEN 1 END)::int AS embedded_chunk_count,
                   COUNT(CASE WHEN kc.embedding IS NOT NULL AND kc.model_name = %s AND kc.model_version = %s THEN 1 END)::int AS matching_embedded_count
            FROM kb_articles ka
            LEFT JOIN kb_chunks kc
              ON kc.tenant_id = ka.tenant_id AND kc.article_id = ka.id
            WHERE ka.tenant_id = %s;
            """,
            (resolved_model_name, resolved_model_version, tenant_id),
        )
        counts = cur.fetchone()
        article_count = counts["article_count"] if counts else 0
        failed_article_count = counts["failed_article_count"] if counts else 0
        processing_article_count = counts["processing_article_count"] if counts else 0
        chunk_count = counts["chunk_count"] if counts else 0
        embedded_chunk_count = counts["embedded_chunk_count"] if counts else 0
        matching_embedded_count = counts["matching_embedded_count"] if counts else 0

        if article_count == 0 or chunk_count == 0:
            return "no_chunks_available", resolved_model_name, resolved_model_version, []

        # If all articles failed, report articles_failed rather than misleading processing label
        if failed_article_count > 0 and failed_article_count == article_count:
            return "articles_failed", resolved_model_name, resolved_model_version, []

        # If chunks are embedded but with a different model
        if matching_embedded_count == 0 and embedded_chunk_count > 0 and processing_article_count == 0:
            return "model_mismatch", resolved_model_name, resolved_model_version, []

        # If no matching embeddings ready yet and articles are processing
        if matching_embedded_count == 0:
            return "articles_processing", resolved_model_name, resolved_model_version, []

        # 3. Exact cosine retrieval via pgvector
        query = """
            SELECT
                kc.id AS chunk_id,
                kc.article_id,
                ka.title AS article_title,
                kc.chunk_index,
                kc.content AS excerpt,
                kc.char_count,
                kc.token_count,
                (1.0 - (kc.embedding <=> %s::vector)) AS similarity_score
            FROM kb_chunks kc
            JOIN kb_articles ka
              ON ka.id = kc.article_id AND ka.tenant_id = kc.tenant_id
            WHERE kc.tenant_id = %s
              AND kc.model_name = %s
              AND kc.model_version = %s
              AND kc.embedding IS NOT NULL
            ORDER BY kc.embedding <=> %s::vector ASC, ka.id ASC, kc.chunk_index ASC
            LIMIT %s;
        """
        cur.execute(
            query,
            (
                source_vec_str,
                tenant_id,
                resolved_model_name,
                resolved_model_version,
                source_vec_str,
                bounded_limit,
            ),
        )
        rows = cur.fetchall()

        candidates = [
            RelevantKnowledgeCandidate(
                chunk_id=uuid.UUID(str(r["chunk_id"])),
                article_id=uuid.UUID(str(r["article_id"])),
                article_title=r["article_title"],
                chunk_index=r["chunk_index"],
                excerpt=r["excerpt"],
                char_count=r["char_count"],
                token_count=r["token_count"],
                similarity_score=max(-1.0, min(1.0, float(r["similarity_score"]))),
            )
            for r in rows
        ]

        return "ready", resolved_model_name, resolved_model_version, candidates
