"""Repository layer for multi-tenant ticket embeddings and duplicate links.

Per SPEC.md Section 5, 6 and Phase 4 foundations:
- All database access requires tenant_id and uses it in every query.
- Composite foreign keys guarantee database-level multi-tenant isolation.
- Undirected duplicate links are enforced canonically (source_ticket_id < target_ticket_id).
- Vector dimension is vector(384) corresponding to all-MiniLM-L6-v2.
- Similarity scores represent cosine similarity bounded in [-1.0, 1.0].
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any
import uuid
import psycopg
from psycopg.rows import dict_row


class SelfLinkError(ValueError):
    """Raised when attempting to link a ticket to itself."""
    pass


class InvalidStatusError(ValueError):
    """Raised when duplicate link status is not in allowed set."""
    pass


class InvalidSimilarityScoreError(ValueError):
    """Raised when similarity score is outside valid range [-1.0, 1.0] or non-finite."""
    pass


class ContradictoryReviewStateError(ValueError):
    """Raised when reviewed_at and status are contradictory (e.g. reviewed_at set for suggested)."""
    pass


ALLOWED_LINK_STATUSES = {"suggested", "confirmed", "rejected"}


@dataclass(frozen=True)
class EmbeddingRecord:
    """Read-only view of a ticket embedding record."""
    id: uuid.UUID
    tenant_id: uuid.UUID
    ticket_id: uuid.UUID
    embedding: list[float]
    model_name: str
    model_version: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class DuplicateLinkRecord:
    """Read-only view of a duplicate ticket link record."""
    id: uuid.UUID
    tenant_id: uuid.UUID
    source_ticket_id: uuid.UUID
    target_ticket_id: uuid.UUID
    similarity_score: float
    model_name: str
    model_version: str
    status: str
    created_at: datetime
    reviewed_at: datetime | None


def _parse_vector(raw: Any) -> list[float]:
    """Parse vector representation returned by postgres/psycopg into list of floats."""
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, str):
        cleaned = raw.strip("[] \t\r\n")
        if not cleaned:
            return []
        return [float(x.strip()) for x in cleaned.split(",")]
    raise TypeError(f"Unexpected vector type: {type(raw)}")


def _format_vector(embedding: list[float] | str) -> str:
    """Format float list into postgres vector literal string '[0.1,0.2,...]'."""
    if isinstance(embedding, str):
        return embedding
    return f"[{','.join(str(x) for x in embedding)}]"


def _row_to_embedding(row: dict[str, Any]) -> EmbeddingRecord:
    return EmbeddingRecord(
        id=uuid.UUID(str(row["id"])),
        tenant_id=uuid.UUID(str(row["tenant_id"])),
        ticket_id=uuid.UUID(str(row["ticket_id"])),
        embedding=_parse_vector(row["embedding"]),
        model_name=row["model_name"],
        model_version=row["model_version"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_duplicate_link(row: dict[str, Any]) -> DuplicateLinkRecord:
    return DuplicateLinkRecord(
        id=uuid.UUID(str(row["id"])),
        tenant_id=uuid.UUID(str(row["tenant_id"])),
        source_ticket_id=uuid.UUID(str(row["source_ticket_id"])),
        target_ticket_id=uuid.UUID(str(row["target_ticket_id"])),
        similarity_score=float(row["similarity_score"]),
        model_name=row["model_name"],
        model_version=row["model_version"],
        status=row["status"],
        created_at=row["created_at"],
        reviewed_at=row["reviewed_at"],
    )


# ==============================================================================
# Ticket Embeddings Operations
# ==============================================================================

def save_ticket_embedding(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    ticket_id: uuid.UUID,
    embedding: list[float] | str,
    model_name: str,
    model_version: str,
) -> EmbeddingRecord:
    """Persist a ticket embedding idempotently for a given tenant, ticket, and model.

    If an embedding already exists for (tenant_id, ticket_id, model_name, model_version),
    it is updated with the new vector and updated_at timestamp.
    Database composite foreign key enforces that ticket_id belongs to tenant_id.
    """
    vec_str = _format_vector(embedding)
    query = """
        INSERT INTO ticket_embeddings (
            tenant_id, ticket_id, embedding, model_name, model_version
        )
        VALUES (%s, %s, %s::vector, %s, %s)
        ON CONFLICT (tenant_id, ticket_id, model_name, model_version)
        DO UPDATE SET
            embedding = EXCLUDED.embedding,
            updated_at = NOW()
        RETURNING id, tenant_id, ticket_id, embedding::text, model_name, model_version, created_at, updated_at;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (tenant_id, ticket_id, vec_str, model_name, model_version))
        row = cur.fetchone()
        assert row is not None
        return _row_to_embedding(row)


def get_ticket_embedding(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    ticket_id: uuid.UUID,
    model_name: str,
    model_version: str,
) -> EmbeddingRecord | None:
    """Retrieve embedding for a specific ticket and model version within a tenant."""
    query = """
        SELECT id, tenant_id, ticket_id, embedding::text, model_name, model_version, created_at, updated_at
        FROM ticket_embeddings
        WHERE tenant_id = %s AND ticket_id = %s AND model_name = %s AND model_version = %s;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (tenant_id, ticket_id, model_name, model_version))
        row = cur.fetchone()
        if not row:
            return None
        return _row_to_embedding(row)


def delete_ticket_embedding(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    ticket_id: uuid.UUID,
    model_name: str | None = None,
    model_version: str | None = None,
) -> int:
    """Delete ticket embedding(s) within a tenant. Returns number of rows deleted."""
    if model_name is not None and model_version is not None:
        query = """
            DELETE FROM ticket_embeddings
            WHERE tenant_id = %s AND ticket_id = %s AND model_name = %s AND model_version = %s;
        """
        params = (tenant_id, ticket_id, model_name, model_version)
    else:
        query = """
            DELETE FROM ticket_embeddings
            WHERE tenant_id = %s AND ticket_id = %s;
        """
        params = (tenant_id, ticket_id)

    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.rowcount


# ==============================================================================
# Duplicate Links Operations
# ==============================================================================

def create_duplicate_link(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    ticket_id_1: uuid.UUID,
    ticket_id_2: uuid.UUID,
    similarity_score: float,
    model_name: str,
    model_version: str,
    status: str = "suggested",
    reviewed_at: datetime | None = None,
    auto_canonicalize: bool = True,
) -> DuplicateLinkRecord:
    """Create a duplicate link between two tickets belonging to the same tenant.

    Semantics:
    - Self-links (ticket_id_1 == ticket_id_2) are rejected.
    - Undirected canonical ordering: if auto_canonicalize is True, tickets are
      ordered such that source_ticket_id = min(t1, t2) and target_ticket_id = max(t1, t2).
      If auto_canonicalize is False, tickets are passed as given, allowing tests or callers
      to verify database check-constraint enforcement.
    - Similarity score must be a finite float within bounds [-1.0, 1.0].
      Rejects NaN, +inf, -inf before SQL execution.
    - Review state contract:
      - status == 'suggested' => reviewed_at must be NULL (raises ContradictoryReviewStateError if passed).
      - status in ('confirmed', 'rejected') => reviewed_at is populated (defaults to NOW()).
    - Repeated links for the same pair will violate uniqueness.
    """
    if ticket_id_1 == ticket_id_2:
        raise SelfLinkError(f"Cannot link ticket {ticket_id_1} to itself.")

    if status not in ALLOWED_LINK_STATUSES:
        raise InvalidStatusError(
            f"Invalid status '{status}'. Must be one of: {sorted(ALLOWED_LINK_STATUSES)}"
        )

    if not math.isfinite(similarity_score) or similarity_score < -1.0 or similarity_score > 1.0:
        raise InvalidSimilarityScoreError(
            f"Similarity score {similarity_score} is not a finite float within bounds [-1.0, 1.0]."
        )

    if status == "suggested":
        if reviewed_at is not None:
            raise ContradictoryReviewStateError(
                "Cannot specify reviewed_at when duplicate link status is 'suggested'."
            )
        effective_reviewed_at = None
    else:
        effective_reviewed_at = reviewed_at or datetime.now(timezone.utc)

    if auto_canonicalize:
        source_id = ticket_id_1 if ticket_id_1 < ticket_id_2 else ticket_id_2
        target_id = ticket_id_2 if ticket_id_1 < ticket_id_2 else ticket_id_1
    else:
        source_id = ticket_id_1
        target_id = ticket_id_2

    query = """
        INSERT INTO duplicate_links (
            tenant_id, source_ticket_id, target_ticket_id,
            similarity_score, model_name, model_version, status, reviewed_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id, tenant_id, source_ticket_id, target_ticket_id,
                  similarity_score, model_name, model_version, status,
                  created_at, reviewed_at;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            query,
            (
                tenant_id,
                source_id,
                target_id,
                similarity_score,
                model_name,
                model_version,
                status,
                effective_reviewed_at,
            ),
        )
        row = cur.fetchone()
        assert row is not None
        return _row_to_duplicate_link(row)


def get_duplicate_link(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    link_id: uuid.UUID,
) -> DuplicateLinkRecord | None:
    """Retrieve duplicate link by id, strictly scoped by tenant_id."""
    query = """
        SELECT id, tenant_id, source_ticket_id, target_ticket_id,
               similarity_score, model_name, model_version, status,
               created_at, reviewed_at
        FROM duplicate_links
        WHERE tenant_id = %s AND id = %s;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (tenant_id, link_id))
        row = cur.fetchone()
        if not row:
            return None
        return _row_to_duplicate_link(row)


def get_duplicate_links_for_ticket(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    ticket_id: uuid.UUID,
) -> list[DuplicateLinkRecord]:
    """Retrieve all duplicate links involving ticket_id (as either source or target).

    Because links are stored canonically (source < target), a ticket may appear in
    either column. Returns links sorted by similarity_score DESC.
    """
    query = """
        SELECT id, tenant_id, source_ticket_id, target_ticket_id,
               similarity_score, model_name, model_version, status,
               created_at, reviewed_at
        FROM duplicate_links
        WHERE tenant_id = %s AND (source_ticket_id = %s OR target_ticket_id = %s)
        ORDER BY similarity_score DESC, created_at DESC;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (tenant_id, ticket_id, ticket_id))
        return [_row_to_duplicate_link(r) for r in cur.fetchall()]


def update_duplicate_link_status(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    link_id: uuid.UUID,
    status: str,
    reviewed_at: datetime | None = None,
) -> DuplicateLinkRecord | None:
    """Update status of duplicate link (e.g., 'confirmed' or 'rejected') with reviewed_at.

    Review state contract:
    - status == 'suggested' => reviewed_at is set to NULL (raises ContradictoryReviewStateError if reviewed_at passed).
    - status in ('confirmed', 'rejected') => reviewed_at is populated (defaults to NOW() if omitted).
    """
    if status not in ALLOWED_LINK_STATUSES:
        raise InvalidStatusError(
            f"Invalid status '{status}'. Must be one of: {sorted(ALLOWED_LINK_STATUSES)}"
        )

    if status == "suggested":
        if reviewed_at is not None:
            raise ContradictoryReviewStateError(
                "Cannot specify reviewed_at when updating duplicate link status to 'suggested'."
            )
        ts = None
    else:
        ts = reviewed_at or datetime.now(timezone.utc)

    query = """
        UPDATE duplicate_links
        SET status = %s, reviewed_at = %s
        WHERE tenant_id = %s AND id = %s
        RETURNING id, tenant_id, source_ticket_id, target_ticket_id,
                  similarity_score, model_name, model_version, status,
                  created_at, reviewed_at;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (status, ts, tenant_id, link_id))
        row = cur.fetchone()
        if not row:
            return None
        return _row_to_duplicate_link(row)


def list_duplicate_links(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[DuplicateLinkRecord]:
    """List duplicate links for a tenant, optionally filtered by status."""
    if status is not None:
        query = """
            SELECT id, tenant_id, source_ticket_id, target_ticket_id,
                   similarity_score, model_name, model_version, status,
                   created_at, reviewed_at
            FROM duplicate_links
            WHERE tenant_id = %s AND status = %s
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s;
        """
        params = (tenant_id, status, limit, offset)
    else:
        query = """
            SELECT id, tenant_id, source_ticket_id, target_ticket_id,
                   similarity_score, model_name, model_version, status,
                   created_at, reviewed_at
            FROM duplicate_links
            WHERE tenant_id = %s
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s;
        """
        params = (tenant_id, limit, offset)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        return [_row_to_duplicate_link(r) for r in cur.fetchall()]


def delete_duplicate_link(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    link_id: uuid.UUID,
) -> bool:
    """Delete a duplicate link scoped strictly to tenant_id. Returns True if deleted."""
    query = """
        DELETE FROM duplicate_links
        WHERE tenant_id = %s AND id = %s;
    """
    with conn.cursor() as cur:
        cur.execute(query, (tenant_id, link_id))
        return cur.rowcount > 0
