"""Ticket repository layer for multi-tenant ticket operations.

Per SPEC.md Section 5, 6, and 7:
- All functions require tenant_id and filter by it in every query.
- New ticket returns created ticket record.
- If external_id exists for the same tenant, returns the existing record without updating it.
- List tickets newest first, with deterministic tie-breaker (created_at DESC, id DESC).
- Detail lookup returns single ticket or None.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
import uuid
import psycopg
from psycopg.rows import dict_row


@dataclass(frozen=True)
class TicketRecord:
    """Read-only view of a ticket record."""
    id: uuid.UUID
    tenant_id: uuid.UUID
    external_id: str | None
    subject: str
    body: str
    customer_email: str | None
    status: str
    category: str | None
    priority: str | None
    created_at: datetime
    first_response_at: datetime | None
    resolved_at: datetime | None
    assignee_id: uuid.UUID | None


class InvalidStatusTransitionError(ValueError):
    """Raised when an invalid ticket lifecycle transition is attempted."""
    pass


@dataclass(frozen=True)
class TicketListItem:
    """Read-only view of a ticket list row with prediction and job processing metadata."""
    id: uuid.UUID
    tenant_id: uuid.UUID
    external_id: str | None
    subject: str
    body: str
    customer_email: str | None
    status: str
    category: str | None
    priority: str | None
    created_at: datetime
    predicted_label: str | None
    model_name: str | None
    model_version: str | None
    job_status: str | None = None
    job_attempts: int | None = None
    job_max_attempts: int | None = None
    job_run_after: datetime | None = None


def _row_to_ticket(row: dict[str, Any]) -> TicketRecord:
    """Convert a database row dict to TicketRecord."""
    return TicketRecord(
        id=uuid.UUID(str(row["id"])),
        tenant_id=uuid.UUID(str(row["tenant_id"])),
        external_id=row["external_id"],
        subject=row["subject"],
        body=row["body"],
        customer_email=row["customer_email"],
        status=row["status"],
        category=row["category"],
        priority=row["priority"],
        created_at=row["created_at"],
        first_response_at=row["first_response_at"],
        resolved_at=row["resolved_at"],
        assignee_id=uuid.UUID(str(row["assignee_id"])) if row["assignee_id"] else None,
    )


def create_or_get_ticket(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    subject: str,
    body: str,
    external_id: str | None = None,
    customer_email: str | None = None,
) -> tuple[TicketRecord, bool]:
    """Insert a new ticket or return existing ticket if external_id matches for this tenant.

    Returns:
        tuple[TicketRecord, bool]: (ticket_record, is_created)
        is_created is True if a new ticket was created, False if an existing ticket was returned.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        # Attempt atomic insert with ON CONFLICT DO NOTHING for external_id
        if external_id is not None:
            cur.execute(
                """
                INSERT INTO tickets (tenant_id, external_id, subject, body, customer_email, status)
                VALUES (%s, %s, %s, %s, %s, 'new')
                ON CONFLICT (tenant_id, external_id) WHERE external_id IS NOT NULL
                DO NOTHING
                RETURNING id, tenant_id, external_id, subject, body, customer_email,
                          status, category, priority, created_at, first_response_at,
                          resolved_at, assignee_id;
                """,
                (tenant_id, external_id, subject, body, customer_email),
            )
            created_row = cur.fetchone()
            if created_row:
                return _row_to_ticket(created_row), True

            # If ON CONFLICT was triggered (ticket already exists), fetch existing ticket
            cur.execute(
                """
                SELECT id, tenant_id, external_id, subject, body, customer_email,
                       status, category, priority, created_at, first_response_at,
                       resolved_at, assignee_id
                FROM tickets
                WHERE tenant_id = %s AND external_id = %s;
                """,
                (tenant_id, external_id),
            )
            return _row_to_ticket(cur.fetchone()), False

        # Independent insertion when external_id is None
        cur.execute(
            """
            INSERT INTO tickets (tenant_id, external_id, subject, body, customer_email, status)
            VALUES (%s, NULL, %s, %s, %s, 'new')
            RETURNING id, tenant_id, external_id, subject, body, customer_email,
                      status, category, priority, created_at, first_response_at,
                      resolved_at, assignee_id;
            """,
            (tenant_id, subject, body, customer_email),
        )
        return _row_to_ticket(cur.fetchone()), True


def list_tickets(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    status_filter: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[TicketRecord]:
    """List tickets for tenant_id, ordered newest first with deterministic tie-breaker."""
    query = """
        SELECT id, tenant_id, external_id, subject, body, customer_email,
               status, category, priority, created_at, first_response_at,
               resolved_at, assignee_id
        FROM tickets
        WHERE tenant_id = %s
    """
    params: list[Any] = [tenant_id]

    if status_filter:
        query += " AND status = %s"
        params.append(status_filter)

    query += " ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s;"
    params.extend([limit, offset])

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, tuple(params))
        rows = cur.fetchall()
        return [_row_to_ticket(r) for r in rows]


def get_ticket_by_id(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    ticket_id: uuid.UUID,
) -> TicketRecord | None:
    """Retrieve single ticket by id strictly scoped to tenant_id."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, tenant_id, external_id, subject, body, customer_email,
                   status, category, priority, created_at, first_response_at,
                   resolved_at, assignee_id
            FROM tickets
            WHERE tenant_id = %s AND id = %s;
            """,
            (tenant_id, ticket_id),
        )
        row = cur.fetchone()
        if not row:
            return None
        return _row_to_ticket(row)


def count_tickets(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
) -> int:
    """Return total count of tickets for tenant_id."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM tickets WHERE tenant_id = %s;", (tenant_id,))
        row = cur.fetchone()
        return int(row[0]) if row else 0


def list_tickets_with_prediction(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    limit: int = 20,
    offset: int = 0,
) -> list[TicketListItem]:
    """List tickets with optional prediction and processing metadata, strictly scoped to tenant_id."""
    query = """
        SELECT t.id, t.tenant_id, t.external_id, t.subject, t.body, t.customer_email,
               t.status, t.category, t.priority, t.created_at,
               p.predicted_label, m.name AS model_name, m.version AS model_version,
               j.status AS job_status, j.attempts AS job_attempts,
               j.max_attempts AS job_max_attempts, j.run_after AS job_run_after
        FROM tickets t
        LEFT JOIN ticket_predictions p
          ON t.id = p.ticket_id AND t.tenant_id = p.tenant_id AND p.task = 'category'
        LEFT JOIN model_versions m
          ON p.model_version_id = m.id AND p.tenant_id = m.tenant_id
        LEFT JOIN jobs j
          ON t.id = j.ticket_id AND t.tenant_id = j.tenant_id AND j.type = 'ticket_ingest'
        WHERE t.tenant_id = %s
        ORDER BY t.created_at DESC, t.id DESC
        LIMIT %s OFFSET %s;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (tenant_id, limit, offset))
        rows = cur.fetchall()
        return [
            TicketListItem(
                id=uuid.UUID(str(r["id"])),
                tenant_id=uuid.UUID(str(r["tenant_id"])),
                external_id=r["external_id"],
                subject=r["subject"],
                body=r["body"],
                customer_email=r["customer_email"],
                status=r["status"],
                category=r["category"],
                priority=r["priority"],
                created_at=r["created_at"],
                predicted_label=r["predicted_label"],
                model_name=r["model_name"],
                model_version=r["model_version"],
                job_status=r["job_status"],
                job_attempts=r["job_attempts"],
                job_max_attempts=r["job_max_attempts"],
                job_run_after=r["job_run_after"],
            )
            for r in rows
        ]


def record_first_response(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    ticket_id: uuid.UUID,
    responded_at: datetime,
) -> TicketRecord | None:
    """Record first_response_at on a ticket, strictly scoped to tenant_id.

    Guarantees:
    - Independent of ticket status (does not alter status).
    - Preserves existing first_response_at on repeated calls (idempotent).
    - Rejects naive timestamps without timezone awareness.
    - Caller owns the transaction; does not call commit().
    - Returns updated TicketRecord or None if ticket not found.
    """
    if responded_at.tzinfo is None or responded_at.tzinfo.utcoffset(responded_at) is None:
        raise ValueError(f"responded_at must be timezone-aware (received naive datetime: {responded_at})")

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, tenant_id, created_at, first_response_at
            FROM tickets
            WHERE tenant_id = %s AND id = %s
            FOR UPDATE;
            """,
            (tenant_id, ticket_id),
        )
        current = cur.fetchone()
        if not current:
            return None

        created_at = current.get("created_at")
        if created_at is not None and responded_at < created_at:
            raise ValueError(
                f"responded_at ({responded_at}) cannot precede ticket created_at ({created_at})."
            )

        cur.execute(
            """
            UPDATE tickets
            SET first_response_at = COALESCE(first_response_at, %s)
            WHERE tenant_id = %s AND id = %s
            RETURNING id, tenant_id, external_id, subject, body, customer_email,
                      status, category, priority, created_at,
                      first_response_at, resolved_at, assignee_id;
            """,
            (responded_at, tenant_id, ticket_id),
        )
        row = cur.fetchone()
        return _row_to_ticket(row) if row else None


def resolve_ticket(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    ticket_id: uuid.UUID,
    resolved_at: datetime,
) -> TicketRecord | None:
    """Transition ticket to 'resolved' and record resolved_at timestamp.

    Guarantees:
    - Strictly scoped to tenant_id.
    - Preserves existing resolved_at timestamp on repeated calls if already resolved.
    - Raises InvalidStatusTransitionError if ticket is already 'closed' (no reopening).
    - Rejects naive timestamps without timezone awareness.
    - Rejects resolved_at preceding ticket created_at.
    - Does not alter or infer first_response_at.
    - Caller owns the transaction; does not call commit().
    - Returns updated TicketRecord or None if ticket not found.
    """
    if resolved_at.tzinfo is None or resolved_at.tzinfo.utcoffset(resolved_at) is None:
        raise ValueError(f"resolved_at must be timezone-aware (received naive datetime: {resolved_at})")

    with conn.cursor(row_factory=dict_row) as cur:
        # Atomic lock on ticket row to verify current state
        cur.execute(
            """
            SELECT id, tenant_id, external_id, subject, body, customer_email,
                   status, category, priority, created_at,
                   first_response_at, resolved_at, assignee_id
            FROM tickets
            WHERE tenant_id = %s AND id = %s
            FOR UPDATE;
            """,
            (tenant_id, ticket_id),
        )
        current = cur.fetchone()
        if not current:
            return None

        current_status = current["status"]
        if current_status == "closed":
            raise InvalidStatusTransitionError(
                f"Cannot resolve ticket {ticket_id}: ticket is already closed."
            )

        created_at = current.get("created_at")
        if created_at is not None and resolved_at < created_at:
            raise ValueError(
                f"resolved_at ({resolved_at}) cannot precede ticket created_at ({created_at})."
            )

        cur.execute(
            """
            UPDATE tickets
            SET status = 'resolved',
                resolved_at = COALESCE(resolved_at, %s)
            WHERE tenant_id = %s AND id = %s
            RETURNING id, tenant_id, external_id, subject, body, customer_email,
                      status, category, priority, created_at,
                      first_response_at, resolved_at, assignee_id;
            """,
            (resolved_at, tenant_id, ticket_id),
        )
        updated = cur.fetchone()
        return _row_to_ticket(updated) if updated else None

def close_ticket(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    ticket_id: uuid.UUID,
) -> TicketRecord | None:
    """Transition ticket to 'closed', requiring prior resolution.

    Guarantees:
    - Strictly scoped to tenant_id.
    - Closing requires prior resolution ('resolved' status).
    - Raises InvalidStatusTransitionError if ticket is in 'new', 'open', or 'pending'.
    - If already 'closed', operation is idempotent and returns the current record.
    - Does not alter or infer first_response_at or resolved_at.
    - Caller owns the transaction; does not call commit().
    - Returns updated TicketRecord or None if ticket not found.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, tenant_id, external_id, subject, body, customer_email,
                   status, category, priority, created_at,
                   first_response_at, resolved_at, assignee_id
            FROM tickets
            WHERE tenant_id = %s AND id = %s
            FOR UPDATE;
            """,
            (tenant_id, ticket_id),
        )
        current = cur.fetchone()
        if not current:
            return None

        current_status = current["status"]
        if current_status == "closed":
            # Idempotent: already closed
            return _row_to_ticket(current)

        if current_status != "resolved":
            raise InvalidStatusTransitionError(
                f"Cannot close ticket {ticket_id}: ticket must be resolved first (current status: '{current_status}')."
            )

        cur.execute(
            """
            UPDATE tickets
            SET status = 'closed'
            WHERE tenant_id = %s AND id = %s
            RETURNING id, tenant_id, external_id, subject, body, customer_email,
                      status, category, priority, created_at,
                      first_response_at, resolved_at, assignee_id;
            """,
            (tenant_id, ticket_id),
        )
        updated = cur.fetchone()
        return _row_to_ticket(updated) if updated else None

