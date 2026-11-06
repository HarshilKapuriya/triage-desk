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


@dataclass(frozen=True)
class TicketListItem:
    """Read-only view of a ticket list row with prediction metadata."""
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
    """List tickets with optional prediction metadata, strictly scoped to tenant_id."""
    query = """
        SELECT t.id, t.tenant_id, t.external_id, t.subject, t.body, t.customer_email,
               t.status, t.category, t.priority, t.created_at,
               p.predicted_label, m.name AS model_name, m.version AS model_version
        FROM tickets t
        LEFT JOIN ticket_predictions p
          ON t.id = p.ticket_id AND t.tenant_id = p.tenant_id AND p.task = 'category'
        LEFT JOIN model_versions m
          ON p.model_version_id = m.id AND p.tenant_id = m.tenant_id
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
            )
            for r in rows
        ]

