"""Ticket repository layer for multi-tenant ticket operations.

Per SPEC.md Section 5, 6, and 7:
- All functions require tenant_id and filter by it in every query.
- New ticket returns created ticket record.
- If external_id exists for the same tenant, returns the existing record without updating it.
- List tickets newest first, with deterministic tie-breaker (created_at DESC, id DESC).
- Detail lookup returns single ticket or None.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import uuid
import psycopg
from psycopg.rows import dict_row

import app.repositories.sla as sla_repo
import app.repositories.ticket_events as ticket_events_repo


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


def _emit_ticket_created_event(
    cur: psycopg.Cursor[Any],
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    ticket: TicketRecord,
    actor_type: str,
    actor_id: str | None,
) -> None:
    """Emit a ticket_created lifecycle event with effective SLA policy snapshot."""
    policy_snapshot = None
    try:
        policy = sla_repo.resolve_effective_policy(conn, tenant_id, ticket.priority)
        if policy is not None:
            policy_snapshot = {
                "is_configured": True,
                "policy_id": str(policy.id),
                "policy_priority": policy.priority,
                "first_response_minutes": policy.first_response_minutes,
                "resolution_minutes": policy.resolution_minutes,
            }
        else:
            policy_snapshot = {
                "is_configured": False,
                "policy_id": None,
                "policy_priority": None,
                "first_response_minutes": None,
                "resolution_minutes": None,
            }
    except Exception:
        # Handles unapplied SLA schema or connection lookup issues gracefully
        policy_snapshot = {
            "is_configured": False,
            "policy_id": None,
            "policy_priority": None,
            "first_response_minutes": None,
            "resolution_minutes": None,
        }

    payload = ticket_events_repo.build_ticket_created_payload(
        policy_snapshot=policy_snapshot,
        priority=ticket.priority,
        category=ticket.category,
    )
    ticket_events_repo.append_ticket_event(
        conn_or_cur=cur,
        tenant_id=tenant_id,
        ticket_id=ticket.id,
        event_type="ticket_created",
        occurred_at=ticket.created_at,
        actor_type=actor_type,
        actor_id=actor_id,
        payload=payload,
    )


def create_or_get_ticket(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    subject: str,
    body: str,
    external_id: str | None = None,
    customer_email: str | None = None,
    actor_type: str = "customer",
    actor_id: str | None = None,
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
                ticket = _row_to_ticket(created_row)
                _emit_ticket_created_event(cur, conn, tenant_id, ticket, actor_type, actor_id)
                return ticket, True

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
        created_row = cur.fetchone()
        ticket = _row_to_ticket(created_row)
        _emit_ticket_created_event(cur, conn, tenant_id, ticket, actor_type, actor_id)
        return ticket, True


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
    actor_type: str = "agent",
    actor_id: str | None = None,
) -> TicketRecord | None:
    """Record first_response_at on a ticket and emit first_response_recorded event.

    Guarantees:
    - Strictly scoped to tenant_id.
    - Preserves existing first_response_at on repeated calls (idempotent, no duplicate event).
    - Rejects naive timestamps without timezone awareness.
    - Caller owns the transaction; does not call commit().
    - Returns updated TicketRecord or None if ticket not found.
    """
    if responded_at.tzinfo is None or responded_at.tzinfo.utcoffset(responded_at) is None:
        raise ValueError(f"responded_at must be timezone-aware (received naive datetime: {responded_at})")

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, tenant_id, created_at, first_response_at, status
            FROM tickets
            WHERE tenant_id = %s AND id = %s
            FOR UPDATE;
            """,
            (tenant_id, ticket_id),
        )
        current = cur.fetchone()
        if not current:
            return None

        # Rejection of new first response event on resolved/closed tickets (while already-recorded response remains idempotent)
        if current.get("first_response_at") is None and current.get("status") in ("resolved", "closed"):
            raise InvalidStatusTransitionError(
                f"Cannot record first response on {current['status']} ticket {ticket_id}."
            )

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
        ticket = _row_to_ticket(row) if row else None
        if ticket and current.get("first_response_at") is None:
            created_at = current.get("created_at")
            elapsed = (responded_at - created_at).total_seconds() if created_at else None
            payload = ticket_events_repo.build_first_response_payload(
                responded_at=responded_at,
                elapsed_calendar_seconds=elapsed,
            )
            ticket_events_repo.append_ticket_event(
                conn_or_cur=cur,
                tenant_id=tenant_id,
                ticket_id=ticket_id,
                event_type="first_response_recorded",
                occurred_at=responded_at,
                actor_type=actor_type,
                actor_id=actor_id,
                payload=payload,
            )
        return ticket


def resolve_ticket(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    ticket_id: uuid.UUID,
    resolved_at: datetime,
    actor_type: str = "agent",
    actor_id: str | None = None,
) -> TicketRecord | None:
    """Transition ticket to 'resolved', record resolved_at timestamp, and emit ticket_resolved event.

    Guarantees:
    - Strictly scoped to tenant_id.
    - Preserves existing resolved_at timestamp on repeated calls if already resolved (idempotent).
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

        if current_status != "resolved":
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
            ticket = _row_to_ticket(updated) if updated else None
            if ticket:
                elapsed = (resolved_at - created_at).total_seconds() if created_at else None
                payload = ticket_events_repo.build_ticket_resolved_payload(
                    resolved_at=resolved_at,
                    from_status=current_status,
                    to_status="resolved",
                    elapsed_calendar_seconds=elapsed,
                )
                ticket_events_repo.append_ticket_event(
                    conn_or_cur=cur,
                    tenant_id=tenant_id,
                    ticket_id=ticket_id,
                    event_type="ticket_resolved",
                    occurred_at=resolved_at,
                    actor_type=actor_type,
                    actor_id=actor_id,
                    payload=payload,
                )
            return ticket
        else:
            return _row_to_ticket(current)


def close_ticket(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    ticket_id: uuid.UUID,
    actor_type: str = "agent",
    actor_id: str | None = None,
    closed_at: datetime | None = None,
) -> TicketRecord | None:
    """Transition ticket to 'closed', requiring prior resolution, and emit ticket_closed event.

    Guarantees:
    - Strictly scoped to tenant_id.
    - Closing requires prior resolution ('resolved' status).
    - Raises InvalidStatusTransitionError if ticket is in 'new', 'open', or 'pending'.
    - If already 'closed', operation is idempotent and returns the current record (no duplicate event).
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

        now = closed_at or datetime.now(timezone.utc)
        if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
            raise ValueError(f"closed_at must be timezone-aware (received naive datetime: {now})")

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
        ticket = _row_to_ticket(updated) if updated else None
        if ticket:
            payload = ticket_events_repo.build_ticket_closed_payload(
                closed_at=now,
                from_status=current_status,
                to_status="closed",
            )
            ticket_events_repo.append_ticket_event(
                conn_or_cur=cur,
                tenant_id=tenant_id,
                ticket_id=ticket_id,
                event_type="ticket_closed",
                occurred_at=now,
                actor_type=actor_type,
                actor_id=actor_id,
                payload=payload,
            )
        return ticket

