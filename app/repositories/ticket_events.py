"""Ticket lifecycle events repository for multi-tenant operational logging and SLA telemetry.

Per Phase 5 specifications & architectural decisions:
- All operations are strictly tenant-isolated using tenant_id.
- Callers own the transaction; this repository never calls commit().
- Database-level partial UNIQUE constraints enforce one-time guarantees for lifecycle transitions.
- Timestamps: occurred_at (business time) vs recorded_at (transaction start time via NOW()).
- Payloads: versioned, minimized raw facts with zero customer PII (no message body or customer emails).
- Append-only repository contract: only append and read methods are provided.
"""

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any
import uuid

import psycopg
from psycopg.rows import dict_row

ALLOWED_EVENT_TYPES = {
    "ticket_created",
    "ticket_processed",
    "first_response_recorded",
    "ticket_resolved",
    "ticket_closed",
}

ALLOWED_ACTOR_TYPES = {
    "system",
    "agent",
    "worker",
    "customer",
}


@dataclass(frozen=True)
class TicketEventRecord:
    """Read-only view of an operational ticket lifecycle event."""
    id: uuid.UUID
    tenant_id: uuid.UUID
    ticket_id: uuid.UUID
    event_type: str
    occurred_at: datetime
    recorded_at: datetime
    actor_type: str
    actor_id: str | None
    schema_version: int
    payload: dict[str, Any]


def _validate_timezone_aware(dt: datetime, name: str) -> None:
    """Ensure datetime has an explicit timezone (reject naive timestamps)."""
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(f"Timestamp '{name}' must be timezone-aware (received naive datetime: {dt})")


def _row_to_event(row: dict[str, Any]) -> TicketEventRecord:
    """Convert a database row dict to TicketEventRecord."""
    raw_payload = row["payload"]
    if isinstance(raw_payload, str):
        payload = json.loads(raw_payload)
    elif isinstance(raw_payload, dict):
        payload = raw_payload
    else:
        payload = {}

    return TicketEventRecord(
        id=row["id"] if isinstance(row["id"], uuid.UUID) else uuid.UUID(str(row["id"])),
        tenant_id=row["tenant_id"] if isinstance(row["tenant_id"], uuid.UUID) else uuid.UUID(str(row["tenant_id"])),
        ticket_id=row["ticket_id"] if isinstance(row["ticket_id"], uuid.UUID) else uuid.UUID(str(row["ticket_id"])),
        event_type=row["event_type"],
        occurred_at=row["occurred_at"],
        recorded_at=row["recorded_at"],
        actor_type=row["actor_type"],
        actor_id=row["actor_id"],
        schema_version=int(row["schema_version"]),
        payload=payload,
    )


def append_ticket_event(
    conn_or_cur: Any,
    tenant_id: uuid.UUID,
    ticket_id: uuid.UUID,
    event_type: str,
    occurred_at: datetime,
    actor_type: str,
    actor_id: str | None = None,
    payload: dict[str, Any] | None = None,
    schema_version: int = 1,
) -> TicketEventRecord:
    """Append a lifecycle event to ticket_events in the caller's transaction.

    Guarantees:
    - Strictly scoped to tenant_id.
    - Validates allowlisted event_type and actor_type.
    - Rejects naive occurred_at timestamps.
    - Requires schema_version >= 1.
    - Caller owns the transaction; never commits.
    - Database constraints handle uniqueness; integrity errors propagate to trigger atomic rollback.
    """
    if event_type not in ALLOWED_EVENT_TYPES:
        raise ValueError(
            f"Invalid event_type '{event_type}'. Must be one of {sorted(ALLOWED_EVENT_TYPES)}."
        )

    if actor_type not in ALLOWED_ACTOR_TYPES:
        raise ValueError(
            f"Invalid actor_type '{actor_type}'. Must be one of {sorted(ALLOWED_ACTOR_TYPES)}."
        )

    _validate_timezone_aware(occurred_at, "occurred_at")

    if schema_version < 1:
        raise ValueError(f"schema_version must be >= 1 (received {schema_version}).")

    clean_payload = payload if payload is not None else {}
    payload_json = json.dumps(clean_payload)

    event_id = uuid.uuid4()
    query = """
        INSERT INTO ticket_events (
            id, tenant_id, ticket_id, event_type, occurred_at, actor_type, actor_id, schema_version, payload
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
        );
    """
    params = (
        event_id,
        tenant_id,
        ticket_id,
        event_type,
        occurred_at,
        actor_type,
        actor_id,
        schema_version,
        payload_json,
    )

    if hasattr(conn_or_cur, "cursor"):
        # Connection object passed
        with conn_or_cur.cursor() as cur:
            cur.execute(query, params)
    else:
        # Cursor object passed
        conn_or_cur.execute(query, params)

    return TicketEventRecord(
        id=event_id,
        tenant_id=tenant_id,
        ticket_id=ticket_id,
        event_type=event_type,
        occurred_at=occurred_at,
        recorded_at=datetime.now(occurred_at.tzinfo),
        actor_type=actor_type,
        actor_id=actor_id,
        schema_version=schema_version,
        payload=clean_payload,
    )


def list_ticket_events(
    conn_or_cur: Any,
    tenant_id: uuid.UUID,
    ticket_id: uuid.UUID,
) -> list[TicketEventRecord]:
    """Retrieve chronological event history for a ticket strictly scoped to tenant_id.

    Deterministic tie-breaker: occurred_at ASC, recorded_at ASC, id ASC.
    """
    query = """
        SELECT id, tenant_id, ticket_id, event_type, occurred_at, recorded_at, actor_type, actor_id, schema_version, payload
        FROM ticket_events
        WHERE tenant_id = %s AND ticket_id = %s
        ORDER BY occurred_at ASC, recorded_at ASC, id ASC;
    """
    params = (tenant_id, ticket_id)

    if hasattr(conn_or_cur, "cursor"):
        with conn_or_cur.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            return [_row_to_event(r) for r in rows]
    else:
        cur = conn_or_cur
        cur.execute(query, params)
        rows = cur.fetchall() if hasattr(cur, "fetchall") else []
        return [_row_to_event(r) for r in rows]


# Payload Construction Helpers

def build_ticket_created_payload(
    policy_snapshot: dict[str, Any] | None = None,
    priority: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """Construct minimal schema_version 1 payload for ticket_created."""
    if policy_snapshot is None:
        p_snap = {
            "is_configured": False,
            "policy_id": None,
            "policy_priority": None,
            "first_response_minutes": None,
            "resolution_minutes": None,
        }
    else:
        p_snap = policy_snapshot

    return {
        "schema_version": 1,
        "policy_snapshot": p_snap,
        "priority": priority,
        "category": category,
    }


def build_ticket_processed_payload(
    from_status: str = "new",
    to_status: str = "open",
    predicted_category: str | None = None,
    model_version_id: str | None = None,
) -> dict[str, Any]:
    """Construct minimal schema_version 1 payload for ticket_processed."""
    return {
        "schema_version": 1,
        "from_status": from_status,
        "to_status": to_status,
        "predicted_category": predicted_category,
        "model_version_id": model_version_id,
    }


def build_first_response_payload(
    responded_at: datetime,
    elapsed_calendar_seconds: float | None = None,
) -> dict[str, Any]:
    """Construct minimal schema_version 1 payload for first_response_recorded."""
    _validate_timezone_aware(responded_at, "responded_at")
    return {
        "schema_version": 1,
        "responded_at": responded_at.isoformat(),
        "elapsed_calendar_seconds": elapsed_calendar_seconds,
    }


def build_ticket_resolved_payload(
    resolved_at: datetime,
    from_status: str,
    to_status: str = "resolved",
    elapsed_calendar_seconds: float | None = None,
) -> dict[str, Any]:
    """Construct minimal schema_version 1 payload for ticket_resolved."""
    _validate_timezone_aware(resolved_at, "resolved_at")
    return {
        "schema_version": 1,
        "from_status": from_status,
        "to_status": to_status,
        "resolved_at": resolved_at.isoformat(),
        "elapsed_calendar_seconds": elapsed_calendar_seconds,
    }


def build_ticket_closed_payload(
    closed_at: datetime,
    from_status: str = "resolved",
    to_status: str = "closed",
) -> dict[str, Any]:
    """Construct minimal schema_version 1 payload for ticket_closed."""
    _validate_timezone_aware(closed_at, "closed_at")
    return {
        "schema_version": 1,
        "from_status": from_status,
        "to_status": to_status,
        "closed_at": closed_at.isoformat(),
    }
