"""SLA policy repository layer for multi-tenant SLA management.

Per SPEC.md Section 5, 6, and Phase 5 requirements:
- All database operations are strictly tenant-isolated using tenant_id.
- Caller owns the database connection and transaction; functions never call commit().
- Target durations are positive integer minutes.
- Priority values are strictly validated ('low', 'normal', 'high', 'urgent', 'default').
- Effective policy resolution prefers exact priority match, falling back to 'default'.
  If neither exists, returns None (never silently invents targets).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
import uuid

import psycopg
from psycopg.rows import dict_row

VALID_POLICY_PRIORITIES = {"low", "normal", "high", "urgent", "default"}
MAX_DURATION_MINUTES = 525600  # 365 days in minutes


@dataclass(frozen=True)
class SLAPolicyRecord:
    """Read-only view of a tenant SLA policy."""
    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    priority: str
    first_response_minutes: int
    resolution_minutes: int
    created_at: datetime
    updated_at: datetime


def _row_to_policy(row: dict[str, Any]) -> SLAPolicyRecord:
    """Convert a database row dict to an SLAPolicyRecord."""
    return SLAPolicyRecord(
        id=row["id"] if isinstance(row["id"], uuid.UUID) else uuid.UUID(str(row["id"])),
        tenant_id=row["tenant_id"] if isinstance(row["tenant_id"], uuid.UUID) else uuid.UUID(str(row["tenant_id"])),
        name=row["name"],
        priority=row["priority"],
        first_response_minutes=row["first_response_minutes"],
        resolution_minutes=row["resolution_minutes"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def upsert_sla_policy(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    name: str,
    priority: str,
    first_response_minutes: int,
    resolution_minutes: int,
) -> SLAPolicyRecord:
    """Upsert an SLA policy scoped strictly to tenant_id and priority.

    Caller owns the transaction.
    """
    clean_name = name.strip() if name else ""
    if not clean_name:
        raise ValueError("SLA policy name must not be empty.")

    clean_priority = priority.strip().lower() if priority else ""
    if clean_priority not in VALID_POLICY_PRIORITIES:
        raise ValueError(
            f"Invalid SLA policy priority '{priority}'. Must be one of {sorted(VALID_POLICY_PRIORITIES)}."
        )

    if not isinstance(first_response_minutes, int) or first_response_minutes <= 0:
        raise ValueError(f"first_response_minutes must be a positive integer (received {first_response_minutes}).")
    if first_response_minutes > MAX_DURATION_MINUTES:
        raise ValueError(f"first_response_minutes exceeds maximum allowed ({MAX_DURATION_MINUTES} minutes).")

    if not isinstance(resolution_minutes, int) or resolution_minutes <= 0:
        raise ValueError(f"resolution_minutes must be a positive integer (received {resolution_minutes}).")
    if resolution_minutes > MAX_DURATION_MINUTES:
        raise ValueError(f"resolution_minutes exceeds maximum allowed ({MAX_DURATION_MINUTES} minutes).")

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO sla_policies (
                tenant_id, name, priority, first_response_minutes, resolution_minutes
            ) VALUES (
                %s, %s, %s, %s, %s
            )
            ON CONFLICT (tenant_id, priority) DO UPDATE SET
                name = EXCLUDED.name,
                first_response_minutes = EXCLUDED.first_response_minutes,
                resolution_minutes = EXCLUDED.resolution_minutes,
                updated_at = NOW()
            RETURNING id, tenant_id, name, priority, first_response_minutes, resolution_minutes, created_at, updated_at;
            """,
            (tenant_id, clean_name, clean_priority, first_response_minutes, resolution_minutes),
        )
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Failed to upsert SLA policy: no row returned.")
        return _row_to_policy(row)


def get_sla_policy_by_priority(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    priority: str,
) -> SLAPolicyRecord | None:
    """Fetch an SLA policy by priority key scoped strictly to tenant_id."""
    clean_priority = priority.strip().lower() if priority else ""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, tenant_id, name, priority, first_response_minutes, resolution_minutes, created_at, updated_at
            FROM sla_policies
            WHERE tenant_id = %s AND priority = %s;
            """,
            (tenant_id, clean_priority),
        )
        row = cur.fetchone()
        return _row_to_policy(row) if row else None

def list_sla_policies(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
) -> list[SLAPolicyRecord]:
    """List all SLA policies scoped strictly to tenant_id."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, tenant_id, name, priority, first_response_minutes, resolution_minutes, created_at, updated_at
            FROM sla_policies
            WHERE tenant_id = %s
            ORDER BY priority ASC, name ASC;
            """,
            (tenant_id,),
        )
        rows = cur.fetchall()
        return [_row_to_policy(r) for r in rows]


def delete_sla_policy(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    priority: str,
) -> bool:
    """Delete an SLA policy by priority key scoped strictly to tenant_id.

    Returns True if a row was deleted, False otherwise. Caller owns the transaction.
    """
    clean_priority = priority.strip().lower() if priority else ""
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM sla_policies
            WHERE tenant_id = %s AND priority = %s
            RETURNING id;
            """,
            (tenant_id, clean_priority),
        )
        return cur.fetchone() is not None


def resolve_effective_policy(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    ticket_priority: str | None,
) -> SLAPolicyRecord | None:
    """Resolve the effective SLA policy for a ticket scoped strictly to tenant_id.

    Resolution Rules:
    1. If ticket_priority is provided (e.g. 'urgent', 'high', 'normal', 'low'):
       - Checks for an exact priority match first.
       - Falls back to the tenant's 'default' policy if exact priority is not configured.
    2. If ticket_priority is None:
       - Checks only for the tenant's 'default' policy.
    3. If neither exact match nor default policy exists:
       - Returns None (not_configured). Never silently invents targets.
    """
    clean_prio = ticket_priority.strip().lower() if ticket_priority else None

    with conn.cursor(row_factory=dict_row) as cur:
        if clean_prio is not None:
            cur.execute(
                """
                SELECT id, tenant_id, name, priority, first_response_minutes, resolution_minutes, created_at, updated_at
                FROM sla_policies
                WHERE tenant_id = %s
                  AND (priority = %s OR priority = 'default')
                ORDER BY (CASE WHEN priority = %s THEN 0 ELSE 1 END) ASC
                LIMIT 1;
                """,
                (tenant_id, clean_prio, clean_prio),
            )
        else:
            cur.execute(
                """
                SELECT id, tenant_id, name, priority, first_response_minutes, resolution_minutes, created_at, updated_at
                FROM sla_policies
                WHERE tenant_id = %s
                  AND priority = 'default'
                LIMIT 1;
                """,
                (tenant_id,),
            )
        row = cur.fetchone()
        return _row_to_policy(row) if row else None
