"""Job repository: handles database operations for the jobs table.

Per SPEC.md Section 6:
- All tenant-owned repository functions take tenant_id as a required argument and filter by it.
- All SQL statements live strictly in the repository layer.
"""

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any
import uuid
import psycopg
from psycopg.rows import dict_row


@dataclass(frozen=True)
class JobRecord:
    """Represents a row from the jobs table."""
    id: uuid.UUID
    tenant_id: uuid.UUID
    ticket_id: uuid.UUID
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


def _row_to_job_record(row: dict[str, Any]) -> JobRecord:
    """Map a dictionary row from jobs to JobRecord."""
    payload_val = row["payload"]
    if isinstance(payload_val, str):
        try:
            payload_dict = json.loads(payload_val)
        except Exception:
            payload_dict = {}
    elif isinstance(payload_val, dict):
        payload_dict = payload_val
    else:
        payload_dict = {}

    return JobRecord(
        id=row["id"],
        tenant_id=row["tenant_id"],
        ticket_id=row["ticket_id"],
        type=row["type"],
        status=row["status"],
        attempts=row["attempts"],
        max_attempts=row["max_attempts"],
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


def enqueue_job(
    conn: psycopg.Connection,
    tenant_id: uuid.UUID,
    ticket_id: uuid.UUID,
    job_type: str = "ticket_ingest",
    payload: dict[str, Any] | None = None,
) -> JobRecord:
    """Enqueue a job for a tenant's ticket within the current transaction.

    Enforces composite tenant-ticket consistency and unique active job constraint.
    """
    payload_json = json.dumps(payload or {})
    query = """
        INSERT INTO jobs (
            tenant_id,
            ticket_id,
            type,
            status,
            payload
        )
        VALUES (%s, %s, %s, 'queued', %s::jsonb)
        RETURNING
            id, tenant_id, ticket_id, type, status,
            attempts, max_attempts, run_after, locked_at, locked_by,
            created_at, updated_at, completed_at, failed_at, error, payload;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (tenant_id, ticket_id, job_type, payload_json))
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Failed to enqueue job; no row returned.")
        return _row_to_job_record(row)


def get_job_by_ticket_id(
    conn: psycopg.Connection,
    tenant_id: uuid.UUID,
    ticket_id: uuid.UUID,
    job_type: str = "ticket_ingest",
) -> JobRecord | None:
    """Retrieve the job for a given ticket strictly scoped by tenant_id."""
    query = """
        SELECT
            id, tenant_id, ticket_id, type, status,
            attempts, max_attempts, run_after, locked_at, locked_by,
            created_at, updated_at, completed_at, failed_at, error, payload
        FROM jobs
        WHERE tenant_id = %s
          AND ticket_id = %s
          AND type = %s;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (tenant_id, ticket_id, job_type))
        row = cur.fetchone()
        if not row:
            return None
        return _row_to_job_record(row)


def get_job_by_id(
    conn: psycopg.Connection,
    tenant_id: uuid.UUID,
    job_id: uuid.UUID,
) -> JobRecord | None:
    """Retrieve a job by its primary key strictly scoped by tenant_id."""
    query = """
        SELECT
            id, tenant_id, ticket_id, type, status,
            attempts, max_attempts, run_after, locked_at, locked_by,
            created_at, updated_at, completed_at, failed_at, error, payload
        FROM jobs
        WHERE tenant_id = %s
          AND id = %s;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (tenant_id, job_id))
        row = cur.fetchone()
        if not row:
            return None
        return _row_to_job_record(row)

