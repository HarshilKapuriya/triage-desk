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
import app.repositories.duplicates as duplicates_repo
import app.repositories.ml as ml_repo



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


def enqueue_embedding_job_if_not_exists(
    conn: psycopg.Connection,
    tenant_id: uuid.UUID,
    ticket_id: uuid.UUID,
    payload: dict[str, Any] | None = None,
) -> JobRecord | None:
    """Atomically enqueue a ticket_embedding job if one does not already exist for this ticket.

    Guarantees idempotency via ON CONFLICT (tenant_id, ticket_id, type) DO NOTHING.
    Operates strictly within the caller's transaction.
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
        VALUES (%s, %s, 'ticket_embedding', 'queued', %s::jsonb)
        ON CONFLICT (tenant_id, ticket_id, type) DO NOTHING
        RETURNING
            id, tenant_id, ticket_id, type, status,
            attempts, max_attempts, run_after, locked_at, locked_by,
            created_at, updated_at, completed_at, failed_at, error, payload;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (tenant_id, ticket_id, payload_json))
        row = cur.fetchone()
        if not row:
            return None
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


# --- Platform-level Worker Operations (Approved Narrow Exception) ---


class StaleClaimError(Exception):
    """Raised when a worker attempts an operation on a job whose lease expired or generation changed."""


def claim_next_job(
    conn: psycopg.Connection,
    worker_id: str,
    lease_seconds: int = 300,
    tenant_id: uuid.UUID | None = None,
    allowed_types: tuple[str, ...] | list[str] = ("ticket_ingest", "ticket_embedding"),
) -> JobRecord | None:
    """Claim the next eligible queued job atomically.

    Approved platform-level exception: operates without tenant_id to service the shared queue.
    An optional tenant_id can be passed by isolated test suites to prevent draining unrelated fixtures.
    Uses SELECT ... FOR UPDATE SKIP LOCKED and increments attempts generation on claim.
    Deterministic ordering: run_after ASC, created_at ASC, id ASC.
    """
    types_list = list(allowed_types)
    query = """
        WITH next_job AS (
            SELECT id
            FROM jobs
            WHERE status = 'queued'
              AND run_after <= NOW()
              AND attempts < max_attempts
              AND (%s::uuid IS NULL OR tenant_id = %s)
              AND type = ANY(%s)
            ORDER BY run_after ASC, created_at ASC, id ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        UPDATE jobs j
        SET status = 'running',
            locked_at = NOW(),
            locked_by = %s,
            attempts = attempts + 1,
            updated_at = NOW()
        FROM next_job
        WHERE j.id = next_job.id
        RETURNING
            j.id, j.tenant_id, j.ticket_id, j.type, j.status,
            j.attempts, j.max_attempts, j.run_after, j.locked_at, j.locked_by,
            j.created_at, j.updated_at, j.completed_at, j.failed_at, j.error, j.payload;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (tenant_id, tenant_id, types_list, worker_id))
        row = cur.fetchone()
        if not row:
            return None
        return _row_to_job_record(row)


def renew_lease(
    conn: psycopg.Connection,
    job_id: uuid.UUID,
    worker_id: str,
    generation: int,
    lease_seconds: int = 300,
) -> bool:
    """Renew the lease on an active job.

    Strictly validates that:
    1. The job is currently in 'running' status.
    2. The caller matches locked_by.
    3. The claim generation matches attempts.
    4. The current lease has NOT already expired (NOW <= locked_at + lease_seconds).
    Returns True if successfully renewed; False otherwise.
    """
    query = """
        UPDATE jobs
        SET locked_at = NOW(),
            updated_at = NOW()
        WHERE id = %s
          AND status = 'running'
          AND locked_by = %s
          AND attempts = %s
          AND locked_at >= NOW() - (%s || ' seconds')::interval;
    """
    with conn.cursor() as cur:
        cur.execute(query, (job_id, worker_id, generation, lease_seconds))
        return cur.rowcount > 0


def complete_job_with_ticket_transition(
    conn: psycopg.Connection,
    tenant_id: uuid.UUID,
    job_id: uuid.UUID,
    ticket_id: uuid.UUID,
    worker_id: str,
    generation: int,
    lease_seconds: int = 300,
    model_version_id: uuid.UUID | None = None,
    task: str | None = None,
    predicted_label: str | None = None,
    confidence: float | None = None,
    decision_scores: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Atomically validate claim, apply ticket state transition and audit log, and complete job.

    All mutations occur inside the caller's transaction. If claim ownership or lease validity
    is lost, raises StaleClaimError ensuring zero side-effects commit.
    """
    # 1. Validate and complete the job claim atomically
    complete_query = """
        UPDATE jobs
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
    """
    with conn.cursor() as cur:
        cur.execute(complete_query, (job_id, tenant_id, worker_id, generation, lease_seconds))
        if cur.rowcount == 0:
            raise StaleClaimError("Cannot complete job: active lease expired or generation mismatch.")

    # 2. Persist ML prediction atomically if provided (idempotent under retries)
    if model_version_id is not None and task is not None and predicted_label is not None:
        ml_repo.save_ticket_prediction(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id=ticket_id,
            model_version_id=model_version_id,
            task=task,
            predicted_label=predicted_label,
            confidence=confidence,
            decision_scores=decision_scores,
            metadata=metadata,
        )

    # 3. Lock and inspect ticket status
    actor = f"worker:{worker_id}"
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT status FROM tickets WHERE tenant_id = %s AND id = %s FOR UPDATE;",
            (tenant_id, ticket_id),
        )
        t_row = cur.fetchone()
        if not t_row:
            raise RuntimeError(f"Referenced ticket {ticket_id} not found for tenant {tenant_id}.")

        current_status = t_row["status"]

        if current_status == "new":
            cur.execute(
                "UPDATE tickets SET status = 'open' WHERE tenant_id = %s AND id = %s AND status = 'new';",
                (tenant_id, ticket_id),
            )
            cur.execute(
                "INSERT INTO audit_log (tenant_id, actor, action) VALUES (%s, %s, 'ticket.processed');",
                (tenant_id, actor),
            )
        else:
            # Leave ticket status unchanged, record skipped audit log
            cur.execute(
                "INSERT INTO audit_log (tenant_id, actor, action) VALUES (%s, %s, 'ticket.ingest_skipped_already_progressed');",
                (tenant_id, actor),
            )

    return True


def complete_embedding_job(
    conn: psycopg.Connection,
    tenant_id: uuid.UUID,
    job_id: uuid.UUID,
    ticket_id: uuid.UUID,
    worker_id: str,
    generation: int,
    embedding: list[float],
    model_name: str,
    model_version: str,
    lease_seconds: int = 300,
) -> bool:
    """Atomically validate claim, persist ticket embedding, and complete job.

    All mutations occur inside the caller's transaction without internal commits.
    If claim ownership or lease validity is lost, raises StaleClaimError ensuring zero side-effects commit.
    """
    # 1. Validate and complete the job claim atomically
    complete_query = """
        UPDATE jobs
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
    """
    with conn.cursor() as cur:
        cur.execute(complete_query, (job_id, tenant_id, worker_id, generation, lease_seconds))
        if cur.rowcount == 0:
            raise StaleClaimError("Cannot complete embedding job: active lease expired or generation mismatch.")

    # 2. Persist embedding atomically via duplicates repository (idempotent under retries)
    duplicates_repo.save_ticket_embedding(
        conn=conn,
        tenant_id=tenant_id,
        ticket_id=ticket_id,
        embedding=embedding,
        model_name=model_name,
        model_version=model_version,
    )
    return True


def release_claimed_job_to_queued(
    conn: psycopg.Connection,
    tenant_id: uuid.UUID,
    job_id: uuid.UUID,
    worker_id: str,
    generation: int,
) -> bool:
    """Release a claimed job back to queued status without incrementing attempts."""
    query = """
        UPDATE jobs
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


def fail_or_retry_job(
    conn: psycopg.Connection,
    tenant_id: uuid.UUID,
    job_id: uuid.UUID,
    worker_id: str,
    generation: int,
    error_message: str,
) -> str:
    """Record job failure or schedule exponential backoff retry.

    Returns:
        str: 'requeued', 'failed', or 'stale_claim'.
    """
    # Sanitize and truncate error
    sanitized_error = error_message.strip()[:255] if error_message else "unknown_error"

    query_check = """
        SELECT attempts, max_attempts
        FROM jobs
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

        if attempts < max_attempts:
            # Exponential backoff: 2s, 4s, 8s...
            delay_seconds = 2 * (2 ** max(0, attempts - 1))
            cur.execute(
                """
                UPDATE jobs
                SET status = 'queued',
                    run_after = NOW() + (%s || ' seconds')::interval,
                    locked_at = NULL,
                    locked_by = NULL,
                    error = %s,
                    updated_at = NOW()
                WHERE id = %s AND tenant_id = %s;
                """,
                (delay_seconds, sanitized_error, job_id, tenant_id),
            )
            return "requeued"
        else:
            cur.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    failed_at = NOW(),
                    locked_at = NULL,
                    locked_by = NULL,
                    error = %s,
                    updated_at = NOW()
                WHERE id = %s AND tenant_id = %s;
                """,
                (sanitized_error, job_id, tenant_id),
            )
            return "failed"


def recover_stale_jobs(
    conn: psycopg.Connection,
    lease_seconds: int = 300,
    tenant_id: uuid.UUID | None = None,
) -> int:
    """Scan and recover orphaned jobs whose lease expired.

    Approved platform-level exception: operates without tenant_id.
    Optional tenant_id limits recovery to a specific tenant in tests.
    Requeues jobs if attempts < max_attempts with exponential backoff;
    otherwise marks them failed.
    """
    query = """
        WITH stale_jobs AS (
            SELECT id
            FROM jobs
            WHERE status = 'running'
              AND locked_at < NOW() - (%s || ' seconds')::interval
              AND (%s::uuid IS NULL OR tenant_id = %s)
            ORDER BY locked_at ASC, id ASC
            FOR UPDATE SKIP LOCKED
        )
        UPDATE jobs j
        SET status = CASE WHEN j.attempts >= j.max_attempts THEN 'failed' ELSE 'queued' END,
            failed_at = CASE WHEN j.attempts >= j.max_attempts THEN NOW() ELSE NULL END,
            run_after = CASE
                WHEN j.attempts >= j.max_attempts THEN j.run_after
                ELSE NOW() + (INTERVAL '2 seconds' * (2 ^ GREATEST(0, j.attempts - 1)))
            END,
            locked_at = NULL,
            locked_by = NULL,
            error = CASE
                WHEN j.attempts >= j.max_attempts THEN 'lease_timeout_exhausted'
                ELSE 'lease_timeout_requeued'
            END,
            updated_at = NOW()
        FROM stale_jobs
        WHERE j.id = stale_jobs.id;
    """
    with conn.cursor() as cur:
        cur.execute(query, (lease_seconds, tenant_id, tenant_id))
        return cur.rowcount
