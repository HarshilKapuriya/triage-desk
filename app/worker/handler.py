"""Worker job handlers: execute domain processing for claimed jobs.

Per SPEC.md & architecture decisions:
- Preserves subject and body exactly (no normalization, no overwriting).
- Database-only handler executes in short transactions without holding DB locks during processing.
- Handles unsupported job types safely without crashing the worker loop.
"""

import logging
import uuid
import psycopg
import app.repositories.jobs as job_repo

logger = logging.getLogger(__name__)


def process_claimed_job(
    conn: psycopg.Connection,
    job: job_repo.JobRecord,
    worker_id: str,
    lease_seconds: int = 300,
) -> str:
    """Process a single claimed job.

    Returns:
        str: 'completed', 'requeued', 'failed', 'stale_claim', or 'unsupported_job_type'.
    """
    if job.type != "ticket_ingest":
        logger.warning(f"Worker {worker_id}: unsupported job type '{job.type}' for job {job.id}")
        try:
            with conn.transaction():
                job_repo.fail_or_retry_job(
                    conn=conn,
                    tenant_id=job.tenant_id,
                    job_id=job.id,
                    worker_id=worker_id,
                    generation=job.attempts,
                    error_message=f"unsupported_job_type: {job.type}",
                )
        except Exception as exc:
            logger.error(f"Error recording failure for unsupported job {job.id}: {exc}")
        return "unsupported_job_type"

    try:
        with conn.transaction():
            job_repo.complete_job_with_ticket_transition(
                conn=conn,
                tenant_id=job.tenant_id,
                job_id=job.id,
                ticket_id=job.ticket_id,
                worker_id=worker_id,
                generation=job.attempts,
                lease_seconds=lease_seconds,
            )
        return "completed"

    except job_repo.StaleClaimError:
        logger.warning(
            f"Worker {worker_id}: lost claim or lease expired for job {job.id} (gen {job.attempts})"
        )
        return "stale_claim"

    except Exception as exc:
        err_msg = f"{exc.__class__.__name__}: {str(exc)}"
        logger.error(f"Worker {worker_id}: processing failed for job {job.id}: {err_msg}")
        try:
            with conn.transaction():
                status = job_repo.fail_or_retry_job(
                    conn=conn,
                    tenant_id=job.tenant_id,
                    job_id=job.id,
                    worker_id=worker_id,
                    generation=job.attempts,
                    error_message=err_msg,
                )
            return status
        except Exception as fail_exc:
            logger.error(f"Worker {worker_id}: failed to record retry/failure for job {job.id}: {fail_exc}")
            return "error_recording_failure"

