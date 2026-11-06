"""Worker job handlers: execute domain processing for claimed jobs.

Per SPEC.md & architecture decisions:
- Preserves subject and body exactly (no normalization, no overwriting).
- Database-only handler executes in short transactions without holding DB locks during processing.
- Handles unsupported job types safely without crashing the worker loop.
- Phase 3 ML Demo: Runs category classification on ticket body using trusted CFPB-trained model.
- Persists category prediction and model version atomically with ticket transition.
"""

import logging
import uuid
import psycopg

import app.repositories.jobs as job_repo
import app.repositories.tickets as ticket_repo
import app.repositories.ml as ml_repo
from app.ml.loader import (
    get_model_manager,
    TRUSTED_TASK,
    DEFAULT_ARTIFACT_REL_PATH,
    EXPECTED_CLASSES,
)

logger = logging.getLogger(__name__)


def process_claimed_job(
    conn: psycopg.Connection,
    job: job_repo.JobRecord,
    worker_id: str,
    lease_seconds: int = 300,
    enable_inference: bool = True,
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
        model_version_id = None
        predicted_label = None
        decision_scores = None
        task = None
        meta = None

        if enable_inference:
            # 1. Fetch ticket strictly by tenant_id to retrieve ticket body
            ticket = ticket_repo.get_ticket_by_id(conn, tenant_id=job.tenant_id, ticket_id=job.ticket_id)
            if not ticket:
                raise RuntimeError(f"Referenced ticket {job.ticket_id} not found for tenant {job.tenant_id}.")

            # 2. Run classification using ticket body as input
            # If missing/corrupt artifact, this raises an explicit operational error caught below
            model_mgr = get_model_manager()
            inference_res = model_mgr.predict_ticket(ticket.body)

            # 3. Ensure active model version record exists for this specific tenant
            model_rec = ml_repo.ensure_tenant_model_version(
                conn=conn,
                tenant_id=job.tenant_id,
                name=inference_res.model_name,
                version=inference_res.model_version,
                task=TRUSTED_TASK,
                artifact_path=DEFAULT_ARTIFACT_REL_PATH,
                sha256_hash=inference_res.sha256_hash,
                classes=EXPECTED_CLASSES,
                metadata={"domain": "cfpb_financial_complaints"},
            )

            model_version_id = model_rec.id
            predicted_label = inference_res.predicted_label
            decision_scores = inference_res.decision_scores
            task = TRUSTED_TASK
            meta = {
                "inference_engine": "scikit-learn",
                "model_name": inference_res.model_name,
                "model_version": inference_res.model_version,
                "sha256": inference_res.sha256_hash,
                "uncalibrated_notice": "LinearSVC decision scores are margin distances, not probabilities.",
            }

        with conn.transaction():
            job_repo.complete_job_with_ticket_transition(
                conn=conn,
                tenant_id=job.tenant_id,
                job_id=job.id,
                ticket_id=job.ticket_id,
                worker_id=worker_id,
                generation=job.attempts,
                lease_seconds=lease_seconds,
                model_version_id=model_version_id,
                task=task,
                predicted_label=predicted_label,
                confidence=None,
                decision_scores=decision_scores,
                metadata=meta,
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
