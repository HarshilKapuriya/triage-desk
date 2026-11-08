"""Worker job handlers: execute domain processing for claimed jobs.

Per SPEC.md & architecture decisions:
- Preserves subject and body exactly (no normalization, no overwriting).
- Database-only handler executes in short transactions without holding DB locks during processing.
- Handles unsupported job types safely without crashing the worker loop.
- Phase 3 ML Demo: Runs category classification on ticket body using trusted CFPB-trained model.
- Persists category prediction and model version atomically with ticket transition.
"""

import logging
from typing import Any
import uuid
import psycopg

import app.repositories.jobs as job_repo
import app.repositories.tickets as ticket_repo
import app.repositories.ml as ml_repo
import app.repositories.kb as kb_repo
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
    enable_embeddings: bool = False,
    embedding_adapter: Any | None = None,
) -> str:
    """Process a single claimed job based on its type.

    Returns:
        str: 'completed', 'requeued', 'failed', 'stale_claim', or 'unsupported_job_type'.
    """
    if job.type == "ticket_ingest":
        return _process_ticket_ingest_job(
            conn=conn,
            job=job,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            enable_inference=enable_inference,
            enable_embeddings=enable_embeddings,
        )
    elif job.type == "ticket_embedding":
        return _process_ticket_embedding_job(
            conn=conn,
            job=job,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            enable_embeddings=enable_embeddings,
            embedding_adapter=embedding_adapter,
        )
    else:
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


def _process_ticket_ingest_job(
    conn: psycopg.Connection,
    job: job_repo.JobRecord,
    worker_id: str,
    lease_seconds: int = 300,
    enable_inference: bool = True,
    enable_embeddings: bool = False,
) -> str:
    """Process ticket_ingest: classify complaint category, transition status, and conditionally enqueue embedding."""
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
            # Atomically enqueue embedding job only when feature is enabled
            if enable_embeddings:
                job_repo.enqueue_embedding_job_if_not_exists(
                    conn=conn,
                    tenant_id=job.tenant_id,
                    ticket_id=job.ticket_id,
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


def _process_ticket_embedding_job(
    conn: psycopg.Connection,
    job: job_repo.JobRecord,
    worker_id: str,
    lease_seconds: int = 300,
    enable_embeddings: bool = False,
    embedding_adapter: Any | None = None,
) -> str:
    """Process ticket_embedding: generate 384-dim dense vector and persist into ticket_embeddings."""
    if not enable_embeddings:
        # Do not mark existing embedding jobs completed without producing an embedding
        # merely because the feature is disabled. Leave them pending and release claim.
        logger.info(
            f"Worker {worker_id}: embedding feature disabled; releasing job {job.id} back to queued."
        )
        with conn.transaction():
            job_repo.release_claimed_job_to_queued(
                conn=conn,
                tenant_id=job.tenant_id,
                job_id=job.id,
                worker_id=worker_id,
                generation=job.attempts,
            )
        return "requeued"

    try:
        # 1. Fetch ticket strictly by tenant_id to retrieve ticket subject and body
        ticket = ticket_repo.get_ticket_by_id(conn, tenant_id=job.tenant_id, ticket_id=job.ticket_id)
        if not ticket:
            raise RuntimeError(f"Referenced ticket {job.ticket_id} not found for tenant {job.tenant_id}.")

        # 2. Resolve embedding adapter (or load lazily)
        adapter = embedding_adapter
        if adapter is None:
            from app.ml.embeddings import EmbeddingModelAdapter
            adapter = EmbeddingModelAdapter()

        # 3. CPU tensor inference executed OUTSIDE database transactions
        embedding_res = adapter.embed_ticket(subject=ticket.subject, body=ticket.body)

        # 4. Atomic completion and embedding persistence inside a short transaction
        with conn.transaction():
            job_repo.complete_embedding_job(
                conn=conn,
                tenant_id=job.tenant_id,
                job_id=job.id,
                ticket_id=job.ticket_id,
                worker_id=worker_id,
                generation=job.attempts,
                embedding=embedding_res.vector,
                model_name=embedding_res.model_name,
                model_version=embedding_res.model_revision,
                lease_seconds=lease_seconds,
            )
        return "completed"

    except job_repo.StaleClaimError:
        logger.warning(
            f"Worker {worker_id}: lost claim or lease expired for embedding job {job.id} (gen {job.attempts})"
        )
        return "stale_claim"

    except Exception as exc:
        err_msg = f"{exc.__class__.__name__}: {str(exc)}"
        logger.error(f"Worker {worker_id}: embedding processing failed for job {job.id}: {err_msg}")
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
            logger.error(
                f"Worker {worker_id}: failed to record retry/failure for embedding job {job.id}: {fail_exc}"
            )
            return "error_recording_failure"


def process_claimed_article_job(
    conn: psycopg.Connection,
    job: kb_repo.ArticleJobRecord,
    worker_id: str,
    lease_seconds: int = 300,
    enable_embeddings: bool = False,
    embedding_adapter: Any | None = None,
) -> str:
    """Process a single claimed article job based on its type.

    Returns:
        str: 'completed', 'requeued', 'failed', 'stale_claim', or 'unsupported_job_type'.
    """
    if job.type == "article_embedding":
        return _process_article_embedding_job(
            conn=conn,
            job=job,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            enable_embeddings=enable_embeddings,
            embedding_adapter=embedding_adapter,
        )
    else:
        logger.warning(f"Worker {worker_id}: unsupported article job type '{job.type}' for job {job.id}")
        try:
            with conn.transaction():
                kb_repo.fail_or_retry_article_job(
                    conn=conn,
                    tenant_id=job.tenant_id,
                    job_id=job.id,
                    worker_id=worker_id,
                    generation=job.attempts,
                    error_message=f"unsupported_job_type: {job.type}",
                    article_id=job.article_id,
                )
        except Exception as exc:
            logger.error(f"Error recording failure for unsupported article job {job.id}: {exc}")
        return "unsupported_job_type"


def _process_article_embedding_job(
    conn: psycopg.Connection,
    job: kb_repo.ArticleJobRecord,
    worker_id: str,
    lease_seconds: int = 300,
    enable_embeddings: bool = False,
    embedding_adapter: Any | None = None,
) -> str:
    """Process article_embedding: embed all chunks and mark article ready."""
    if not enable_embeddings:
        logger.info(
            f"Worker {worker_id}: embedding feature disabled; releasing article job {job.id} back to queued."
        )
        with conn.transaction():
            kb_repo.release_claimed_article_job_to_queued(
                conn=conn,
                tenant_id=job.tenant_id,
                job_id=job.id,
                worker_id=worker_id,
                generation=job.attempts,
            )
        return "requeued"

    try:
        # 1. Fetch chunks for article strictly scoped to tenant_id
        chunks = kb_repo.get_kb_chunks_for_article(
            conn=conn,
            tenant_id=job.tenant_id,
            article_id=job.article_id,
        )
        if not chunks:
            raise RuntimeError(f"No chunks found for article {job.article_id} (tenant {job.tenant_id}).")

        # 2. Resolve embedding adapter (or load lazily)
        adapter = embedding_adapter
        if adapter is None:
            from app.ml.embeddings import EmbeddingModelAdapter
            adapter = EmbeddingModelAdapter()

        # 3. CPU tensor inference executed OUTSIDE database transactions without holding locks
        chunk_embeddings: list[tuple[uuid.UUID, list[float], str, str]] = []
        for chunk in chunks:
            res = adapter.embed_text(chunk.content)
            chunk_embeddings.append((chunk.id, res.vector, res.model_name, res.model_revision))

        # 4. Atomic fenced completion and embedding persistence inside a short transaction
        with conn.transaction():
            kb_repo.complete_article_job(
                conn=conn,
                tenant_id=job.tenant_id,
                job_id=job.id,
                article_id=job.article_id,
                worker_id=worker_id,
                generation=job.attempts,
                chunk_embeddings=chunk_embeddings,
                lease_seconds=lease_seconds,
            )
        return "completed"

    except kb_repo.StaleClaimError:
        logger.warning(
            f"Worker {worker_id}: lost claim or lease expired for article job {job.id} (gen {job.attempts})"
        )
        return "stale_claim"

    except Exception as exc:
        err_msg = f"{exc.__class__.__name__}: {str(exc)}"
        logger.error(f"Worker {worker_id}: article embedding processing failed for job {job.id}: {err_msg}")
        try:
            with conn.transaction():
                status = kb_repo.fail_or_retry_article_job(
                    conn=conn,
                    tenant_id=job.tenant_id,
                    job_id=job.id,
                    worker_id=worker_id,
                    generation=job.attempts,
                    error_message=err_msg,
                    article_id=job.article_id,
                )
            return status
        except Exception as fail_exc:
            logger.error(
                f"Worker {worker_id}: failed to record retry/failure for article job {job.id}: {fail_exc}"
            )
            return "error_recording_failure"
