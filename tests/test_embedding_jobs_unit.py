"""Offline unit tests for independent ticket_embedding job integration.

Verifies:
1. Ingest job enqueues ticket_embedding atomically when ENABLE_EMBEDDINGS=true.
2. Ingest job skips embedding enqueue when ENABLE_EMBEDDINGS=false.
3. Embedding job releases claim back to 'queued' when ENABLE_EMBEDDINGS=false without failure.
4. Embedding job computes embedding outside DB transaction and calls complete_embedding_job.
5. Missing or corrupted embedding artifacts trigger fail_or_retry_job without affecting category predictions.
6. complete_embedding_job enforces lease and generation fencing (StaleClaimError on 0 rowcount).
7. release_claimed_job_to_queued safely clears worker lock.
8. renew_lease verifies lease validity and worker ownership.
9. enqueue_embedding_job_if_not_exists uses idempotent ON CONFLICT DO NOTHING.
10. claim_next_job respects allowed_types filtering.
11. WorkerService dispatches allowed_types based on enable_embeddings setting.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch
import uuid
import pytest

from app.ml.embeddings import EmbeddingResult
from app.repositories.jobs import (
    JobRecord,
    StaleClaimError,
    claim_next_job,
    complete_embedding_job,
    enqueue_embedding_job_if_not_exists,
    release_claimed_job_to_queued,
    renew_lease,
)
from app.repositories.tickets import TicketRecord
from app.worker.handler import (
    process_claimed_job,
    _process_ticket_ingest_job,
    _process_ticket_embedding_job,
)
from app.worker.main import WorkerService


@contextmanager
def dummy_transaction():
    yield


def make_dummy_job(
    job_type: str = "ticket_embedding",
    tenant_id: uuid.UUID | None = None,
    ticket_id: uuid.UUID | None = None,
    attempts: int = 1,
    max_attempts: int = 3,
) -> JobRecord:
    return JobRecord(
        id=uuid.uuid4(),
        tenant_id=tenant_id or uuid.uuid4(),
        ticket_id=ticket_id or uuid.uuid4(),
        type=job_type,
        status="running",
        attempts=attempts,
        max_attempts=max_attempts,
        run_after=None,
        locked_at=None,
        locked_by="worker_test",
        created_at=None,
        updated_at=None,
        completed_at=None,
        failed_at=None,
        error=None,
        payload={},
    )


def make_dummy_ticket(
    ticket_id: uuid.UUID,
    tenant_id: uuid.UUID,
    subject: str = "Test Subject",
    body: str = "Test Body",
) -> TicketRecord:
    return TicketRecord(
        id=ticket_id,
        tenant_id=tenant_id,
        external_id="ext-1",
        subject=subject,
        body=body,
        customer_email="user@example.com",
        status="open",
        category="Billing",
        priority="medium",
        created_at=datetime.now(timezone.utc),
        first_response_at=None,
        resolved_at=None,
        assignee_id=None,
    )


class FakeEmbeddingAdapter:
    """Offline fake adapter returning deterministic 384-d vectors."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.model_name = "all-MiniLM-L6-v2"
        self.model_revision = "rev_test123"
        self.embed_calls: list[tuple[str | None, str | None]] = []

    def load(self):
        if self.should_fail:
            raise RuntimeError("Fake artifact loading failed: missing file")

    def embed_ticket(self, subject: str | None, body: str | None) -> EmbeddingResult:
        if self.should_fail:
            raise RuntimeError("Inference session failure: corrupt ONNX graph")
        self.embed_calls.append((subject, body))
        return EmbeddingResult(
            vector=[0.05 * (i % 10) for i in range(384)],
            pipeline_version="all-MiniLM-L6-v2:rev_test123:mean_pooling:l2_norm",
            model_name=self.model_name,
            model_revision=self.model_revision,
            char_count=len((subject or "") + (body or "")),
            token_count=10,
        )


# -------------------------------------------------------------------------
# 1. Ingest Job: Embedding Enqueue Behavior
# -------------------------------------------------------------------------

def test_ingest_job_enqueues_embedding_when_enabled():
    """When enable_embeddings=True, category ingest atomically enqueues ticket_embedding."""
    mock_conn = MagicMock()
    mock_conn.transaction.return_value = dummy_transaction()

    job = make_dummy_job(job_type="ticket_ingest")
    ticket = make_dummy_ticket(ticket_id=job.ticket_id, tenant_id=job.tenant_id)

    with patch("app.repositories.tickets.get_ticket_by_id", return_value=ticket), \
         patch("app.worker.handler.get_model_manager") as mock_mm, \
         patch("app.repositories.ml.ensure_tenant_model_version", return_value=MagicMock(id=uuid.uuid4())), \
         patch("app.repositories.jobs.complete_job_with_ticket_transition", return_value=True), \
         patch("app.repositories.jobs.enqueue_embedding_job_if_not_exists") as mock_enqueue:

        mock_manager_instance = MagicMock()
        mock_manager_instance.predict_ticket.return_value = MagicMock(
            predicted_label="Billing",
            decision_scores={"Billing": 1.2},
            model_name="tfidf_linearsvc",
            model_version="v1",
            sha256_hash="abc123hash",
        )
        mock_mm.return_value = mock_manager_instance

        status = _process_ticket_ingest_job(
            conn=mock_conn,
            job=job,
            worker_id="worker_test",
            lease_seconds=300,
            enable_inference=True,
            enable_embeddings=True,
        )

        assert status == "completed"
        mock_enqueue.assert_called_once_with(
            conn=mock_conn,
            tenant_id=job.tenant_id,
            ticket_id=job.ticket_id,
        )


def test_ingest_job_does_not_enqueue_embedding_when_disabled():
    """When enable_embeddings=False, category ingest does not enqueue ticket_embedding."""
    mock_conn = MagicMock()
    mock_conn.transaction.return_value = dummy_transaction()

    job = make_dummy_job(job_type="ticket_ingest")
    ticket = make_dummy_ticket(ticket_id=job.ticket_id, tenant_id=job.tenant_id)

    with patch("app.repositories.tickets.get_ticket_by_id", return_value=ticket), \
         patch("app.worker.handler.get_model_manager") as mock_mm, \
         patch("app.repositories.ml.ensure_tenant_model_version", return_value=MagicMock(id=uuid.uuid4())), \
         patch("app.repositories.jobs.complete_job_with_ticket_transition", return_value=True), \
         patch("app.repositories.jobs.enqueue_embedding_job_if_not_exists") as mock_enqueue:

        mock_manager_instance = MagicMock()
        mock_manager_instance.predict_ticket.return_value = MagicMock(
            predicted_label="Billing",
            decision_scores={"Billing": 1.2},
            model_name="tfidf_linearsvc",
            model_version="v1",
            sha256_hash="abc123hash",
        )
        mock_mm.return_value = mock_manager_instance

        status = _process_ticket_ingest_job(
            conn=mock_conn,
            job=job,
            worker_id="worker_test",
            lease_seconds=300,
            enable_inference=True,
            enable_embeddings=False,
        )

        assert status == "completed"
        mock_enqueue.assert_not_called()


# -------------------------------------------------------------------------
# 2. Embedding Job: Processing & Disabled Handling
# -------------------------------------------------------------------------

def test_embedding_job_released_to_queued_when_disabled():
    """If a ticket_embedding job is claimed while disabled, it is released to queued without failure."""
    mock_conn = MagicMock()
    mock_conn.transaction.return_value = dummy_transaction()

    job = make_dummy_job(job_type="ticket_embedding")

    with patch("app.repositories.jobs.release_claimed_job_to_queued", return_value=True) as mock_release, \
         patch("app.repositories.jobs.complete_embedding_job") as mock_complete, \
         patch("app.repositories.jobs.fail_or_retry_job") as mock_fail:

        status = _process_ticket_embedding_job(
            conn=mock_conn,
            job=job,
            worker_id="worker_test",
            lease_seconds=300,
            enable_embeddings=False,
            embedding_adapter=None,
        )

        assert status == "requeued"
        mock_release.assert_called_once_with(
            conn=mock_conn,
            tenant_id=job.tenant_id,
            job_id=job.id,
            worker_id="worker_test",
            generation=job.attempts,
        )
        mock_complete.assert_not_called()
        mock_fail.assert_not_called()


def test_embedding_job_computes_and_completes_when_enabled():
    """When enabled, ticket_embedding computes vector outside tx and completes atomically."""
    mock_conn = MagicMock()
    mock_conn.transaction.return_value = dummy_transaction()

    job = make_dummy_job(job_type="ticket_embedding")
    ticket = make_dummy_ticket(ticket_id=job.ticket_id, tenant_id=job.tenant_id, subject="Login Issue", body="Cannot login")
    fake_adapter = FakeEmbeddingAdapter()

    with patch("app.repositories.tickets.get_ticket_by_id", return_value=ticket) as mock_get_ticket, \
         patch("app.repositories.jobs.complete_embedding_job", return_value=True) as mock_complete, \
         patch("app.repositories.jobs.fail_or_retry_job") as mock_fail:

        status = _process_ticket_embedding_job(
            conn=mock_conn,
            job=job,
            worker_id="worker_test",
            lease_seconds=300,
            enable_embeddings=True,
            embedding_adapter=fake_adapter,
        )

        assert status == "completed"
        mock_get_ticket.assert_called_once_with(mock_conn, tenant_id=job.tenant_id, ticket_id=job.ticket_id)
        assert len(fake_adapter.embed_calls) == 1
        assert fake_adapter.embed_calls[0] == ("Login Issue", "Cannot login")

        mock_complete.assert_called_once()
        call_kwargs = mock_complete.call_args.kwargs
        assert call_kwargs["tenant_id"] == job.tenant_id
        assert call_kwargs["job_id"] == job.id
        assert call_kwargs["ticket_id"] == job.ticket_id
        assert call_kwargs["worker_id"] == "worker_test"
        assert call_kwargs["generation"] == job.attempts
        assert len(call_kwargs["embedding"]) == 384
        assert call_kwargs["model_name"] == "all-MiniLM-L6-v2"
        assert call_kwargs["model_version"] == "rev_test123"

        mock_fail.assert_not_called()


def test_embedding_job_missing_ticket_fails():
    """If ticket does not exist for the job tenant, fail_or_retry_job is called."""
    mock_conn = MagicMock()
    mock_conn.transaction.return_value = dummy_transaction()

    job = make_dummy_job(job_type="ticket_embedding")
    fake_adapter = FakeEmbeddingAdapter()

    with patch("app.repositories.tickets.get_ticket_by_id", return_value=None), \
         patch("app.repositories.jobs.fail_or_retry_job", return_value="failed") as mock_fail:

        status = _process_ticket_embedding_job(
            conn=mock_conn,
            job=job,
            worker_id="worker_test",
            lease_seconds=300,
            enable_embeddings=True,
            embedding_adapter=fake_adapter,
        )

        assert status == "failed"
        mock_fail.assert_called_once()
        err = mock_fail.call_args.kwargs["error_message"]
        assert f"Referenced ticket {job.ticket_id} not found" in err


def test_embedding_job_adapter_failure_isolated_from_category():
    """If adapter throws (missing model / corrupt artifact), fail_or_retry_job is called and category untouched."""
    mock_conn = MagicMock()
    mock_conn.transaction.return_value = dummy_transaction()

    job = make_dummy_job(job_type="ticket_embedding")
    ticket = make_dummy_ticket(ticket_id=job.ticket_id, tenant_id=job.tenant_id)
    failing_adapter = FakeEmbeddingAdapter(should_fail=True)

    with patch("app.repositories.tickets.get_ticket_by_id", return_value=ticket), \
         patch("app.repositories.jobs.fail_or_retry_job", return_value="requeued") as mock_fail, \
         patch("app.repositories.jobs.complete_job_with_ticket_transition") as mock_complete_tx:

        status = _process_ticket_embedding_job(
            conn=mock_conn,
            job=job,
            worker_id="worker_test",
            lease_seconds=300,
            enable_embeddings=True,
            embedding_adapter=failing_adapter,
        )

        assert status == "requeued"
        mock_fail.assert_called_once()
        assert "Inference session failure" in mock_fail.call_args.kwargs["error_message"]
        # Category / ticket transition must NOT be called
        mock_complete_tx.assert_not_called()


# -------------------------------------------------------------------------
# 3. Repository Fencing & Transactions
# -------------------------------------------------------------------------

def test_complete_embedding_job_success():
    """complete_embedding_job updates jobs row and saves embedding when claim is valid."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.rowcount = 1
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    tenant_id = uuid.uuid4()
    job_id = uuid.uuid4()
    ticket_id = uuid.uuid4()

    with patch("app.repositories.duplicates.save_ticket_embedding") as mock_save:
        res = complete_embedding_job(
            conn=mock_conn,
            tenant_id=tenant_id,
            job_id=job_id,
            ticket_id=ticket_id,
            worker_id="worker_1",
            generation=2,
            embedding=[0.1] * 384,
            model_name="all-MiniLM-L6-v2",
            model_version="rev123",
            lease_seconds=300,
        )

        assert res is True
        assert mock_cur.execute.called
        query_text = mock_cur.execute.call_args[0][0]
        assert "UPDATE jobs" in query_text
        assert "status = 'completed'" in query_text
        assert "tenant_id = %s" in query_text

        mock_save.assert_called_once_with(
            conn=mock_conn,
            tenant_id=tenant_id,
            ticket_id=ticket_id,
            embedding=[0.1] * 384,
            model_name="all-MiniLM-L6-v2",
            model_version="rev123",
        )
        # Ensure no internal commits
        mock_conn.commit.assert_not_called()


def test_complete_embedding_job_stale_claim_raises():
    """complete_embedding_job raises StaleClaimError when rowcount == 0 and does not save embedding."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.rowcount = 0  # Stale claim or expired lease
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with patch("app.repositories.duplicates.save_ticket_embedding") as mock_save:
        with pytest.raises(StaleClaimError, match="lease expired or generation mismatch"):
            complete_embedding_job(
                conn=mock_conn,
                tenant_id=uuid.uuid4(),
                job_id=uuid.uuid4(),
                ticket_id=uuid.uuid4(),
                worker_id="worker_stale",
                generation=1,
                embedding=[0.1] * 384,
                model_name="all-MiniLM-L6-v2",
                model_version="rev123",
            )
        mock_save.assert_not_called()
        mock_conn.commit.assert_not_called()


def test_release_claimed_job_to_queued():
    """release_claimed_job_to_queued transitions job back to queued status."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.rowcount = 1
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    res = release_claimed_job_to_queued(
        conn=mock_conn,
        tenant_id=uuid.uuid4(),
        job_id=uuid.uuid4(),
        worker_id="worker_1",
        generation=1,
    )

    assert res is True
    query_text = mock_cur.execute.call_args[0][0]
    assert "status = 'queued'" in query_text
    assert "locked_by = NULL" in query_text


def test_renew_lease():
    """renew_lease returns True if rowcount > 0, False if rowcount == 0."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    mock_cur.rowcount = 1
    assert renew_lease(mock_conn, uuid.uuid4(), "worker_1", 1, 300) is True

    mock_cur.rowcount = 0
    assert renew_lease(mock_conn, uuid.uuid4(), "worker_1", 1, 300) is False


def test_enqueue_embedding_job_if_not_exists_idempotent():
    """enqueue_embedding_job_if_not_exists includes ON CONFLICT DO NOTHING."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = None  # Conflict occurred, no new row returned
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    result = enqueue_embedding_job_if_not_exists(
        conn=mock_conn,
        tenant_id=uuid.uuid4(),
        ticket_id=uuid.uuid4(),
    )

    assert result is None
    query_text = mock_cur.execute.call_args[0][0]
    assert "ticket_embedding" in query_text
    assert "ON CONFLICT (tenant_id, ticket_id, type) DO NOTHING" in query_text


def test_claim_next_job_allowed_types():
    """claim_next_job filters by allowed_types using ANY(%s)."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = None
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    claim_next_job(
        conn=mock_conn,
        worker_id="worker_1",
        allowed_types=("ticket_ingest",),
    )

    query_text = mock_cur.execute.call_args[0][0]
    params = mock_cur.execute.call_args[0][1]
    assert "type = ANY(%s)" in query_text
    assert params[2] == ["ticket_ingest"]


# -------------------------------------------------------------------------
# 4. WorkerService Allowed Types Dispatch
# -------------------------------------------------------------------------

def test_worker_service_cycle_passes_correct_allowed_types():
    """WorkerService passes allowed_types to claim_next_job according to enable_embeddings."""
    mock_conn = MagicMock()

    with patch("app.repositories.jobs.recover_stale_jobs", return_value=0), \
         patch("app.repositories.jobs.claim_next_job", return_value=None) as mock_claim:

        # Case A: enable_embeddings = False
        worker_disabled = WorkerService(db_url="postgresql://fake", enable_embeddings=False)
        worker_disabled.run_one_cycle(mock_conn)
        assert mock_claim.call_args.kwargs["allowed_types"] == ("ticket_ingest",)

        mock_claim.reset_mock()

        # Case B: enable_embeddings = True
        worker_enabled = WorkerService(db_url="postgresql://fake", enable_embeddings=True)
        worker_enabled.run_one_cycle(mock_conn)
        assert mock_claim.call_args.kwargs["allowed_types"] == ("ticket_ingest", "ticket_embedding")
