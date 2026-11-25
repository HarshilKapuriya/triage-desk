"""Database integration tests for independent ticket_embedding job integration.

Strict safety constraints:
- Database tests run ONLY against an already established disposable local PostgreSQL
  instance (localhost or 127.0.0.1).
- Hosted databases (e.g. Supabase poolers) are strictly prohibited.
- Local DB test guards do not assume localhost alone proves disposability:
  they verify target separation from DATABASE_URL, dedicated test naming,
  and actual reachability.
- If no established disposable local test database exists, tests are cleanly skipped.
"""

import os
import uuid
import pytest
import psycopg
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

from app.config import get_settings
from app.repositories.auth import create_tenant_and_admin_user
import app.repositories.jobs as job_repo
from app.repositories.jobs import StaleClaimError
import app.repositories.duplicates as dup_repo
import app.repositories.ticket_events as ticket_events_repo
import app.repositories.tickets as ticket_repo
from app.security import hash_password
from tests.test_auth_db import resolve_safe_test_db_url, IS_CI


def resolve_disposable_local_test_db() -> tuple[str | None, str]:
    """Resolve TEST_DATABASE_URL ensuring it is strictly local, distinct, disposable, and reachable.

    Does not assume localhost alone proves disposability.
    """
    safe_url, safety_status = resolve_safe_test_db_url()
    if not safe_url:
        return None, f"TEST_DATABASE_URL unsafe or unconfigured: {safety_status}"

    try:
        params = conninfo_to_dict(safe_url)
    except Exception as exc:
        return None, f"Cannot parse TEST_DATABASE_URL: {exc}"

    host = params.get("host", "localhost")
    port = params.get("port", "5432")
    dbname = params.get("dbname", "")

    # 1. Strictly reject hosted / remote databases (e.g. Supabase poolers)
    if host not in ("localhost", "127.0.0.1", "postgres"):
        return None, f"Hosted database ({host}) is strictly prohibited for embedding integration tests. Requires an established local disposable instance."

    # 2. Verify disposability (do not assume localhost alone proves disposability)
    if not IS_CI:
        cfg = get_settings()
        dev_raw = cfg.database_url.get_secret_value() if cfg.database_url else ""
        if dev_raw:
            try:
                dev_params = conninfo_to_dict(dev_raw)
                if (
                    dev_params.get("host", "localhost") == host
                    and dev_params.get("port", "5432") == port
                    and dev_params.get("dbname", "") == dbname
                ):
                    return None, "SAFETY VIOLATION: localhost test database targets same dbname as DATABASE_URL; not disposable."
            except Exception:
                pass
        if "test" not in dbname.lower():
            return None, f"Database '{dbname}' does not contain 'test' in its name. Disposability cannot be assumed from localhost alone."

    # 3. Active reachability probe (1 second connect timeout)
    try:
        with psycopg.connect(safe_url, connect_timeout=1) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
    except Exception as exc:
        return None, f"No established local PostgreSQL instance is reachable on {host}:{port}: {exc}"

    return safe_url, "Available and verified disposable"


LOCAL_TEST_DB_URL, LOCAL_SAFETY_STATUS = resolve_disposable_local_test_db()
pytestmark = pytest.mark.skipif(
    LOCAL_TEST_DB_URL is None,
    reason=f"Database integration tests require an established disposable local PostgreSQL instance: {LOCAL_SAFETY_STATUS}",
)


def _create_test_tenant(company_name: str) -> uuid.UUID:
    """Helper to create an isolated test tenant."""
    assert LOCAL_TEST_DB_URL is not None
    suffix = uuid.uuid4().hex[:8]
    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        tenant_id, _ = create_tenant_and_admin_user(
            conn=conn,
            company_name=company_name,
            normalized_email=f"user_{suffix}@test.com",
            password_hash=hash_password("Pass123!"),
        )
    return tenant_id


def _create_test_ticket(tenant_id: uuid.UUID, subject: str = "Test Subject") -> uuid.UUID:
    """Helper to create a ticket for a tenant."""
    assert LOCAL_TEST_DB_URL is not None
    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        ticket, _ = ticket_repo.create_or_get_ticket(
            conn=conn,
            tenant_id=tenant_id,
            subject=subject,
            body="Test complaint body text regarding unauthorized debit.",
        )
        return ticket.id


def test_atomic_enqueue_and_rollback():
    """Verify category completion and embedding enqueue share one atomic transaction."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Atomic Enqueue Co")
    ticket_id = _create_test_ticket(tenant_id, "Atomic Enqueue Ticket")

    # autocommit=True guarantees each `with conn.transaction():` is an outermost transaction
    # (BEGIN/COMMIT/ROLLBACK) rather than a nested savepoint after intermediate reads.
    with psycopg.connect(LOCAL_TEST_DB_URL, autocommit=True) as conn:
        # Explicitly enqueue ticket_ingest job and claim it prior to completion attempts
        with conn.transaction():
            enqueued_ingest = job_repo.enqueue_job(
                conn=conn,
                tenant_id=tenant_id,
                ticket_id=ticket_id,
                job_type="ticket_ingest",
            )
            assert enqueued_ingest.status == "queued"

        with conn.transaction():
            ingest_job = job_repo.claim_next_job(
                conn=conn,
                worker_id="worker_atomic",
                allowed_types=("ticket_ingest",),
                tenant_id=tenant_id,
            )
        assert ingest_job is not None
        assert ingest_job.id == enqueued_ingest.id
        assert ingest_job.status == "running"
        assert ingest_job.attempts == 1

        # Case A: Rollback inside transaction drops completion, ticket transition,
        # lifecycle event, audit entry, and enqueued embedding job atomically
        try:
            with conn.transaction():
                job_repo.complete_job_with_ticket_transition(
                    conn=conn,
                    tenant_id=tenant_id,
                    job_id=ingest_job.id,
                    ticket_id=ticket_id,
                    worker_id="worker_atomic",
                    generation=ingest_job.attempts,
                )
                job_repo.enqueue_embedding_job_if_not_exists(conn, tenant_id, ticket_id)
                raise RuntimeError("Simulated failure after enqueue")
        except RuntimeError:
            pass

        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM jobs WHERE tenant_id = %s AND ticket_id = %s AND type = 'ticket_embedding';",
                (tenant_id, ticket_id),
            )
            count = cur.fetchone()[0]
            assert count == 0, "Rolled back transaction must not commit embedding job"

            cur.execute(
                "SELECT count(*) FROM audit_log WHERE tenant_id = %s AND action = 'ticket.processed';",
                (tenant_id,),
            )
            assert cur.fetchone()[0] == 0, "Rolled back transaction must not commit audit entry"

        ticket_after_rollback = ticket_repo.get_ticket_by_id(conn, tenant_id, ticket_id)
        assert ticket_after_rollback is not None
        assert ticket_after_rollback.status == "new", "Rolled back transaction must keep ticket in 'new' status"

        ingest_after_rollback = job_repo.get_job_by_id(conn, tenant_id, ingest_job.id)
        assert ingest_after_rollback is not None
        assert ingest_after_rollback.status == "running"
        assert ingest_after_rollback.attempts == 1
        assert ingest_after_rollback.completed_at is None

        events_after_rollback = ticket_events_repo.list_ticket_events(conn, tenant_id, ticket_id)
        assert [e.event_type for e in events_after_rollback] == ["ticket_created"], (
            "Rolled back transaction must not commit ticket_processed lifecycle event"
        )

        # Case B: Ingest completion and embedding enqueue commit atomically together
        with conn.transaction():
            completed = job_repo.complete_job_with_ticket_transition(
                conn=conn,
                tenant_id=tenant_id,
                job_id=ingest_job.id,
                ticket_id=ticket_id,
                worker_id="worker_atomic",
                generation=ingest_job.attempts,
            )
            assert completed is True
            emb_job = job_repo.enqueue_embedding_job_if_not_exists(conn, tenant_id, ticket_id)
            assert emb_job is not None
            assert emb_job.type == "ticket_embedding"
            assert emb_job.status == "queued"

        # Verify ticket is open, ingest job is completed, lifecycle event recorded, and embedding job exists
        ticket = ticket_repo.get_ticket_by_id(conn, tenant_id, ticket_id)
        assert ticket is not None
        assert ticket.status == "open"

        ingest_after_commit = job_repo.get_job_by_id(conn, tenant_id, ingest_job.id)
        assert ingest_after_commit is not None
        assert ingest_after_commit.status == "completed"
        assert ingest_after_commit.completed_at is not None

        events_after_commit = ticket_events_repo.list_ticket_events(conn, tenant_id, ticket_id)
        assert len(events_after_commit) == 2
        assert {e.event_type for e in events_after_commit} == {"ticket_created", "ticket_processed"}

        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM jobs WHERE tenant_id = %s AND ticket_id = %s AND type = 'ticket_embedding';",
                (tenant_id, ticket_id),
            )
            assert cur.fetchone()[0] == 1

            cur.execute(
                "SELECT count(*) FROM audit_log WHERE tenant_id = %s AND action = 'ticket.processed';",
                (tenant_id,),
            )
            assert cur.fetchone()[0] == 1

        # Verify idempotent enqueue replay does not add duplicate embedding job
        with conn.transaction():
            duplicate_emb = job_repo.enqueue_embedding_job_if_not_exists(conn, tenant_id, ticket_id)
            assert duplicate_emb is None


def test_embedding_persistence():
    """Verify complete_embedding_job atomically completes job and persists embedding vector."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Persistence Co")
    ticket_id = _create_test_ticket(tenant_id, "Persistence Ticket")

    with psycopg.connect(LOCAL_TEST_DB_URL, autocommit=True) as conn:
        with conn.transaction():
            enqueued = job_repo.enqueue_embedding_job_if_not_exists(conn, tenant_id, ticket_id)
            assert enqueued is not None

        # Claim the embedding job
        with conn.transaction():
            job = job_repo.claim_next_job(
                conn=conn,
                worker_id="worker_persist",
                allowed_types=("ticket_embedding",),
                tenant_id=tenant_id,
            )
        assert job is not None
        assert job.type == "ticket_embedding"
        assert job.attempts == 1

        dummy_vector = [0.05 * (i % 10) for i in range(384)]
        with conn.transaction():
            completed = job_repo.complete_embedding_job(
                conn=conn,
                tenant_id=tenant_id,
                job_id=job.id,
                ticket_id=ticket_id,
                worker_id="worker_persist",
                generation=job.attempts,
                embedding=dummy_vector,
                model_name="all-MiniLM-L6-v2",
                model_version="rev_db_persist",
            )
            assert completed is True

        # Verify job is completed
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT status, completed_at FROM jobs WHERE tenant_id = %s AND id = %s;",
                (tenant_id, job.id),
            )
            row = cur.fetchone()
            assert row is not None
            assert row["status"] == "completed"
            assert row["completed_at"] is not None

        # Verify embedding row persisted via duplicates repo
        stored_emb = dup_repo.get_ticket_embedding(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id=ticket_id,
            model_name="all-MiniLM-L6-v2",
            model_version="rev_db_persist",
        )
        assert stored_emb is not None
        assert stored_emb.tenant_id == tenant_id
        assert stored_emb.ticket_id == ticket_id
        assert stored_emb.model_name == "all-MiniLM-L6-v2"
        assert stored_emb.model_version == "rev_db_persist"
        assert len(stored_emb.embedding) == 384
        assert stored_emb.embedding == pytest.approx(dummy_vector, abs=1e-5)

        # Verify cross-tenant isolation and model/version specificity
        assert (
            dup_repo.get_ticket_embedding(
                conn=conn,
                tenant_id=uuid.uuid4(),
                ticket_id=ticket_id,
                model_name="all-MiniLM-L6-v2",
                model_version="rev_db_persist",
            )
            is None
        )
        assert (
            dup_repo.get_ticket_embedding(
                conn=conn,
                tenant_id=tenant_id,
                ticket_id=ticket_id,
                model_name="all-MiniLM-L6-v2",
                model_version="rev_other",
            )
            is None
        )


def test_independent_retry_and_terminal_failure():
    """Verify embedding job retry and terminal failure do not roll back or modify ticket status."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Failure Isolation Co")
    ticket_id = _create_test_ticket(tenant_id, "Failure Isolation Ticket")

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        # Enqueue embedding job
        with conn.transaction():
            job_repo.enqueue_embedding_job_if_not_exists(conn, tenant_id, ticket_id)

        # Claim job (attempt 1)
        with conn.transaction():
            job = job_repo.claim_next_job(
                conn=conn,
                worker_id="worker_fail",
                allowed_types=("ticket_embedding",),
                tenant_id=tenant_id,
            )
            assert job is not None

        # Record retryable error
        with conn.transaction():
            res = job_repo.fail_or_retry_job(
                conn=conn,
                tenant_id=tenant_id,
                job_id=job.id,
                worker_id="worker_fail",
                generation=job.attempts,
                error_message="RuntimeError: Transient artifact download failure",
            )
            assert res == "requeued"

        # Force attempts to max_attempts to test terminal failure
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET attempts = max_attempts, status = 'running', locked_by = 'worker_fail' WHERE id = %s;",
                (job.id,),
            )

        with conn.transaction():
            res_terminal = job_repo.fail_or_retry_job(
                conn=conn,
                tenant_id=tenant_id,
                job_id=job.id,
                worker_id="worker_fail",
                generation=job.max_attempts,
                error_message="RuntimeError: Corrupt ONNX session graph",
            )
            assert res_terminal == "failed"

        # Check job status is 'failed'
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT status, failed_at, error FROM jobs WHERE id = %s;", (job.id,))
            row = cur.fetchone()
            assert row["status"] == "failed"
            assert row["failed_at"] is not None
            assert "Corrupt ONNX session" in row["error"]

        # Ensure ticket status remains untouched
        ticket = ticket_repo.get_ticket_by_id(conn, tenant_id, ticket_id)
        assert ticket is not None
        assert ticket.status == "new"


def test_stale_claim_rejection():
    """Verify complete_embedding_job, renew_lease, and fail_or_retry_job reject stale claims."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Stale Claim Co")
    ticket_id = _create_test_ticket(tenant_id, "Stale Claim Ticket")

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        with conn.transaction():
            job_repo.enqueue_embedding_job_if_not_exists(conn, tenant_id, ticket_id)
            job = job_repo.claim_next_job(
                conn=conn,
                worker_id="worker_primary",
                allowed_types=("ticket_embedding",),
                tenant_id=tenant_id,
            )
            assert job is not None

        # Expired lease / generation mismatch rejection on renew_lease
        assert not job_repo.renew_lease(
            conn=conn,
            job_id=job.id,
            worker_id="worker_imposter",
            generation=job.attempts,
        )

        # Stale claim rejection on fail_or_retry_job
        status = job_repo.fail_or_retry_job(
            conn=conn,
            tenant_id=tenant_id,
            job_id=job.id,
            worker_id="worker_imposter",
            generation=job.attempts,
            error_message="Should be rejected",
        )
        assert status == "stale_claim"

        # Stale claim rejection on complete_embedding_job raises StaleClaimError
        with pytest.raises(StaleClaimError):
            job_repo.complete_embedding_job(
                conn=conn,
                tenant_id=tenant_id,
                job_id=job.id,
                ticket_id=ticket_id,
                worker_id="worker_imposter",
                generation=job.attempts,
                embedding=[0.1] * 384,
                model_name="all-MiniLM-L6-v2",
                model_version="rev_stale",
            )


def test_disabled_claims_filtering():
    """Verify claim_next_job respects allowed_types and excludes ticket_embedding when disabled."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Disabled Claim Co")
    ticket_id = _create_test_ticket(tenant_id, "Disabled Claim Ticket")

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        # Enqueue embedding job
        with conn.transaction():
            job_repo.enqueue_embedding_job_if_not_exists(conn, tenant_id, ticket_id)

        # Worker claiming with allowed_types=("ticket_ingest",) must find nothing
        with conn.transaction():
            claimed_disabled = job_repo.claim_next_job(
                conn=conn,
                worker_id="worker_disabled",
                allowed_types=("ticket_ingest",),
                tenant_id=tenant_id,
            )
            assert claimed_disabled is None

        # Verify job remains queued
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT status FROM jobs WHERE tenant_id = %s AND ticket_id = %s AND type = 'ticket_embedding';",
                (tenant_id, ticket_id),
            )
            assert cur.fetchone()["status"] == "queued"

        # Worker claiming with allowed_types including ticket_embedding claims it
        with conn.transaction():
            claimed_enabled = job_repo.claim_next_job(
                conn=conn,
                worker_id="worker_enabled",
                allowed_types=("ticket_ingest", "ticket_embedding"),
                tenant_id=tenant_id,
            )
            assert claimed_enabled is not None
            assert claimed_enabled.type == "ticket_embedding"


def test_tenant_integrity():
    """Verify database foreign keys reject cross-tenant embedding jobs and embeddings."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_a = _create_test_tenant("Tenant A Co")
    tenant_b = _create_test_tenant("Tenant B Co")
    ticket_b = _create_test_ticket(tenant_b, "Tenant B Ticket")

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        # Attempt to insert job associating Tenant A with Ticket B must fail
        with pytest.raises(psycopg.Error):
            with conn.transaction():
                job_repo.enqueue_embedding_job_if_not_exists(conn, tenant_a, ticket_b)

        # Attempt to persist embedding for Tenant A referencing Ticket B must fail
        with pytest.raises(psycopg.Error):
            with conn.transaction():
                dup_repo.save_ticket_embedding(
                    conn=conn,
                    tenant_id=tenant_a,
                    ticket_id=ticket_b,
                    embedding=[0.1] * 384,
                    model_name="all-MiniLM-L6-v2",
                    model_version="rev_cross",
                )


def test_browser_list_and_status_non_interference():
    """Verify ticket list query returns exactly one row per ticket and derives status only from ingest job."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Browser Query Co")
    other_tenant_id = _create_test_tenant("Browser Isolation Co")
    ticket_id = _create_test_ticket(tenant_id, "Browser Non-Interference Ticket")

    with psycopg.connect(LOCAL_TEST_DB_URL, autocommit=True) as conn:
        # Explicitly enqueue, claim, and complete ticket_ingest job
        with conn.transaction():
            enqueued_ingest = job_repo.enqueue_job(
                conn=conn,
                tenant_id=tenant_id,
                ticket_id=ticket_id,
                job_type="ticket_ingest",
            )
            assert enqueued_ingest.status == "queued"

        with conn.transaction():
            ingest_job = job_repo.claim_next_job(
                conn=conn,
                worker_id="worker_browser",
                allowed_types=("ticket_ingest",),
                tenant_id=tenant_id,
            )
        assert ingest_job is not None
        assert ingest_job.id == enqueued_ingest.id
        assert ingest_job.attempts == 1

        with conn.transaction():
            completed = job_repo.complete_job_with_ticket_transition(
                conn=conn,
                tenant_id=tenant_id,
                job_id=ingest_job.id,
                ticket_id=ticket_id,
                worker_id="worker_browser",
                generation=ingest_job.attempts,
            )
            assert completed is True

        # Enqueue embedding job in queued status
        with conn.transaction():
            emb_job = job_repo.enqueue_embedding_job_if_not_exists(conn, tenant_id, ticket_id)
            assert emb_job is not None
            assert emb_job.status == "queued"

        # Query ticket list via supported browser repository listing interface
        items = ticket_repo.list_tickets_with_prediction(conn, tenant_id, limit=20, offset=0)
        matching = [t for t in items if t.id == ticket_id]

        assert ticket_repo.count_tickets(conn, tenant_id) == 1
        assert len(matching) == 1, "Must return exactly one row without duplication from embedding job"
        assert matching[0].tenant_id == tenant_id
        assert matching[0].status == "open", "Ticket status must reflect completed ingest transition"
        assert matching[0].job_status == "completed", "Job status must reflect ticket_ingest job, not ticket_embedding"
        assert matching[0].job_attempts == 1

        # Pagination offset beyond available items returns empty list
        assert ticket_repo.list_tickets_with_prediction(conn, tenant_id, limit=20, offset=1) == []

        # Cross-tenant isolation check on ticket listing
        other_items = ticket_repo.list_tickets_with_prediction(conn, other_tenant_id, limit=20, offset=0)
        assert all(t.id != ticket_id for t in other_items)
