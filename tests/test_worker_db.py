"""Database integration tests for Phase 2 background worker process and queue operations.

Tests against TEST_DATABASE_URL with controlled database timestamps (no sleep):
1. Eligible job claim and attempts increment.
2. Future run_after jobs are not claimed.
3. Competing connections cannot claim the same job generation.
4. Successful processing and matching audit event.
5. Non-new tickets remain unchanged.
6. Atomic rollback during finalization.
7. Retry scheduling and exhaustion.
8. Expired lease recovery.
9. Valid renewal and expired renewal rejection.
10. Stale worker cannot renew, fail, complete, or change ticket/audit data.
11. Tenant-scoped processing remains isolated.
12. Unsupported job type handled safely without crashing.
"""

from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import uuid
import pytest
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app
import app.db as app_db
import app.repositories.jobs as job_repo
import app.repositories.tickets as ticket_repo
from app.worker.handler import process_claimed_job
from app.worker.main import WorkerService
from tests.test_auth_db import TEST_DB_URL, SAFETY_STATUS
from tests.test_tickets_db import _create_tenant_and_api_key

skip_no_test_db = pytest.mark.skipif(TEST_DB_URL is None, reason=SAFETY_STATUS)


@asynccontextmanager
async def _worker_test_db_lifespan(test_app: FastAPI) -> AsyncIterator[None]:
    """Lifespan fixture connecting app.db to TEST_DATABASE_URL."""
    assert TEST_DB_URL is not None
    pool = ConnectionPool(
        conninfo=TEST_DB_URL,
        min_size=1,
        max_size=5,
        open=False,
    )
    app_db.set_pool(pool)
    pool.open(wait=True)
    try:
        yield
    finally:
        pool.close()
        app_db.set_pool(None)


@pytest.fixture
def http_client() -> TestClient:
    """FastAPI TestClient wired to the test database pool."""
    orig_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _worker_test_db_lifespan
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.router.lifespan_context = orig_lifespan


def _create_ticket_and_job(
    http_client: TestClient,
    company_name: str,
    subject: str = "Test Subject",
    body: str = "Test Body",
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID, str]:
    """Helper to create a tenant, ticket, and queued job via HTTP endpoint.

    Returns:
        tuple[tenant_id, ticket_id, job_id, api_key]
    """
    assert TEST_DB_URL is not None
    tenant_id, _, api_key, _ = _create_tenant_and_api_key(company_name)
    resp = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": api_key},
        json={"subject": subject, "body": body},
    )
    assert resp.status_code == 202
    ticket_id = uuid.UUID(resp.json()["id"])

    with psycopg.connect(TEST_DB_URL) as conn:
        job = job_repo.get_job_by_ticket_id(conn, tenant_id=tenant_id, ticket_id=ticket_id)
        assert job is not None
        job_id = job.id

    return tenant_id, ticket_id, job_id, api_key


@skip_no_test_db
def test_eligible_job_claim_and_attempts_increment(http_client: TestClient) -> None:
    """Eligible queued job is claimed atomically with attempts incremented to 1."""
    assert TEST_DB_URL is not None
    tenant_id, ticket_id, job_id, _ = _create_ticket_and_job(http_client, "Claim Corp")

    worker_id = f"worker_{uuid.uuid4().hex[:6]}"
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.transaction():
            job = job_repo.claim_next_job(conn, worker_id=worker_id, lease_seconds=300, tenant_id=tenant_id)

        assert job is not None
        assert job.id == job_id
        assert job.tenant_id == tenant_id
        assert job.ticket_id == ticket_id
        assert job.status == "running"
        assert job.attempts == 1
        assert job.locked_by == worker_id
        assert job.locked_at is not None


@skip_no_test_db
def test_future_run_after_jobs_are_not_claimed(http_client: TestClient) -> None:
    """Jobs scheduled in the future (run_after > NOW()) are ignored by claim_next_job."""
    assert TEST_DB_URL is not None
    tenant_id, ticket_id, job_id, _ = _create_ticket_and_job(http_client, "Future Corp")

    # Set run_after to 1 hour in the future
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET run_after = NOW() + INTERVAL '1 hour' WHERE id = %s;",
                (job_id,),
            )
        conn.commit()

        # Attempt to claim
        worker_id = f"worker_{uuid.uuid4().hex[:6]}"
        with conn.transaction():
            job = job_repo.claim_next_job(conn, worker_id=worker_id, tenant_id=tenant_id)
            assert job is None, "Future job was unexpectedly claimed!"

        # Verify job is still queued with attempts=0
        persisted = job_repo.get_job_by_id(conn, tenant_id=tenant_id, job_id=job_id)
        assert persisted is not None
        assert persisted.status == "queued"
        assert persisted.attempts == 0


@skip_no_test_db
def test_competing_connections_cannot_claim_same_job_generation(http_client: TestClient) -> None:
    """Concurrent claims with FOR UPDATE SKIP LOCKED guarantee mutual exclusion."""
    assert TEST_DB_URL is not None
    tenant_id, ticket_id, job_id, _ = _create_ticket_and_job(http_client, "Contention Corp")

    claimed_by: list[str | None] = []

    def claim_attempt(w_id: str) -> None:
        with psycopg.connect(TEST_DB_URL) as conn:
            with conn.transaction():
                job = job_repo.claim_next_job(conn, worker_id=w_id, tenant_id=tenant_id)
                claimed_by.append(job.locked_by if job else None)

    w1 = f"worker_A_{uuid.uuid4().hex[:4]}"
    w2 = f"worker_B_{uuid.uuid4().hex[:4]}"

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(claim_attempt, w1)
        f2 = executor.submit(claim_attempt, w2)
        f1.result()
        f2.result()

    # Exactly one connection claimed the job
    successful_claims = [w for w in claimed_by if w is not None]
    assert len(successful_claims) == 1
    assert None in claimed_by


@skip_no_test_db
def test_successful_processing_and_matching_audit_event(http_client: TestClient) -> None:
    """Processing a ticket moves status from new -> open and creates a matching audit event."""
    assert TEST_DB_URL is not None
    tenant_id, ticket_id, job_id, _ = _create_ticket_and_job(http_client, "Success Corp")

    worker_id = f"worker_{uuid.uuid4().hex[:6]}"
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.transaction():
            job = job_repo.claim_next_job(conn, worker_id=worker_id, tenant_id=tenant_id)
        assert job is not None
        assert job.id == job_id

        # Process the claimed job
        result = process_claimed_job(conn, job=job, worker_id=worker_id)
        assert result == "completed"

        # Verify job completed and lease fields cleared
        updated_job = job_repo.get_job_by_id(conn, tenant_id=tenant_id, job_id=job_id)
        assert updated_job is not None
        assert updated_job.status == "completed"
        assert updated_job.completed_at is not None
        assert updated_job.locked_at is None
        assert updated_job.locked_by is None

        # Verify ticket transitioned to open
        ticket = ticket_repo.get_ticket_by_id(conn, tenant_id=tenant_id, ticket_id=ticket_id)
        assert ticket is not None
        assert ticket.status == "open"

        # Verify audit log entry
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT actor, action FROM audit_log WHERE tenant_id = %s AND action = 'ticket.processed';",
                (tenant_id,),
            )
            logs = cur.fetchall()
            assert len(logs) == 1
            assert logs[0]["actor"] == f"worker:{worker_id}"


@skip_no_test_db
def test_non_new_tickets_remain_unchanged(http_client: TestClient) -> None:
    """If ticket is already open, pending, resolved, or closed, its status is not overwritten."""
    assert TEST_DB_URL is not None
    tenant_id, ticket_id, job_id, _ = _create_ticket_and_job(http_client, "Progressed Corp")

    # Manually move ticket status to 'pending' before the worker runs
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE tickets SET status = 'pending' WHERE tenant_id = %s AND id = %s;",
                (tenant_id, ticket_id),
            )
        conn.commit()

        worker_id = f"worker_{uuid.uuid4().hex[:6]}"
        with conn.transaction():
            job = job_repo.claim_next_job(conn, worker_id=worker_id, tenant_id=tenant_id)
        assert job is not None

        result = process_claimed_job(conn, job=job, worker_id=worker_id)
        assert result == "completed"

        # Verify ticket status remained 'pending'
        ticket = ticket_repo.get_ticket_by_id(conn, tenant_id=tenant_id, ticket_id=ticket_id)
        assert ticket is not None
        assert ticket.status == "pending"

        # Verify audit event indicates skipped transition
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT actor, action FROM audit_log WHERE tenant_id = %s AND action = 'ticket.ingest_skipped_already_progressed';",
                (tenant_id,),
            )
            logs = cur.fetchall()
            assert len(logs) == 1
            assert logs[0]["actor"] == f"worker:{worker_id}"


@skip_no_test_db
def test_atomic_rollback_during_finalization(http_client: TestClient) -> None:
    """If an error occurs during finalization, ticket status change and audit logs roll back."""
    assert TEST_DB_URL is not None
    tenant_id, ticket_id, job_id, _ = _create_ticket_and_job(http_client, "Rollback Final Corp")

    worker_id = f"worker_{uuid.uuid4().hex[:6]}"
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.transaction():
            job = job_repo.claim_next_job(conn, worker_id=worker_id, tenant_id=tenant_id)
        assert job is not None

        # Simulate finalization failure inside transaction
        with pytest.raises(RuntimeError):
            with conn.transaction():
                # Step 1: Execute complete_job_with_ticket_transition
                job_repo.complete_job_with_ticket_transition(
                    conn=conn,
                    tenant_id=tenant_id,
                    job_id=job_id,
                    ticket_id=ticket_id,
                    worker_id=worker_id,
                    generation=job.attempts,
                )
                # Forced error before commit
                raise RuntimeError("Simulated finalization crash")

        # Verify ticket was NOT transitioned (still new)
        ticket = ticket_repo.get_ticket_by_id(conn, tenant_id=tenant_id, ticket_id=ticket_id)
        assert ticket is not None
        assert ticket.status == "new"

        # Verify audit log for ticket.processed was NOT committed
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM audit_log WHERE tenant_id = %s AND action = 'ticket.processed';",
                (tenant_id,),
            )
            assert cur.fetchone()[0] == 0


@skip_no_test_db
def test_retry_scheduling_and_exhaustion(http_client: TestClient) -> None:
    """Job retries with exponential backoff and eventually marks failed upon exhaustion."""
    assert TEST_DB_URL is not None
    tenant_id, ticket_id, job_id, _ = _create_ticket_and_job(http_client, "Retry Corp")

    worker_id = f"worker_{uuid.uuid4().hex[:6]}"
    with psycopg.connect(TEST_DB_URL) as conn:
        # Attempt 1: Fail and requeue
        with conn.transaction():
            job = job_repo.claim_next_job(conn, worker_id=worker_id, tenant_id=tenant_id)
        assert job is not None
        assert job.attempts == 1

        with conn.transaction():
            res1 = job_repo.fail_or_retry_job(
                conn=conn,
                tenant_id=tenant_id,
                job_id=job_id,
                worker_id=worker_id,
                generation=1,
                error_message="Error attempt 1",
            )
        assert res1 == "requeued"

        j1 = job_repo.get_job_by_id(conn, tenant_id=tenant_id, job_id=job_id)
        assert j1 is not None
        assert j1.status == "queued"
        assert j1.locked_at is None
        assert j1.locked_by is None
        assert j1.error == "Error attempt 1"

        # Fast-forward run_after into past (no sleep)
        with conn.cursor() as cur:
            cur.execute("UPDATE jobs SET run_after = NOW() - INTERVAL '1 second' WHERE id = %s;", (job_id,))
        conn.commit()

        # Attempt 2: Fail and requeue
        with conn.transaction():
            job2 = job_repo.claim_next_job(conn, worker_id=worker_id, tenant_id=tenant_id)
        assert job2 is not None
        assert job2.attempts == 2

        with conn.transaction():
            res2 = job_repo.fail_or_retry_job(
                conn=conn,
                tenant_id=tenant_id,
                job_id=job_id,
                worker_id=worker_id,
                generation=2,
                error_message="Error attempt 2",
            )
        assert res2 == "requeued"

        # Fast-forward again
        with conn.cursor() as cur:
            cur.execute("UPDATE jobs SET run_after = NOW() - INTERVAL '1 second' WHERE id = %s;", (job_id,))
        conn.commit()

        # Attempt 3: Final attempt -> exhaustion
        with conn.transaction():
            job3 = job_repo.claim_next_job(conn, worker_id=worker_id, tenant_id=tenant_id)
        assert job3 is not None
        assert job3.attempts == 3

        with conn.transaction():
            res3 = job_repo.fail_or_retry_job(
                conn=conn,
                tenant_id=tenant_id,
                job_id=job_id,
                worker_id=worker_id,
                generation=3,
                error_message="Final fatal crash",
            )
        assert res3 == "failed"

        j_final = job_repo.get_job_by_id(conn, tenant_id=tenant_id, job_id=job_id)
        assert j_final is not None
        assert j_final.status == "failed"
        assert j_final.failed_at is not None
        assert j_final.error == "Final fatal crash"
        assert j_final.locked_at is None
        assert j_final.locked_by is None


@skip_no_test_db
def test_expired_lease_recovery(http_client: TestClient) -> None:
    """Expired running leases are recovered: requeued if attempts remain, failed if exhausted."""
    assert TEST_DB_URL is not None
    tenant_id, ticket_id, job_id, _ = _create_ticket_and_job(http_client, "Recovery Corp")

    with psycopg.connect(TEST_DB_URL) as conn:
        # Simulate a crashed worker at attempts=1 with expired lease (> 300s in past)
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    attempts = 1,
                    locked_by = 'crashed_worker',
                    locked_at = NOW() - INTERVAL '301 seconds'
                WHERE id = %s;
                """,
                (job_id,),
            )
        conn.commit()

        # Run recovery targeted at this tenant
        with conn.transaction():
            recovered_count = job_repo.recover_stale_jobs(conn, lease_seconds=300, tenant_id=tenant_id)
        assert recovered_count == 1

        rec = job_repo.get_job_by_id(conn, tenant_id=tenant_id, job_id=job_id)
        assert rec is not None
        assert rec.status == "queued"
        assert rec.locked_by is None
        assert rec.locked_at is None
        assert rec.error == "lease_timeout_requeued"

        # Now simulate exhaustion scenario: attempts=3
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    attempts = 3,
                    max_attempts = 3,
                    locked_by = 'crashed_worker_2',
                    locked_at = NOW() - INTERVAL '301 seconds'
                WHERE id = %s;
                """,
                (job_id,),
            )
        conn.commit()

        with conn.transaction():
            rec_exhaust = job_repo.recover_stale_jobs(conn, lease_seconds=300, tenant_id=tenant_id)
        assert rec_exhaust == 1

        rec_final = job_repo.get_job_by_id(conn, tenant_id=tenant_id, job_id=job_id)
        assert rec_final is not None
        assert rec_final.status == "failed"
        assert rec_final.failed_at is not None
        assert rec_final.error == "lease_timeout_exhausted"


@skip_no_test_db
def test_valid_renewal_and_expired_renewal_rejection(http_client: TestClient) -> None:
    """Active leases can be renewed; already-expired leases are strictly rejected."""
    assert TEST_DB_URL is not None
    tenant_id, ticket_id, job_id, _ = _create_ticket_and_job(http_client, "Renew Corp")

    worker_id = "worker_renew_test"
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.transaction():
            job = job_repo.claim_next_job(conn, worker_id=worker_id, tenant_id=tenant_id)
        assert job is not None

        # 1. Valid renewal before expiration
        renew_ok = job_repo.renew_lease(
            conn=conn,
            job_id=job_id,
            worker_id=worker_id,
            generation=job.attempts,
            lease_seconds=300,
        )
        assert renew_ok is True

        # 2. Set locked_at in the past (> 300s)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET locked_at = NOW() - INTERVAL '301 seconds' WHERE id = %s;",
                (job_id,),
            )
        conn.commit()

        # Renewal must now reject because lease is already expired!
        renew_expired = job_repo.renew_lease(
            conn=conn,
            job_id=job_id,
            worker_id=worker_id,
            generation=job.attempts,
            lease_seconds=300,
        )
        assert renew_expired is False


@skip_no_test_db
def test_stale_worker_cannot_renew_fail_complete_or_change_data(http_client: TestClient) -> None:
    """A stale worker whose lease expired cannot renew, fail, or complete, and makes no ticket updates."""
    assert TEST_DB_URL is not None
    tenant_id, ticket_id, job_id, _ = _create_ticket_and_job(http_client, "Stale Preempt Corp")

    w1 = "worker_stale_1"
    w2 = "worker_active_2"

    with psycopg.connect(TEST_DB_URL) as conn:
        # Worker 1 claims at generation 1
        with conn.transaction():
            job1 = job_repo.claim_next_job(conn, worker_id=w1, tenant_id=tenant_id)
        assert job1 is not None
        assert job1.attempts == 1

        # Simulate Worker 1 timing out: lease expired
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET locked_at = NOW() - INTERVAL '301 seconds' WHERE id = %s;",
                (job_id,),
            )
        conn.commit()

        # Recovery requeues it
        with conn.transaction():
            job_repo.recover_stale_jobs(conn, lease_seconds=300, tenant_id=tenant_id)

        # Clear run_after for Worker 2 to claim immediately
        with conn.cursor() as cur:
            cur.execute("UPDATE jobs SET run_after = NOW() - INTERVAL '1 second' WHERE id = %s;", (job_id,))
        conn.commit()

        # Worker 2 claims at generation 2
        with conn.transaction():
            job2 = job_repo.claim_next_job(conn, worker_id=w2, tenant_id=tenant_id)
        assert job2 is not None
        assert job2.attempts == 2
        assert job2.locked_by == w2

        # Stale Worker 1 wakes up and attempts renewal -> rejected
        assert job_repo.renew_lease(conn, job_id, w1, generation=1, lease_seconds=300) is False

        # Stale Worker 1 attempts fail_or_retry -> rejected
        fail_res = job_repo.fail_or_retry_job(conn, tenant_id, job_id, w1, generation=1, error_message="stale")
        assert fail_res == "stale_claim"

        # Stale Worker 1 attempts complete_job_with_ticket_transition -> raises StaleClaimError
        with pytest.raises(job_repo.StaleClaimError):
            with conn.transaction():
                job_repo.complete_job_with_ticket_transition(
                    conn=conn,
                    tenant_id=tenant_id,
                    job_id=job_id,
                    ticket_id=ticket_id,
                    worker_id=w1,
                    generation=1,
                    lease_seconds=300,
                )

        # Worker 2 processes successfully
        res = process_claimed_job(conn, job=job2, worker_id=w2)
        assert res == "completed"

        # Final state verification
        final_job = job_repo.get_job_by_id(conn, tenant_id=tenant_id, job_id=job_id)
        assert final_job is not None
        assert final_job.status == "completed"
        assert final_job.attempts == 2


@skip_no_test_db
def test_tenant_scoped_processing_remains_isolated(http_client: TestClient) -> None:
    """Processing a job for Tenant A modifies only Tenant A's ticket and audit log."""
    assert TEST_DB_URL is not None
    tA_id, tA_ticket_id, tA_job_id, _ = _create_ticket_and_job(http_client, "Tenant A Worker")
    tB_id, tB_ticket_id, tB_job_id, _ = _create_ticket_and_job(http_client, "Tenant B Worker")

    worker_id = "worker_isolation_test"
    with psycopg.connect(TEST_DB_URL) as conn:
        # Claim Tenant A's job directly with tenant_id filter
        with conn.transaction():
            job_A = job_repo.claim_next_job(conn, worker_id=worker_id, tenant_id=tA_id)
        assert job_A is not None
        assert job_A.id == tA_job_id

        # Process Tenant A's job
        result = process_claimed_job(conn, job=job_A, worker_id=worker_id)
        assert result == "completed"

        # Tenant A's ticket transitioned to open
        ticket_A = ticket_repo.get_ticket_by_id(conn, tenant_id=tA_id, ticket_id=tA_ticket_id)
        assert ticket_A is not None
        assert ticket_A.status == "open"

        # Tenant B's ticket is strictly unchanged (still 'new')
        ticket_B = ticket_repo.get_ticket_by_id(conn, tenant_id=tB_id, ticket_id=tB_ticket_id)
        assert ticket_B is not None
        assert ticket_B.status == "new"

        # Audit log verification: Tenant A has ticket.processed row, Tenant B has zero ticket.processed rows
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM audit_log WHERE tenant_id = %s AND action = 'ticket.processed';", (tA_id,))
            assert cur.fetchone()[0] == 1

            cur.execute("SELECT COUNT(*) FROM audit_log WHERE tenant_id = %s AND action = 'ticket.processed';", (tB_id,))
            assert cur.fetchone()[0] == 0


@skip_no_test_db
def test_worker_service_single_cycle_execution(http_client: TestClient) -> None:
    """WorkerService.run_one_cycle claims, processes, and completes an eligible job."""
    assert TEST_DB_URL is not None
    tenant_id, ticket_id, job_id, _ = _create_ticket_and_job(http_client, "WorkerService Corp")

    worker = WorkerService(db_url=TEST_DB_URL, worker_id="service_worker_1")
    with psycopg.connect(TEST_DB_URL, autocommit=True) as conn:
        res = worker.run_one_cycle(conn, tenant_id=tenant_id)
        assert res == "completed"

        # Verify job is completed
        job = job_repo.get_job_by_id(conn, tenant_id=tenant_id, job_id=job_id)
        assert job is not None
        assert job.status == "completed"


@skip_no_test_db
def test_unsupported_job_type_handled_safely(http_client: TestClient) -> None:
    """An unsupported job type records error and fails without crashing worker."""
    assert TEST_DB_URL is not None
    tenant_id, ticket_id, job_id, _ = _create_ticket_and_job(http_client, "Unsupported Corp")

    # Claim job
    worker_id = "worker_unsupported_test"
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.transaction():
            job = job_repo.claim_next_job(conn, worker_id=worker_id, tenant_id=tenant_id)
        assert job is not None

        # Create a JobRecord with unsupported type
        fake_unsupported_job = job_repo.JobRecord(
            id=job.id,
            tenant_id=job.tenant_id,
            ticket_id=job.ticket_id,
            type="unsupported_type_xyz",
            status=job.status,
            attempts=job.attempts,
            max_attempts=job.max_attempts,
            run_after=job.run_after,
            locked_at=job.locked_at,
            locked_by=job.locked_by,
            created_at=job.created_at,
            updated_at=job.updated_at,
            completed_at=job.completed_at,
            failed_at=job.failed_at,
            error=job.error,
            payload=job.payload,
        )

        res = process_claimed_job(conn, job=fake_unsupported_job, worker_id=worker_id)
        assert res == "unsupported_job_type"
