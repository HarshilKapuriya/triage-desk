"""Database integration tests for Phase 2 jobs queue schema and transactional enqueueing.

Per SPEC.md and Phase 2 requirements:
1. New ticket produces exactly one matching queued job.
2. Enqueue failure rolls back the new ticket atomically.
3. Idempotent replay does not add another job, including when the original job is completed or failed.
4. Concurrent same-tenant/external_id submissions produce one ticket and one job.
5. Same external_id across tenants produces independent tickets and jobs.
6. The database rejects a job attaching tenant A to tenant B's ticket (composite foreign key).
7. Tickets without external_id each receive their own job.
8. Regression check: all existing tests remain green.
"""

from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from unittest.mock import patch
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
from tests.test_auth_db import TEST_DB_URL, SAFETY_STATUS
from tests.test_tickets_db import _create_tenant_and_api_key

skip_no_test_db = pytest.mark.skipif(TEST_DB_URL is None, reason=SAFETY_STATUS)


@asynccontextmanager
async def _jobs_test_db_lifespan(test_app: FastAPI) -> AsyncIterator[None]:
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
    app.router.lifespan_context = _jobs_test_db_lifespan
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.router.lifespan_context = orig_lifespan


@skip_no_test_db
def test_new_ticket_produces_exactly_one_matching_queued_job(http_client: TestClient) -> None:
    """A new ticket ingestion transactionally enqueues exactly one matching queued job."""
    assert TEST_DB_URL is not None
    tenant_id, _, api_key, _ = _create_tenant_and_api_key("Job Ingest Corp")

    resp = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": api_key},
        json={"subject": "Billing issue", "body": "Need invoice for March."},
    )
    assert resp.status_code == 202
    ticket_data = resp.json()
    ticket_id = uuid.UUID(ticket_data["id"])

    with psycopg.connect(TEST_DB_URL) as conn:
        job = job_repo.get_job_by_ticket_id(conn, tenant_id=tenant_id, ticket_id=ticket_id)
        assert job is not None
        assert job.tenant_id == tenant_id
        assert job.ticket_id == ticket_id
        assert job.type == "ticket_ingest"
        assert job.status == "queued"
        assert job.attempts == 0
        assert job.max_attempts == 3
        assert job.locked_at is None
        assert job.locked_by is None
        assert job.completed_at is None
        assert job.failed_at is None
        assert job.error is None


@skip_no_test_db
def test_enqueue_failure_rolls_back_new_ticket(http_client: TestClient) -> None:
    """If job enqueueing fails, the transaction rolls back and the ticket is not created."""
    assert TEST_DB_URL is not None
    tenant_id, _, api_key, _ = _create_tenant_and_api_key("Rollback Corp")
    ext_id = f"ext_fail_{uuid.uuid4().hex[:8]}"

    # Patch job_repo.enqueue_job to fail
    with patch("app.routers.tickets.job_repo.enqueue_job", side_effect=RuntimeError("Simulated job failure")):
        with pytest.raises(RuntimeError):
            http_client.post(
                "/v1/tickets",
                headers={"X-API-Key": api_key},
                json={"subject": "Failed Subject", "body": "Failed Body", "external_id": ext_id},
            )

    # Verify that the ticket was NOT created
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM tickets WHERE tenant_id = %s AND external_id = %s;",
                (tenant_id, ext_id),
            )
            count = cur.fetchone()[0]
            assert count == 0, "Ticket row was committed despite enqueue failure!"


@skip_no_test_db
def test_idempotent_replay_does_not_add_job_including_completed_or_failed(http_client: TestClient) -> None:
    """Replaying an existing external_id returns 200 and does not enqueue another job in any state."""
    assert TEST_DB_URL is not None
    tenant_id, _, api_key, _ = _create_tenant_and_api_key("Replay Corp")
    ext_id = f"order_{uuid.uuid4().hex[:8]}"

    # 1. Initial creation
    resp1 = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": api_key},
        json={"subject": "Order Issue", "body": "Where is my item?", "external_id": ext_id},
    )
    assert resp1.status_code == 202
    ticket_id = uuid.UUID(resp1.json()["id"])

    # 2. Replay while job is queued
    resp2 = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": api_key},
        json={"subject": "Order Issue", "body": "Where is my item?", "external_id": ext_id},
    )
    assert resp2.status_code == 200
    assert resp2.json()["id"] == str(ticket_id)

    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM jobs WHERE tenant_id = %s AND ticket_id = %s;",
                (tenant_id, ticket_id),
            )
            assert cur.fetchone()[0] == 1

        # 3. Simulate job completed by worker
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status = 'completed', completed_at = NOW() WHERE ticket_id = %s;",
                (ticket_id,),
            )
        conn.commit()

    # Replay while job is completed
    resp3 = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": api_key},
        json={"subject": "Order Issue", "body": "Where is my item?", "external_id": ext_id},
    )
    assert resp3.status_code == 200
    assert resp3.json()["id"] == str(ticket_id)

    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM jobs WHERE tenant_id = %s AND ticket_id = %s;",
                (tenant_id, ticket_id),
            )
            assert cur.fetchone()[0] == 1

    # 4. Another ticket with failed job
    ext_id_fail = f"fail_{uuid.uuid4().hex[:8]}"
    resp_fail = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": api_key},
        json={"subject": "Failed Ticket", "body": "Error occurred", "external_id": ext_id_fail},
    )
    assert resp_fail.status_code == 202
    ticket_fail_id = uuid.UUID(resp_fail.json()["id"])

    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status = 'failed', failed_at = NOW(), error = 'max_retries' WHERE ticket_id = %s;",
                (ticket_fail_id,),
            )
        conn.commit()

    # Replay when job is in failed state
    resp_replay_fail = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": api_key},
        json={"subject": "Failed Ticket", "body": "Error occurred", "external_id": ext_id_fail},
    )
    assert resp_replay_fail.status_code == 200
    assert resp_replay_fail.json()["id"] == str(ticket_fail_id)

    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM jobs WHERE tenant_id = %s AND ticket_id = %s;",
                (tenant_id, ticket_fail_id),
            )
            assert cur.fetchone()[0] == 1


@skip_no_test_db
def test_concurrent_same_tenant_external_id_submissions_produce_one_ticket_and_one_job(http_client: TestClient) -> None:
    """Concurrent requests with same (tenant_id, external_id) create exactly 1 ticket and 1 job."""
    assert TEST_DB_URL is not None
    tenant_id, _, api_key, _ = _create_tenant_and_api_key("Race Jobs Corp")
    ext_id = f"concurrent_ext_{uuid.uuid4().hex[:8]}"

    def send_request(subject: str) -> dict:
        return http_client.post(
            "/v1/tickets",
            headers={"X-API-Key": api_key},
            json={"subject": subject, "body": "Concurrent race body", "external_id": ext_id},
        ).json()

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(send_request, "Subject 1")
        f2 = executor.submit(send_request, "Subject 2")
        res1 = f1.result()
        res2 = f2.result()

    assert res1["id"] == res2["id"]
    ticket_id = uuid.UUID(res1["id"])

    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM tickets WHERE tenant_id = %s AND external_id = %s;",
                (tenant_id, ext_id),
            )
            assert cur.fetchone()[0] == 1

            cur.execute(
                "SELECT COUNT(*) FROM jobs WHERE tenant_id = %s AND ticket_id = %s;",
                (tenant_id, ticket_id),
            )
            assert cur.fetchone()[0] == 1


@skip_no_test_db
def test_same_external_id_across_tenants_produces_independent_tickets_and_jobs(http_client: TestClient) -> None:
    """Same external_id across two tenants produces independent tickets and jobs."""
    assert TEST_DB_URL is not None
    tA_id, _, key_A, _ = _create_tenant_and_api_key("Tenant A Jobs")
    tB_id, _, key_B, _ = _create_tenant_and_api_key("Tenant B Jobs")
    shared_ext_id = f"ext_multi_{uuid.uuid4().hex[:8]}"

    resp_A = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": key_A},
        json={"subject": "Tenant A Subject", "body": "Body A", "external_id": shared_ext_id},
    )
    resp_B = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": key_B},
        json={"subject": "Tenant B Subject", "body": "Body B", "external_id": shared_ext_id},
    )

    assert resp_A.status_code == 202
    assert resp_B.status_code == 202
    tA_ticket_id = uuid.UUID(resp_A.json()["id"])
    tB_ticket_id = uuid.UUID(resp_B.json()["id"])
    assert tA_ticket_id != tB_ticket_id

    with psycopg.connect(TEST_DB_URL) as conn:
        job_A = job_repo.get_job_by_ticket_id(conn, tenant_id=tA_id, ticket_id=tA_ticket_id)
        job_B = job_repo.get_job_by_ticket_id(conn, tenant_id=tB_id, ticket_id=tB_ticket_id)
        assert job_A is not None
        assert job_B is not None
        assert job_A.id != job_B.id
        assert job_A.tenant_id == tA_id
        assert job_B.tenant_id == tB_id


@skip_no_test_db
def test_database_rejects_job_attaching_tenant_a_to_tenant_b_ticket() -> None:
    """Database composite foreign key rejects inserting a job referencing another tenant's ticket."""
    assert TEST_DB_URL is not None
    tA_id, _, _, _ = _create_tenant_and_api_key("Tenant A Isolation")
    tB_id, _, _, _ = _create_tenant_and_api_key("Tenant B Isolation")

    # Create Ticket for Tenant A
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.transaction():
            rec_A, _ = ticket_repo.create_or_get_ticket(
                conn=conn,
                tenant_id=tA_id,
                subject="Ticket for A",
                body="Body for A",
            )
            ticket_A_id = rec_A.id

    # Attempt to insert job with Tenant B's tenant_id and Tenant A's ticket_id
    with psycopg.connect(TEST_DB_URL) as conn:
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            with conn.transaction():
                job_repo.enqueue_job(
                    conn=conn,
                    tenant_id=tB_id,  # Tenant B
                    ticket_id=ticket_A_id,  # Ticket belongs to Tenant A!
                )


@skip_no_test_db
def test_tickets_without_external_id_each_receive_their_own_job(http_client: TestClient) -> None:
    """Unindexed tickets without external_id each receive their own independent job."""
    assert TEST_DB_URL is not None
    tenant_id, _, api_key, _ = _create_tenant_and_api_key("NoExt Jobs Corp")

    resp1 = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": api_key},
        json={"subject": "Inquiry 1", "body": "Body 1"},
    )
    resp2 = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": api_key},
        json={"subject": "Inquiry 2", "body": "Body 2"},
    )

    assert resp1.status_code == 202
    assert resp2.status_code == 202
    t1_id = uuid.UUID(resp1.json()["id"])
    t2_id = uuid.UUID(resp2.json()["id"])
    assert t1_id != t2_id

    with psycopg.connect(TEST_DB_URL) as conn:
        job1 = job_repo.get_job_by_ticket_id(conn, tenant_id=tenant_id, ticket_id=t1_id)
        job2 = job_repo.get_job_by_ticket_id(conn, tenant_id=tenant_id, ticket_id=t2_id)
        assert job1 is not None
        assert job2 is not None
        assert job1.id != job2.id
        assert job1.ticket_id == t1_id
        assert job2.ticket_id == t2_id
