"""PostgreSQL integration tests for CSV bulk ticket ingestion (POST /v1/tickets/bulk).

Tests against TEST_DATABASE_URL:
1. Valid CSV ingestion: creates tickets and matching queued jobs atomically.
2. Quoted commas and newlines: preserves exact contents without parsing errors.
3. Duplicate external_id within same CSV: creates 1 ticket and 1 job, replays second without duplicate job.
4. Idempotency against pre-existing tickets: replays existing ticket, no duplicate job enqueued.
5. Cross-tenant isolation: identical external_ids in two tenants create independent tickets and jobs.
6. Whole-file rejection: invalid row rejects entire request with 422 and commits zero rows.
"""

from collections.abc import AsyncIterator
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
from app.repositories.auth import create_tenant_and_admin_user
import app.repositories.api_keys as api_key_repo
import app.repositories.tickets as ticket_repo
import app.repositories.jobs as job_repo
from app.security import hash_password, generate_api_key
import app.db as app_db
from tests.test_auth_db import TEST_DB_URL, SAFETY_STATUS

skip_no_test_db = pytest.mark.skipif(TEST_DB_URL is None, reason=SAFETY_STATUS)


@asynccontextmanager
async def _test_db_lifespan(test_app: FastAPI) -> AsyncIterator[None]:
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
    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _test_db_lifespan
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.router.lifespan_context = original_lifespan


def _create_test_tenant_and_api_key(company_suffix: str) -> tuple[uuid.UUID, str]:
    """Helper to register a tenant and generate an API key."""
    assert TEST_DB_URL is not None
    unique_company = f"Bulk Co {company_suffix} {uuid.uuid4().hex[:6]}"
    unique_email = f"admin_{company_suffix}_{uuid.uuid4().hex[:6]}@example.com"
    full_key, key_prefix, key_hash = generate_api_key()

    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.transaction():
            tenant_id, admin_user_id = create_tenant_and_admin_user(
                conn=conn,
                company_name=unique_company,
                normalized_email=unique_email,
                password_hash=hash_password("ValidPassword123!"),
            )
            api_key_repo.create_api_key(
                conn=conn,
                tenant_id=tenant_id,
                key_prefix=key_prefix,
                key_hash=key_hash,
                label=f"Bulk Ingestion Key {company_suffix}",
            )
            return tenant_id, full_key


@skip_no_test_db
def test_db_bulk_tickets_valid_ingestion(http_client: TestClient) -> None:
    """Valid CSV creates tickets and atomic matching queued jobs in database."""
    tenant_id, api_key = _create_test_tenant_and_api_key("ValidIngest")

    csv_data = (
        "subject,body,external_id,customer_email\n"
        "Broken checkout,Cannot complete payment,ext-checkout-1,buyer@example.com\n"
        "Missing invoice,Where is my receipt?,,\n"
        "App crash,Crashing on start,ext-crash-2,user@domain.com\n"
    )

    resp = http_client.post(
        "/v1/tickets/bulk",
        content=csv_data,
        headers={
            "X-API-Key": api_key,
            "Content-Type": "text/csv",
        },
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["received"] == 3
    assert data["created"] == 3
    assert data["replayed"] == 0
    assert len(data["ticket_ids"]) == 3

    created_uuids = [uuid.UUID(tid) for tid in data["ticket_ids"]]

    # Verify directly in test database
    with psycopg.connect(TEST_DB_URL, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            # Check tickets table
            cur.execute(
                "SELECT id, tenant_id, external_id, subject, status FROM tickets WHERE tenant_id = %s ORDER BY created_at ASC;",
                (tenant_id,),
            )
            tickets = cur.fetchall()
            assert len(tickets) == 3
            assert set(t["id"] for t in tickets) == set(created_uuids)
            assert all(t["status"] == "new" for t in tickets)

            # Check jobs table
            cur.execute(
                "SELECT id, tenant_id, ticket_id, type, status, attempts FROM jobs WHERE tenant_id = %s ORDER BY created_at ASC;",
                (tenant_id,),
            )
            jobs = cur.fetchall()
            assert len(jobs) == 3
            assert all(j["type"] == "ticket_ingest" for j in jobs)
            assert all(j["status"] == "queued" for j in jobs)
            assert all(j["attempts"] == 0 for j in jobs)
            job_ticket_ids = {j["ticket_id"] for j in jobs}
            assert job_ticket_ids == set(created_uuids)


@skip_no_test_db
def test_db_bulk_tickets_quoted_commas_and_newlines(http_client: TestClient) -> None:
    """Quoted commas and embedded newlines are preserved accurately."""
    tenant_id, api_key = _create_test_tenant_and_api_key("QuotedCSV")

    csv_data = (
        'subject,body,external_id,customer_email\n'
        '"Error, code 404, page missing","Detailed issue:\nLine 1: 404 error\nLine 2: try again later.",ext-quoted,quoted@example.com\n'
    )

    resp = http_client.post(
        "/v1/tickets/bulk",
        content=csv_data,
        headers={
            "X-API-Key": api_key,
            "Content-Type": "text/csv",
        },
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["received"] == 1
    assert data["created"] == 1
    ticket_id = uuid.UUID(data["ticket_ids"][0])

    with psycopg.connect(TEST_DB_URL, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT subject, body FROM tickets WHERE tenant_id = %s AND id = %s;",
                (tenant_id, ticket_id),
            )
            row = cur.fetchone()
            assert row is not None
            assert row["subject"] == "Error, code 404, page missing"
            assert row["body"] == "Detailed issue:\nLine 1: 404 error\nLine 2: try again later."


@skip_no_test_db
def test_db_bulk_tickets_duplicate_external_id_in_same_file(http_client: TestClient) -> None:
    """Duplicate external_id within the same CSV creates 1 ticket/job and replays the duplicate."""
    tenant_id, api_key = _create_test_tenant_and_api_key("InternalDups")

    csv_data = (
        "subject,body,external_id\n"
        "First appearance,Body one,dup-external-ref\n"
        "Unrelated ticket,Body two,other-ref\n"
        "Second appearance,Body three,dup-external-ref\n"
    )

    resp = http_client.post(
        "/v1/tickets/bulk",
        content=csv_data,
        headers={
            "X-API-Key": api_key,
            "Content-Type": "text/csv",
        },
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["received"] == 3
    assert data["created"] == 2
    assert data["replayed"] == 1
    assert len(data["ticket_ids"]) == 3
    # Rows 0 and 2 reference the exact same ticket
    assert data["ticket_ids"][0] == data["ticket_ids"][2]

    # Verify only 2 tickets and 2 jobs exist in DB
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM tickets WHERE tenant_id = %s;", (tenant_id,))
            assert cur.fetchone()[0] == 2
            cur.execute("SELECT count(*) FROM jobs WHERE tenant_id = %s;", (tenant_id,))
            assert cur.fetchone()[0] == 2


@skip_no_test_db
def test_db_bulk_tickets_idempotent_replay_against_existing_tickets(http_client: TestClient) -> None:
    """Replaying pre-existing external_id returns existing ticket without creating a new job."""
    tenant_id, api_key = _create_test_tenant_and_api_key("PreExisting")

    # Pre-create ticket via single-ticket endpoint
    single_resp = http_client.post(
        "/v1/tickets",
        json={
            "subject": "Pre-existing ticket",
            "body": "Pre-existing body",
            "external_id": "pre-existing-ref",
        },
        headers={"X-API-Key": api_key},
    )
    assert single_resp.status_code == 202
    pre_existing_id = single_resp.json()["id"]

    # Now upload CSV containing the pre-existing external_id and a brand-new external_id
    csv_data = (
        "subject,body,external_id\n"
        "Attempt to overwrite,New body,pre-existing-ref\n"
        "Brand new ticket,Fresh body,fresh-ref\n"
    )

    resp = http_client.post(
        "/v1/tickets/bulk",
        content=csv_data,
        headers={
            "X-API-Key": api_key,
            "Content-Type": "text/csv",
        },
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["received"] == 2
    assert data["created"] == 1
    assert data["replayed"] == 1
    assert data["ticket_ids"][0] == pre_existing_id

    # Verify database state
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            # Exactly 2 tickets exist for this tenant
            cur.execute("SELECT count(*) FROM tickets WHERE tenant_id = %s;", (tenant_id,))
            assert cur.fetchone()[0] == 2
            # Exactly 2 jobs exist for this tenant (1 from pre-create, 1 from bulk upload)
            cur.execute("SELECT count(*) FROM jobs WHERE tenant_id = %s;", (tenant_id,))
            assert cur.fetchone()[0] == 2


@skip_no_test_db
def test_db_bulk_tickets_cross_tenant_isolation(http_client: TestClient) -> None:
    """Same external_id across two tenants creates distinct tickets and jobs for each tenant."""
    tenant_a_id, api_key_a = _create_test_tenant_and_api_key("TenantA")
    tenant_b_id, api_key_b = _create_test_tenant_and_api_key("TenantB")

    shared_csv = (
        "subject,body,external_id\n"
        "Shared order,Support request,shared-order-123\n"
    )

    resp_a = http_client.post(
        "/v1/tickets/bulk",
        content=shared_csv,
        headers={"X-API-Key": api_key_a, "Content-Type": "text/csv"},
    )
    assert resp_a.status_code == 202
    ticket_a_id = resp_a.json()["ticket_ids"][0]

    resp_b = http_client.post(
        "/v1/tickets/bulk",
        content=shared_csv,
        headers={"X-API-Key": api_key_b, "Content-Type": "text/csv"},
    )
    assert resp_b.status_code == 202
    ticket_b_id = resp_b.json()["ticket_ids"][0]

    assert ticket_a_id != ticket_b_id

    # Verify each tenant sees only their own ticket via GET /v1/tickets
    list_a = http_client.get("/v1/tickets", headers={"X-API-Key": api_key_a})
    assert list_a.status_code == 200
    assert len(list_a.json()) == 1
    assert list_a.json()[0]["id"] == ticket_a_id

    list_b = http_client.get("/v1/tickets", headers={"X-API-Key": api_key_b})
    assert list_b.status_code == 200
    assert len(list_b.json()) == 1
    assert list_b.json()[0]["id"] == ticket_b_id


@skip_no_test_db
def test_db_bulk_tickets_atomic_rollback_on_invalid_row(http_client: TestClient) -> None:
    """Invalid row in CSV causes whole-file rejection with zero writes committed."""
    tenant_id, api_key = _create_test_tenant_and_api_key("AtomicFail")

    csv_data = (
        "subject,body,external_id\n"
        "Valid row 1,Valid body 1,ext-atomic-1\n"
        "   ,Invalid blank subject row 2,ext-atomic-2\n"
    )

    resp = http_client.post(
        "/v1/tickets/bulk",
        content=csv_data,
        headers={"X-API-Key": api_key, "Content-Type": "text/csv"},
    )
    assert resp.status_code == 422
    assert "Row 2: subject is required" in resp.json()["error"]["message"]

    # Verify that row 1 was NOT inserted
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM tickets WHERE tenant_id = %s;", (tenant_id,))
            assert cur.fetchone()[0] == 0
            cur.execute("SELECT count(*) FROM jobs WHERE tenant_id = %s;", (tenant_id,))
            assert cur.fetchone()[0] == 0


@skip_no_test_db
def test_db_bulk_tickets_mid_batch_enqueue_failure_rolls_back_entire_batch(http_client: TestClient) -> None:
    """Forces an enqueue failure on row 2 after row 1 was inserted; verifies full rollback in database."""
    tenant_id, api_key = _create_test_tenant_and_api_key("MidBatchFail")

    csv_data = (
        "subject,body,external_id\n"
        "Row 1 Subject,Row 1 Body,ext-rollback-1\n"
        "Row 2 Subject,Row 2 Body,ext-rollback-2\n"
    )

    real_enqueue_job = job_repo.enqueue_job
    call_count = 0

    def mock_enqueue(conn, tenant_id, ticket_id, job_type="ticket_ingest", run_after=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Let row 1 succeed normally in PostgreSQL
            return real_enqueue_job(conn, tenant_id, ticket_id, job_type, run_after)
        else:
            # Force failure on row 2
            raise RuntimeError("Simulated mid-batch queue insertion failure on row 2")

    with patch("app.routers.tickets.job_repo.enqueue_job", side_effect=mock_enqueue):
        with pytest.raises(RuntimeError, match="Simulated mid-batch queue insertion failure on row 2"):
            http_client.post(
                "/v1/tickets/bulk",
                content=csv_data,
                headers={"X-API-Key": api_key, "Content-Type": "text/csv"},
            )

    # Verify that NEITHER row 1 nor row 2 was committed in tickets or jobs
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM tickets WHERE tenant_id = %s;", (tenant_id,))
            ticket_count = cur.fetchone()[0]
            assert ticket_count == 0, f"Expected 0 tickets, found {ticket_count} committed despite mid-batch failure!"

            cur.execute("SELECT count(*) FROM jobs WHERE tenant_id = %s;", (tenant_id,))
            job_count = cur.fetchone()[0]
            assert job_count == 0, f"Expected 0 jobs, found {job_count} committed despite mid-batch failure!"

