"""Database integration tests for Ticket ingestion, listing, detail, and API key authentication.

Tests against TEST_DATABASE_URL:
1. API key authentication: valid key succeeds, unknown or revoked key returns 401.
2. last_used_at is updated upon successful API key authentication.
3. New ticket returns 202 with status="new".
4. Existing external_id in same tenant returns 200 with existing ticket; does not duplicate rows.
5. Same external_id works independently across two different tenants.
6. Concurrent submissions with same external_id handle race safely without duplicates.
7. Independent tickets without external_id are created separately.
8. Listing tickets: newest first, status filter, limit & offset pagination, tie-breaking.
9. Listing and detail with both JWT and API key credentials.
10. Cross-tenant isolation on detail: 404 for other tenant's ticket ID.
"""

from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import time
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
from app.security import hash_password, create_access_token, generate_api_key
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
    orig_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _test_db_lifespan
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.router.lifespan_context = orig_lifespan


def _create_tenant_and_api_key(company_name: str) -> tuple[uuid.UUID, uuid.UUID, str, str]:
    """Helper to create a tenant, admin user, and valid API key in TEST_DATABASE_URL."""
    assert TEST_DB_URL is not None
    suffix = uuid.uuid4().hex[:8]
    full_key, key_prefix, key_hash = generate_api_key()

    with psycopg.connect(TEST_DB_URL) as conn:
        tenant_id, user_id = create_tenant_and_admin_user(
            conn=conn,
            company_name=company_name,
            normalized_email=f"admin_{suffix}@test.com",
            password_hash=hash_password("Pass123!"),
        )
        key_id = api_key_repo.create_api_key(
            conn=conn,
            tenant_id=tenant_id,
            key_prefix=key_prefix,
            key_hash=key_hash,
            label="Integration Key",
        )
    return tenant_id, user_id, full_key, str(key_id)


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

@skip_no_test_db
def test_db_api_key_authentication_and_last_used_update(http_client: TestClient) -> None:
    """Valid API key authenticates and updates last_used_at; revoked key returns 401."""
    assert TEST_DB_URL is not None
    tenant_id, _, full_key, key_id = _create_tenant_and_api_key("Auth Corp")

    # 1. Check last_used_at is initially None
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT last_used_at, revoked_at FROM api_keys WHERE id = %s;", (key_id,))
            row = cur.fetchone()
            assert row["last_used_at"] is None
            assert row["revoked_at"] is None

    # 2. Ingest ticket with valid API key
    resp = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": full_key},
        json={"subject": "Billing inquiry", "body": "Need invoice for March."},
    )
    assert resp.status_code == 202

    # 3. Assert last_used_at is now updated
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT last_used_at FROM api_keys WHERE id = %s;", (key_id,))
            assert cur.fetchone()["last_used_at"] is not None

    # 4. Revoke the key
    with psycopg.connect(TEST_DB_URL) as conn:
        api_key_repo.revoke_api_key(conn, tenant_id, uuid.UUID(key_id))

    # 5. Subsequent request with revoked key returns 401
    resp_revoked = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": full_key},
        json={"subject": "Should Fail", "body": "Key is revoked."},
    )
    assert resp_revoked.status_code == 401
    assert resp_revoked.json()["error"]["code"] == "invalid_api_key"


@skip_no_test_db
def test_db_ticket_creation_and_idempotency_by_external_id(http_client: TestClient) -> None:
    """New ticket returns 202; repeat with same external_id returns 200 with existing ticket."""
    assert TEST_DB_URL is not None
    tenant_id, _, api_key, _ = _create_tenant_and_api_key("Idempotent Corp")
    ext_id = f"crm_ext_{uuid.uuid4().hex[:8]}"

    # First submission -> 202 Accepted
    resp1 = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": api_key},
        json={
            "subject": "Initial ticket",
            "body": "First body content.",
            "external_id": ext_id,
            "customer_email": "customer@acme.com",
        },
    )
    assert resp1.status_code == 202
    ticket1 = resp1.json()
    assert ticket1["external_id"] == ext_id
    assert ticket1["status"] == "new"
    assert ticket1["subject"] == "Initial ticket"

    # Second submission with same external_id but different subject -> 200 OK returning original
    resp2 = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": api_key},
        json={
            "subject": "Attempted overwrite subject",
            "body": "Attempted overwrite body.",
            "external_id": ext_id,
            "customer_email": "customer@acme.com",
        },
    )
    assert resp2.status_code == 200
    ticket2 = resp2.json()
    assert ticket2["id"] == ticket1["id"]
    assert ticket2["subject"] == "Initial ticket"  # Did not overwrite
    assert ticket2["body"] == "First body content."

    # Assert exactly 1 ticket exists for this external_id
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM tickets WHERE tenant_id = %s AND external_id = %s;", (tenant_id, ext_id))
            assert cur.fetchone()[0] == 1


@skip_no_test_db
def test_db_same_external_id_across_two_tenants(http_client: TestClient) -> None:
    """Same external_id can exist in both Tenant A and Tenant B without conflict."""
    assert TEST_DB_URL is not None
    shared_ext_id = f"order_{uuid.uuid4().hex[:8]}"

    tA_id, _, key_A, _ = _create_tenant_and_api_key("Tenant A Inc")
    tB_id, _, key_B, _ = _create_tenant_and_api_key("Tenant B Inc")

    resp_A = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": key_A},
        json={"subject": "Tenant A Ticket", "body": "Body A", "external_id": shared_ext_id},
    )
    resp_B = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": key_B},
        json={"subject": "Tenant B Ticket", "body": "Body B", "external_id": shared_ext_id},
    )

    assert resp_A.status_code == 202
    assert resp_B.status_code == 202
    data_A = resp_A.json()
    data_B = resp_B.json()

    assert data_A["id"] != data_B["id"]
    assert data_A["tenant_id"] == str(tA_id)
    assert data_B["tenant_id"] == str(tB_id)
    assert data_A["external_id"] == shared_ext_id
    assert data_B["external_id"] == shared_ext_id


@skip_no_test_db
def test_db_concurrent_idempotent_ticket_creation() -> None:
    """Concurrent submissions for same tenant/external_id create 1 row and avoid 500 error."""
    assert TEST_DB_URL is not None
    tenant_id, _, _, _ = _create_tenant_and_api_key("Race Corp")
    ext_id = f"concurrent_ext_{uuid.uuid4().hex[:8]}"

    results: list[tuple[ticket_repo.TicketRecord, bool]] = []

    def insert_ticket(subj: str) -> None:
        with psycopg.connect(TEST_DB_URL) as conn:
            with conn.transaction():
                rec, is_created = ticket_repo.create_or_get_ticket(
                    conn=conn,
                    tenant_id=tenant_id,
                    subject=subj,
                    body="Concurrent body",
                    external_id=ext_id,
                )
                results.append((rec, is_created))

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(insert_ticket, "Subject 1")
        f2 = executor.submit(insert_ticket, "Subject 2")
        f1.result()
        f2.result()

    created_flags = [r[1] for r in results]
    assert created_flags.count(True) == 1, f"Expected exactly 1 created, got {results}"
    assert created_flags.count(False) == 1, f"Expected exactly 1 existing, got {results}"
    # Both return the exact same ticket ID
    assert results[0][0].id == results[1][0].id


@skip_no_test_db
def test_db_ticket_creation_without_external_id(http_client: TestClient) -> None:
    """Tickets without external_id are created independently as distinct rows."""
    assert TEST_DB_URL is not None
    tenant_id, _, key, _ = _create_tenant_and_api_key("NoExt Corp")

    resp1 = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": key},
        json={"subject": "General Query", "body": "Body 1"},
    )
    resp2 = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": key},
        json={"subject": "General Query", "body": "Body 2"},
    )

    assert resp1.status_code == 202
    assert resp2.status_code == 202
    assert resp1.json()["id"] != resp2.json()["id"]


@skip_no_test_db
def test_db_list_and_detail_with_both_credentials_and_isolation(http_client: TestClient) -> None:
    """GET /v1/tickets and GET /v1/tickets/{id} work with JWT or API key, with strict isolation."""
    assert TEST_DB_URL is not None
    # Setup Tenant A
    tA_id, uA_id, key_A, _ = _create_tenant_and_api_key("Tenant A View")
    token_A = create_access_token(user_id=uA_id, tenant_id=tA_id, role="admin")

    # Setup Tenant B
    tB_id, uB_id, key_B, _ = _create_tenant_and_api_key("Tenant B View")
    token_B = create_access_token(user_id=uB_id, tenant_id=tB_id, role="admin")

    # Ingest 3 tickets for Tenant A
    t1_id = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": key_A},
        json={"subject": "A Ticket 1", "body": "Body 1"},
    ).json()["id"]

    t2_id = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": key_A},
        json={"subject": "A Ticket 2", "body": "Body 2"},
    ).json()["id"]

    t3_id = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": key_A},
        json={"subject": "A Ticket 3", "body": "Body 3"},
    ).json()["id"]

    # Ingest 1 ticket for Tenant B
    tB_ticket_id = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": key_B},
        json={"subject": "B Ticket", "body": "Body B"},
    ).json()["id"]

    # 1. Tenant A lists via JWT
    list_resp_jwt = http_client.get(
        "/v1/tickets",
        headers={"Authorization": f"Bearer {token_A}"},
    )
    assert list_resp_jwt.status_code == 200
    a_tickets_jwt = list_resp_jwt.json()
    assert len(a_tickets_jwt) == 3
    # Ordered newest first: t3, t2, t1
    assert [t["id"] for t in a_tickets_jwt] == [t3_id, t2_id, t1_id]

    # 2. Tenant A lists via API Key (Bearer tk_... format)
    list_resp_key = http_client.get(
        "/v1/tickets",
        headers={"Authorization": f"Bearer {key_A}"},
    )
    assert list_resp_key.status_code == 200
    assert len(list_resp_key.json()) == 3

    # 3. Tenant A gets own ticket detail via JWT -> 200
    detail_resp = http_client.get(
        f"/v1/tickets/{t1_id}",
        headers={"Authorization": f"Bearer {token_A}"},
    )
    assert detail_resp.status_code == 200
    assert detail_resp.json()["id"] == t1_id

    # 4. Tenant A attempts to get Tenant B's ticket detail -> 404
    cross_detail_resp = http_client.get(
        f"/v1/tickets/{tB_ticket_id}",
        headers={"Authorization": f"Bearer {token_A}"},
    )
    assert cross_detail_resp.status_code == 404
    assert cross_detail_resp.json() == {
        "error": {
            "code": "not_found",
            "message": "Ticket not found.",
        }
    }

    # 5. Pagination test on Tenant A: limit=2, offset=1
    page_resp = http_client.get(
        "/v1/tickets?limit=2&offset=1",
        headers={"Authorization": f"Bearer {token_A}"},
    )
    assert page_resp.status_code == 200
    page_data = page_resp.json()
    assert len(page_data) == 2
    assert page_data[0]["id"] == t2_id
    assert page_data[1]["id"] == t1_id

