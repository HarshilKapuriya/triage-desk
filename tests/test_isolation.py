"""End-to-end two-tenant isolation and credential boundary tests.

Per SPEC.md Section 10:
1. Sign up tenant A and tenant B. Create one API key for each.
2. Using A's key, create three tickets. Using B's key, create two tickets.
3. List tickets with A's credentials (API key & JWT) and assert that exactly
   A's three tickets come back and none of B's ids appear. Repeat with B's credentials.
4. Request each of B's ticket ids with A's credentials and assert 404.
   Repeat in the other direction.
5. Using A's admin token, try to revoke B's API key id and assert 404,
   then confirm B's key still works.
6. Assert that the same external_id can exist in both tenants without conflict.
7. Revoke a tenant's own key and prove it now returns 401 on ticket create, list, and detail.

Dual-header precedence:
- Tests deterministic resolution when X-API-Key and Authorization are both supplied.
- Ensures single-source resolution: X-API-Key takes strict precedence and cannot
  mix tenant from one credential with identity/role from another, nor fall back to JWT.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import uuid
import pytest
from psycopg_pool import ConnectionPool
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app
import app.db as app_db
from tests.test_auth_db import TEST_DB_URL, SAFETY_STATUS

skip_no_test_db = pytest.mark.skipif(TEST_DB_URL is None, reason=SAFETY_STATUS)


@asynccontextmanager
async def _isolation_test_db_lifespan(test_app: FastAPI) -> AsyncIterator[None]:
    """Lifespan fixture connecting app.db to the verified TEST_DATABASE_URL."""
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
    """FastAPI TestClient wired to the real test database pool.

    No mocks or test doubles are used for authentication or repository layers.
    """
    orig_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _isolation_test_db_lifespan
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.router.lifespan_context = orig_lifespan


@skip_no_test_db
def test_two_tenant_isolation_full_lifecycle(http_client: TestClient) -> None:
    """Execute the complete two-tenant isolation scenario specified in SPEC.md Section 10."""
    assert TEST_DB_URL is not None
    rand_suffix = uuid.uuid4().hex[:8]

    # 1. Sign up Tenant A and Tenant B via actual HTTP endpoints
    email_a = f"admin_a_{rand_suffix}@corp-a.com"
    email_b = f"admin_b_{rand_suffix}@corp-b.com"

    resp_signup_a = http_client.post(
        "/v1/auth/signup",
        json={"company_name": f"Corp A {rand_suffix}", "email": email_a, "password": "PasswordA123!"},
    )
    assert resp_signup_a.status_code == 201, resp_signup_a.text
    data_a = resp_signup_a.json()
    tenant_a_id = data_a["tenant_id"]
    jwt_a = data_a["access_token"]

    resp_signup_b = http_client.post(
        "/v1/auth/signup",
        json={"company_name": f"Corp B {rand_suffix}", "email": email_b, "password": "PasswordB123!"},
    )
    assert resp_signup_b.status_code == 201, resp_signup_b.text
    data_b = resp_signup_b.json()
    tenant_b_id = data_b["tenant_id"]
    jwt_b = data_b["access_token"]

    assert tenant_a_id != tenant_b_id

    # Create one API key for Tenant A
    resp_key_a = http_client.post(
        "/v1/api-keys",
        headers={"Authorization": f"Bearer {jwt_a}"},
        json={"label": "Tenant A Ingest Key"},
    )
    assert resp_key_a.status_code == 201, resp_key_a.text
    key_a_data = resp_key_a.json()
    key_a_id = key_a_data["id"]
    key_a_secret = key_a_data["key"]

    # Create one API key for Tenant B
    resp_key_b = http_client.post(
        "/v1/api-keys",
        headers={"Authorization": f"Bearer {jwt_b}"},
        json={"label": "Tenant B Ingest Key"},
    )
    assert resp_key_b.status_code == 201, resp_key_b.text
    key_b_data = resp_key_b.json()
    key_b_id = key_b_data["id"]
    key_b_secret = key_b_data["key"]

    assert key_a_id != key_b_id

    # 2 & 6. Create three tickets with A's key, two with B's key.
    # Share one identical external_id across both tenants to verify multi-tenant scoping.
    shared_ext_id = f"EXT-ORDER-{rand_suffix}"

    # A's 3 tickets
    resp_t_a1 = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": key_a_secret},
        json={"subject": "Tenant A - Ticket 1", "body": "Body A1", "external_id": shared_ext_id},
    )
    assert resp_t_a1.status_code == 202
    ticket_a1 = resp_t_a1.json()

    resp_t_a2 = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": key_a_secret},
        json={"subject": "Tenant A - Ticket 2", "body": "Body A2", "external_id": f"EXT-A2-{rand_suffix}"},
    )
    assert resp_t_a2.status_code == 202
    ticket_a2 = resp_t_a2.json()

    resp_t_a3 = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": key_a_secret},
        json={"subject": "Tenant A - Ticket 3", "body": "Body A3"},
    )
    assert resp_t_a3.status_code == 202
    ticket_a3 = resp_t_a3.json()

    # B's 2 tickets (B1 uses identical shared_ext_id)
    resp_t_b1 = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": key_b_secret},
        json={"subject": "Tenant B - Ticket 1", "body": "Body B1", "external_id": shared_ext_id},
    )
    assert resp_t_b1.status_code == 202
    ticket_b1 = resp_t_b1.json()

    resp_t_b2 = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": key_b_secret},
        json={"subject": "Tenant B - Ticket 2", "body": "Body B2"},
    )
    assert resp_t_b2.status_code == 202
    ticket_b2 = resp_t_b2.json()

    # Step 6 assertions: Same external_id coexists in both tenants independently
    assert ticket_a1["external_id"] == shared_ext_id
    assert ticket_b1["external_id"] == shared_ext_id
    assert ticket_a1["id"] != ticket_b1["id"]
    assert ticket_a1["tenant_id"] == tenant_a_id
    assert ticket_b1["tenant_id"] == tenant_b_id

    # Re-submitting shared_ext_id for Tenant A returns 200 with ticket_a1
    resp_replay_a = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": key_a_secret},
        json={"subject": "Replay A", "body": "Replay Body", "external_id": shared_ext_id},
    )
    assert resp_replay_a.status_code == 200
    assert resp_replay_a.json()["id"] == ticket_a1["id"]

    # Re-submitting shared_ext_id for Tenant B returns 200 with ticket_b1
    resp_replay_b = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": key_b_secret},
        json={"subject": "Replay B", "body": "Replay Body", "external_id": shared_ext_id},
    )
    assert resp_replay_b.status_code == 200
    assert resp_replay_b.json()["id"] == ticket_b1["id"]

    a_ids = {ticket_a1["id"], ticket_a2["id"], ticket_a3["id"]}
    b_ids = {ticket_b1["id"], ticket_b2["id"]}
    assert a_ids.isdisjoint(b_ids)

    # 3. List using each tenant's API key and JWT
    # 3a. Tenant A via API Key
    resp_list_a_key = http_client.get(
        "/v1/tickets",
        headers={"X-API-Key": key_a_secret},
    )
    assert resp_list_a_key.status_code == 200
    list_a_key = resp_list_a_key.json()
    assert isinstance(list_a_key, list)
    assert len(list_a_key) == 3
    found_a_key_ids = {t["id"] for t in list_a_key}
    assert found_a_key_ids == a_ids
    assert found_a_key_ids.isdisjoint(b_ids)
    assert all(t["tenant_id"] == tenant_a_id for t in list_a_key)

    # 3b. Tenant A via JWT
    resp_list_a_jwt = http_client.get(
        "/v1/tickets",
        headers={"Authorization": f"Bearer {jwt_a}"},
    )
    assert resp_list_a_jwt.status_code == 200
    list_a_jwt = resp_list_a_jwt.json()
    assert isinstance(list_a_jwt, list)
    assert len(list_a_jwt) == 3
    found_a_jwt_ids = {t["id"] for t in list_a_jwt}
    assert found_a_jwt_ids == a_ids
    assert found_a_jwt_ids.isdisjoint(b_ids)
    # Check ordering: newest first (ticket_a3, ticket_a2, ticket_a1)
    assert [t["id"] for t in list_a_jwt] == [ticket_a3["id"], ticket_a2["id"], ticket_a1["id"]]

    # 3c. Tenant B via API Key
    resp_list_b_key = http_client.get(
        "/v1/tickets",
        headers={"X-API-Key": key_b_secret},
    )
    assert resp_list_b_key.status_code == 200
    list_b_key = resp_list_b_key.json()
    assert isinstance(list_b_key, list)
    assert len(list_b_key) == 2
    found_b_key_ids = {t["id"] for t in list_b_key}
    assert found_b_key_ids == b_ids
    assert found_b_key_ids.isdisjoint(a_ids)
    assert all(t["tenant_id"] == tenant_b_id for t in list_b_key)

    # 3d. Tenant B via JWT
    resp_list_b_jwt = http_client.get(
        "/v1/tickets",
        headers={"Authorization": f"Bearer {jwt_b}"},
    )
    assert resp_list_b_jwt.status_code == 200
    list_b_jwt = resp_list_b_jwt.json()
    assert isinstance(list_b_jwt, list)
    assert len(list_b_jwt) == 2
    found_b_jwt_ids = {t["id"] for t in list_b_jwt}
    assert found_b_jwt_ids == b_ids
    assert found_b_jwt_ids.isdisjoint(a_ids)
    assert [t["id"] for t in list_b_jwt] == [ticket_b2["id"], ticket_b1["id"]]

    # 4. Fetch the other tenant's ticket IDs in both directions (API Key & JWT)
    # Tenant A attempts to fetch Tenant B's tickets -> 404
    for b_id in b_ids:
        # A via API key
        resp_a_cross_key = http_client.get(
            f"/v1/tickets/{b_id}",
            headers={"X-API-Key": key_a_secret},
        )
        assert resp_a_cross_key.status_code == 404
        assert resp_a_cross_key.json() == {
            "error": {"code": "not_found", "message": "Ticket not found."}
        }

        # A via JWT
        resp_a_cross_jwt = http_client.get(
            f"/v1/tickets/{b_id}",
            headers={"Authorization": f"Bearer {jwt_a}"},
        )
        assert resp_a_cross_jwt.status_code == 404
        assert resp_a_cross_jwt.json() == {
            "error": {"code": "not_found", "message": "Ticket not found."}
        }

    # Tenant B attempts to fetch Tenant A's tickets -> 404
    for a_id in a_ids:
        # B via API key
        resp_b_cross_key = http_client.get(
            f"/v1/tickets/{a_id}",
            headers={"X-API-Key": key_b_secret},
        )
        assert resp_b_cross_key.status_code == 404
        assert resp_b_cross_key.json() == {
            "error": {"code": "not_found", "message": "Ticket not found."}
        }

        # B via JWT
        resp_b_cross_jwt = http_client.get(
            f"/v1/tickets/{a_id}",
            headers={"Authorization": f"Bearer {jwt_b}"},
        )
        assert resp_b_cross_jwt.status_code == 404
        assert resp_b_cross_jwt.json() == {
            "error": {"code": "not_found", "message": "Ticket not found."}
        }

    # 5. Using A's admin JWT, attempt to revoke B's API key id
    resp_cross_revoke = http_client.delete(
        f"/v1/api-keys/{key_b_id}",
        headers={"Authorization": f"Bearer {jwt_a}"},
    )
    assert resp_cross_revoke.status_code == 404
    assert resp_cross_revoke.json() == {
        "error": {"code": "not_found", "message": "API key not found."}
    }

    # Prove B's key still works for listing and detail
    resp_b_still_works = http_client.get(
        "/v1/tickets",
        headers={"X-API-Key": key_b_secret},
    )
    assert resp_b_still_works.status_code == 200
    assert len(resp_b_still_works.json()) == 2

    resp_b_detail_works = http_client.get(
        f"/v1/tickets/{ticket_b1['id']}",
        headers={"X-API-Key": key_b_secret},
    )
    assert resp_b_detail_works.status_code == 200

    # 7. Revoke Tenant A's own key and prove it now returns 401 on create, list, and detail
    resp_revoke_own = http_client.delete(
        f"/v1/api-keys/{key_a_id}",
        headers={"Authorization": f"Bearer {jwt_a}"},
    )
    assert resp_revoke_own.status_code == 204

    # Idempotent re-revocation
    resp_revoke_again = http_client.delete(
        f"/v1/api-keys/{key_a_id}",
        headers={"Authorization": f"Bearer {jwt_a}"},
    )
    assert resp_revoke_again.status_code == 204

    # Revoked key returns 401 on ticket create
    resp_create_revoked = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": key_a_secret},
        json={"subject": "Revoked Subject", "body": "Revoked Body"},
    )
    assert resp_create_revoked.status_code == 401
    assert resp_create_revoked.json()["error"]["code"] == "invalid_api_key"

    # Revoked key returns 401 on ticket list
    resp_list_revoked = http_client.get(
        "/v1/tickets",
        headers={"X-API-Key": key_a_secret},
    )
    assert resp_list_revoked.status_code == 401
    assert resp_list_revoked.json()["error"]["code"] == "invalid_api_key"

    # Revoked key returns 401 on ticket detail
    resp_detail_revoked = http_client.get(
        f"/v1/tickets/{ticket_a1['id']}",
        headers={"X-API-Key": key_a_secret},
    )
    assert resp_detail_revoked.status_code == 401
    assert resp_detail_revoked.json()["error"]["code"] == "invalid_api_key"

    # Confirm Tenant B's key remains completely valid and unaffected
    resp_b_unaffected = http_client.get(
        "/v1/tickets",
        headers={"X-API-Key": key_b_secret},
    )
    assert resp_b_unaffected.status_code == 200
    assert len(resp_b_unaffected.json()) == 2

    # Confirm Tenant A can still access tickets via JWT
    resp_a_jwt_still_works = http_client.get(
        "/v1/tickets",
        headers={"Authorization": f"Bearer {jwt_a}"},
    )
    assert resp_a_jwt_still_works.status_code == 200
    assert len(resp_a_jwt_still_works.json()) == 3


@skip_no_test_db
def test_dual_header_precedence_and_credential_isolation(http_client: TestClient) -> None:
    """Test deterministic dual-header precedence and prevent credential/tenant mixing."""
    assert TEST_DB_URL is not None
    rand_suffix = uuid.uuid4().hex[:8]

    # Create Tenant Alpha
    resp_a = http_client.post(
        "/v1/auth/signup",
        json={"company_name": f"Alpha Corp {rand_suffix}", "email": f"alpha_{rand_suffix}@corp.com", "password": "PasswordA123!"},
    )
    assert resp_a.status_code == 201
    tenant_a_id = resp_a.json()["tenant_id"]
    jwt_a = resp_a.json()["access_token"]

    # Create Tenant Beta
    resp_b = http_client.post(
        "/v1/auth/signup",
        json={"company_name": f"Beta Corp {rand_suffix}", "email": f"beta_{rand_suffix}@corp.com", "password": "PasswordB123!"},
    )
    assert resp_b.status_code == 201
    tenant_b_id = resp_b.json()["tenant_id"]

    # Create API key for Tenant Beta
    resp_key_b = http_client.post(
        "/v1/api-keys",
        headers={"Authorization": f"Bearer {resp_b.json()['access_token']}"},
        json={"label": "Beta Key"},
    )
    assert resp_key_b.status_code == 201
    key_b_secret = resp_key_b.json()["key"]

    # Ingest 1 ticket for Tenant Beta using Beta's key
    resp_t_b = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": key_b_secret},
        json={"subject": "Beta Ticket", "body": "Beta ticket content"},
    )
    assert resp_t_b.status_code == 202
    ticket_b_id = resp_t_b.json()["id"]

    # Scenario 1: Dual headers supplied with conflicting tenants
    # X-API-Key is Tenant Beta, Authorization Bearer is Tenant Alpha admin JWT
    # X-API-Key must take strict precedence: resolves strictly as Tenant Beta.
    # It must NOT resolve as Tenant Alpha, nor mix Alpha's admin role into Beta's key.
    dual_headers = {
        "X-API-Key": key_b_secret,
        "Authorization": f"Bearer {jwt_a}",
    }
    resp_list = http_client.get("/v1/tickets", headers=dual_headers)
    assert resp_list.status_code == 200
    tickets = resp_list.json()
    assert isinstance(tickets, list)
    assert len(tickets) == 1
    assert tickets[0]["id"] == ticket_b_id
    assert tickets[0]["tenant_id"] == tenant_b_id

    # Scenario 2: Ingestion with dual conflicting headers
    # X-API-Key (Beta) + Authorization (Alpha) -> creates ticket under Tenant Beta
    resp_ingest_dual = http_client.post(
        "/v1/tickets",
        headers=dual_headers,
        json={"subject": "Dual Header Ticket", "body": "Body content"},
    )
    assert resp_ingest_dual.status_code == 202
    assert resp_ingest_dual.json()["tenant_id"] == tenant_b_id

    # Scenario 3: Invalid X-API-Key supplied with valid Authorization header
    # X-API-Key is present but invalid. It must NOT fall back to Authorization.
    # Must fail immediately with 401 invalid_api_key.
    conflicting_invalid_key_headers = {
        "X-API-Key": "tk_invalid_key_does_not_exist_at_all_12345",
        "Authorization": f"Bearer {jwt_a}",
    }
    resp_invalid_dual = http_client.get("/v1/tickets", headers=conflicting_invalid_key_headers)
    assert resp_invalid_dual.status_code == 401
    assert resp_invalid_dual.json()["error"]["code"] == "invalid_api_key"

    # Scenario 4: Admin-only route with dual headers
    # POST /v1/api-keys is strictly admin-only and requires user JWT.
    # An API key cannot create API keys.
    # If only X-API-Key is supplied to /v1/api-keys -> 401 missing_authorization_header
    resp_key_only = http_client.post(
        "/v1/api-keys",
        headers={"X-API-Key": key_b_secret},
        json={"label": "Unauthorized Key Creation Attempt"},
    )
    assert resp_key_only.status_code == 401
    assert resp_key_only.json()["error"]["code"] == "missing_authorization_header"

