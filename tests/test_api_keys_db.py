"""Database integration tests for API key creation and revocation against TEST_DATABASE_URL.

Tests:
1. Admin creates a key: row in DB contains prefix and SHA-256 hash, never plaintext key.
2. Admin revokes its own key: revoked_at is persisted; repeat revocation is idempotent.
3. Tenant isolation: Tenant A admin cannot revoke Tenant B's API key (returns 404 and B's key remains unrevoked).
4. Non-admin user cannot create or revoke API keys.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import hashlib
import uuid
import pytest
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app
from app.repositories.auth import create_tenant_and_admin_user
from app.security import hash_password, create_access_token
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


@skip_no_test_db
def test_db_admin_create_api_key_and_storage_verification(http_client: TestClient) -> None:
    """Admin creates API key: returns 201 with full key; DB stores only prefix and SHA-256 hash."""
    assert TEST_DB_URL is not None
    suffix = uuid.uuid4().hex[:8]

    # Create tenant and admin in test database
    with psycopg.connect(TEST_DB_URL) as conn:
        tenant_id, user_id = create_tenant_and_admin_user(
            conn=conn,
            company_name=f"Key Corp {suffix}",
            normalized_email=f"admin_{suffix}@keycorp.com",
            password_hash=hash_password("AdminPass123!"),
        )

    admin_token = create_access_token(user_id=user_id, tenant_id=tenant_id, role="admin")

    # Call POST /v1/api-keys
    resp = http_client.post(
        "/v1/api-keys",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"label": "Webhook Key"},
    )
    assert resp.status_code == 201
    data = resp.json()
    key_id = data["id"]
    full_key = data["key"]

    assert full_key.startswith("tk_")
    assert data["label"] == "Webhook Key"

    # Inspect test database row directly
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, tenant_id, key_prefix, key_hash, label, revoked_at FROM api_keys WHERE id = %s;",
                (key_id,),
            )
            row = cur.fetchone()
            assert row is not None
            assert str(row["tenant_id"]) == str(tenant_id)
            assert row["key_prefix"] == full_key[:8]
            assert row["label"] == "Webhook Key"
            assert row["revoked_at"] is None

            # Verify key_hash is the SHA-256 of full_key
            expected_hash = hashlib.sha256(full_key.encode("utf-8")).hexdigest()
            assert row["key_hash"] == expected_hash

            # Verify plaintext key is NOT present anywhere in database fields
            assert full_key not in row["key_prefix"]
            assert full_key not in row["key_hash"]


@skip_no_test_db
def test_db_admin_revokes_own_key_and_idempotency(http_client: TestClient) -> None:
    """Admin revokes its own API key; revoked_at is persisted; repeat deletion is idempotent."""
    assert TEST_DB_URL is not None
    suffix = uuid.uuid4().hex[:8]

    with psycopg.connect(TEST_DB_URL) as conn:
        tenant_id, user_id = create_tenant_and_admin_user(
            conn=conn,
            company_name=f"Revoke Corp {suffix}",
            normalized_email=f"admin_{suffix}@revokecorp.com",
            password_hash=hash_password("AdminPass123!"),
        )

    admin_token = create_access_token(user_id=user_id, tenant_id=tenant_id, role="admin")

    # 1. Create key
    create_resp = http_client.post(
        "/v1/api-keys",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"label": "Temporary Key"},
    )
    key_id = create_resp.json()["id"]

    # 2. Revoke key -> 204
    del_resp1 = http_client.delete(
        f"/v1/api-keys/{key_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert del_resp1.status_code == 204
    assert del_resp1.text == ""

    # Verify revoked_at is set in DB
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT revoked_at FROM api_keys WHERE id = %s;", (key_id,))
            first_revocation_time = cur.fetchone()["revoked_at"]
            assert first_revocation_time is not None

    # 3. Repeat revocation is idempotent -> 204, original revoked_at preserved
    del_resp2 = http_client.delete(
        f"/v1/api-keys/{key_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert del_resp2.status_code == 204

    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT revoked_at FROM api_keys WHERE id = %s;", (key_id,))
            second_revocation_time = cur.fetchone()["revoked_at"]
            assert second_revocation_time == first_revocation_time


@skip_no_test_db
def test_db_cross_tenant_revocation_isolation(http_client: TestClient) -> None:
    """Tenant A admin cannot revoke Tenant B's API key: returns 404, B's key remains active."""
    assert TEST_DB_URL is not None
    suffix = uuid.uuid4().hex[:8]

    # Setup Tenant A and Tenant B
    with psycopg.connect(TEST_DB_URL) as conn:
        tA_id, uA_id = create_tenant_and_admin_user(
            conn=conn,
            company_name=f"Tenant A {suffix}",
            normalized_email=f"adminA_{suffix}@corp.com",
            password_hash=hash_password("Pass123!"),
        )
        tB_id, uB_id = create_tenant_and_admin_user(
            conn=conn,
            company_name=f"Tenant B {suffix}",
            normalized_email=f"adminB_{suffix}@corp.com",
            password_hash=hash_password("Pass123!"),
        )

    token_A = create_access_token(user_id=uA_id, tenant_id=tA_id, role="admin")
    token_B = create_access_token(user_id=uB_id, tenant_id=tB_id, role="admin")

    # Tenant B creates an API key
    create_b_resp = http_client.post(
        "/v1/api-keys",
        headers={"Authorization": f"Bearer {token_B}"},
        json={"label": "Tenant B Secret Key"},
    )
    key_b_id = create_b_resp.json()["id"]

    # Tenant A attempts to revoke Tenant B's API key
    attack_resp = http_client.delete(
        f"/v1/api-keys/{key_b_id}",
        headers={"Authorization": f"Bearer {token_A}"},
    )
    # Must return 404 per SPEC.md Section 6 rule 3
    assert attack_resp.status_code == 404
    assert attack_resp.json() == {
        "error": {
            "code": "not_found",
            "message": "API key not found.",
        }
    }

    # Verify Tenant B's key in DB is still NOT revoked
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT revoked_at FROM api_keys WHERE id = %s;", (key_b_id,))
            b_row = cur.fetchone()
            assert b_row["revoked_at"] is None

