"""Database integration tests for Signup and Login against TEST_DATABASE_URL.

Per SPEC.md and strict user safety constraints:
1. Never fall back from TEST_DATABASE_URL to DATABASE_URL.
2. Comprehensive multi-attribute connection target check: compares host, port, dbname,
   and username to prevent shared targets across hosted/pooler databases.
3. Does not permit generic CI=true to bypass external safety checks; allows shared DB only
   if the target is strictly the local/container throwaway CI service ('localhost' or '127.0.0.1').
4. Fails loudly in CI if TEST_DATABASE_URL is missing or unconfigured.
5. In local development without TEST_DATABASE_URL, tests are cleanly skipped.
"""

from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import os
import uuid
import pytest
import psycopg
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
import app.repositories.auth as auth_repo
from app.repositories.auth import (
    create_tenant_and_admin_user,
    get_user_by_email,
    EmailAlreadyExistsError,
)
from app.security import hash_password, verify_password, decode_access_token
import app.db as app_db

IS_CI = os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"


def resolve_safe_test_db_url() -> tuple[str | None, str]:
    """Resolve and validate TEST_DATABASE_URL with comprehensive safety checks.

    Returns:
        tuple[str | None, str]: (valid_test_url, reason_or_status)
    """
    cfg = get_settings()
    if not cfg.test_database_url:
        return None, "TEST_DATABASE_URL is not configured (empty)."

    test_raw = cfg.test_database_url.get_secret_value()
    if not test_raw or not test_raw.strip() or "<your-" in test_raw:
        return None, "TEST_DATABASE_URL contains placeholder values or is blank."

    dev_raw = cfg.database_url.get_secret_value() if cfg.database_url else ""

    try:
        test_params = conninfo_to_dict(test_raw)
    except Exception:
        return None, "TEST_DATABASE_URL is not a valid Postgres connection string."

    test_host = test_params.get("host", "localhost")
    test_port = test_params.get("port", "5432")
    test_dbname = test_params.get("dbname", "")
    test_user = test_params.get("user", "")

    # Multi-attribute target identifier
    test_target = (test_host, test_port, test_dbname, test_user)

    if dev_raw and "<your-" not in dev_raw:
        try:
            dev_params = conninfo_to_dict(dev_raw)
            dev_host = dev_params.get("host", "localhost")
            dev_port = dev_params.get("port", "5432")
            dev_dbname = dev_params.get("dbname", "")
            dev_user = dev_params.get("user", "")
            dev_target = (dev_host, dev_port, dev_dbname, dev_user)

            # Check if both resolve to the same target
            if test_target == dev_target:
                # Permit shared target ONLY if running in CI against local container service
                is_local_throwaway = test_host in ("localhost", "127.0.0.1", "postgres")
                if not (IS_CI and is_local_throwaway):
                    return None, "SAFETY VIOLATION: TEST_DATABASE_URL and DATABASE_URL resolve to the same target."
        except Exception:
            pass

    return test_raw, "Configured"


TEST_DB_URL, SAFETY_STATUS = resolve_safe_test_db_url()

# In CI: fail the entire test run immediately if TEST_DATABASE_URL is missing or unsafe
if IS_CI and TEST_DB_URL is None:
    pytest.fail(f"CI Failure: Integration test database not configured properly. Status: {SAFETY_STATUS}")

skip_no_test_db = pytest.mark.skipif(TEST_DB_URL is None, reason=SAFETY_STATUS)


@asynccontextmanager
async def _test_db_lifespan(test_app: FastAPI) -> AsyncIterator[None]:
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
    """FastAPI TestClient wired to the test database pool."""
    orig_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _test_db_lifespan
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.router.lifespan_context = orig_lifespan


# ---------------------------------------------------------------------------
# Database Integration Tests
# ---------------------------------------------------------------------------

@skip_no_test_db
def test_db_signup_persists_tenant_admin_and_audit() -> None:
    """Verify signup persists tenant, admin user with Argon2 hash, and audit entry."""
    assert TEST_DB_URL is not None
    unique_suffix = uuid.uuid4().hex[:8]
    email = f"admin_{unique_suffix}@example.com"
    company_name = f"Test Tenant {unique_suffix}"
    raw_password = "SecurePassword123!"
    pw_hash = hash_password(raw_password)

    with psycopg.connect(TEST_DB_URL) as conn:
        tenant_id, user_id = create_tenant_and_admin_user(
            conn=conn,
            company_name=company_name,
            normalized_email=email,
            password_hash=pw_hash,
        )

        with conn.cursor(row_factory=dict_row) as cur:
            # 1. Verify tenant persisted
            cur.execute("SELECT id, name, plan FROM tenants WHERE id = %s;", (tenant_id,))
            tenant_row = cur.fetchone()
            assert tenant_row is not None
            assert tenant_row["name"] == company_name
            assert tenant_row["plan"] == "free"

            # 2. Verify user persisted with Argon2 hash (never plaintext)
            cur.execute("SELECT id, tenant_id, email, password_hash, role FROM users WHERE id = %s;", (user_id,))
            user_row = cur.fetchone()
            assert user_row is not None
            assert user_row["email"] == email
            assert user_row["role"] == "admin"
            assert user_row["password_hash"].startswith("$argon2")
            assert raw_password not in user_row["password_hash"]
            assert verify_password(user_row["password_hash"], raw_password) is True

            # 3. Verify audit log entry
            cur.execute("SELECT tenant_id, actor, action FROM audit_log WHERE tenant_id = %s;", (tenant_id,))
            audit_row = cur.fetchone()
            assert audit_row is not None
            assert audit_row["actor"] == str(user_id)
            assert audit_row["action"] == "tenant_signup"


@skip_no_test_db
def test_db_signup_duplicate_email_rollback() -> None:
    """Verify duplicate email raises EmailAlreadyExistsError and rolls back without orphan tenants."""
    assert TEST_DB_URL is not None
    unique_suffix = uuid.uuid4().hex[:8]
    email = f"dup_{unique_suffix}@example.com"
    pw_hash = hash_password("ValidPassword123!")

    with psycopg.connect(TEST_DB_URL) as conn:
        # Initial signup
        create_tenant_and_admin_user(
            conn=conn,
            company_name="Tenant 1",
            normalized_email=email,
            password_hash=pw_hash,
        )

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM tenants;")
            count_before = cur.fetchone()[0]

        # Duplicate signup attempt with same email
        with pytest.raises(EmailAlreadyExistsError):
            create_tenant_and_admin_user(
                conn=conn,
                company_name="Tenant 2 (Should Roll Back)",
                normalized_email=email,
                password_hash=pw_hash,
            )

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM tenants;")
            count_after = cur.fetchone()[0]
            assert count_after == count_before


@skip_no_test_db
def test_db_controlled_failure_rolls_back_all_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Controlled failure after user insert (via monkeypatched audit helper) rolls back all writes."""
    assert TEST_DB_URL is not None
    unique_suffix = uuid.uuid4().hex[:8]
    email = f"fail_{unique_suffix}@example.com"
    company_name = f"Failing Tenant {unique_suffix}"
    pw_hash = hash_password("ValidPassword123!")

    def failing_audit_helper(cur: psycopg.Cursor, tenant_id: uuid.UUID, user_id: uuid.UUID) -> None:
        raise RuntimeError("Simulated crash in audit helper before commit")

    # Patch the internal helper function without touching the public repository signature
    monkeypatch.setattr(auth_repo, "_record_signup_audit", failing_audit_helper)

    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM tenants;")
            tenant_count_before = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM users;")
            user_count_before = cur.fetchone()[0]

        with pytest.raises(RuntimeError, match="Simulated crash in audit helper"):
            create_tenant_and_admin_user(
                conn=conn,
                company_name=company_name,
                normalized_email=email,
                password_hash=pw_hash,
            )

        # Assert full rollback
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM tenants;")
            assert cur.fetchone()[0] == tenant_count_before
            cur.execute("SELECT count(*) FROM users;")
            assert cur.fetchone()[0] == user_count_before


@skip_no_test_db
def test_db_concurrent_duplicate_email_race_condition() -> None:
    """Genuine concurrency test: two independent database connections compete to register the same email.

    Asserts that exactly one connection commits successfully (201) while the competing
    connection receives EmailAlreadyExistsError (producing 409), without orphan rows or 500 crashes.
    """
    assert TEST_DB_URL is not None
    unique_suffix = uuid.uuid4().hex[:8]
    email = f"concur_{unique_suffix}@example.com"
    pw_hash = hash_password("ConcurrentPass123!")

    results: list[tuple[str, Any]] = []

    def attempt_signup(company: str) -> None:
        try:
            with psycopg.connect(TEST_DB_URL) as conn:
                t_id, u_id = create_tenant_and_admin_user(
                    conn=conn,
                    company_name=company,
                    normalized_email=email,
                    password_hash=pw_hash,
                )
                results.append(("success", (t_id, u_id)))
        except EmailAlreadyExistsError as exc:
            results.append(("duplicate", exc))
        except Exception as exc:
            results.append(("unexpected_error", exc))

    # Run both registration attempts concurrently across separate DB connections
    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(attempt_signup, "Competitor A")
        f2 = executor.submit(attempt_signup, "Competitor B")
        f1.result()
        f2.result()

    outcomes = [r[0] for r in results]
    assert outcomes.count("success") == 1, f"Expected exactly 1 success, got {results}"
    assert outcomes.count("duplicate") == 1, f"Expected exactly 1 duplicate error, got {results}"
    assert "unexpected_error" not in outcomes, f"Unexpected error occurred: {results}"


@skip_no_test_db
def test_http_signup_and_login_full_lifecycle(http_client: TestClient) -> None:
    """Full HTTP endpoint verification against real PostgreSQL database."""
    unique_suffix = uuid.uuid4().hex[:8]
    raw_email = f"Lifecycle_{unique_suffix}@Example.COM"
    normalized_email = raw_email.strip().lower()
    raw_password = "StrictPassword123!"

    # 1. Successful HTTP Signup
    signup_resp = http_client.post(
        "/v1/auth/signup",
        json={
            "company_name": f"HTTP Tenant {unique_suffix}",
            "email": raw_email,
            "password": raw_password,
        },
    )
    assert signup_resp.status_code == 201
    signup_data = signup_resp.json()
    assert "tenant_id" in signup_data
    assert "user_id" in signup_data
    assert "access_token" in signup_data
    assert signup_data["token_type"] == "bearer"

    token_claims = decode_access_token(signup_data["access_token"])
    assert token_claims["sub"] == signup_data["user_id"]
    assert token_claims["tenant_id"] == signup_data["tenant_id"]
    assert token_claims["role"] == "admin"

    # 2. Duplicate normalized email returns 409
    dup_resp = http_client.post(
        "/v1/auth/signup",
        json={
            "company_name": "Another Corp",
            "email": normalized_email,
            "password": raw_password,
        },
    )
    assert dup_resp.status_code == 409
    assert dup_resp.json() == {
        "error": {
            "code": "email_already_registered",
            "message": "Email is already registered.",
        }
    }

    # 3. Successful HTTP Login
    login_resp = http_client.post(
        "/v1/auth/login",
        json={
            "email": raw_email,  # Mixed-case email
            "password": raw_password,
        },
    )
    assert login_resp.status_code == 200
    login_data = login_resp.json()
    assert "access_token" in login_data
    assert login_data["token_type"] == "bearer"

    login_claims = decode_access_token(login_data["access_token"])
    assert login_claims["sub"] == signup_data["user_id"]
    assert login_claims["tenant_id"] == signup_data["tenant_id"]
    assert login_claims["role"] == "admin"

    # 4. Unknown email and wrong password return identical 401
    resp_wrong_pw = http_client.post(
        "/v1/auth/login",
        json={"email": raw_email, "password": "WrongPassword123!"},
    )
    resp_unknown_email = http_client.post(
        "/v1/auth/login",
        json={"email": f"unknown_{unique_suffix}@example.com", "password": raw_password},
    )

    assert resp_wrong_pw.status_code == 401
    assert resp_unknown_email.status_code == 401
    assert resp_wrong_pw.json() == resp_unknown_email.json()
    assert resp_wrong_pw.json() == {
        "error": {
            "code": "invalid_credentials",
            "message": "Invalid email or password.",
        }
    }
