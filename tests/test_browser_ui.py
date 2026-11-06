"""Browser JSON API integration tests for React frontend (/browser-api/*).

Covers:
1. Anonymous access rejection (401 JSON, never redirects to HTML).
2. CSRF token issuance, pre-login cookie setup, and recovery from stale/expired cookies.
3. Strict CSRF validation returning 403 Forbidden (distinguished from 422 validation).
4. Login credentials validation without echoing passwords, setting HttpOnly cookie with
   Max-Age derived from JWT expiry, and omitting JWT from JSON bodies.
5. Logout CSRF validation and cookie clearance.
6. Expired and tampered session handling (401 JSON and cookie clearance).
7. Tenant-scoped ticket list and bounded pagination (page >= 1, 1 <= per_page <= 100).
8. Ticket detail with prediction, "Confidence unavailable", and cross-tenant isolation (404).
9. API /v1/* endpoints strictly reject browser cookies.
10. Explicit SPA routing rules: never returning HTML for missing API routes or assets.
11. Cache-Control: no-store header on all browser API responses.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
import re
import uuid
import pytest
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
import app.db as app_db
from app.repositories.auth import create_tenant_and_admin_user
from app.repositories.tickets import create_or_get_ticket
from app.repositories.ml import ensure_tenant_model_version, save_ticket_prediction
from app.security import (
    create_access_token,
    derive_session_identifier,
    generate_csrf_token,
    hash_password,
)
from tests.test_auth_db import TEST_DB_URL, SAFETY_STATUS

skip_no_test_db = pytest.mark.skipif(TEST_DB_URL is None, reason=SAFETY_STATUS)


@asynccontextmanager
async def _browser_test_db_lifespan(test_app: FastAPI) -> AsyncIterator[None]:
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
def client() -> TestClient:
    """FastAPI TestClient wired to the test database."""
    orig_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _browser_test_db_lifespan
    try:
        with TestClient(app, follow_redirects=False) as c:
            yield c
    finally:
        app.router.lifespan_context = orig_lifespan


def _create_test_user(
    company_name: str | None = None,
    email: str | None = None,
    password: str = "CorrectPassword123!",
) -> tuple[uuid.UUID, uuid.UUID, str, str]:
    """Helper creating a test tenant and admin user."""
    assert TEST_DB_URL is not None
    company = company_name or f"BrowserCo {uuid.uuid4().hex[:8]}"
    user_email = email or f"user_{uuid.uuid4().hex[:8]}@example.com"
    pwd_hash = hash_password(password)

    with psycopg.connect(TEST_DB_URL) as conn:
        tenant_id, user_id = create_tenant_and_admin_user(
            conn=conn,
            company_name=company,
            normalized_email=user_email,
            password_hash=pwd_hash,
        )
    return tenant_id, user_id, user_email, password


# ===========================================================================
# 1. Anonymous Access Rejection & Cache-Control
# ===========================================================================

@skip_no_test_db
def test_browser_api_anonymous_access_returns_401_json(client: TestClient):
    """Anonymous callers receive 401 JSON (not HTML redirects) with Cache-Control: no-store."""
    r_tickets = client.get("/browser-api/tickets")
    assert r_tickets.status_code == 401
    assert r_tickets.headers["content-type"].startswith("application/json")
    assert r_tickets.headers["cache-control"] == "no-store"
    assert r_tickets.json()["error"]["code"] == "unauthenticated"

    random_id = uuid.uuid4()
    r_detail = client.get(f"/browser-api/tickets/{random_id}")
    assert r_detail.status_code == 401
    assert r_detail.json()["error"]["code"] == "unauthenticated"

    r_session = client.get("/browser-api/auth/session")
    assert r_session.status_code == 401
    assert r_session.json()["error"]["code"] == "unauthenticated"


# ===========================================================================
# 2. CSRF Token Issuance & Stale Cookie Recovery
# ===========================================================================

@skip_no_test_db
def test_csrf_token_issuance_and_stale_cookie_recovery(client: TestClient):
    """GET /browser-api/csrf-token issues token and recovers cleanly from stale session cookies."""
    # 1. Initial anonymous request issues pre_login_session cookie and valid CSRF token
    r1 = client.get("/browser-api/csrf-token")
    assert r1.status_code == 200
    assert r1.headers["cache-control"] == "no-store"
    assert "pre_login_session" in r1.cookies
    token1 = r1.json()["csrf_token"]
    assert len(token1.split(".")) == 3

    # 2. Request with expired/stale triage_session cookie
    client.cookies.set("triage_session", "malformed.or.expired.jwt.cookie")
    r2 = client.get("/browser-api/csrf-token")
    assert r2.status_code == 200
    # Stale triage_session is cleared on the response
    set_cookie = r2.headers.get("set-cookie", "")
    assert "triage_session=" in set_cookie
    assert "max-age=0" in set_cookie.lower() or "expires=" in set_cookie.lower()
    # A fresh pre_login CSRF token is issued, allowing the user to log in
    assert "csrf_token" in r2.json()


# ===========================================================================
# 3. CSRF Validation on Login
# ===========================================================================

@skip_no_test_db
def test_login_csrf_validation_rules(client: TestClient):
    """POST /browser-api/auth/login returns 403 Forbidden on missing, malformed, or cross-session CSRF."""
    # 1. Missing cookie
    r1 = client.post(
        "/browser-api/auth/login",
        json={"email": "a@b.com", "password": "pass", "csrf_token": "fake.fake.fake"},
    )
    assert r1.status_code == 403
    assert r1.json()["error"]["code"] == "csrf_forbidden"

    # 2. Missing token
    r_csrf = client.get("/browser-api/csrf-token")
    r2 = client.post(
        "/browser-api/auth/login",
        json={"email": "a@b.com", "password": "pass"},
    )
    assert r2.status_code == 403
    assert r2.json()["error"]["code"] == "csrf_forbidden"

    # 3. Malformed token via X-CSRF-Token header
    r3 = client.post(
        "/browser-api/auth/login",
        headers={"X-CSRF-Token": "bad.token.signature"},
        json={"email": "a@b.com", "password": "pass"},
    )
    assert r3.status_code == 403
    assert r3.json()["error"]["code"] == "csrf_forbidden"

    # 4. Cross-session token
    client_a = TestClient(app, follow_redirects=False)
    r_a = client_a.get("/browser-api/csrf-token")
    token_a = r_a.json()["csrf_token"]

    r4 = client.post(
        "/browser-api/auth/login",
        headers={"X-CSRF-Token": token_a},
        json={"email": "a@b.com", "password": "pass"},
    )
    assert r4.status_code == 403
    assert r4.json()["error"]["code"] == "csrf_forbidden"


# ===========================================================================
# 4. Login Credentials, Cookie Flags & Security Hygiene
# ===========================================================================

@skip_no_test_db
def test_login_credentials_and_security(client: TestClient):
    """Login sets HttpOnly cookie with configured Max-Age, omits JWT from body, and verifies credentials."""
    _tenant_id, _user_id, email, password = _create_test_user()

    r_csrf = client.get("/browser-api/csrf-token")
    csrf_token = r_csrf.json()["csrf_token"]

    # 1. Invalid credentials -> 401 JSON
    secret_typo = "SuperSecretTypo123!"
    r_bad = client.post(
        "/browser-api/auth/login",
        headers={"X-CSRF-Token": csrf_token},
        json={"email": email, "password": secret_typo},
    )
    assert r_bad.status_code == 401
    assert r_bad.json()["error"]["code"] == "invalid_credentials"
    assert secret_typo not in r_bad.text

    # 2. Valid credentials -> 200 OK
    r_ok = client.post(
        "/browser-api/auth/login",
        headers={"X-CSRF-Token": csrf_token},
        json={"email": email, "password": password},
    )
    assert r_ok.status_code == 200
    body = r_ok.json()

    # JWT is strictly NOT in the JSON response
    assert "access_token" not in body
    assert "token" not in body
    assert "jwt" not in body

    # User info is present
    assert body["user"]["email"] == email
    assert "csrf_token" in body

    # triage_session cookie is set with HttpOnly, SameSite=Lax, and Max-Age matching config
    settings = get_settings()
    expected_max_age = settings.jwt_expire_minutes * 60

    set_cookies = r_ok.headers.get_list("set-cookie") if hasattr(r_ok.headers, "get_list") else [r_ok.headers.get("set-cookie", "")]
    combined = " ".join(set_cookies)

    assert "triage_session=" in combined
    assert "httponly" in combined.lower()
    assert "samesite=lax" in combined.lower()
    assert f"max-age={expected_max_age}" in combined.lower()

    # pre_login_session is deleted
    assert 'pre_login_session=""' in combined or "pre_login_session=;" in combined


# ===========================================================================
# 5. Logout Flow
# ===========================================================================

@skip_no_test_db
def test_logout_csrf_and_cookie_clearance(client: TestClient):
    """POST /browser-api/auth/logout requires CSRF and deletes triage_session."""
    tenant_id, user_id, _email, _pwd = _create_test_user()
    token = create_access_token(user_id=user_id, tenant_id=tenant_id, role="admin")
    client.cookies.set("triage_session", token)

    # 1. Obtain authenticated CSRF token
    r_sess = client.get("/browser-api/auth/session")
    assert r_sess.status_code == 200
    csrf_token = r_sess.json()["csrf_token"]

    # 2. Logout without CSRF -> 403 Forbidden
    r_no_csrf = client.post("/browser-api/auth/logout", json={})
    assert r_no_csrf.status_code == 403

    # 3. Logout with CSRF -> 200 OK and cookie deleted
    r_logout = client.post(
        "/browser-api/auth/logout",
        headers={"X-CSRF-Token": csrf_token},
        json={},
    )
    assert r_logout.status_code == 200
    assert r_logout.json()["status"] == "ok"
    set_cookie = r_logout.headers.get("set-cookie", "")
    assert "triage_session=" in set_cookie
    assert "max-age=0" in set_cookie.lower() or "expires=" in set_cookie.lower()


# ===========================================================================
# 6. Expired / Tampered Session Handling
# ===========================================================================

@skip_no_test_db
def test_expired_session_clears_cookie_on_401(client: TestClient):
    """Expired JWT returns 401 JSON and deletes triage_session cookie on that response."""
    tenant_id, user_id, _email, _pwd = _create_test_user()
    expired_token = create_access_token(
        user_id=user_id,
        tenant_id=tenant_id,
        role="admin",
        expires_delta=timedelta(minutes=-10),
    )
    client.cookies.set("triage_session", expired_token)

    resp = client.get("/browser-api/tickets")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthenticated"
    set_cookie = resp.headers.get("set-cookie", "")
    assert "triage_session=" in set_cookie
    assert "max-age=0" in set_cookie.lower() or "expires=" in set_cookie.lower()


# ===========================================================================
# 7. Tenant-Scoped Ticket List & Bounded Pagination
# ===========================================================================

@skip_no_test_db
def test_tenant_isolation_and_bounded_pagination(client: TestClient):
    """Ticket list is isolated to authenticated tenant and enforces bounded pagination."""
    tenant_a, user_a, _email_a, _pwd_a = _create_test_user(company_name="Tenant A")
    tenant_b, _user_b, _email_b, _pwd_b = _create_test_user(company_name="Tenant B")

    with psycopg.connect(TEST_DB_URL) as conn:
        for i in range(25):
            create_or_get_ticket(
                conn=conn,
                tenant_id=tenant_a,
                subject=f"Tenant A Issue #{i:02d}",
                body=f"Narrative description {i}.",
                customer_email=f"client{i}@tenanta.com",
            )
        for i in range(3):
            create_or_get_ticket(
                conn=conn,
                tenant_id=tenant_b,
                subject=f"Tenant B Secret #{i}",
                body="Tenant B secret body.",
            )

    token_a = create_access_token(user_id=user_a, tenant_id=tenant_a, role="admin")
    client.cookies.set("triage_session", token_a)

    # 1. Page 1: 20 tickets
    r1 = client.get("/browser-api/tickets?page=1&per_page=20")
    assert r1.status_code == 200
    data1 = r1.json()
    assert len(data1["tickets"]) == 20
    assert data1["pagination"]["total_count"] == 25
    assert data1["pagination"]["total_pages"] == 2

    # None of Tenant B's tickets appear
    subjects = [t["subject"] for t in data1["tickets"]]
    assert not any("Tenant B" in s for s in subjects)

    # 2. Input bounding validation
    r_bad_page = client.get("/browser-api/tickets?page=0")
    assert r_bad_page.status_code == 422

    r_bad_per_page = client.get("/browser-api/tickets?per_page=150")
    assert r_bad_per_page.status_code == 422


# ===========================================================================
# 8. Ticket Detail, Diagnostics & Isolation
# ===========================================================================

@skip_no_test_db
def test_ticket_detail_json_and_isolation(client: TestClient):
    """Detail endpoint returns model diagnostics, preserves raw text, and returns 404 across tenants."""
    tenant_a, user_a, _email_a, _pwd_a = _create_test_user(company_name="Tenant A")
    tenant_b, _user_b, _email_b, _pwd_b = _create_test_user(company_name="Tenant B")

    malicious_text = "<script>alert('XSS')</script>"
    with psycopg.connect(TEST_DB_URL) as conn:
        mv = ensure_tenant_model_version(
            conn=conn,
            tenant_id=tenant_a,
            name="cfpb_financial_product_classifier",
            version="run_004",
            task="category",
            artifact_path="experiments/models/cfpb_linearsvc_run_004.joblib",
            sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            classes=["Checking or savings account", "Mortgage"],
        )
        ticket_a, _ = create_or_get_ticket(
            conn=conn,
            tenant_id=tenant_a,
            subject=f"Dispute {malicious_text}",
            body=f"Complaint narrative with {malicious_text}.",
            customer_email="victim@example.com",
        )
        save_ticket_prediction(
            conn=conn,
            tenant_id=tenant_a,
            ticket_id=ticket_a.id,
            model_version_id=mv.id,
            task="category",
            predicted_label="Checking or savings account",
            confidence=None,
            decision_scores={"Checking or savings account": 1.594, "Mortgage": -1.167},
        )
        ticket_b, _ = create_or_get_ticket(
            conn=conn,
            tenant_id=tenant_b,
            subject="Tenant B Private Ticket",
            body="Tenant B private body.",
        )

    token_a = create_access_token(user_id=user_a, tenant_id=tenant_a, role="admin")
    client.cookies.set("triage_session", token_a)

    # 1. Detail endpoint returns literal text (not HTML-interpreted)
    r_detail = client.get(f"/browser-api/tickets/{ticket_a.id}")
    assert r_detail.status_code == 200
    detail = r_detail.json()
    assert detail["ticket"]["subject"] == f"Dispute {malicious_text}"
    assert detail["ticket"]["body"] == f"Complaint narrative with {malicious_text}."

    # Prediction diagnostics
    pred = detail["prediction"]
    assert pred["predicted_label"] == "Checking or savings account"
    assert pred["model_version"] == "run_004"
    assert pred["confidence"] is None
    assert pred["confidence_display"] == "Confidence unavailable"
    assert pred["decision_scores"]["Checking or savings account"] == 1.594

    # Domain notices
    assert "CFPB financial product classifier" in detail["notices"]["domain_notice"]
    assert "Priority prediction deferred" in detail["notices"]["priority_notice"]

    # 2. Cross-tenant access returns 404
    r_cross = client.get(f"/browser-api/tickets/{ticket_b.id}")
    assert r_cross.status_code == 404
    assert r_cross.json()["error"]["code"] == "not_found"


# ===========================================================================
# 9. API /v1/* Rejects Browser Cookies
# ===========================================================================

@skip_no_test_db
def test_api_v1_rejects_browser_session_cookie(client: TestClient):
    """Endpoints under /v1/* reject cookies; require Bearer JWT or API key."""
    tenant_id, user_id, _email, _pwd = _create_test_user()
    token = create_access_token(user_id=user_id, tenant_id=tenant_id, role="admin")
    client.cookies.set("triage_session", token)

    resp = client.get("/v1/tickets")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "missing_credentials"


# ===========================================================================
# 10. SPA Routing Exclusions
# ===========================================================================

@skip_no_test_db
def test_spa_routing_exclusions(client: TestClient):
    """Unknown API routes return 404 JSON, never SPA HTML; UI routes return index.html."""
    r_unknown_api = client.get("/browser-api/unknown-endpoint")
    assert r_unknown_api.status_code == 404
    assert r_unknown_api.json()["error"]["code"] == "http_404"

    r_unknown_v1 = client.get("/v1/unknown-endpoint")
    assert r_unknown_v1.status_code == 404
    assert r_unknown_v1.json()["error"]["code"] == "http_404"

    # UI routes return index.html
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert '<div id="root"></div>' in r_root.text

    r_tickets = client.get("/tickets")
    assert r_tickets.status_code == 200
    assert '<div id="root"></div>' in r_tickets.text

    r_detail = client.get(f"/tickets/{uuid.uuid4()}")
    assert r_detail.status_code == 200
    assert '<div id="root"></div>' in r_detail.text


# ===========================================================================
# 11. Raw Request Body Size Protection (32 KiB Bound)
# ===========================================================================

@skip_no_test_db
def test_login_body_size_normal_login_succeeds(client: TestClient):
    """Normal-sized login request within 32 KiB succeeds."""
    _tenant_id, _user_id, email, password = _create_test_user()
    r_csrf = client.get("/browser-api/csrf-token")
    csrf_token = r_csrf.json()["csrf_token"]

    resp = client.post(
        "/browser-api/auth/login",
        headers={"X-CSRF-Token": csrf_token},
        json={"email": email, "password": password},
    )
    assert resp.status_code == 200
    assert resp.json()["user"]["email"] == email


@skip_no_test_db
def test_login_oversized_json_string_rejected(client: TestClient):
    """Oversized JSON string exceeding 32 KiB is rejected with 413 and no-store."""
    r_csrf = client.get("/browser-api/csrf-token")
    csrf_token = r_csrf.json()["csrf_token"]

    oversized_email = "a" * (33 * 1024) + "@example.com"
    resp = client.post(
        "/browser-api/auth/login",
        headers={"X-CSRF-Token": csrf_token},
        json={"email": oversized_email, "password": "password"},
    )
    assert resp.status_code == 413
    assert resp.headers.get("Cache-Control") == "no-store"
    body = resp.json()
    assert body["error"]["code"] == "payload_too_large"
    assert "32768" in body["error"]["message"]


@skip_no_test_db
def test_login_oversized_whitespace_rejected(client: TestClient):
    """Oversized JSON trailing/leading whitespace exceeding 32 KiB is rejected with 413."""
    r_csrf = client.get("/browser-api/csrf-token")
    csrf_token = r_csrf.json()["csrf_token"]

    # Small valid JSON payload padded with > 32 KiB whitespace
    valid_json = '{"email": "user@example.com", "password": "secret"}'
    oversized_raw = valid_json.encode("utf-8") + (b" " * (33 * 1024))
    assert len(oversized_raw) > 32 * 1024

    resp = client.post(
        "/browser-api/auth/login",
        headers={
            "X-CSRF-Token": csrf_token,
            "Content-Type": "application/json",
        },
        content=oversized_raw,
    )
    assert resp.status_code == 413
    assert resp.headers.get("Cache-Control") == "no-store"
    body = resp.json()
    assert body["error"]["code"] == "payload_too_large"


@skip_no_test_db
def test_login_oversized_streamed_input_without_content_length_rejected(client: TestClient):
    """Oversized streamed body without Content-Length is rejected with 413 before completing."""
    r_csrf = client.get("/browser-api/csrf-token")
    csrf_token = r_csrf.json()["csrf_token"]

    # Generator streaming 35 KiB in 1 KiB chunks (httpx sends Transfer-Encoding: chunked without Content-Length)
    def chunk_generator():
        for _ in range(35):
            yield b" " * 1024

    resp = client.post(
        "/browser-api/auth/login",
        headers={
            "X-CSRF-Token": csrf_token,
            "Content-Type": "application/json",
        },
        content=chunk_generator(),
    )
    assert resp.status_code == 413
    assert resp.headers.get("Cache-Control") == "no-store"
    body = resp.json()
    assert body["error"]["code"] == "payload_too_large"


# ===========================================================================
# 12. Authenticated Browser Ticket Submission (POST /browser-api/tickets)
# ===========================================================================

@skip_no_test_db
def test_browser_create_ticket_authenticated_success(client: TestClient):
    """Authenticated browser user creates a ticket and enqueues a background job."""
    tenant_id, user_id, _email, _pwd = _create_test_user()
    token = create_access_token(user_id=user_id, tenant_id=tenant_id, role="admin")
    client.cookies.set("triage_session", token)

    r_csrf = client.get("/browser-api/csrf-token")
    csrf_token = r_csrf.json()["csrf_token"]

    subject = "Dispute over uncredited checking deposit"
    body = "Deposited $1,200 check at ATM on Monday with extended hold."

    resp = client.post(
        "/browser-api/tickets",
        headers={"X-CSRF-Token": csrf_token},
        json={"subject": subject, "body": body},
    )
    assert resp.status_code == 201
    assert resp.headers.get("Cache-Control") == "no-store"
    data = resp.json()
    ticket_id = uuid.UUID(data["id"])
    assert data["ticket"]["subject"] == subject
    assert data["ticket"]["status"] == "new"

    # Verify database ticket and job state
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT tenant_id, subject, status FROM tickets WHERE id = %s;", (ticket_id,))
            row = cur.fetchone()
            assert row is not None
            assert row[0] == tenant_id
            assert row[1] == subject
            assert row[2] == "new"

            cur.execute("SELECT tenant_id, ticket_id, type, status, attempts FROM jobs WHERE ticket_id = %s;", (ticket_id,))
            j_row = cur.fetchone()
            assert j_row is not None
            assert j_row[0] == tenant_id
            assert j_row[1] == ticket_id
            assert j_row[2] == "ticket_ingest"
            assert j_row[3] == "queued"
            assert j_row[4] == 0


@skip_no_test_db
def test_browser_create_ticket_unauthenticated_returns_401(client: TestClient):
    """Missing or invalid triage_session cookie returns 401 unauthenticated."""
    resp = client.post(
        "/browser-api/tickets",
        json={"subject": "Valid Subject", "body": "Valid complaint narrative body."},
    )
    assert resp.status_code == 401
    assert resp.headers.get("Cache-Control") == "no-store"
    assert resp.json()["error"]["code"] == "unauthenticated"


@skip_no_test_db
def test_browser_create_ticket_csrf_validation(client: TestClient):
    """Missing, invalid, or cross-session CSRF token returns 403 csrf_forbidden."""
    tenant_a, user_a, _email_a, _pwd_a = _create_test_user()
    token_a = create_access_token(user_id=user_a, tenant_id=tenant_a, role="admin")

    tenant_b, user_b, _email_b, _pwd_b = _create_test_user()
    token_b = create_access_token(user_id=user_b, tenant_id=tenant_b, role="admin")

    # 1. Missing CSRF token -> 403
    client.cookies.set("triage_session", token_a)
    r_missing = client.post(
        "/browser-api/tickets",
        json={"subject": "Valid Subject", "body": "Valid body narrative."},
    )
    assert r_missing.status_code == 403
    assert r_missing.json()["error"]["code"] == "csrf_forbidden"

    # 2. Tampered CSRF token -> 403
    r_csrf = client.get("/browser-api/csrf-token")
    csrf_a = r_csrf.json()["csrf_token"]
    tampered_csrf = csrf_a[:-4] + "fake"
    r_bad = client.post(
        "/browser-api/tickets",
        headers={"X-CSRF-Token": tampered_csrf},
        json={"subject": "Valid Subject", "body": "Valid body narrative."},
    )
    assert r_bad.status_code == 403
    assert r_bad.json()["error"]["code"] == "csrf_forbidden"

    # 3. Cross-session CSRF token -> 403
    session_b_id = derive_session_identifier(token_b)
    csrf_b = generate_csrf_token(session_b_id)
    r_cross = client.post(
        "/browser-api/tickets",
        headers={"X-CSRF-Token": csrf_b},
        json={"subject": "Valid Subject", "body": "Valid body narrative."},
    )
    assert r_cross.status_code == 403
    assert r_cross.json()["error"]["code"] == "csrf_forbidden"


@skip_no_test_db
def test_browser_create_ticket_validation_and_oversized_rejection(client: TestClient):
    """Rejects empty/blank fields (422) and oversized raw request bodies (413)."""
    tenant_id, user_id, _email, _pwd = _create_test_user()
    token = create_access_token(user_id=user_id, tenant_id=tenant_id, role="admin")
    client.cookies.set("triage_session", token)

    r_csrf = client.get("/browser-api/csrf-token")
    csrf_token = r_csrf.json()["csrf_token"]

    # 1. Blank subject -> 422
    r_blank_sub = client.post(
        "/browser-api/tickets",
        headers={"X-CSRF-Token": csrf_token},
        json={"subject": "   ", "body": "Valid complaint narrative body."},
    )
    assert r_blank_sub.status_code == 422
    assert r_blank_sub.json()["error"]["code"] == "invalid_input"

    # 2. Blank body -> 422
    r_blank_body = client.post(
        "/browser-api/tickets",
        headers={"X-CSRF-Token": csrf_token},
        json={"subject": "Valid Subject", "body": "   "},
    )
    assert r_blank_body.status_code == 422
    assert r_blank_body.json()["error"]["code"] == "invalid_input"

    # 3. Subject exceeding 300 chars -> 422
    r_long_sub = client.post(
        "/browser-api/tickets",
        headers={"X-CSRF-Token": csrf_token},
        json={"subject": "x" * 301, "body": "Valid complaint narrative body."},
    )
    assert r_long_sub.status_code == 422
    assert r_long_sub.json()["error"]["code"] == "invalid_input"

    # 4. Raw body exceeding 32 KiB -> 413
    oversized_body = "x" * (33 * 1024)
    r_oversized = client.post(
        "/browser-api/tickets",
        headers={"X-CSRF-Token": csrf_token},
        json={"subject": "Valid Subject", "body": oversized_body},
    )
    assert r_oversized.status_code == 413
    assert r_oversized.headers.get("Cache-Control") == "no-store"
    assert r_oversized.json()["error"]["code"] == "payload_too_large"


@skip_no_test_db
def test_browser_create_ticket_client_tenant_id_cannot_control_ownership(client: TestClient):
    """Client cannot supply tenant_id in request to hijack ownership; principal strictly governs."""
    tenant_a, user_a, _email_a, _pwd_a = _create_test_user()
    tenant_b, user_b, _email_b, _pwd_b = _create_test_user()

    token_a = create_access_token(user_id=user_a, tenant_id=tenant_a, role="admin")
    client.cookies.set("triage_session", token_a)

    r_csrf = client.get("/browser-api/csrf-token")
    csrf_token = r_csrf.json()["csrf_token"]

    # Attacker attempts to spoof tenant_id = tenant_b in the body
    resp = client.post(
        "/browser-api/tickets",
        headers={"X-CSRF-Token": csrf_token},
        json={
            "subject": "Spoofed Ownership Ticket",
            "body": "Attempting to create ticket in tenant B while logged in as tenant A.",
            "tenant_id": str(tenant_b),
        },
    )
    assert resp.status_code == 201
    created_id = uuid.UUID(resp.json()["id"])

    # Verify in DB: ticket belongs strictly to tenant_a
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT tenant_id FROM tickets WHERE id = %s;", (created_id,))
            row = cur.fetchone()
            assert row is not None
            assert row[0] == tenant_a
            assert row[0] != tenant_b

    # Verify tenant_b cannot see this ticket
    token_b = create_access_token(user_id=user_b, tenant_id=tenant_b, role="admin")
    client.cookies.set("triage_session", token_b)
    r_b_detail = client.get(f"/browser-api/tickets/{created_id}")
    assert r_b_detail.status_code == 404


# ===========================================================================
# 13. Accurate Ticket Processing Status & Error Sanitization
# ===========================================================================

@skip_no_test_db
def test_browser_processing_status_queued_initial(client: TestClient):
    """Initial ticket submission reports queued processing state and null prediction."""
    tenant_id, user_id, _email, _pwd = _create_test_user()
    token = create_access_token(user_id=user_id, tenant_id=tenant_id, role="admin")
    client.cookies.set("triage_session", token)

    r_csrf = client.get("/browser-api/csrf-token")
    csrf_token = r_csrf.json()["csrf_token"]

    r_create = client.post(
        "/browser-api/tickets",
        headers={"X-CSRF-Token": csrf_token},
        json={"subject": "Billing issue", "body": "Overcharged by bank on statement."},
    )
    assert r_create.status_code == 201
    ticket_id = r_create.json()["id"]

    # Detail check
    r_detail = client.get(f"/browser-api/tickets/{ticket_id}")
    assert r_detail.status_code == 200
    data = r_detail.json()
    assert data["processing"]["state"] == "queued"
    assert data["processing"]["label"] == "Queued"
    assert "Waiting in queue" in data["processing"]["details"]
    assert data["prediction"] is None

    # List check
    r_list = client.get("/browser-api/tickets")
    assert r_list.status_code == 200
    list_item = next(item for item in r_list.json()["tickets"] if item["id"] == ticket_id)
    assert list_item["processing_status"]["state"] == "queued"
    assert list_item["processing_status"]["label"] == "Queued"
    assert list_item["predicted_label"] is None


@skip_no_test_db
def test_browser_processing_status_running_processing(client: TestClient):
    """Running worker job reports processing state and never leaks worker_id."""
    tenant_id, user_id, _email, _pwd = _create_test_user()
    token = create_access_token(user_id=user_id, tenant_id=tenant_id, role="admin")
    client.cookies.set("triage_session", token)

    r_csrf = client.get("/browser-api/csrf-token")
    csrf_token = r_csrf.json()["csrf_token"]

    r_create = client.post(
        "/browser-api/tickets",
        headers={"X-CSRF-Token": csrf_token},
        json={"subject": "In-flight job ticket", "body": "Narrative currently being processed by worker."},
    )
    assert r_create.status_code == 201
    ticket_id = r_create.json()["id"]

    secret_worker_id = "worker_internal_hostname_node_99"
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    attempts = 1,
                    locked_at = NOW(),
                    locked_by = %s
                WHERE tenant_id = %s AND ticket_id = %s;
                """,
                (secret_worker_id, tenant_id, ticket_id),
            )
        conn.commit()

    # Detail check
    r_detail = client.get(f"/browser-api/tickets/{ticket_id}")
    assert r_detail.status_code == 200
    data = r_detail.json()
    assert data["processing"]["state"] == "processing"
    assert data["processing"]["label"] == "Processing"
    assert "currently analyzing" in data["processing"]["details"]
    # Verify internal worker identity is not leaked
    assert secret_worker_id not in r_detail.text

    # List check
    r_list = client.get("/browser-api/tickets")
    assert r_list.status_code == 200
    list_item = next(item for item in r_list.json()["tickets"] if item["id"] == ticket_id)
    assert list_item["processing_status"]["state"] == "processing"
    assert list_item["processing_status"]["label"] == "Processing"
    assert secret_worker_id not in r_list.text


@skip_no_test_db
def test_browser_processing_status_retry_scheduled(client: TestClient):
    """Job awaiting retry reports retry_scheduled and sanitizes internal exception."""
    tenant_id, user_id, _email, _pwd = _create_test_user()
    token = create_access_token(user_id=user_id, tenant_id=tenant_id, role="admin")
    client.cookies.set("triage_session", token)

    r_csrf = client.get("/browser-api/csrf-token")
    csrf_token = r_csrf.json()["csrf_token"]

    r_create = client.post(
        "/browser-api/tickets",
        headers={"X-CSRF-Token": csrf_token},
        json={"subject": "Transient error ticket", "body": "Narrative with transient worker error."},
    )
    assert r_create.status_code == 201
    ticket_id = r_create.json()["id"]

    raw_error_message = "psycopg.errors.OperationalError: could not connect to server: Connection refused (0x0000274D/10061)"
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE jobs
                SET status = 'queued',
                    attempts = 1,
                    run_after = NOW() + interval '30 seconds',
                    error = %s
                WHERE tenant_id = %s AND ticket_id = %s;
                """,
                (raw_error_message, tenant_id, ticket_id),
            )
        conn.commit()

    # Detail check
    r_detail = client.get(f"/browser-api/tickets/{ticket_id}")
    assert r_detail.status_code == 200
    data = r_detail.json()
    assert data["processing"]["state"] == "retry_scheduled"
    assert data["processing"]["label"] == "Retry Scheduled"
    assert "retry scheduled" in data["processing"]["details"]
    # Verify raw database connection error is never leaked
    assert "OperationalError" not in r_detail.text
    assert "Connection refused" not in r_detail.text

    # List check
    r_list = client.get("/browser-api/tickets")
    assert r_list.status_code == 200
    list_item = next(item for item in r_list.json()["tickets"] if item["id"] == ticket_id)
    assert list_item["processing_status"]["state"] == "retry_scheduled"
    assert list_item["processing_status"]["label"] == "Retry Scheduled"
    assert "OperationalError" not in r_list.text


@skip_no_test_db
def test_browser_processing_status_queued_after_retry_delay_elapsed(client: TestClient):
    """Queued job with attempts > 0 whose run_after is due/past maps to waiting queued state, not unknown or missing."""
    tenant_id, user_id, _email, _pwd = _create_test_user()
    token = create_access_token(user_id=user_id, tenant_id=tenant_id, role="admin")
    client.cookies.set("triage_session", token)

    r_csrf = client.get("/browser-api/csrf-token")
    csrf_token = r_csrf.json()["csrf_token"]

    r_create = client.post(
        "/browser-api/tickets",
        headers={"X-CSRF-Token": csrf_token},
        json={"subject": "Due for retry ticket", "body": "Complaint whose scheduled retry interval has elapsed."},
    )
    assert r_create.status_code == 201
    ticket_id = r_create.json()["id"]

    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE jobs
                SET status = 'queued',
                    attempts = 1,
                    run_after = NOW() - interval '10 seconds',
                    error = 'Temporary network timeout'
                WHERE tenant_id = %s AND ticket_id = %s;
                """,
                (tenant_id, ticket_id),
            )
        conn.commit()

    # Detail check: must map to sensible queued waiting state, never unknown or not_found
    r_detail = client.get(f"/browser-api/tickets/{ticket_id}")
    assert r_detail.status_code == 200
    data = r_detail.json()
    assert data["processing"]["state"] == "queued"
    assert data["processing"]["label"] == "Queued"
    assert "waiting in queue" in data["processing"]["details"].lower()
    assert data["prediction"] is None

    # List check: must map to queued waiting state
    r_list = client.get("/browser-api/tickets")
    assert r_list.status_code == 200
    list_item = next(item for item in r_list.json()["tickets"] if item["id"] == ticket_id)
    assert list_item["processing_status"]["state"] == "queued"
    assert list_item["processing_status"]["label"] == "Queued"


@skip_no_test_db
def test_browser_processing_status_completed_with_prediction(client: TestClient):
    """Completed job with prediction reports completed state and active prediction."""
    tenant_id, user_id, _email, _pwd = _create_test_user()
    token = create_access_token(user_id=user_id, tenant_id=tenant_id, role="admin")
    client.cookies.set("triage_session", token)

    r_csrf = client.get("/browser-api/csrf-token")
    csrf_token = r_csrf.json()["csrf_token"]

    r_create = client.post(
        "/browser-api/tickets",
        headers={"X-CSRF-Token": csrf_token},
        json={"subject": "Completed ticket", "body": "Complaint that was classified successfully."},
    )
    assert r_create.status_code == 201
    ticket_id = uuid.UUID(r_create.json()["id"])

    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE jobs
                SET status = 'completed',
                    completed_at = NOW(),
                    attempts = 1
                WHERE tenant_id = %s AND ticket_id = %s;
                """,
                (tenant_id, ticket_id),
            )
        # Ensure tenant model version and save prediction
        mv = ensure_tenant_model_version(
            conn=conn,
            tenant_id=tenant_id,
            name="cfpb_financial_product_classifier",
            version="run_004",
            task="category",
            artifact_path="experiments/models/cfpb_linearsvc_run_004.joblib",
            sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            classes=["Credit reporting", "Debt collection"],
        )
        save_ticket_prediction(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id=ticket_id,
            model_version_id=mv.id,
            task="category",
            predicted_label="Credit reporting",
            confidence=None,
            decision_scores={"Credit reporting": 1.25, "Debt collection": -0.42},
        )
        conn.commit()

    # Detail check
    r_detail = client.get(f"/browser-api/tickets/{ticket_id}")
    assert r_detail.status_code == 200
    data = r_detail.json()
    assert data["processing"]["state"] == "completed"
    assert data["processing"]["label"] == "Completed"
    assert data["processing"]["details"] is None
    assert data["prediction"] is not None
    assert data["prediction"]["predicted_label"] == "Credit reporting"

    # List check
    r_list = client.get("/browser-api/tickets")
    assert r_list.status_code == 200
    list_item = next(item for item in r_list.json()["tickets"] if item["id"] == str(ticket_id))
    assert list_item["processing_status"]["state"] == "completed"
    assert list_item["processing_status"]["label"] == "Completed"
    assert list_item["predicted_label"] == "Credit reporting"


@skip_no_test_db
def test_browser_processing_status_completed_without_prediction(client: TestClient):
    """Completed job with no prediction record reports completed state and null prediction (not pending)."""
    tenant_id, user_id, _email, _pwd = _create_test_user()
    token = create_access_token(user_id=user_id, tenant_id=tenant_id, role="admin")
    client.cookies.set("triage_session", token)

    r_csrf = client.get("/browser-api/csrf-token")
    csrf_token = r_csrf.json()["csrf_token"]

    r_create = client.post(
        "/browser-api/tickets",
        headers={"X-CSRF-Token": csrf_token},
        json={"subject": "No pred ticket", "body": "Job completed but no prediction was saved."},
    )
    assert r_create.status_code == 201
    ticket_id = r_create.json()["id"]

    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE jobs
                SET status = 'completed',
                    completed_at = NOW(),
                    attempts = 1
                WHERE tenant_id = %s AND ticket_id = %s;
                """,
                (tenant_id, ticket_id),
            )
        conn.commit()

    # Detail check
    r_detail = client.get(f"/browser-api/tickets/{ticket_id}")
    assert r_detail.status_code == 200
    data = r_detail.json()
    assert data["processing"]["state"] == "completed"
    assert data["processing"]["label"] == "Completed"
    assert data["prediction"] is None

    # List check
    r_list = client.get("/browser-api/tickets")
    assert r_list.status_code == 200
    list_item = next(item for item in r_list.json()["tickets"] if item["id"] == ticket_id)
    assert list_item["processing_status"]["state"] == "completed"
    assert list_item["processing_status"]["label"] == "Completed"
    assert list_item["predicted_label"] is None


@skip_no_test_db
def test_browser_processing_status_failed_error_sanitization(client: TestClient):
    """Permanently failed job reports failed state and strictly sanitizes raw exceptions and credentials."""
    tenant_id, user_id, _email, _pwd = _create_test_user()
    token = create_access_token(user_id=user_id, tenant_id=tenant_id, role="admin")
    client.cookies.set("triage_session", token)

    r_csrf = client.get("/browser-api/csrf-token")
    csrf_token = r_csrf.json()["csrf_token"]

    r_create = client.post(
        "/browser-api/tickets",
        headers={"X-CSRF-Token": csrf_token},
        json={"subject": "Permanent failure ticket", "body": "Narrative where retries were exhausted."},
    )
    assert r_create.status_code == 201
    ticket_id = r_create.json()["id"]

    secret_leak_error = (
        'psycopg.errors.UndefinedTable: relation "ticket_predictions" does not exist '
        'LINE 1: INSERT INTO ticket_predictions ... user="admin_db" password="super_secret_password_123"'
    )
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    failed_at = NOW(),
                    attempts = 3,
                    max_attempts = 3,
                    error = %s
                WHERE tenant_id = %s AND ticket_id = %s;
                """,
                (secret_leak_error, tenant_id, ticket_id),
            )
        conn.commit()

    # Detail check
    r_detail = client.get(f"/browser-api/tickets/{ticket_id}")
    assert r_detail.status_code == 200
    data = r_detail.json()
    assert data["processing"]["state"] == "failed"
    assert data["processing"]["label"] == "Failed"
    assert data["processing"]["details"] == "Processing encountered an error and could not complete automated triage."
    assert data["prediction"] is None

    # Strict assertion: sensitive strings from error column must NEVER appear anywhere in response text
    assert "super_secret_password_123" not in r_detail.text
    assert "admin_db" not in r_detail.text
    assert "UndefinedTable" not in r_detail.text
    assert "ticket_predictions" not in r_detail.json()["processing"]["details"]

    # List check
    r_list = client.get("/browser-api/tickets")
    assert r_list.status_code == 200
    list_item = next(item for item in r_list.json()["tickets"] if item["id"] == ticket_id)
    assert list_item["processing_status"]["state"] == "failed"
    assert list_item["processing_status"]["label"] == "Failed"
    assert "super_secret_password_123" not in r_list.text


@skip_no_test_db
def test_browser_processing_status_no_job_found(client: TestClient):
    """Ticket created without an ingestion job reports not_found state."""
    tenant_id, user_id, _email, _pwd = _create_test_user()
    token = create_access_token(user_id=user_id, tenant_id=tenant_id, role="admin")
    client.cookies.set("triage_session", token)

    # Insert a ticket directly without an ingest job
    with psycopg.connect(TEST_DB_URL) as conn:
        ticket_rec, _ = create_or_get_ticket(
            conn=conn,
            tenant_id=tenant_id,
            subject="Legacy ticket without job",
            body="Inserted before Phase 2 jobs queue was introduced.",
        )
        conn.commit()

    ticket_id = str(ticket_rec.id)

    # Detail check
    r_detail = client.get(f"/browser-api/tickets/{ticket_id}")
    assert r_detail.status_code == 200
    data = r_detail.json()
    assert data["processing"]["state"] == "not_found"
    assert data["processing"]["label"] == "No Job Found"
    assert "No background processing job was found" in data["processing"]["details"]
    assert data["prediction"] is None

    # List check
    r_list = client.get("/browser-api/tickets")
    assert r_list.status_code == 200
    list_item = next(item for item in r_list.json()["tickets"] if item["id"] == ticket_id)
    assert list_item["processing_status"]["state"] == "not_found"
    assert list_item["processing_status"]["label"] == "No Job Found"
    assert list_item["predicted_label"] is None


@skip_no_test_db
def test_browser_processing_status_tenant_isolation(client: TestClient):
    """Processing status and job details cannot leak across tenants."""
    tenant_a, user_a, _email_a, _pwd_a = _create_test_user()
    tenant_b, user_b, _email_b, _pwd_b = _create_test_user()

    # Create ticket in Tenant A
    token_a = create_access_token(user_id=user_a, tenant_id=tenant_a, role="admin")
    client.cookies.set("triage_session", token_a)

    r_csrf_a = client.get("/browser-api/csrf-token")
    r_create_a = client.post(
        "/browser-api/tickets",
        headers={"X-CSRF-Token": r_csrf_a.json()["csrf_token"]},
        json={"subject": "Tenant A Secret Complaint", "body": "Confidential financial issue in Tenant A."},
    )
    assert r_create_a.status_code == 201
    ticket_a_id = r_create_a.json()["id"]

    # Switch session to Tenant B
    token_b = create_access_token(user_id=user_b, tenant_id=tenant_b, role="admin")
    client.cookies.set("triage_session", token_b)

    # Tenant B detail attempt -> 404
    r_detail_b = client.get(f"/browser-api/tickets/{ticket_a_id}")
    assert r_detail_b.status_code == 404

    # Tenant B ticket list does not contain Tenant A's ticket
    r_list_b = client.get("/browser-api/tickets")
    assert r_list_b.status_code == 200
    ticket_ids_b = [t["id"] for t in r_list_b.json()["tickets"]]
    assert ticket_a_id not in ticket_ids_b
