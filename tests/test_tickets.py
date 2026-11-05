"""Unit tests for tickets endpoints and API-key credential validation.

Tests:
1. POST /v1/tickets rejects missing or invalid API keys with 401.
2. POST /v1/tickets rejects JWT authentication (API-key-only route) with 401.
3. POST /v1/tickets input validation: subject > 300 chars, body > 20000 chars, blank fields return 422.
4. GET /v1/tickets rejects unauthenticated requests with 401.
5. GET /v1/tickets query validation: limit > 200 or offset < 0 returns 422.
6. GET /v1/tickets/{id} invalid UUID format returns 404.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import uuid
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app
from app.deps import Principal, get_api_key_principal
from app.security import create_access_token


@asynccontextmanager
async def noop_lifespan(test_app: FastAPI) -> AsyncIterator[None]:
    """Substitute lifespan for unit tests to bypass database connection attempts."""
    yield


@pytest.fixture
def client() -> TestClient:
    """Fixture providing isolated TestClient with no-op database lifespan."""
    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = noop_lifespan
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.router.lifespan_context = original_lifespan


def _mock_api_key_principal() -> Principal:
    """Mock Principal for validating request body models without hitting database."""
    return Principal(
        tenant_id=uuid.uuid4(),
        kind="api_key",
        id=uuid.uuid4(),
        role="ingestion",
    )


def test_create_ticket_missing_api_key_rejected(client: TestClient) -> None:
    """Requesting ticket ingestion without API key returns 401."""
    resp = client.post(
        "/v1/tickets",
        json={"subject": "Login Issue", "body": "Cannot login to app."},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "missing_api_key"


def test_create_ticket_jwt_token_rejected(client: TestClient) -> None:
    """Providing a JWT token to API-key-only ticket ingestion route returns 401."""
    token = create_access_token(user_id=uuid.uuid4(), tenant_id=uuid.uuid4(), role="admin")
    resp = client.post(
        "/v1/tickets",
        headers={"Authorization": f"Bearer {token}"},
        json={"subject": "Login Issue", "body": "Cannot login."},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "missing_api_key"


def test_create_ticket_subject_length_validation(client: TestClient) -> None:
    """Subject longer than 300 characters is rejected with 422 in error envelope."""
    app.dependency_overrides[get_api_key_principal] = _mock_api_key_principal
    try:
        too_long_subject = "A" * 301
        resp = client.post(
            "/v1/tickets",
            headers={"X-API-Key": "tk_valid_prefix_format_1234567890"},
            json={"subject": too_long_subject, "body": "Valid body text."},
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "invalid_input"
        assert "subject" in resp.json()["error"]["message"].lower()
    finally:
        app.dependency_overrides.pop(get_api_key_principal, None)


def test_create_ticket_body_length_validation(client: TestClient) -> None:
    """Body longer than 20,000 characters is rejected with 422 in error envelope."""
    app.dependency_overrides[get_api_key_principal] = _mock_api_key_principal
    try:
        too_long_body = "B" * 20001
        resp = client.post(
            "/v1/tickets",
            headers={"X-API-Key": "tk_valid_prefix_format_1234567890"},
            json={"subject": "Valid Subject", "body": too_long_body},
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "invalid_input"
        assert "body" in resp.json()["error"]["message"].lower()
    finally:
        app.dependency_overrides.pop(get_api_key_principal, None)


def test_create_ticket_blank_fields_validation(client: TestClient) -> None:
    """Blank subject or body returns 422."""
    app.dependency_overrides[get_api_key_principal] = _mock_api_key_principal
    try:
        resp = client.post(
            "/v1/tickets",
            headers={"X-API-Key": "tk_valid_prefix_format_1234567890"},
            json={"subject": "   ", "body": "   "},
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "invalid_input"
    finally:
        app.dependency_overrides.pop(get_api_key_principal, None)


def test_list_tickets_missing_credentials_rejected(client: TestClient) -> None:
    """Listing tickets without any authentication returns 401."""
    resp = client.get("/v1/tickets")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "missing_credentials"


def test_list_tickets_pagination_limit_validation(client: TestClient) -> None:
    """Limit > 200 or offset < 0 is rejected with 422."""
    token = create_access_token(user_id=uuid.uuid4(), tenant_id=uuid.uuid4(), role="agent")
    resp_limit = client.get(
        "/v1/tickets?limit=201",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp_limit.status_code == 422
    assert resp_limit.json()["error"]["code"] == "invalid_input"

    resp_offset = client.get(
        "/v1/tickets?offset=-1",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp_offset.status_code == 422
    assert resp_offset.json()["error"]["code"] == "invalid_input"


def test_get_ticket_invalid_uuid_returns_404(client: TestClient) -> None:
    """Querying ticket with malformed UUID returns 404."""
    token = create_access_token(user_id=uuid.uuid4(), tenant_id=uuid.uuid4(), role="agent")
    resp = client.get(
        "/v1/tickets/not-a-valid-uuid",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"
