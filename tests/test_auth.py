"""Unit tests for signup and login endpoint contracts and error envelopes.

These tests run purely in-memory / unit-mocked without requiring real database credentials
or a network connection. They verify:
- Standardized error envelope format {"error": {"code": ..., "message": ...}}.
- Validation errors return 422 with the envelope without echoing passwords.
- Signup validation rules (company_name, email format, password min length 8).
- Email normalization (lowercase).
- Unknown email and wrong password returning identical 401 response bodies.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch
import uuid
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app
from app.repositories.auth import EmailAlreadyExistsError, UserRecord
from app.security import hash_password, decode_access_token


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


def test_signup_validation_short_password(client: TestClient) -> None:
    """Password with fewer than 8 characters is rejected with 422 in standard error envelope."""
    response = client.post(
        "/v1/auth/signup",
        json={
            "company_name": "Acme Corp",
            "email": "admin@acme.com",
            "password": "short",
        },
    )
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "invalid_input"
    assert "at least 8" in data["error"]["message"].lower() or "password" in data["error"]["message"].lower()
    # Ensure raw password is NOT echoed
    assert "short" not in str(data)


def test_signup_validation_invalid_email(client: TestClient) -> None:
    """Invalid email format is rejected with 422 in standard error envelope."""
    response = client.post(
        "/v1/auth/signup",
        json={
            "company_name": "Acme Corp",
            "email": "not-an-email",
            "password": "ValidPassword123!",
        },
    )
    assert response.status_code == 422
    data = response.json()
    assert data["error"]["code"] == "invalid_input"
    assert "email" in data["error"]["message"].lower()


def test_signup_validation_empty_company_name(client: TestClient) -> None:
    """Empty company name is rejected with 422 in standard error envelope."""
    response = client.post(
        "/v1/auth/signup",
        json={
            "company_name": "   ",
            "email": "admin@acme.com",
            "password": "ValidPassword123!",
        },
    )
    assert response.status_code == 422
    data = response.json()
    assert data["error"]["code"] == "invalid_input"


@patch("app.routers.auth.get_connection")
@patch("app.routers.auth.create_tenant_and_admin_user")
def test_signup_success_unit(mock_create: MagicMock, mock_conn: MagicMock, client: TestClient) -> None:
    """Successful signup returns 201 with tenant_id, user_id, access_token, and bearer type."""
    mock_conn.return_value.__enter__.return_value = MagicMock()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    mock_create.return_value = (tenant_id, user_id)

    response = client.post(
        "/v1/auth/signup",
        json={
            "company_name": "Acme Corp",
            "email": "Admin@Acme.COM ",
            "password": "StrongPassword123!",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["tenant_id"] == str(tenant_id)
    assert data["user_id"] == str(user_id)
    assert data["token_type"] == "bearer"

    # Verify email was normalized to lowercase when calling repository
    mock_create.assert_called_once()
    called_args = mock_create.call_args[1]
    assert called_args["normalized_email"] == "admin@acme.com"
    assert called_args["company_name"] == "Acme Corp"

    # Verify returned token decodes properly with expected claims
    token_claims = decode_access_token(data["access_token"])
    assert token_claims["sub"] == str(user_id)
    assert token_claims["tenant_id"] == str(tenant_id)
    assert token_claims["role"] == "admin"


@patch("app.routers.auth.get_connection")
@patch("app.routers.auth.create_tenant_and_admin_user")
def test_signup_duplicate_email_conflict(mock_create: MagicMock, mock_conn: MagicMock, client: TestClient) -> None:
    """Duplicate email raises 409 with standardized error envelope."""
    mock_conn.return_value.__enter__.return_value = MagicMock()
    mock_create.side_effect = EmailAlreadyExistsError("Email is already registered.")

    response = client.post(
        "/v1/auth/signup",
        json={
            "company_name": "Acme Corp",
            "email": "existing@acme.com",
            "password": "StrongPassword123!",
        },
    )
    assert response.status_code == 409
    data = response.json()
    assert data == {
        "error": {
            "code": "email_already_registered",
            "message": "Email is already registered.",
        }
    }


@patch("app.routers.auth.get_connection")
@patch("app.routers.auth.get_user_by_email")
def test_login_success_unit(mock_get_user: MagicMock, mock_conn: MagicMock, client: TestClient) -> None:
    """Successful login returns 200 with access_token and bearer type."""
    mock_conn.return_value.__enter__.return_value = MagicMock()
    user_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    raw_password = "SecretPassword123!"

    mock_get_user.return_value = UserRecord(
        id=user_id,
        tenant_id=tenant_id,
        email="user@acme.com",
        password_hash=hash_password(raw_password),
        role="agent",
    )

    response = client.post(
        "/v1/auth/login",
        json={
            "email": "USER@Acme.COM",
            "password": raw_password,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["token_type"] == "bearer"
    assert "access_token" in data

    # Verify decoded claims
    claims = decode_access_token(data["access_token"])
    assert claims["sub"] == str(user_id)
    assert claims["tenant_id"] == str(tenant_id)
    assert claims["role"] == "agent"


@patch("app.routers.auth.get_connection")
@patch("app.routers.auth.get_user_by_email")
def test_login_unknown_email_and_wrong_password_return_identical_401(
    mock_get_user: MagicMock,
    mock_conn: MagicMock,
    client: TestClient,
) -> None:
    """Unknown email and wrong password return identical 401 error envelope responses."""
    mock_conn.return_value.__enter__.return_value = MagicMock()

    # Case 1: Unknown email
    mock_get_user.return_value = None
    resp_unknown = client.post(
        "/v1/auth/login",
        json={"email": "nobody@nowhere.com", "password": "AnyPassword123!"},
    )

    # Case 2: Wrong password
    mock_get_user.return_value = UserRecord(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        email="existing@acme.com",
        password_hash=hash_password("RealPassword123!"),
        role="admin",
    )
    resp_wrong_pw = client.post(
        "/v1/auth/login",
        json={"email": "existing@acme.com", "password": "WrongPassword123!"},
    )

    assert resp_unknown.status_code == 401
    assert resp_wrong_pw.status_code == 401
    assert resp_unknown.json() == resp_wrong_pw.json()
    assert resp_unknown.json() == {
        "error": {
            "code": "invalid_credentials",
            "message": "Invalid email or password.",
        }
    }

