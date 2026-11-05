"""Unit tests for JWT Principal resolution and API-key endpoints.

Tests:
1. Missing, malformed, or invalid Authorization headers return 401.
2. Expired JWT returns 401.
3. Agent role receives 403 on admin-only routes (POST /v1/api-keys, DELETE /v1/api-keys/{id}).
4. Admin successfully creates an API key with shape {id, label, key}.
5. Returned key starts with 'tk_' and is 35+ characters long.
6. Validation errors (e.g. empty label) return 422 in standard error envelope.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from unittest.mock import MagicMock, patch
import uuid
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app
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


def test_api_keys_missing_authorization_header(client: TestClient) -> None:
    """Request without Authorization header returns 401 in error envelope."""
    resp = client.post("/v1/api-keys", json={"label": "Production Ingestion"})
    assert resp.status_code == 401
    assert resp.json() == {
        "error": {
            "code": "missing_authorization_header",
            "message": "Authorization header is required.",
        }
    }


def test_api_keys_malformed_authorization_header(client: TestClient) -> None:
    """Request with non-Bearer Authorization header returns 401 in error envelope."""
    resp = client.post(
        "/v1/api-keys",
        headers={"Authorization": "Token not_a_bearer_token"},
        json={"label": "Production Ingestion"},
    )
    assert resp.status_code == 401
    assert resp.json() == {
        "error": {
            "code": "invalid_authorization_header",
            "message": "Authorization header must be in format 'Bearer <token>'.",
        }
    }


def test_api_keys_expired_jwt_rejected(client: TestClient) -> None:
    """Expired JWT token returns 401."""
    expired_token = create_access_token(
        user_id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        role="admin",
        expires_delta=timedelta(seconds=-10),
    )
    resp = client.post(
        "/v1/api-keys",
        headers={"Authorization": f"Bearer {expired_token}"},
        json={"label": "Production Ingestion"},
    )
    assert resp.status_code == 401
    data = resp.json()
    assert data["error"]["code"] == "invalid_token"
    assert "expired" in data["error"]["message"].lower()


def test_api_keys_agent_role_forbidden(client: TestClient) -> None:
    """Agent role attempting to create or revoke API keys returns 403."""
    agent_token = create_access_token(
        user_id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        role="agent",
    )
    # Attempt create
    resp_create = client.post(
        "/v1/api-keys",
        headers={"Authorization": f"Bearer {agent_token}"},
        json={"label": "Agent Key"},
    )
    assert resp_create.status_code == 403
    assert resp_create.json() == {
        "error": {
            "code": "forbidden",
            "message": "Admin privileges are required for this action.",
        }
    }

    # Attempt revoke
    resp_delete = client.delete(
        f"/v1/api-keys/{uuid.uuid4()}",
        headers={"Authorization": f"Bearer {agent_token}"},
    )
    assert resp_delete.status_code == 403
    assert resp_delete.json()["error"]["code"] == "forbidden"


def test_api_keys_create_empty_label_validation(client: TestClient) -> None:
    """Empty or blank label returns 422 validation error."""
    admin_token = create_access_token(
        user_id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        role="admin",
    )
    resp = client.post(
        "/v1/api-keys",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"label": "   "},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_input"


@patch("app.routers.api_keys.get_connection")
@patch("app.routers.api_keys.api_key_repo.create_api_key")
def test_api_keys_create_success_unit(
    mock_create_key: MagicMock,
    mock_conn: MagicMock,
    client: TestClient,
) -> None:
    """Admin successfully creates API key, receiving required shape with full key."""
    mock_conn.return_value.__enter__.return_value = MagicMock()
    mock_conn.return_value.__enter__.return_value.transaction.return_value.__enter__.return_value = None
    generated_id = uuid.uuid4()
    mock_create_key.return_value = generated_id

    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    admin_token = create_access_token(user_id=user_id, tenant_id=tenant_id, role="admin")

    resp = client.post(
        "/v1/api-keys",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"label": "CI Ingestion Key"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"] == str(generated_id)
    assert data["label"] == "CI Ingestion Key"
    assert data["key"].startswith("tk_")
    assert len(data["key"]) > 35

    # Verify repository was called strictly with principal.tenant_id
    mock_create_key.assert_called_once()
    called_kwargs = mock_create_key.call_args[1]
    assert called_kwargs["tenant_id"] == tenant_id
    assert called_kwargs["key_prefix"] == data["key"][:8]
    assert called_kwargs["label"] == "CI Ingestion Key"


@patch("app.routers.api_keys.get_connection")
@patch("app.routers.api_keys.api_key_repo.revoke_api_key")
def test_api_keys_revoke_success_and_not_found_unit(
    mock_revoke_key: MagicMock,
    mock_conn: MagicMock,
    client: TestClient,
) -> None:
    """Admin revokes key returning 204 with empty body; non-existent key returns 404."""
    mock_conn.return_value.__enter__.return_value = MagicMock()
    mock_conn.return_value.__enter__.return_value.transaction.return_value.__enter__.return_value = None
    tenant_id = uuid.uuid4()
    admin_token = create_access_token(user_id=uuid.uuid4(), tenant_id=tenant_id, role="admin")
    key_id = uuid.uuid4()

    # Success case
    mock_revoke_key.return_value = True
    resp_success = client.delete(
        f"/v1/api-keys/{key_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp_success.status_code == 204
    assert resp_success.text == ""

    # Not found / cross-tenant case
    mock_revoke_key.return_value = False
    resp_not_found = client.delete(
        f"/v1/api-keys/{uuid.uuid4()}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp_not_found.status_code == 404
    assert resp_not_found.json() == {
        "error": {
            "code": "not_found",
            "message": "API key not found.",
        }
    }

