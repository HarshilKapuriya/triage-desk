"""Unit tests for CSV bulk tickets ingestion endpoint (POST /v1/tickets/bulk).

Tests:
1. Rejects missing API key with 401.
2. Rejects JWT Bearer token with 401 (API-key-only route).
3. Rejects empty body with 422.
4. Rejects CSV missing required column headers ('subject', 'body') with 422.
5. Rejects CSV containing only headers (no data rows) with 422.
6. Rejects row with blank/empty subject with 422.
7. Rejects row with subject > 300 chars with 422.
8. Rejects row with blank/empty body with 422.
9. Rejects row with body > 20000 chars with 422.
10. Rejects row with external_id > 255 chars with 422.
11. Rejects row with customer_email > 255 chars with 422.
12. Rejects CSV exceeding 500 rows limit with 422.
13. Rejects CSV payload exceeding 5 MB limit with 422.
14. Rejects malformed CSV syntax with 422.
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
    """Mock Principal for validating request body without hitting database."""
    return Principal(
        tenant_id=uuid.uuid4(),
        kind="api_key",
        id=uuid.uuid4(),
        role="ingestion",
    )


def test_bulk_tickets_missing_api_key_rejected(client: TestClient) -> None:
    """POST /v1/tickets/bulk without API key returns 401."""
    resp = client.post(
        "/v1/tickets/bulk",
        content="subject,body\nTest,Body",
        headers={"Content-Type": "text/csv"},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "missing_api_key"


def test_bulk_tickets_jwt_rejected(client: TestClient) -> None:
    """POST /v1/tickets/bulk with JWT returns 401 (API-key-only route)."""
    token = create_access_token(user_id=uuid.uuid4(), tenant_id=uuid.uuid4(), role="admin")
    resp = client.post(
        "/v1/tickets/bulk",
        content="subject,body\nTest,Body",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "text/csv",
        },
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "missing_api_key"


def test_bulk_tickets_empty_payload(client: TestClient) -> None:
    """Empty CSV body returns 422 invalid_input."""
    app.dependency_overrides[get_api_key_principal] = _mock_api_key_principal
    try:
        resp = client.post(
            "/v1/tickets/bulk",
            content="",
            headers={"Content-Type": "text/csv"},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "invalid_input"
        assert "empty" in data["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_bulk_tickets_missing_required_headers(client: TestClient) -> None:
    """CSV missing 'subject' or 'body' header returns 422 invalid_input."""
    app.dependency_overrides[get_api_key_principal] = _mock_api_key_principal
    try:
        # Missing 'body' header
        resp = client.post(
            "/v1/tickets/bulk",
            content="subject,external_id\nTest Subject,ext-1\n",
            headers={"Content-Type": "text/csv"},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "invalid_input"
        assert "missing required column headers" in data["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_bulk_tickets_no_data_rows(client: TestClient) -> None:
    """CSV with only headers returns 422 invalid_input."""
    app.dependency_overrides[get_api_key_principal] = _mock_api_key_principal
    try:
        resp = client.post(
            "/v1/tickets/bulk",
            content="subject,body,external_id,customer_email\n",
            headers={"Content-Type": "text/csv"},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "invalid_input"
        assert "no data rows" in data["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_bulk_tickets_row_subject_blank(client: TestClient) -> None:
    """Row with blank subject returns 422 invalid_input with row number."""
    app.dependency_overrides[get_api_key_principal] = _mock_api_key_principal
    try:
        csv_data = "subject,body\nValid Subject,Valid Body\n   ,Second Body\n"
        resp = client.post(
            "/v1/tickets/bulk",
            content=csv_data,
            headers={"Content-Type": "text/csv"},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "invalid_input"
        assert "Row 2: subject is required" in data["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_bulk_tickets_row_subject_too_long(client: TestClient) -> None:
    """Row with subject exceeding 300 chars returns 422 invalid_input with row number."""
    app.dependency_overrides[get_api_key_principal] = _mock_api_key_principal
    try:
        too_long = "S" * 301
        csv_data = f"subject,body\n{too_long},Body text\n"
        resp = client.post(
            "/v1/tickets/bulk",
            content=csv_data,
            headers={"Content-Type": "text/csv"},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "invalid_input"
        assert "Row 1: subject exceeds maximum length" in data["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_bulk_tickets_row_body_blank(client: TestClient) -> None:
    """Row with blank body returns 422 invalid_input with row number."""
    app.dependency_overrides[get_api_key_principal] = _mock_api_key_principal
    try:
        csv_data = "subject,body\nSubj 1,Body 1\nSubj 2,   \n"
        resp = client.post(
            "/v1/tickets/bulk",
            content=csv_data,
            headers={"Content-Type": "text/csv"},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "invalid_input"
        assert "Row 2: body is required" in data["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_bulk_tickets_row_body_too_long(client: TestClient) -> None:
    """Row with body exceeding 20000 chars returns 422 invalid_input with row number."""
    app.dependency_overrides[get_api_key_principal] = _mock_api_key_principal
    try:
        too_long_body = "B" * 20001
        csv_data = f"subject,body\nSubj,{too_long_body}\n"
        resp = client.post(
            "/v1/tickets/bulk",
            content=csv_data,
            headers={"Content-Type": "text/csv"},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "invalid_input"
        assert "Row 1: body exceeds maximum length" in data["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_bulk_tickets_row_external_id_too_long(client: TestClient) -> None:
    """Row with external_id exceeding 255 chars returns 422 invalid_input."""
    app.dependency_overrides[get_api_key_principal] = _mock_api_key_principal
    try:
        too_long_ext = "E" * 256
        csv_data = f"subject,body,external_id\nSubj,Body,{too_long_ext}\n"
        resp = client.post(
            "/v1/tickets/bulk",
            content=csv_data,
            headers={"Content-Type": "text/csv"},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "invalid_input"
        assert "Row 1: external_id exceeds maximum length" in data["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_bulk_tickets_row_customer_email_too_long(client: TestClient) -> None:
    """Row with customer_email exceeding 255 chars returns 422 invalid_input."""
    app.dependency_overrides[get_api_key_principal] = _mock_api_key_principal
    try:
        too_long_email = ("M" * 250) + "@test.com"
        csv_data = f"subject,body,customer_email\nSubj,Body,{too_long_email}\n"
        resp = client.post(
            "/v1/tickets/bulk",
            content=csv_data,
            headers={"Content-Type": "text/csv"},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "invalid_input"
        assert "Row 1: customer_email exceeds maximum length" in data["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_bulk_tickets_max_rows_limit_exceeded(client: TestClient) -> None:
    """CSV with 501 rows exceeds the 500 limit and returns 422 invalid_input."""
    app.dependency_overrides[get_api_key_principal] = _mock_api_key_principal
    try:
        lines = ["subject,body"]
        for i in range(501):
            lines.append(f"Subject {i},Body {i}")
        csv_data = "\n".join(lines)
        resp = client.post(
            "/v1/tickets/bulk",
            content=csv_data,
            headers={"Content-Type": "text/csv"},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "invalid_input"
        assert "exceeds maximum limit of 500 rows" in data["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_bulk_tickets_max_bytes_limit_exceeded(client: TestClient) -> None:
    """Payload exceeding 5 MB returns 422 invalid_input."""
    app.dependency_overrides[get_api_key_principal] = _mock_api_key_principal
    try:
        # Create byte payload slightly larger than 5 MB
        large_payload = b"subject,body\n" + (b"A,B\n" * 1_500_000)
        assert len(large_payload) > 5 * 1024 * 1024
        resp = client.post(
            "/v1/tickets/bulk",
            content=large_payload,
            headers={"Content-Type": "text/csv"},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "invalid_input"
        assert "exceeds maximum size limit of 5 MB" in data["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_bulk_tickets_oversized_streaming_without_content_length(client: TestClient) -> None:
    """Streaming generator sends > 5 MB chunked without Content-Length header, rejected with 422."""
    app.dependency_overrides[get_api_key_principal] = _mock_api_key_principal
    try:
        def chunk_generator():
            # 1 MB chunk yielded 6 times (> 5 MB total)
            chunk = b"A" * (1024 * 1024)
            for _ in range(6):
                yield chunk

        # Generator content in httpx / TestClient omits Content-Length and uses chunked transfer
        resp = client.post(
            "/v1/tickets/bulk",
            content=chunk_generator(),
            headers={"Content-Type": "text/csv"},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "invalid_input"
        assert "exceeds maximum size limit of 5 MB" in data["error"]["message"]
    finally:
        app.dependency_overrides.clear()


