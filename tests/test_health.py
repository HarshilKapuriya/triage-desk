"""Focused unit test for GET /healthz endpoint.

Verifies:
- GET /healthz returns HTTP 200 with {"status": "ok"}.
- Runs cleanly without requiring real database credentials or a network connection.
- Injects a no-op lifespan context so no database connections are attempted.

NOTE: This is a unit test of the application routing and liveness contract.
It does NOT verify PostgreSQL database connectivity.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app


@asynccontextmanager
async def noop_lifespan(test_app: FastAPI) -> AsyncIterator[None]:
    """Substitute lifespan for unit tests to bypass database connection attempts."""
    yield


def test_healthz_returns_ok_unit() -> None:
    """Test that /healthz returns 200 OK and {"status": "ok"} without connecting to DB."""
    # Override the app lifespan with no-op to isolate the endpoint from external network/DB
    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = noop_lifespan
    try:
        with TestClient(app) as client:
            response = client.get("/healthz")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
    finally:
        app.router.lifespan_context = original_lifespan

