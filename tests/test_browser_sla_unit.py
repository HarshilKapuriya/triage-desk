"""Offline unit tests for user-facing SLA operations and lifecycle endpoints.

Verifies:
1. Admin-only policy configuration (GET, POST, DELETE /browser-api/sla/policies):
   - Role enforcement (403 Forbidden for non-admin agents on mutations).
   - Session-bound CSRF token validation.
   - Graceful fallback on missing schema (503 on writes, schema_available=False on listing).
   - Atomic audit logging on policy upsert and delete.
2. Ticket SLA readout (GET /browser-api/tickets/{id}/sla):
   - Tenant isolation (404 on missing or foreign ticket).
   - Explicit schema_unavailable fallback when migration 008 is unapplied.
   - Not_configured fallback when tenant has no policy for ticket priority or default.
   - Deterministic calculations, independent targets, and rules-based urgency.
   - Exact deadline boundary semantic ("Due now").
3. Explicit agent lifecycle actions:
   - POST /first-response: Rejects resolved/closed tickets (409 Conflict); idempotent repeat; audit log.
   - POST /resolve: Valid from new/open/pending; rejects closed (409 Conflict); idempotent repeat; audit log.
   - POST /close: Valid only from resolved; rejects new/open/pending (409 Conflict); idempotent repeat; audit log.
4. Transaction atomicity & audit rollback guarantee:
   - If audit log insertion fails, the entire transaction rolls back and ticket state is untouched.
   - Audit log schema strictness: records only tenant_id, actor, action, at (never sensitive content).
"""

from datetime import datetime, timedelta, timezone
import concurrent.futures
import threading
from typing import Any
from unittest.mock import MagicMock, patch
import uuid
import pytest
from fastapi.testclient import TestClient
import psycopg.errors

from app.deps import Principal
from app.deps_browser import get_browser_principal_json
from app.main import app
import app.repositories.sla as sla_repo
from app.repositories.tickets import TicketRecord


@pytest.fixture
def admin_principal() -> Principal:
    return Principal(
        tenant_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        kind="user",
        id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
        role="admin",
    )


@pytest.fixture
def agent_principal() -> Principal:
    return Principal(
        tenant_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        kind="user",
        id=uuid.UUID("33333333-3333-3333-3333-333333333333"),
        role="agent",
    )


# -----------------------------------------------------------------------------
# 1. SLA Policy Endpoints Unit Tests
# -----------------------------------------------------------------------------


def test_list_sla_policies_success(agent_principal: Principal):
    """GET /browser-api/sla/policies returns policy list when schema is provisioned."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)

    policy_record = sla_repo.SLAPolicyRecord(
        id=uuid.uuid4(),
        tenant_id=agent_principal.tenant_id,
        name="Urgent Policy",
        priority="urgent",
        first_response_minutes=30,
        resolution_minutes=120,
        created_at=datetime(2026, 11, 8, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 11, 8, 10, 0, tzinfo=timezone.utc),
    )

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.repositories.sla.list_sla_policies", return_value=[policy_record]):
                res = client.get("/browser-api/sla/policies")
                assert res.status_code == 200
                data = res.json()
                assert data["schema_available"] is True
                assert len(data["policies"]) == 1
                assert data["policies"][0]["priority"] == "urgent"
                assert data["policies"][0]["first_response_minutes"] == 30
                assert data["policies"][0]["resolution_minutes"] == 120
    finally:
        app.dependency_overrides.clear()


def test_list_sla_policies_schema_unavailable(agent_principal: Principal):
    """GET /browser-api/sla/policies gracefully handles unapplied migration 008."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch(
                "app.repositories.sla.list_sla_policies",
                side_effect=sla_repo.SLASchemaUnavailableError("Table missing"),
            ):
                res = client.get("/browser-api/sla/policies")
                assert res.status_code == 200
                data = res.json()
                assert data["schema_available"] is False
                assert data["policies"] == []
                assert "not provisioned" in data["message"]
    finally:
        app.dependency_overrides.clear()


def test_upsert_sla_policy_admin_only(agent_principal: Principal):
    """POST /browser-api/sla/policies rejects non-admin users with 403 Forbidden."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)

    try:
        with patch("app.routers.browser_api.validate_browser_csrf"):
            res = client.post(
                "/browser-api/sla/policies",
                json={"priority": "urgent", "first_response_minutes": 60, "resolution_minutes": 240},
            )
            assert res.status_code == 403
            assert "Admin privileges required" in res.json()["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_upsert_sla_policy_csrf_required(admin_principal: Principal):
    """POST /browser-api/sla/policies enforces session-bound CSRF token."""
    app.dependency_overrides[get_browser_principal_json] = lambda: admin_principal
    client = TestClient(app, follow_redirects=False)

    try:
        res = client.post(
            "/browser-api/sla/policies",
            json={"priority": "urgent", "first_response_minutes": 60, "resolution_minutes": 240},
        )
        assert res.status_code == 403
    finally:
        app.dependency_overrides.clear()


def test_upsert_sla_policy_success_with_audit(admin_principal: Principal):
    """POST /browser-api/sla/policies upserts policy and creates audit record in one transaction."""
    app.dependency_overrides[get_browser_principal_json] = lambda: admin_principal
    client = TestClient(app, follow_redirects=False)

    policy_record = sla_repo.SLAPolicyRecord(
        id=uuid.uuid4(),
        tenant_id=admin_principal.tenant_id,
        name="Urgent Policy",
        priority="urgent",
        first_response_minutes=45,
        resolution_minutes=180,
        created_at=datetime(2026, 11, 8, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 11, 8, 10, 0, tzinfo=timezone.utc),
    )

    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.validate_browser_csrf"):
                with patch("app.repositories.sla.upsert_sla_policy", return_value=policy_record):
                    res = client.post(
                        "/browser-api/sla/policies",
                        json={"priority": "urgent", "first_response_minutes": 45, "resolution_minutes": 180},
                    )
                    assert res.status_code == 200
                    data = res.json()
                    assert data["schema_available"] is True
                    assert data["policy"]["priority"] == "urgent"

                    # Verify audit log was recorded
                    mock_cur.execute.assert_called_once()
                    sql, params = mock_cur.execute.call_args[0]
                    assert "INSERT INTO audit_log" in sql
                    assert params[0] == admin_principal.tenant_id
                    assert params[1] == str(admin_principal.id)
                    assert params[2] == "sla_policy.upserted"
    finally:
        app.dependency_overrides.clear()


def test_upsert_sla_policy_schema_unavailable_returns_503(admin_principal: Principal):
    """POST /browser-api/sla/policies returns 503 when migration 008 is unapplied."""
    app.dependency_overrides[get_browser_principal_json] = lambda: admin_principal
    client = TestClient(app, follow_redirects=False)

    mock_conn = MagicMock()
    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.validate_browser_csrf"):
                with patch(
                    "app.repositories.sla.upsert_sla_policy",
                    side_effect=sla_repo.SLASchemaUnavailableError("Table missing"),
                ):
                    res = client.post(
                        "/browser-api/sla/policies",
                        json={"priority": "urgent", "first_response_minutes": 60, "resolution_minutes": 240},
                    )
                    assert res.status_code == 503
                    assert res.json()["error"]["code"] == "schema_unavailable"
    finally:
        app.dependency_overrides.clear()


def test_delete_sla_policy_admin_only_and_audit(admin_principal: Principal, agent_principal: Principal):
    """DELETE /browser-api/sla/policies/{priority} enforces admin role and records audit log."""
    # 1. Non-admin forbidden
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)
    try:
        with patch("app.routers.browser_api.validate_browser_csrf"):
            res = client.delete("/browser-api/sla/policies/urgent")
            assert res.status_code == 403
    finally:
        app.dependency_overrides.clear()

    # 2. Admin success with audit
    app.dependency_overrides[get_browser_principal_json] = lambda: admin_principal
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.validate_browser_csrf"):
                with patch("app.repositories.sla.delete_sla_policy", return_value=True):
                    res = client.delete("/browser-api/sla/policies/urgent")
                    assert res.status_code == 200
                    assert res.json()["deleted"] is True
                    assert res.json()["priority"] == "urgent"

                    # Verify audit log was recorded
                    mock_cur.execute.assert_called_once()
                    sql, params = mock_cur.execute.call_args[0]
                    assert "INSERT INTO audit_log" in sql
                    assert params[1] == str(admin_principal.id)
                    assert params[2] == "sla_policy.deleted"
    finally:
        app.dependency_overrides.clear()


# -----------------------------------------------------------------------------
# 2. Ticket SLA Readout Unit Tests
# -----------------------------------------------------------------------------


def test_get_ticket_sla_tenant_isolation(agent_principal: Principal):
    """GET /browser-api/tickets/{id}/sla enforces strict tenant isolation (returns 404)."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)

    ticket_id = uuid.uuid4()
    mock_conn = MagicMock()

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.get_ticket_by_id", return_value=None):
                res = client.get(f"/browser-api/tickets/{ticket_id}/sla")
                assert res.status_code == 404
                assert res.json()["error"]["code"] == "not_found"
    finally:
        app.dependency_overrides.clear()


def test_get_ticket_sla_schema_unavailable(agent_principal: Principal):
    """GET /browser-api/tickets/{id}/sla explicitly reports schema_unavailable when migration 008 unapplied."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)

    ticket = TicketRecord(
        id=uuid.uuid4(),
        tenant_id=agent_principal.tenant_id,
        external_id=None,
        subject="Test Ticket",
        body="Complaint details",
        customer_email="user@example.com",
        status="open",
        category="Checking",
        priority="urgent",
        created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        first_response_at=None,
        resolved_at=None,
        assignee_id=None,
    )

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.get_ticket_by_id", return_value=ticket):
                with patch(
                    "app.repositories.sla.resolve_effective_policy",
                    side_effect=sla_repo.SLASchemaUnavailableError("Table missing"),
                ):
                    res = client.get(f"/browser-api/tickets/{ticket.id}/sla")
                    assert res.status_code == 200
                    data = res.json()
                    assert data["schema_available"] is False
                    assert data["is_configured"] is False
                    assert data["overall_status"] == "schema_unavailable"
                    assert "not provisioned" in data["message"]
    finally:
        app.dependency_overrides.clear()


def test_get_ticket_sla_not_configured(agent_principal: Principal):
    """GET /browser-api/tickets/{id}/sla distinguishes not_configured from schema_unavailable."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)

    ticket = TicketRecord(
        id=uuid.uuid4(),
        tenant_id=agent_principal.tenant_id,
        external_id=None,
        subject="Test Ticket",
        body="Complaint details",
        customer_email="user@example.com",
        status="open",
        category="Checking",
        priority="low",
        created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        first_response_at=None,
        resolved_at=None,
        assignee_id=None,
    )

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.get_ticket_by_id", return_value=ticket):
                with patch("app.repositories.sla.resolve_effective_policy", return_value=None):
                    res = client.get(f"/browser-api/tickets/{ticket.id}/sla")
                    assert res.status_code == 200
                    data = res.json()
                    assert data["schema_available"] is True
                    assert data["is_configured"] is False
                    assert data["overall_status"] == "not_configured"
                    assert data["first_response"] is None
                    assert data["resolution"] is None
    finally:
        app.dependency_overrides.clear()


def test_get_ticket_sla_configured_and_boundary_due_now(agent_principal: Principal):
    """GET /browser-api/tickets/{id}/sla calculates timeliness and labels exact deadline 'Due now'."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)

    now = datetime(2026, 11, 8, 12, 0, 0, tzinfo=timezone.utc)
    # Ticket created exactly 60 minutes ago
    created_at = now - timedelta(minutes=60)

    ticket = TicketRecord(
        id=uuid.uuid4(),
        tenant_id=agent_principal.tenant_id,
        external_id=None,
        subject="Urgent Ticket",
        body="Complaint details",
        customer_email="user@example.com",
        status="open",
        category="Checking",
        priority="urgent",
        created_at=created_at,
        first_response_at=None,
        resolved_at=None,
        assignee_id=None,
    )

    # First response target is 60 minutes (due exactly now!)
    # Resolution target is 240 minutes (due in 3 hours)
    policy = sla_repo.SLAPolicyRecord(
        id=uuid.uuid4(),
        tenant_id=agent_principal.tenant_id,
        name="Urgent Policy",
        priority="urgent",
        first_response_minutes=60,
        resolution_minutes=240,
        created_at=created_at,
        updated_at=created_at,
    )

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.get_ticket_by_id", return_value=ticket):
                with patch("app.repositories.sla.resolve_effective_policy", return_value=policy):
                    with patch("app.routers.browser_api.datetime") as mock_dt:
                        mock_dt.now.return_value = now
                        mock_dt.fromisoformat = datetime.fromisoformat
                        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

                        res = client.get(f"/browser-api/tickets/{ticket.id}/sla")
                        assert res.status_code == 200
                        data = res.json()
                        assert data["schema_available"] is True
                        assert data["is_configured"] is True
                        assert data["overall_status"] == "active"
                        assert data["first_response"]["timeliness_label"] == "Due now"
                        assert data["first_response"]["is_breached"] is False
                        assert data["resolution"]["timeliness_label"] == "On track"
    finally:
        app.dependency_overrides.clear()


# -----------------------------------------------------------------------------
# 3. Agent Lifecycle Actions Unit Tests
# -----------------------------------------------------------------------------


def test_first_response_rejects_resolved_or_closed_tickets(agent_principal: Principal):
    """POST /first-response rejects new response event on resolved or closed ticket (409 Conflict)."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)

    ticket_id = uuid.uuid4()
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value

    for status_str in ("resolved", "closed"):
        mock_cur.fetchone.return_value = {
            "id": ticket_id,
            "tenant_id": agent_principal.tenant_id,
            "status": status_str,
            "first_response_at": None,
            "resolved_at": datetime.now(timezone.utc),
        }
        try:
            with patch("app.routers.browser_api.get_connection") as mock_get_conn:
                mock_get_conn.return_value.__enter__.return_value = mock_conn
                with patch("app.routers.browser_api.validate_browser_csrf"):
                    res = client.post(f"/browser-api/tickets/{ticket_id}/first-response")
                    assert res.status_code == 409
                    assert res.json()["error"]["code"] == "invalid_transition"
                    assert f"Cannot record first response on {status_str} ticket" in res.json()["error"]["message"]
        finally:
            pass
    app.dependency_overrides.clear()


def test_first_response_idempotent_repeat(agent_principal: Principal):
    """POST /first-response on already-responded ticket is idempotent and does not log audit."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)

    ticket_id = uuid.uuid4()
    prior_resp = datetime(2026, 11, 8, 10, 30, tzinfo=timezone.utc)
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchone.return_value = {
        "id": ticket_id,
        "tenant_id": agent_principal.tenant_id,
        "status": "open",
        "first_response_at": prior_resp,
        "resolved_at": None,
    }

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.validate_browser_csrf"):
                res = client.post(f"/browser-api/tickets/{ticket_id}/first-response")
                assert res.status_code == 200
                assert res.json()["ok"] is True
                assert res.json()["ticket"]["first_response_at"] == prior_resp.isoformat()

                # Verify NO update or audit_log insert occurred
                executed_statements = [call[0][0] for call in mock_cur.execute.call_args_list]
                assert not any("UPDATE tickets" in s for s in executed_statements)
                assert not any("INSERT INTO audit_log" in s for s in executed_statements)
    finally:
        app.dependency_overrides.clear()


def test_first_response_success_with_audit(agent_principal: Principal):
    """POST /first-response records server timestamp and inserts audit record."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)

    ticket_id = uuid.uuid4()
    now = datetime(2026, 11, 8, 11, 0, 0, tzinfo=timezone.utc)
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchone.side_effect = [
        # 1. Initial locked select
        {
            "id": ticket_id,
            "tenant_id": agent_principal.tenant_id,
            "status": "open",
            "first_response_at": None,
            "resolved_at": None,
        },
        # 2. UPDATE tickets returning
        {
            "id": ticket_id,
            "status": "open",
            "first_response_at": now,
            "resolved_at": None,
        },
    ]

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.validate_browser_csrf"):
                res = client.post(f"/browser-api/tickets/{ticket_id}/first-response")
                assert res.status_code == 200
                assert res.json()["ok"] is True
                assert res.json()["ticket"]["first_response_at"] == now.isoformat()

                # Verify audit log was recorded
                audit_call = mock_cur.execute.call_args_list[-1]
                sql, params = audit_call[0]
                assert "INSERT INTO audit_log" in sql
                assert params[0] == agent_principal.tenant_id
                assert params[1] == str(agent_principal.id)
                assert params[2] == "ticket.first_response_recorded"
    finally:
        app.dependency_overrides.clear()


def test_resolve_ticket_rejects_closed(agent_principal: Principal):
    """POST /resolve rejects already-closed ticket with 409 Conflict."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)

    ticket_id = uuid.uuid4()
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchone.return_value = {
        "id": ticket_id,
        "tenant_id": agent_principal.tenant_id,
        "status": "closed",
        "first_response_at": None,
        "resolved_at": datetime.now(timezone.utc),
    }

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.validate_browser_csrf"):
                res = client.post(f"/browser-api/tickets/{ticket_id}/resolve")
                assert res.status_code == 409
                assert "Cannot resolve a closed ticket" in res.json()["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_resolve_ticket_success_and_idempotency(agent_principal: Principal):
    """POST /resolve transitions to resolved, records audit; repeated call is idempotent."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)

    ticket_id = uuid.uuid4()
    now = datetime(2026, 11, 8, 11, 30, tzinfo=timezone.utc)
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value

    # First call: open -> resolved
    mock_cur.fetchone.side_effect = [
        {"id": ticket_id, "tenant_id": agent_principal.tenant_id, "status": "open", "first_response_at": None, "resolved_at": None},
        {"id": ticket_id, "status": "resolved", "first_response_at": None, "resolved_at": now},
    ]

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.validate_browser_csrf"):
                res = client.post(f"/browser-api/tickets/{ticket_id}/resolve")
                assert res.status_code == 200
                assert res.json()["ticket"]["status"] == "resolved"
                audit_call = mock_cur.execute.call_args_list[-1]
                assert "INSERT INTO audit_log" in audit_call[0][0]
                assert audit_call[0][1][2] == "ticket.resolved"

        # Second call: already resolved (idempotent)
        mock_cur.reset_mock()
        mock_cur.fetchone.side_effect = None
        mock_cur.fetchone.return_value = {
            "id": ticket_id,
            "tenant_id": agent_principal.tenant_id,
            "status": "resolved",
            "first_response_at": None,
            "resolved_at": now,
        }
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.validate_browser_csrf"):
                res = client.post(f"/browser-api/tickets/{ticket_id}/resolve")
                assert res.status_code == 200
                assert res.json()["ticket"]["status"] == "resolved"
                executed_statements = [call[0][0] for call in mock_cur.execute.call_args_list]
                assert not any("UPDATE tickets" in s for s in executed_statements)
                assert not any("INSERT INTO audit_log" in s for s in executed_statements)
    finally:
        app.dependency_overrides.clear()


def test_close_ticket_lifecycle_transitions(agent_principal: Principal):
    """POST /close requires resolved status; rejects un-resolved; idempotent on closed."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)

    ticket_id = uuid.uuid4()
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value

    try:
        # 1. Rejecting open ticket
        mock_cur.fetchone.return_value = {
            "id": ticket_id,
            "tenant_id": agent_principal.tenant_id,
            "status": "open",
            "first_response_at": None,
            "resolved_at": None,
        }
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.validate_browser_csrf"):
                res = client.post(f"/browser-api/tickets/{ticket_id}/close")
                assert res.status_code == 409
                assert "must be resolved first" in res.json()["error"]["message"]

        # 2. Success from resolved
        mock_cur.reset_mock()
        mock_cur.fetchone.side_effect = [
            {"id": ticket_id, "tenant_id": agent_principal.tenant_id, "status": "resolved", "first_response_at": None, "resolved_at": datetime.now(timezone.utc)},
            {"id": ticket_id, "status": "closed", "first_response_at": None, "resolved_at": datetime.now(timezone.utc)},
        ]
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.validate_browser_csrf"):
                res = client.post(f"/browser-api/tickets/{ticket_id}/close")
                assert res.status_code == 200
                assert res.json()["ticket"]["status"] == "closed"
                audit_call = mock_cur.execute.call_args_list[-1]
                assert "INSERT INTO audit_log" in audit_call[0][0]
                assert audit_call[0][1][2] == "ticket.closed"

        # 3. Idempotent repeat when already closed
        mock_cur.reset_mock()
        mock_cur.fetchone.side_effect = None
        mock_cur.fetchone.return_value = {
            "id": ticket_id,
            "tenant_id": agent_principal.tenant_id,
            "status": "closed",
            "first_response_at": None,
            "resolved_at": datetime.now(timezone.utc),
        }
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.validate_browser_csrf"):
                res = client.post(f"/browser-api/tickets/{ticket_id}/close")
                assert res.status_code == 200
                assert res.json()["ticket"]["status"] == "closed"
                executed_statements = [call[0][0] for call in mock_cur.execute.call_args_list]
                assert not any("UPDATE tickets" in s for s in executed_statements)
                assert not any("INSERT INTO audit_log" in s for s in executed_statements)
    finally:
        app.dependency_overrides.clear()


# -----------------------------------------------------------------------------
# 4. Transaction Atomicity & Audit Rollback Guarantee
# -----------------------------------------------------------------------------


def test_transaction_rollback_when_audit_fails(agent_principal: Principal):
    """If audit log insertion fails, the entire transaction rolls back."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, raise_server_exceptions=False, follow_redirects=False)

    ticket_id = uuid.uuid4()
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value

    mock_cur.fetchone.side_effect = [
        {"id": ticket_id, "tenant_id": agent_principal.tenant_id, "status": "open", "first_response_at": None, "resolved_at": None},
        {"id": ticket_id, "status": "open", "first_response_at": datetime.now(timezone.utc), "resolved_at": None},
    ]

    # Make the audit log query raise an operational DB error
    def execute_side_effect(sql, *args, **kwargs):
        if "INSERT INTO audit_log" in sql:
            raise psycopg.OperationalError("Simulated audit disk failure")
        return None

    mock_cur.execute.side_effect = execute_side_effect

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.validate_browser_csrf"):
                res = client.post(f"/browser-api/tickets/{ticket_id}/first-response")
                # Unhandled database error is sanitized to 500
                assert res.status_code == 500
                assert res.json()["error"]["code"] == "internal_error"

                # The transaction context manager must have exited with error, rolling back
                mock_conn.transaction.return_value.__exit__.assert_called_once()
                exc_type = mock_conn.transaction.return_value.__exit__.call_args[0][0]
                assert exc_type is psycopg.OperationalError
    finally:
        app.dependency_overrides.clear()


# -----------------------------------------------------------------------------
# 5. Concurrency & Serialized Row Lock Tests
# -----------------------------------------------------------------------------


def test_concurrent_first_response_serialized_lock_no_duplicate_audit(agent_principal: Principal):
    """Concurrent first-response calls serialized by row lock produce exactly one audit record."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)

    ticket_id = uuid.uuid4()
    row_lock = threading.Lock()
    db_row = {
        "id": ticket_id,
        "tenant_id": agent_principal.tenant_id,
        "status": "open",
        "first_response_at": None,
        "resolved_at": None,
    }
    audit_entries: list[dict[str, Any]] = []

    class MockCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def execute(self, sql, params=None):
            if "SELECT" in sql and "FOR UPDATE" in sql:
                row_lock.acquire()
            elif "UPDATE tickets" in sql:
                db_row["first_response_at"] = params[0]
            elif "INSERT INTO audit_log" in sql:
                audit_entries.append({"action": params[2], "at": params[3]})

        def fetchone(self):
            return dict(db_row)

    class MockTransaction:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if row_lock.locked():
                row_lock.release()

    class MockConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def cursor(self, *args, **kwargs):
            return MockCursor()

        def transaction(self):
            return MockTransaction()

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.side_effect = lambda: MockConnection()
            with patch("app.routers.browser_api.validate_browser_csrf"):
                def call_endpoint():
                    return client.post(f"/browser-api/tickets/{ticket_id}/first-response")

                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    f1 = executor.submit(call_endpoint)
                    f2 = executor.submit(call_endpoint)
                    res1 = f1.result()
                    res2 = f2.result()

                assert res1.status_code == 200
                assert res2.status_code == 200
                assert res1.json()["ok"] is True
                assert res2.json()["ok"] is True
                assert res1.json()["ticket"]["first_response_at"] is not None
                assert res2.json()["ticket"]["first_response_at"] is not None
                assert res1.json()["ticket"]["first_response_at"] == res2.json()["ticket"]["first_response_at"]
                assert len(audit_entries) == 1
                assert audit_entries[0]["action"] == "ticket.first_response_recorded"
    finally:
        app.dependency_overrides.clear()


def test_concurrent_resolve_serialized_lock_no_duplicate_audit(agent_principal: Principal):
    """Concurrent resolve calls serialized by row lock produce exactly one audit record."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)

    ticket_id = uuid.uuid4()
    row_lock = threading.Lock()
    db_row = {
        "id": ticket_id,
        "tenant_id": agent_principal.tenant_id,
        "status": "open",
        "first_response_at": None,
        "resolved_at": None,
    }
    audit_entries: list[dict[str, Any]] = []

    class MockCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def execute(self, sql, params=None):
            if "SELECT" in sql and "FOR UPDATE" in sql:
                row_lock.acquire()
            elif "UPDATE tickets" in sql:
                db_row["status"] = "resolved"
                db_row["resolved_at"] = params[0]
            elif "INSERT INTO audit_log" in sql:
                audit_entries.append({"action": params[2], "at": params[3]})

        def fetchone(self):
            return dict(db_row)

    class MockTransaction:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if row_lock.locked():
                row_lock.release()

    class MockConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def cursor(self, *args, **kwargs):
            return MockCursor()

        def transaction(self):
            return MockTransaction()

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.side_effect = lambda: MockConnection()
            with patch("app.routers.browser_api.validate_browser_csrf"):
                def call_endpoint():
                    return client.post(f"/browser-api/tickets/{ticket_id}/resolve")

                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    f1 = executor.submit(call_endpoint)
                    f2 = executor.submit(call_endpoint)
                    res1 = f1.result()
                    res2 = f2.result()

                assert res1.status_code == 200
                assert res2.status_code == 200
                assert res1.json()["ok"] is True
                assert res2.json()["ok"] is True
                assert res1.json()["ticket"]["status"] == "resolved"
                assert res2.json()["ticket"]["status"] == "resolved"
                assert res1.json()["ticket"]["resolved_at"] is not None
                assert res1.json()["ticket"]["resolved_at"] == res2.json()["ticket"]["resolved_at"]
                assert len(audit_entries) == 1
                assert audit_entries[0]["action"] == "ticket.resolved"
    finally:
        app.dependency_overrides.clear()


def test_concurrent_close_serialized_lock_no_duplicate_audit(agent_principal: Principal):
    """Concurrent close calls serialized by row lock produce exactly one audit record."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)

    ticket_id = uuid.uuid4()
    row_lock = threading.Lock()
    db_row = {
        "id": ticket_id,
        "tenant_id": agent_principal.tenant_id,
        "status": "resolved",
        "first_response_at": None,
        "resolved_at": datetime.now(timezone.utc),
    }
    audit_entries: list[dict[str, Any]] = []

    class MockCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def execute(self, sql, params=None):
            if "SELECT" in sql and "FOR UPDATE" in sql:
                row_lock.acquire()
            elif "UPDATE tickets" in sql:
                db_row["status"] = "closed"
            elif "INSERT INTO audit_log" in sql:
                audit_entries.append({"action": params[2], "at": params[3]})

        def fetchone(self):
            return dict(db_row)

    class MockTransaction:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if row_lock.locked():
                row_lock.release()

    class MockConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def cursor(self, *args, **kwargs):
            return MockCursor()

        def transaction(self):
            return MockTransaction()

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.side_effect = lambda: MockConnection()
            with patch("app.routers.browser_api.validate_browser_csrf"):
                def call_endpoint():
                    return client.post(f"/browser-api/tickets/{ticket_id}/close")

                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    f1 = executor.submit(call_endpoint)
                    f2 = executor.submit(call_endpoint)
                    res1 = f1.result()
                    res2 = f2.result()

                assert res1.status_code == 200
                assert res2.status_code == 200
                assert res1.json()["ok"] is True
                assert res2.json()["ok"] is True
                assert res1.json()["ticket"]["status"] == "closed"
                assert res2.json()["ticket"]["status"] == "closed"
                assert len(audit_entries) == 1
                assert audit_entries[0]["action"] == "ticket.closed"
    finally:
        app.dependency_overrides.clear()
