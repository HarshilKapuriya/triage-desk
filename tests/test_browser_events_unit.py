"""Offline unit tests for tenant-scoped ticket lifecycle events endpoint and repository read path.

Note on scope & honesty:
- Uses mocked connections, cursors, and repositories (no live PostgreSQL connection).
- Does NOT prove PostgreSQL engine-level transaction/SAVEPOINT recovery or database RLS/FK enforcement
  (those require disposable PostgreSQL integration tests).
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import uuid

from fastapi.testclient import TestClient
import psycopg
import psycopg.errors
import pytest

from app.deps import Principal
from app.deps_browser import get_browser_principal_json
from app.main import app
import app.repositories.ticket_events as ticket_events_repo
from app.repositories.tickets import TicketRecord


def _make_ticket(tenant_id: uuid.UUID, ticket_id: uuid.UUID | None = None) -> TicketRecord:
    tid = ticket_id or uuid.uuid4()
    return TicketRecord(
        id=tid,
        tenant_id=tenant_id,
        external_id="EXT-SECRET-999",
        subject="Sensitive Subject",
        body="Sensitive customer narrative with email user@example.com",
        customer_email="customer@example.com",
        status="open",
        category="Checking or savings account",
        priority="high",
        created_at=datetime(2026, 11, 10, 9, 0, 0, tzinfo=timezone.utc),
        first_response_at=None,
        resolved_at=None,
        assignee_id=None,
    )


@pytest.fixture
def agent_principal() -> Principal:
    return Principal(
        tenant_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        kind="user",
        id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
        role="agent",
    )


@pytest.fixture
def admin_principal() -> Principal:
    return Principal(
        tenant_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        kind="user",
        id=uuid.UUID("33333333-3333-3333-3333-333333333333"),
        role="admin",
    )


def test_get_ticket_events_requires_authentication() -> None:
    """Unauthenticated GET /browser-api/tickets/{id}/events returns 401 JSON with no-store."""
    app.dependency_overrides.clear()
    client = TestClient(app, follow_redirects=False)
    ticket_id = uuid.uuid4()

    res = client.get(f"/browser-api/tickets/{ticket_id}/events")
    assert res.status_code == 401
    assert res.json()["error"]["code"] == "unauthenticated"
    assert res.headers.get("cache-control") == "no-store"


def test_get_ticket_events_allows_agent_and_admin_roles(
    agent_principal: Principal,
    admin_principal: Principal,
) -> None:
    """Both agent and admin browser roles can read tenant-scoped ticket lifecycle events."""
    client = TestClient(app, follow_redirects=False)

    for principal in (agent_principal, admin_principal):
        ticket = _make_ticket(principal.tenant_id)
        app.dependency_overrides[get_browser_principal_json] = lambda p=principal: p
        mock_conn = MagicMock()
        try:
            with patch("app.routers.browser_api.get_connection") as mock_get_conn:
                mock_get_conn.return_value.__enter__.return_value = mock_conn
                with patch("app.routers.browser_api.get_ticket_by_id", return_value=ticket):
                    with patch("app.repositories.ticket_events.list_ticket_events", return_value=[]):
                        res = client.get(f"/browser-api/tickets/{ticket.id}/events")
                        assert res.status_code == 200
                        assert res.headers.get("cache-control") == "no-store"
        finally:
            app.dependency_overrides.clear()


def test_get_ticket_events_invalid_uuid_and_cross_tenant_indistinguishable(
    agent_principal: Principal,
) -> None:
    """Invalid UUID, missing ticket, and cross-tenant ticket all return identical 404 without reading events."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)

    try:
        # Invalid UUID
        res_invalid = client.get("/browser-api/tickets/not-a-valid-uuid/events")
        assert res_invalid.status_code == 404
        assert res_invalid.json() == {"error": {"code": "not_found", "message": "Ticket not found."}}

        # Missing or foreign-tenant ticket (tenant-scoped lookup returns None)
        foreign_ticket_id = uuid.uuid4()
        mock_conn = MagicMock()
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.get_ticket_by_id", return_value=None) as mock_get_ticket:
                with patch("app.repositories.ticket_events.list_ticket_events") as mock_list_events:
                    res_foreign = client.get(f"/browser-api/tickets/{foreign_ticket_id}/events")
                    assert res_foreign.status_code == 404
                    assert res_foreign.json() == res_invalid.json()
                    mock_get_ticket.assert_called_once_with(mock_conn, agent_principal.tenant_id, foreign_ticket_id)
                    mock_list_events.assert_not_called()
    finally:
        app.dependency_overrides.clear()


def test_get_ticket_events_passes_principal_tenant_and_ticket_uuid_to_repo(
    agent_principal: Principal,
) -> None:
    """Router resolves tenant_id strictly from authenticated principal and passes bounded read limit."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)
    ticket = _make_ticket(agent_principal.tenant_id)
    mock_conn = MagicMock()

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.get_ticket_by_id", return_value=ticket) as mock_get_ticket:
                with patch("app.repositories.ticket_events.list_ticket_events", return_value=[]) as mock_list_events:
                    # Client-supplied tenant_id query parameter must have zero effect
                    bogus_tenant = uuid.uuid4()
                    res = client.get(f"/browser-api/tickets/{ticket.id}/events?tenant_id={bogus_tenant}")
                    assert res.status_code == 200
                    mock_get_ticket.assert_called_once_with(mock_conn, agent_principal.tenant_id, ticket.id)
                    mock_list_events.assert_called_once_with(
                        mock_conn,
                        agent_principal.tenant_id,
                        ticket.id,
                        limit=ticket_events_repo.BOUNDED_EVENT_READ_LIMIT,
                    )
    finally:
        app.dependency_overrides.clear()


def test_get_ticket_events_allowlists_fields_and_strips_identifiers_and_pii(
    agent_principal: Principal,
) -> None:
    """Response exposes only allowlisted fields; actor_id, policy_id, model_version_id, and PII are stripped."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)
    ticket = _make_ticket(agent_principal.tenant_id)

    secret_actor_uuid = "99999999-9999-9999-9999-999999999999"
    secret_worker_id = "worker-prod-node-42-pid-888"
    secret_policy_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    secret_model_ver_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    t0 = datetime(2026, 11, 10, 9, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(seconds=12)
    t2 = t0 + timedelta(minutes=15)
    t3 = t0 + timedelta(minutes=45)
    t4 = t0 + timedelta(hours=2)

    records = [
        ticket_events_repo.TicketEventRecord(
            id=uuid.uuid4(),
            tenant_id=agent_principal.tenant_id,
            ticket_id=ticket.id,
            event_type="ticket_created",
            occurred_at=t0,
            recorded_at=t0 + timedelta(milliseconds=5),
            actor_type="agent",
            actor_id=secret_actor_uuid,
            schema_version=1,
            payload={
                "schema_version": 1,
                "priority": "high",
                "category": "Checking or savings account",
                "policy_snapshot": {
                    "is_configured": True,
                    "policy_id": secret_policy_id,
                    "policy_priority": "high",
                    "first_response_minutes": 60,
                    "resolution_minutes": 480,
                },
                # Extra unexpected/PII keys must never leak
                "body": "DO NOT LEAK BODY",
                "customer_email": "secret@example.com",
                "external_id": "EXT-SECRET-999",
            },
        ),
        ticket_events_repo.TicketEventRecord(
            id=uuid.uuid4(),
            tenant_id=agent_principal.tenant_id,
            ticket_id=ticket.id,
            event_type="ticket_processed",
            occurred_at=t1,
            recorded_at=t1 + timedelta(seconds=1),
            actor_type="worker",
            actor_id=secret_worker_id,
            schema_version=1,
            payload={
                "schema_version": 1,
                "from_status": "new",
                "to_status": "open",
                "predicted_category": "Credit card",
                "model_version_id": secret_model_ver_id,
                "internal_trace": "raw stack trace",
            },
        ),
        ticket_events_repo.TicketEventRecord(
            id=uuid.uuid4(),
            tenant_id=agent_principal.tenant_id,
            ticket_id=ticket.id,
            event_type="first_response_recorded",
            occurred_at=t2,
            recorded_at=t2 + timedelta(minutes=2),
            actor_type="agent",
            actor_id=secret_actor_uuid,
            schema_version=1,
            payload={
                "schema_version": 1,
                "responded_at": t2.isoformat(),
                "elapsed_calendar_seconds": 900.0,
            },
        ),
        ticket_events_repo.TicketEventRecord(
            id=uuid.uuid4(),
            tenant_id=agent_principal.tenant_id,
            ticket_id=ticket.id,
            event_type="ticket_resolved",
            occurred_at=t3,
            recorded_at=t3,
            actor_type="agent",
            actor_id=secret_actor_uuid,
            schema_version=1,
            payload={
                "schema_version": 1,
                "from_status": "open",
                "to_status": "resolved",
                "resolved_at": t3.isoformat(),
                "elapsed_calendar_seconds": 2700.0,
            },
        ),
        ticket_events_repo.TicketEventRecord(
            id=uuid.uuid4(),
            tenant_id=agent_principal.tenant_id,
            ticket_id=ticket.id,
            event_type="ticket_closed",
            occurred_at=t4,
            recorded_at=t4,
            actor_type="agent",
            actor_id=secret_actor_uuid,
            schema_version=1,
            payload={
                "schema_version": 1,
                "from_status": "resolved",
                "to_status": "closed",
                "closed_at": t4.isoformat(),
            },
        ),
    ]

    mock_conn = MagicMock()
    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.get_ticket_by_id", return_value=ticket):
                with patch("app.repositories.ticket_events.list_ticket_events", return_value=records):
                    res = client.get(f"/browser-api/tickets/{ticket.id}/events")
                    assert res.status_code == 200
                    assert res.headers.get("cache-control") == "no-store"

                    raw_text = res.text
                    for forbidden in (
                        secret_actor_uuid,
                        secret_worker_id,
                        secret_policy_id,
                        secret_model_ver_id,
                        "DO NOT LEAK BODY",
                        "secret@example.com",
                        "customer@example.com",
                        "EXT-SECRET-999",
                        "Sensitive Subject",
                        "raw stack trace",
                        '"actor_id"',
                        '"policy_id"',
                        '"model_version_id"',
                        '"tenant_id"',
                    ):
                        assert forbidden not in raw_text

                    body = res.json()
                    assert body["schema_available"] is True
                    assert body["ticket_id"] == str(ticket.id)
                    assert body["ordering"] == "occurred_at ASC, recorded_at ASC, id ASC"
                    assert len(body["events"]) == 5

                    ev0, ev1, ev2, ev3, ev4 = body["events"]
                    assert ev0["event_type"] == "ticket_created"
                    assert ev0["label"] == "Ticket created"
                    assert ev0["schema_version"] == 1
                    assert ev0["schema_supported"] is True
                    assert ev0["payload_valid"] is True
                    assert ev0["actor_type"] == "agent"
                    assert ev0["details"]["intake_policy"] == {
                        "is_configured": True,
                        "policy_priority": "high",
                        "first_response_minutes": 60,
                        "resolution_minutes": 480,
                    }

                    assert ev1["event_type"] == "ticket_processed"
                    assert ev1["label"] == "Processing completed"
                    assert ev1["details"]["from_status"] == "new"
                    assert ev1["details"]["to_status"] == "open"
                    assert ev1["details"]["predicted_category"] == "Credit card"

                    assert ev2["event_type"] == "first_response_recorded"
                    assert ev2["label"] == "First response recorded"
                    assert ev2["details"]["elapsed_calendar_seconds"] == 900.0
                    assert ev2["occurred_at"] != ev2["recorded_at"]

                    assert ev3["event_type"] == "ticket_resolved"
                    assert ev3["label"] == "Ticket resolved"
                    assert ev3["details"]["from_status"] == "open"
                    assert ev3["details"]["to_status"] == "resolved"
                    assert ev3["details"]["elapsed_calendar_seconds"] == 2700.0

                    assert ev4["event_type"] == "ticket_closed"
                    assert ev4["label"] == "Ticket closed"
                    assert ev4["details"]["from_status"] == "resolved"
                    assert ev4["details"]["to_status"] == "closed"
    finally:
        app.dependency_overrides.clear()


def test_get_ticket_events_unconfigured_intake_policy_and_empty_legacy_history(
    agent_principal: Principal,
) -> None:
    """Supports unconfigured policy_snapshot on creation and returns empty list for legacy tickets."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)
    ticket = _make_ticket(agent_principal.tenant_id)
    now = datetime(2026, 11, 10, 9, 0, 0, tzinfo=timezone.utc)

    unconfigured_created = ticket_events_repo.TicketEventRecord(
        id=uuid.uuid4(),
        tenant_id=agent_principal.tenant_id,
        ticket_id=ticket.id,
        event_type="ticket_created",
        occurred_at=now,
        recorded_at=now,
        actor_type="system",
        actor_id=None,
        schema_version=1,
        payload={
            "schema_version": 1,
            "priority": None,
            "category": None,
            "policy_snapshot": {
                "is_configured": False,
                "policy_id": None,
                "policy_priority": None,
                "first_response_minutes": None,
                "resolution_minutes": None,
            },
        },
    )

    mock_conn = MagicMock()
    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.get_ticket_by_id", return_value=ticket):
                # 1. Unconfigured intake policy
                with patch("app.repositories.ticket_events.list_ticket_events", return_value=[unconfigured_created]):
                    res = client.get(f"/browser-api/tickets/{ticket.id}/events")
                    assert res.status_code == 200
                    intake = res.json()["events"][0]["details"]["intake_policy"]
                    assert intake == {
                        "is_configured": False,
                        "policy_priority": None,
                        "first_response_minutes": None,
                        "resolution_minutes": None,
                    }

                # 2. Empty legacy history -> must NOT fabricate creation or other events
                with patch("app.repositories.ticket_events.list_ticket_events", return_value=[]):
                    res_empty = client.get(f"/browser-api/tickets/{ticket.id}/events")
                    assert res_empty.status_code == 200
                    data_empty = res_empty.json()
                    assert data_empty["schema_available"] is True
                    assert data_empty["events"] == []
    finally:
        app.dependency_overrides.clear()


def test_get_ticket_events_unsupported_schema_version_omits_details_without_guessing(
    agent_principal: Principal,
) -> None:
    """Unsupported schema versions return schema_supported=False and details=None without leaking raw JSON."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)
    ticket = _make_ticket(agent_principal.tenant_id)
    now = datetime(2026, 11, 10, 9, 0, 0, tzinfo=timezone.utc)

    future_event = ticket_events_repo.TicketEventRecord(
        id=uuid.uuid4(),
        tenant_id=agent_principal.tenant_id,
        ticket_id=ticket.id,
        event_type="ticket_resolved",
        occurred_at=now,
        recorded_at=now,
        actor_type="agent",
        actor_id="should-not-leak",
        schema_version=2,
        payload={
            "schema_version": 2,
            "from_status": "open",
            "to_status": "resolved",
            "future_untrusted_field": "secret_future_blob",
        },
    )

    mock_conn = MagicMock()
    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.get_ticket_by_id", return_value=ticket):
                with patch("app.repositories.ticket_events.list_ticket_events", return_value=[future_event]):
                    res = client.get(f"/browser-api/tickets/{ticket.id}/events")
                    assert res.status_code == 200
                    assert "secret_future_blob" not in res.text
                    assert "should-not-leak" not in res.text

                    ev = res.json()["events"][0]
                    assert ev["schema_version"] == 2
                    assert ev["schema_supported"] is False
                    assert ev["payload_valid"] is False
                    assert ev["details"] is None
                    assert ev["label"] == "Ticket resolved"
    finally:
        app.dependency_overrides.clear()


def test_get_ticket_events_malformed_supported_payload_not_presented_as_valid(
    agent_principal: Principal,
) -> None:
    """Malformed schema_version == 1 payloads set payload_valid=False and details=None instead of silent valid state."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)
    ticket = _make_ticket(agent_principal.tenant_id)
    now = datetime(2026, 11, 10, 9, 0, 0, tzinfo=timezone.utc)

    malformed_events = [
        ticket_events_repo.TicketEventRecord(
            id=uuid.uuid4(),
            tenant_id=agent_principal.tenant_id,
            ticket_id=ticket.id,
            event_type="ticket_created",
            occurred_at=now,
            recorded_at=now,
            actor_type="system",
            actor_id=None,
            schema_version=1,
            payload={"schema_version": 1, "policy_snapshot": "corrupted"},
        ),
        ticket_events_repo.TicketEventRecord(
            id=uuid.uuid4(),
            tenant_id=agent_principal.tenant_id,
            ticket_id=ticket.id,
            event_type="ticket_created",
            occurred_at=now + timedelta(seconds=1),
            recorded_at=now + timedelta(seconds=1),
            actor_type="system",
            actor_id=None,
            schema_version=1,
            payload={
                "schema_version": 1,
                "policy_snapshot": {"is_configured": True, "first_response_minutes": -5, "resolution_minutes": 100},
            },
        ),
        ticket_events_repo.TicketEventRecord(
            id=uuid.uuid4(),
            tenant_id=agent_principal.tenant_id,
            ticket_id=ticket.id,
            event_type="ticket_processed",
            occurred_at=now + timedelta(seconds=2),
            recorded_at=now + timedelta(seconds=2),
            actor_type="worker",
            actor_id=None,
            schema_version=1,
            payload={"schema_version": 1},  # missing from_status/to_status
        ),
        ticket_events_repo.TicketEventRecord(
            id=uuid.uuid4(),
            tenant_id=agent_principal.tenant_id,
            ticket_id=ticket.id,
            event_type="first_response_recorded",
            occurred_at=now + timedelta(seconds=3),
            recorded_at=now + timedelta(seconds=3),
            actor_type="agent",
            actor_id=None,
            schema_version=1,
            payload={"schema_version": 1, "responded_at": now.isoformat(), "elapsed_calendar_seconds": -12.0},
        ),
        ticket_events_repo.TicketEventRecord(
            id=uuid.uuid4(),
            tenant_id=agent_principal.tenant_id,
            ticket_id=ticket.id,
            event_type="ticket_closed",
            occurred_at=now + timedelta(seconds=4),
            recorded_at=now + timedelta(seconds=4),
            actor_type="agent",
            actor_id=None,
            schema_version=1,
            payload={"schema_version": 2, "from_status": "resolved", "to_status": "closed"},
        ),
    ]

    mock_conn = MagicMock()
    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.get_ticket_by_id", return_value=ticket):
                with patch("app.repositories.ticket_events.list_ticket_events", return_value=malformed_events):
                    res = client.get(f"/browser-api/tickets/{ticket.id}/events")
                    assert res.status_code == 200
                    items = res.json()["events"]
                    assert len(items) == len(malformed_events)
                    for item in items:
                        assert item["schema_version"] == 1
                        assert item["schema_supported"] is True
                        assert item["payload_valid"] is False
                        assert item["details"] is None
    finally:
        app.dependency_overrides.clear()


def test_get_ticket_events_missing_relation_fallback_and_ordering_preserved(
    agent_principal: Principal,
) -> None:
    """Missing ticket_events relation returns schema_available=False; repository ordering is preserved."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, follow_redirects=False)
    ticket = _make_ticket(agent_principal.tenant_id)
    mock_conn = MagicMock()

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.get_ticket_by_id", return_value=ticket):
                # 1. Domain exception TicketEventsSchemaUnavailableError
                with patch(
                    "app.repositories.ticket_events.list_ticket_events",
                    side_effect=ticket_events_repo.TicketEventsSchemaUnavailableError("Migration 009 pending"),
                ):
                    res = client.get(f"/browser-api/tickets/{ticket.id}/events")
                    assert res.status_code == 200
                    assert res.headers.get("cache-control") == "no-store"
                    body = res.json()
                    assert body["schema_available"] is False
                    assert body["events"] == []
                    assert "009" in body["message"]

                # 2. Direct psycopg UndefinedTable specifically naming ticket_events
                with patch(
                    "app.repositories.ticket_events.list_ticket_events",
                    side_effect=psycopg.errors.UndefinedTable('relation "ticket_events" does not exist'),
                ):
                    res_undef = client.get(f"/browser-api/tickets/{ticket.id}/events")
                    assert res_undef.status_code == 200
                    assert res_undef.json()["schema_available"] is False
                    assert mock_conn.transaction.called

                # 3. Repository ordering preserved verbatim
                t0 = datetime(2026, 11, 10, 9, 0, 0, tzinfo=timezone.utc)
                ev_a = ticket_events_repo.TicketEventRecord(
                    id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                    tenant_id=agent_principal.tenant_id,
                    ticket_id=ticket.id,
                    event_type="ticket_created",
                    occurred_at=t0,
                    recorded_at=t0,
                    actor_type="system",
                    actor_id=None,
                    schema_version=1,
                    payload={"schema_version": 1, "policy_snapshot": {"is_configured": False}},
                )
                ev_b = ticket_events_repo.TicketEventRecord(
                    id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                    tenant_id=agent_principal.tenant_id,
                    ticket_id=ticket.id,
                    event_type="ticket_resolved",
                    occurred_at=t0,
                    recorded_at=t0 + timedelta(seconds=1),
                    actor_type="agent",
                    actor_id=None,
                    schema_version=1,
                    payload={"schema_version": 1, "from_status": "open", "to_status": "resolved"},
                )
                with patch("app.repositories.ticket_events.list_ticket_events", return_value=[ev_a, ev_b]):
                    res_order = client.get(f"/browser-api/tickets/{ticket.id}/events")
                    assert res_order.status_code == 200
                    ids = [e["id"] for e in res_order.json()["events"]]
                    assert ids == [str(ev_a.id), str(ev_b.id)]
    finally:
        app.dependency_overrides.clear()


def test_get_ticket_events_does_not_mask_other_tables_or_db_failures(
    agent_principal: Principal,
) -> None:
    """Unrelated UndefinedTable, OperationalError, or unexpected errors are never masked as schema_unavailable."""
    app.dependency_overrides[get_browser_principal_json] = lambda: agent_principal
    client = TestClient(app, raise_server_exceptions=False)
    ticket = _make_ticket(agent_principal.tenant_id)
    mock_conn = MagicMock()

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            with patch("app.routers.browser_api.get_ticket_by_id", return_value=ticket):
                for exc in (
                    psycopg.errors.UndefinedTable('relation "some_other_table" does not exist'),
                    psycopg.OperationalError("connection timed out"),
                    psycopg.errors.InsufficientPrivilege("permission denied for table ticket_events"),
                    RuntimeError("unexpected failure"),
                ):
                    with patch("app.repositories.ticket_events.list_ticket_events", side_effect=exc):
                        res = client.get(f"/browser-api/tickets/{ticket.id}/events")
                        assert res.status_code == 500
    finally:
        app.dependency_overrides.clear()


def test_repo_list_ticket_events_bounded_limit_and_specific_undefined_table() -> None:
    """Repository bounded limit binds LIMIT %s and translates only missing ticket_events relation inside transaction."""
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchall.return_value = []
    tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()

    events = ticket_events_repo.list_ticket_events(mock_conn, tenant_id, ticket_id, limit=20)
    assert events == []
    mock_conn.transaction.assert_called_once()
    sql, params = mock_cur.execute.call_args[0]
    assert "ORDER BY occurred_at ASC, recorded_at ASC, id ASC" in sql
    assert "LIMIT %s" in sql
    assert params == (tenant_id, ticket_id, 20)

    with pytest.raises(ValueError, match="limit"):
        ticket_events_repo.list_ticket_events(mock_conn, tenant_id, ticket_id, limit=0)

    # Specific missing ticket_events table translates to TicketEventsSchemaUnavailableError
    mock_cur.execute.side_effect = psycopg.errors.UndefinedTable('relation "ticket_events" does not exist')
    with pytest.raises(ticket_events_repo.TicketEventsSchemaUnavailableError):
        ticket_events_repo.list_ticket_events(mock_conn, tenant_id, ticket_id, limit=20)

    # Different missing table must NOT be converted to TicketEventsSchemaUnavailableError
    mock_cur.execute.side_effect = psycopg.errors.UndefinedTable('relation "unrelated_table" does not exist')
    with pytest.raises(psycopg.errors.UndefinedTable):
        ticket_events_repo.list_ticket_events(mock_conn, tenant_id, ticket_id, limit=20)
