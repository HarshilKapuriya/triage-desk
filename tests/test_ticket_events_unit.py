"""Unit tests for ticket lifecycle event foundations and SLA operational telemetry.

Offline unit tests verifying:
1. Payload construction, schema versioning, and validation rules.
2. Tenant-scoped query parameters and deterministic tie-breaker sorting.
3. Event emission only on actual state transitions (no duplicates on idempotent no-ops).
4. Explicit unconfigured representation when SLA policy is missing or unavailable.
5. Error propagation: event write failures bubble up to caller transactions without suppression.
6. Legacy tickets without fabricated creation events.
7. Backdated occurrence vs recording-time representation.
8. Static analysis of migration DDL constraints.
"""

from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import uuid
import psycopg
import pytest

from app.repositories import (
    ticket_events as ticket_events_repo,
    tickets as ticket_repo,
    jobs as job_repo,
    sla as sla_repo,
)


# 1. Payload construction & validation tests

def test_payload_builder_with_configured_policy() -> None:
    """build_ticket_created_payload records explicit configured policy snapshot."""
    policy_id = str(uuid.uuid4())
    snapshot = {
        "is_configured": True,
        "policy_id": policy_id,
        "policy_priority": "urgent",
        "first_response_minutes": 15,
        "resolution_minutes": 60,
    }
    payload = ticket_events_repo.build_ticket_created_payload(
        policy_snapshot=snapshot,
        priority="urgent",
        category="Billing",
    )
    assert payload["schema_version"] == 1
    assert payload["priority"] == "urgent"
    assert payload["category"] == "Billing"
    assert payload["policy_snapshot"]["is_configured"] is True
    assert payload["policy_snapshot"]["policy_id"] == policy_id
    assert payload["policy_snapshot"]["first_response_minutes"] == 15


def test_payload_builder_with_unconfigured_policy() -> None:
    """build_ticket_created_payload explicitly records unconfigured policy when None."""
    payload = ticket_events_repo.build_ticket_created_payload(
        policy_snapshot=None,
        priority=None,
        category=None,
    )
    assert payload["schema_version"] == 1
    assert payload["priority"] is None
    assert payload["policy_snapshot"]["is_configured"] is False
    assert payload["policy_snapshot"]["policy_id"] is None
    assert payload["policy_snapshot"]["first_response_minutes"] is None


def test_append_ticket_event_validates_event_and_actor_types() -> None:
    """append_ticket_event rejects unallowlisted event or actor types."""
    mock_conn = MagicMock()
    tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    # Invalid event type
    with pytest.raises(ValueError, match="Invalid event_type"):
        ticket_events_repo.append_ticket_event(
            conn_or_cur=mock_conn,
            tenant_id=tenant_id,
            ticket_id=ticket_id,
            event_type="invalid_event",
            occurred_at=now,
            actor_type="agent",
        )

    # Invalid actor type
    with pytest.raises(ValueError, match="Invalid actor_type"):
        ticket_events_repo.append_ticket_event(
            conn_or_cur=mock_conn,
            tenant_id=tenant_id,
            ticket_id=ticket_id,
            event_type="ticket_created",
            occurred_at=now,
            actor_type="unknown_actor",
        )

    # Naive timestamp rejection
    naive_dt = datetime(2026, 11, 8, 12, 0, 0)
    with pytest.raises(ValueError, match="must be timezone-aware"):
        ticket_events_repo.append_ticket_event(
            conn_or_cur=mock_conn,
            tenant_id=tenant_id,
            ticket_id=ticket_id,
            event_type="ticket_created",
            occurred_at=naive_dt,
            actor_type="agent",
        )


# 2. Tenant isolation and query parameters

def test_append_ticket_event_executes_tenant_scoped_sql() -> None:
    """append_ticket_event passes tenant_id and sanitized params to cursor."""
    mock_cur = MagicMock(spec=psycopg.Cursor)
    tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    record = ticket_events_repo.append_ticket_event(
        conn_or_cur=mock_cur,
        tenant_id=tenant_id,
        ticket_id=ticket_id,
        event_type="first_response_recorded",
        occurred_at=now,
        actor_type="agent",
        actor_id="user-123",
        payload={"schema_version": 1, "responded_at": now.isoformat()},
    )

    assert record.tenant_id == tenant_id
    assert record.ticket_id == ticket_id
    assert record.event_type == "first_response_recorded"
    assert record.actor_type == "agent"
    assert record.actor_id == "user-123"

    mock_cur.execute.assert_called_once()
    sql, params = mock_cur.execute.call_args[0]
    assert "INSERT INTO ticket_events" in sql
    assert params[1] == tenant_id
    assert params[2] == ticket_id
    assert params[3] == "first_response_recorded"
    assert params[4] == now
    assert params[5] == "agent"
    assert params[6] == "user-123"


def test_list_ticket_events_scopes_by_tenant_and_deterministic_order() -> None:
    """list_ticket_events enforces tenant filter and deterministic tie-breaker sorting."""
    mock_cur = MagicMock(spec=psycopg.Cursor)
    tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()
    event_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    mock_cur.fetchall.return_value = [
        {
            "id": event_id,
            "tenant_id": tenant_id,
            "ticket_id": ticket_id,
            "event_type": "ticket_created",
            "occurred_at": now,
            "recorded_at": now,
            "actor_type": "system",
            "actor_id": None,
            "schema_version": 1,
            "payload": {"schema_version": 1},
        }
    ]

    events = ticket_events_repo.list_ticket_events(mock_cur, tenant_id, ticket_id)
    assert len(events) == 1
    assert events[0].id == event_id

    mock_cur.execute.assert_called_once()
    sql, params = mock_cur.execute.call_args[0]
    assert "FROM ticket_events" in sql
    assert "WHERE tenant_id = %s AND ticket_id = %s" in sql
    assert "ORDER BY occurred_at ASC, recorded_at ASC, id ASC" in sql
    assert params == (tenant_id, ticket_id)


# 3. State transition idempotency and event emission

def test_create_ticket_emits_event_only_when_created() -> None:
    """create_or_get_ticket emits ticket_created on insertion, skips on existing external_id."""
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    # 1. Fresh ticket creation: row returned
    mock_cur.fetchone.return_value = {
        "id": ticket_id,
        "tenant_id": tenant_id,
        "external_id": "ext-1",
        "subject": "Help",
        "body": "Need help",
        "customer_email": None,
        "status": "new",
        "category": None,
        "priority": None,
        "created_at": now,
        "first_response_at": None,
        "resolved_at": None,
        "assignee_id": None,
    }

    with patch("app.repositories.tickets._emit_ticket_created_event") as mock_emit:
        rec, is_created = ticket_repo.create_or_get_ticket(
            conn=mock_conn,
            tenant_id=tenant_id,
            subject="Help",
            body="Need help",
            external_id="ext-1",
        )
        assert is_created is True
        mock_emit.assert_called_once()

    # 2. Existing ticket replay: first fetchone returns None (conflict), second returns existing
    mock_cur.fetchone.side_effect = [
        None,  # ON CONFLICT DO NOTHING
        {
            "id": ticket_id,
            "tenant_id": tenant_id,
            "external_id": "ext-1",
            "subject": "Help",
            "body": "Need help",
            "customer_email": None,
            "status": "new",
            "category": None,
            "priority": None,
            "created_at": now,
            "first_response_at": None,
            "resolved_at": None,
            "assignee_id": None,
        },
    ]

    with patch("app.repositories.tickets._emit_ticket_created_event") as mock_emit:
        rec, is_created = ticket_repo.create_or_get_ticket(
            conn=mock_conn,
            tenant_id=tenant_id,
            subject="Help",
            body="Need help",
            external_id="ext-1",
        )
        assert is_created is False
        mock_emit.assert_not_called()


def test_first_response_emits_event_only_on_first_record() -> None:
    """record_first_response emits event when first recorded, no-op on repeat."""
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()
    created_at = datetime.now(timezone.utc) - timedelta(minutes=10)
    responded_at = datetime.now(timezone.utc)

    # 1. New response: first_response_at is NULL
    mock_cur.fetchone.side_effect = [
        {"id": ticket_id, "tenant_id": tenant_id, "created_at": created_at, "first_response_at": None, "status": "open"},
        {"id": ticket_id, "tenant_id": tenant_id, "external_id": None, "subject": "S", "body": "B", "customer_email": None,
         "status": "open", "category": None, "priority": None, "created_at": created_at, "first_response_at": responded_at,
         "resolved_at": None, "assignee_id": None},
    ]

    with patch("app.repositories.ticket_events.append_ticket_event") as mock_append:
        rec = ticket_repo.record_first_response(mock_conn, tenant_id, ticket_id, responded_at)
        assert rec is not None
        assert rec.first_response_at == responded_at
        mock_append.assert_called_once()
        assert mock_append.call_args[1]["event_type"] == "first_response_recorded"

    # 2. Repeated response: first_response_at already set
    mock_cur.fetchone.side_effect = None
    mock_cur.fetchone.return_value = {
        "id": ticket_id, "tenant_id": tenant_id, "external_id": None, "subject": "S", "body": "B", "customer_email": None,
        "status": "open", "category": None, "priority": None, "created_at": created_at, "first_response_at": responded_at,
        "resolved_at": None, "assignee_id": None,
    }

    with patch("app.repositories.ticket_events.append_ticket_event") as mock_append:
        rec = ticket_repo.record_first_response(mock_conn, tenant_id, ticket_id, responded_at)
        assert rec is not None
        mock_append.assert_not_called()


def test_resolve_and_close_ticket_idempotency() -> None:
    """resolve_ticket and close_ticket emit events once and are idempotent on repeats."""
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()
    created_at = datetime.now(timezone.utc) - timedelta(hours=2)
    resolved_at = datetime.now(timezone.utc) - timedelta(hours=1)
    closed_at = datetime.now(timezone.utc)

    # 1. Resolve from open -> emits ticket_resolved
    mock_cur.fetchone.side_effect = [
        {"id": ticket_id, "tenant_id": tenant_id, "external_id": None, "subject": "S", "body": "B", "customer_email": None,
         "status": "open", "category": None, "priority": None, "created_at": created_at, "first_response_at": None,
         "resolved_at": None, "assignee_id": None},
        {"id": ticket_id, "tenant_id": tenant_id, "external_id": None, "subject": "S", "body": "B", "customer_email": None,
         "status": "resolved", "category": None, "priority": None, "created_at": created_at, "first_response_at": None,
         "resolved_at": resolved_at, "assignee_id": None},
    ]
    with patch("app.repositories.ticket_events.append_ticket_event") as mock_append:
        rec = ticket_repo.resolve_ticket(mock_conn, tenant_id, ticket_id, resolved_at)
        assert rec.status == "resolved"
        mock_append.assert_called_once()
        assert mock_append.call_args[1]["event_type"] == "ticket_resolved"

    # Repeat resolve on already-resolved ticket -> no duplicate event
    mock_cur.fetchone.side_effect = None
    mock_cur.fetchone.return_value = {
        "id": ticket_id, "tenant_id": tenant_id, "external_id": None, "subject": "S", "body": "B", "customer_email": None,
        "status": "resolved", "category": None, "priority": None, "created_at": created_at, "first_response_at": None,
        "resolved_at": resolved_at, "assignee_id": None,
    }
    with patch("app.repositories.ticket_events.append_ticket_event") as mock_append:
        rec = ticket_repo.resolve_ticket(mock_conn, tenant_id, ticket_id, resolved_at)
        assert rec.status == "resolved"
        mock_append.assert_not_called()

    # 2. Close from resolved -> emits ticket_closed
    mock_cur.fetchone.side_effect = [
        {"id": ticket_id, "tenant_id": tenant_id, "external_id": None, "subject": "S", "body": "B", "customer_email": None,
         "status": "resolved", "category": None, "priority": None, "created_at": created_at, "first_response_at": None,
         "resolved_at": resolved_at, "assignee_id": None},
        {"id": ticket_id, "tenant_id": tenant_id, "external_id": None, "subject": "S", "body": "B", "customer_email": None,
         "status": "closed", "category": None, "priority": None, "created_at": created_at, "first_response_at": None,
         "resolved_at": resolved_at, "assignee_id": None},
    ]
    with patch("app.repositories.ticket_events.append_ticket_event") as mock_append:
        rec = ticket_repo.close_ticket(mock_conn, tenant_id, ticket_id, closed_at=closed_at)
        assert rec.status == "closed"
        mock_append.assert_called_once()
        assert mock_append.call_args[1]["event_type"] == "ticket_closed"

    # Repeat close on already-closed ticket -> no duplicate event
    mock_cur.fetchone.side_effect = None
    mock_cur.fetchone.return_value = {
        "id": ticket_id, "tenant_id": tenant_id, "external_id": None, "subject": "S", "body": "B", "customer_email": None,
        "status": "closed", "category": None, "priority": None, "created_at": created_at, "first_response_at": None,
        "resolved_at": resolved_at, "assignee_id": None,
    }
    with patch("app.repositories.ticket_events.append_ticket_event") as mock_append:
        rec = ticket_repo.close_ticket(mock_conn, tenant_id, ticket_id, closed_at=closed_at)
        assert rec.status == "closed"
        mock_append.assert_not_called()


def test_worker_complete_job_emits_ticket_processed() -> None:
    """complete_job_with_ticket_transition emits ticket_processed when status changes from new to open."""
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    mock_cur.rowcount = 1  # Job claim and update rowcount
    tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()
    job_id = uuid.uuid4()

    mock_cur.fetchone.return_value = {"status": "new"}

    with patch("app.repositories.ticket_events.append_ticket_event") as mock_append:
        res = job_repo.complete_job_with_ticket_transition(
            conn=mock_conn,
            tenant_id=tenant_id,
            job_id=job_id,
            ticket_id=ticket_id,
            worker_id="test-worker",
            generation=0,
            predicted_label="Mortgage",
        )
        assert res is True
        mock_append.assert_called_once()
        kwargs = mock_append.call_args[1]
        assert kwargs["event_type"] == "ticket_processed"
        assert kwargs["actor_type"] == "worker"
        assert kwargs["actor_id"] == "test-worker"
        assert kwargs["payload"]["from_status"] == "new"
        assert kwargs["payload"]["to_status"] == "open"
        assert kwargs["payload"]["predicted_category"] == "Mortgage"


# 4. Error propagation and atomic rollback

def test_event_write_failure_propagates_to_transaction_owner() -> None:
    """If append_ticket_event raises an exception, it propagates to the transaction owner without suppression."""
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    mock_cur.fetchone.side_effect = [
        {"id": ticket_id, "tenant_id": tenant_id, "created_at": now, "first_response_at": None, "status": "open"},
        {"id": ticket_id, "tenant_id": tenant_id, "external_id": None, "subject": "S", "body": "B", "customer_email": None,
         "status": "open", "category": None, "priority": None, "created_at": now, "first_response_at": now,
         "resolved_at": None, "assignee_id": None},
    ]

    with patch("app.repositories.ticket_events.append_ticket_event", side_effect=RuntimeError("DB constraint violation")):
        with pytest.raises(RuntimeError, match="DB constraint violation"):
            ticket_repo.record_first_response(mock_conn, tenant_id, ticket_id, now)


# 5. Legacy tickets & bitemporal representation

def test_legacy_ticket_has_empty_event_history() -> None:
    """Existing tickets without events return an empty list; no synthetic history is fabricated."""
    mock_cur = MagicMock(spec=psycopg.Cursor)
    tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()
    mock_cur.fetchall.return_value = []

    events = ticket_events_repo.list_ticket_events(mock_cur, tenant_id, ticket_id)
    assert events == []


def test_bitemporal_time_representation() -> None:
    """occurred_at captures business event time, recorded_at captures transaction recording time."""
    mock_cur = MagicMock(spec=psycopg.Cursor)
    tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()
    business_time = datetime(2026, 11, 8, 10, 0, 0, tzinfo=timezone.utc)

    rec = ticket_events_repo.append_ticket_event(
        conn_or_cur=mock_cur,
        tenant_id=tenant_id,
        ticket_id=ticket_id,
        event_type="first_response_recorded",
        occurred_at=business_time,
        actor_type="agent",
    )
    assert rec.occurred_at == business_time
    assert rec.recorded_at.tzinfo == timezone.utc
    assert rec.recorded_at >= business_time


# 6. Static SQL migration checks

def test_static_migration_009_ddl() -> None:
    """Verify migrations/009_ticket_events.sql defines composite foreign key and partial unique indexes."""
    migration_path = Path(__file__).resolve().parent.parent / "migrations" / "009_ticket_events.sql"
    assert migration_path.exists(), "009_ticket_events.sql must exist."
    content = migration_path.read_text(encoding="utf-8")

    assert "CREATE TABLE ticket_events" in content
    assert "CONSTRAINT fk_ticket_events_tickets FOREIGN KEY (tenant_id, ticket_id)" in content
    assert "REFERENCES tickets (tenant_id, id) ON DELETE CASCADE" in content
    assert "CONSTRAINT uq_ticket_events_tenant_id_id UNIQUE (tenant_id, id)" in content
    assert "uq_ticket_events_ticket_created" in content
    assert "uq_ticket_events_ticket_processed" in content
    assert "uq_ticket_events_first_response" in content
    assert "uq_ticket_events_resolved" in content
    assert "uq_ticket_events_closed" in content
    assert "idx_ticket_events_lookup" in content
    assert "WHERE event_type = 'ticket_created'" in content
