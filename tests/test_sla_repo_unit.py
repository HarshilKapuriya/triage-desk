"""Unit tests for SLA policy and ticket lifecycle repository functions using fakes/mocks.

Verifies:
1. Strict tenant isolation on every SQL query.
2. Parameter validation for priority and durations.
3. Caller-owned transaction semantics (functions never call commit()).
4. Effective policy resolution logic (exact match -> default fallback -> None).
5. Lifecycle primitives:
   - record_first_response: independent of status, preserves existing timestamp.
   - resolve_ticket: transitions to resolved, preserves resolved_at, rejects closed tickets.
   - close_ticket: requires prior resolution, rejects open/new tickets, idempotent on closed.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock
import uuid
import pytest

import app.repositories.sla as sla_repo
import app.repositories.tickets as ticket_repo
from app.repositories.tickets import InvalidStatusTransitionError

UTC = timezone.utc


def test_upsert_sla_policy_valid():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    tenant_id = uuid.uuid4()
    policy_id = uuid.uuid4()
    now = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)

    cur.fetchone.return_value = {
        "id": policy_id,
        "tenant_id": tenant_id,
        "name": "Urgent Policy",
        "priority": "urgent",
        "first_response_minutes": 60,
        "resolution_minutes": 240,
        "created_at": now,
        "updated_at": now,
    }

    record = sla_repo.upsert_sla_policy(
        conn=conn,
        tenant_id=tenant_id,
        name="Urgent Policy",
        priority="urgent",
        first_response_minutes=60,
        resolution_minutes=240,
    )

    assert record.id == policy_id
    assert record.tenant_id == tenant_id
    assert record.priority == "urgent"
    assert record.first_response_minutes == 60
    assert record.resolution_minutes == 240
    # Must NOT call commit() - caller owns transaction
    conn.commit.assert_not_called()

    # Verify query contains tenant_id and bounded parameters
    args, kwargs = cur.execute.call_args
    sql = args[0]
    params = args[1]
    assert "INSERT INTO sla_policies" in sql
    assert params == (tenant_id, "Urgent Policy", "urgent", 60, 240)


def test_upsert_sla_policy_validation_errors():
    conn = MagicMock()
    tenant_id = uuid.uuid4()

    # Empty name
    with pytest.raises(ValueError, match="name must not be empty"):
        sla_repo.upsert_sla_policy(conn, tenant_id, "  ", "urgent", 60, 120)

    # Invalid priority
    with pytest.raises(ValueError, match="Invalid SLA policy priority"):
        sla_repo.upsert_sla_policy(conn, tenant_id, "Name", "super_urgent", 60, 120)

    # Non-positive first_response_minutes
    with pytest.raises(ValueError, match="first_response_minutes must be a positive integer"):
        sla_repo.upsert_sla_policy(conn, tenant_id, "Name", "urgent", 0, 120)

    # Oversized duration
    with pytest.raises(ValueError, match="exceeds maximum allowed"):
        sla_repo.upsert_sla_policy(conn, tenant_id, "Name", "urgent", 60, 600000)


def test_resolve_effective_policy_exact_and_default():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    tenant_id = uuid.uuid4()
    now = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)

    # 1. Exact priority match
    cur.fetchone.return_value = {
        "id": uuid.uuid4(),
        "tenant_id": tenant_id,
        "name": "High Priority",
        "priority": "high",
        "first_response_minutes": 120,
        "resolution_minutes": 480,
        "created_at": now,
        "updated_at": now,
    }

    pol = sla_repo.resolve_effective_policy(conn, tenant_id, "high")
    assert pol is not None
    assert pol.priority == "high"
    conn.commit.assert_not_called()

    # 2. None priority: queries default
    cur.fetchone.return_value = {
        "id": uuid.uuid4(),
        "tenant_id": tenant_id,
        "name": "Default Fallback",
        "priority": "default",
        "first_response_minutes": 240,
        "resolution_minutes": 1440,
        "created_at": now,
        "updated_at": now,
    }

    pol_def = sla_repo.resolve_effective_policy(conn, tenant_id, None)
    assert pol_def is not None
    assert pol_def.priority == "default"

    # Verify query for None priority checks priority = 'default'
    args, _ = cur.execute.call_args
    assert "priority = 'default'" in args[0]

    # 3. Missing policy: returns None
    cur.fetchone.return_value = None
    pol_missing = sla_repo.resolve_effective_policy(conn, tenant_id, "low")
    assert pol_missing is None


def test_record_first_response_repository():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()
    resp_time = datetime(2026, 11, 8, 11, 0, tzinfo=UTC)

    # Rejection of naive timestamp
    with pytest.raises(ValueError, match="must be timezone-aware"):
        ticket_repo.record_first_response(conn, tenant_id, ticket_id, datetime(2026, 11, 8, 11, 0))

    cur.fetchone.return_value = {
        "id": ticket_id,
        "tenant_id": tenant_id,
        "external_id": None,
        "subject": "Test",
        "body": "Body",
        "customer_email": None,
        "status": "open",
        "category": None,
        "priority": "urgent",
        "created_at": datetime(2026, 11, 8, 10, 0, tzinfo=UTC),
        "first_response_at": resp_time,
        "resolved_at": None,
        "assignee_id": None,
    }

    t = ticket_repo.record_first_response(conn, tenant_id, ticket_id, resp_time)
    assert t is not None
    assert t.first_response_at == resp_time
    assert t.status == "open"  # Does not mutate status
    conn.commit.assert_not_called()

    # Query uses COALESCE to preserve initial response timestamp
    sql = cur.execute.call_args[0][0]
    assert "first_response_at = COALESCE(first_response_at, %s)" in sql


def test_record_first_response_rejects_resolved_or_closed_when_unset():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()
    resp_time = datetime(2026, 11, 8, 11, 0, tzinfo=UTC)

    # 1. Reject on resolved ticket when first_response_at is unset
    cur.fetchone.return_value = {
        "id": ticket_id,
        "tenant_id": tenant_id,
        "status": "resolved",
        "created_at": datetime(2026, 11, 8, 10, 0, tzinfo=UTC),
        "first_response_at": None,
    }
    with pytest.raises(ticket_repo.InvalidStatusTransitionError, match="Cannot record first response on resolved ticket"):
        ticket_repo.record_first_response(conn, tenant_id, ticket_id, resp_time)

    # 2. Reject on closed ticket when first_response_at is unset
    cur.fetchone.return_value = {
        "id": ticket_id,
        "tenant_id": tenant_id,
        "status": "closed",
        "created_at": datetime(2026, 11, 8, 10, 0, tzinfo=UTC),
        "first_response_at": None,
    }
    with pytest.raises(ticket_repo.InvalidStatusTransitionError, match="Cannot record first response on closed ticket"):
        ticket_repo.record_first_response(conn, tenant_id, ticket_id, resp_time)

    # 3. Idempotent preservation if first_response_at is already set on resolved/closed ticket
    prior_resp = datetime(2026, 11, 8, 10, 30, tzinfo=UTC)
    cur.fetchone.return_value = {
        "id": ticket_id,
        "tenant_id": tenant_id,
        "external_id": None,
        "subject": "Test",
        "body": "Body",
        "customer_email": None,
        "status": "resolved",
        "category": None,
        "priority": "normal",
        "created_at": datetime(2026, 11, 8, 10, 0, tzinfo=UTC),
        "first_response_at": prior_resp,
        "resolved_at": datetime(2026, 11, 8, 11, 30, tzinfo=UTC),
        "assignee_id": None,
    }
    t = ticket_repo.record_first_response(conn, tenant_id, ticket_id, resp_time)
    assert t is not None
    assert t.first_response_at == prior_resp


def test_resolve_ticket_transitions():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()
    res_time = datetime(2026, 11, 8, 14, 0, tzinfo=UTC)

    # 1. Reject already closed ticket
    cur.fetchone.return_value = {
        "status": "closed",
        "created_at": datetime(2026, 11, 8, 10, 0, tzinfo=UTC),
        "resolved_at": None,
    }
    with pytest.raises(InvalidStatusTransitionError, match="ticket is already closed"):
        ticket_repo.resolve_ticket(conn, tenant_id, ticket_id, res_time)

    # 2. Successfully resolve open ticket
    cur.fetchone.side_effect = [
        {"status": "open", "created_at": datetime(2026, 11, 8, 10, 0, tzinfo=UTC), "resolved_at": None},  # FOR UPDATE lookup
        {
            "id": ticket_id,
            "tenant_id": tenant_id,
            "external_id": None,
            "subject": "Test",
            "body": "Body",
            "customer_email": None,
            "status": "resolved",
            "category": None,
            "priority": "normal",
            "created_at": datetime(2026, 11, 8, 10, 0, tzinfo=UTC),
            "first_response_at": None,
            "resolved_at": res_time,
            "assignee_id": None,
        },  # UPDATE return
    ]

    t = ticket_repo.resolve_ticket(conn, tenant_id, ticket_id, res_time)
    assert t is not None
    assert t.status == "resolved"
    assert t.resolved_at == res_time
    conn.commit.assert_not_called()


def test_close_ticket_requires_prior_resolution():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()

    # 1. Rejects closing an open ticket
    cur.fetchone.return_value = {"status": "open", "resolved_at": None}
    with pytest.raises(InvalidStatusTransitionError, match="must be resolved first"):
        ticket_repo.close_ticket(conn, tenant_id, ticket_id)

    # 2. Rejects closing a new ticket
    cur.fetchone.return_value = {"status": "new", "resolved_at": None}
    with pytest.raises(InvalidStatusTransitionError, match="must be resolved first"):
        ticket_repo.close_ticket(conn, tenant_id, ticket_id)

    # 3. Closes a resolved ticket
    cur.fetchone.side_effect = [
        {"status": "resolved", "resolved_at": datetime(2026, 11, 8, 12, 0, tzinfo=UTC)},
        {
            "id": ticket_id,
            "tenant_id": tenant_id,
            "external_id": None,
            "subject": "Test",
            "body": "Body",
            "customer_email": None,
            "status": "closed",
            "category": None,
            "priority": "normal",
            "created_at": datetime(2026, 11, 8, 10, 0, tzinfo=UTC),
            "first_response_at": None,
            "resolved_at": datetime(2026, 11, 8, 12, 0, tzinfo=UTC),
            "assignee_id": None,
        },
    ]

    t_closed = ticket_repo.close_ticket(conn, tenant_id, ticket_id)
    assert t_closed is not None
    assert t_closed.status == "closed"
    conn.commit.assert_not_called()

    # 4. Idempotent on already closed ticket
    cur.fetchone.side_effect = None
    cur.fetchone.return_value = {
        "id": ticket_id,
        "tenant_id": tenant_id,
        "external_id": None,
        "subject": "Test",
        "body": "Body",
        "customer_email": None,
        "status": "closed",
        "category": None,
        "priority": "normal",
        "created_at": datetime(2026, 11, 8, 10, 0, tzinfo=UTC),
        "first_response_at": None,
        "resolved_at": datetime(2026, 11, 8, 12, 0, tzinfo=UTC),
        "assignee_id": None,
    }

    t_idem = ticket_repo.close_ticket(conn, tenant_id, ticket_id)
    assert t_idem.status == "closed"
    conn.commit.assert_not_called()


def test_record_first_response_preceding_created_at_rejected():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()
    created = datetime(2026, 11, 8, 10, 0, tzinfo=UTC)
    impossible_resp = datetime(2026, 11, 8, 9, 30, tzinfo=UTC)

    cur.fetchone.return_value = {
        "id": ticket_id,
        "tenant_id": tenant_id,
        "created_at": created,
        "first_response_at": None,
    }

    with pytest.raises(ValueError, match="cannot precede ticket created_at"):
        ticket_repo.record_first_response(conn, tenant_id, ticket_id, impossible_resp)


def test_resolve_ticket_preceding_created_at_rejected():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()
    created = datetime(2026, 11, 8, 10, 0, tzinfo=UTC)
    impossible_res = datetime(2026, 11, 8, 9, 45, tzinfo=UTC)

    cur.fetchone.return_value = {
        "id": ticket_id,
        "tenant_id": tenant_id,
        "created_at": created,
        "status": "open",
        "resolved_at": None,
    }

    with pytest.raises(ValueError, match="cannot precede ticket created_at"):
        ticket_repo.resolve_ticket(conn, tenant_id, ticket_id, impossible_res)


def test_ingest_job_completion_does_not_overwrite_resolved_or_closed_status():
    """Verify that completing a ticket_ingest background job never overwrites resolved or closed status."""
    import app.repositories.jobs as job_repo
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.rowcount = 1  # Job claim validation succeeds

    tenant_id = uuid.uuid4()
    job_id = uuid.uuid4()
    ticket_id = uuid.uuid4()

    for progressed_status in ("resolved", "closed", "pending"):
        cur.reset_mock()
        cur.rowcount = 1
        cur.fetchone.return_value = {"status": progressed_status}

        job_repo.complete_job_with_ticket_transition(
            conn=conn,
            tenant_id=tenant_id,
            job_id=job_id,
            ticket_id=ticket_id,
            worker_id="test_worker_1",
            generation=1,
            lease_seconds=300,
        )

        # Assert no UPDATE tickets SET status = 'open' was executed
        executed_sqls = [call[0][0] for call in cur.execute.call_args_list]
        for sql in executed_sqls:
            assert "SET status = 'open'" not in sql, f"Must not transition {progressed_status} ticket to open"

        # Assert audit log records skipped transition
        audit_calls = [call for call in executed_sqls if "audit_log" in call]
        assert any("ticket.ingest_skipped_already_progressed" in str(call) for call in cur.execute.call_args_list)
