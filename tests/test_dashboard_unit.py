"""Unit tests for Phase 5 Slice 3: Tenant Operational Dashboard.

Covers:
- Session-based authentication & principal tenant derivation.
- Payload structure, Cache-Control header, and evaluation timestamp.
- Pure arithmetic validation (denominators, rates, complete partitioning).
- Hand-built comparison fixtures with the existing pure SLA evaluator (evaluate_target).
- Exact deadline behavior (now == due_at is incomplete_due_now, not overdue).
- Data quality issues (created_at in future, completed_at < created_at, completed_at > now).
- Zero denominators returning None (never dividing by zero or assuming 100%).
- Empty tenant and unconfigured policy edge cases.
- Schema unavailable savepoint fallback handling.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import uuid
import pytest
from fastapi.testclient import TestClient
import psycopg.errors

from app.main import app
from app.deps import Principal
import app.repositories.dashboard as dashboard_repo
from app.services.sla import evaluate_target


UTC = timezone.utc


def test_calculate_target_metrics_complete_partition():
    """Verify mutual exclusivity and complete partitioning of target metrics."""
    metrics = dashboard_repo._calculate_target_metrics(
        eligible_total=100,
        dq_issues=5,
        completed_on_time=50,
        completed_late=15,
        incomplete_before=20,
        incomplete_due_now=2,
        incomplete_overdue=8,
    )

    # 1. Valid eligible exclusion
    assert metrics.eligible_total == 100
    assert metrics.data_quality_issues == 5
    assert metrics.valid_eligible_total == 95

    # 2. Completed partition
    assert metrics.completed_on_time == 50
    assert metrics.completed_late == 15
    assert metrics.completed_total == 65

    # 3. Incomplete partition
    assert metrics.incomplete_before_deadline == 20
    assert metrics.incomplete_due_now == 2
    assert metrics.incomplete_overdue == 8
    assert metrics.incomplete_total == 30

    # 4. Identity: completed_total + incomplete_total == valid_eligible_total
    assert metrics.completed_total + metrics.incomplete_total == metrics.valid_eligible_total

    # 5. Rates
    # 50 / 65 = 0.7692
    assert metrics.completed_timeliness_rate == round(50 / 65, 4)
    # (20 + 2) / 30 = 22 / 30 = 0.7333
    assert metrics.incomplete_not_overdue_rate == round(22 / 30, 4)


def test_calculate_target_metrics_zero_denominators_return_none():
    """Verify rates are None (never 0.0 or 1.0) when denominators are zero."""
    # Zero completed tickets
    metrics_no_completed = dashboard_repo._calculate_target_metrics(
        eligible_total=10,
        dq_issues=0,
        completed_on_time=0,
        completed_late=0,
        incomplete_before=8,
        incomplete_due_now=1,
        incomplete_overdue=1,
    )
    assert metrics_no_completed.completed_total == 0
    assert metrics_no_completed.completed_timeliness_rate is None
    assert metrics_no_completed.incomplete_not_overdue_rate == 0.9000

    # Zero incomplete tickets (all completed)
    metrics_all_completed = dashboard_repo._calculate_target_metrics(
        eligible_total=10,
        dq_issues=0,
        completed_on_time=8,
        completed_late=2,
        incomplete_before=0,
        incomplete_due_now=0,
        incomplete_overdue=0,
    )
    assert metrics_all_completed.incomplete_total == 0
    assert metrics_all_completed.incomplete_not_overdue_rate is None
    assert metrics_all_completed.completed_timeliness_rate == 0.8000

    # Zero eligible tickets
    metrics_empty = dashboard_repo._calculate_target_metrics(
        eligible_total=0,
        dq_issues=0,
        completed_on_time=0,
        completed_late=0,
        incomplete_before=0,
        incomplete_due_now=0,
        incomplete_overdue=0,
    )
    assert metrics_empty.valid_eligible_total == 0
    assert metrics_empty.completed_timeliness_rate is None
    assert metrics_empty.incomplete_not_overdue_rate is None


def test_classification_parity_with_pure_evaluator():
    """Verify hand-built test scenarios match both SQL classification rules and pure evaluator."""
    now = datetime(2026, 11, 8, 12, 0, 0, tzinfo=UTC)
    created_at = now - timedelta(hours=2)
    target_minutes = 120  # due exactly at now!
    due_at = created_at + timedelta(minutes=target_minutes)
    assert due_at == now

    # 1. Exact deadline behavior: now == due_at with completed_at=None
    eval_due_now = evaluate_target(
        target_name="first_response",
        target_minutes=target_minutes,
        created_at=created_at,
        completed_at=None,
        current_time=now,
    )
    # Evaluator considers it not breached yet at exactly due_at
    assert not eval_due_now.is_breached
    assert eval_due_now.timeliness == "on_track"

    # SQL classification rules:
    is_dq = created_at > now
    assert not is_dq
    is_incomplete_due_now = eval_due_now.completed_at is None and now == due_at
    assert is_incomplete_due_now

    # 2. Overdue: now > due_at by 1 second
    now_past = now + timedelta(seconds=1)
    eval_overdue = evaluate_target(
        target_name="first_response",
        target_minutes=target_minutes,
        created_at=created_at,
        completed_at=None,
        current_time=now_past,
    )
    assert eval_overdue.is_breached
    assert eval_overdue.timeliness == "overdue"
    assert now_past > due_at

    # 3. Completed on time: completed 10 minutes before due_at
    completed_early = due_at - timedelta(minutes=10)
    eval_on_time = evaluate_target(
        target_name="first_response",
        target_minutes=target_minutes,
        created_at=created_at,
        completed_at=completed_early,
        current_time=now,
    )
    assert not eval_on_time.is_breached
    assert eval_on_time.timeliness == "on_time"
    assert completed_early <= due_at

    # 4. Completed late: completed 10 minutes after due_at
    completed_late = due_at + timedelta(minutes=10)
    eval_late = evaluate_target(
        target_name="first_response",
        target_minutes=target_minutes,
        created_at=created_at,
        completed_at=completed_late,
        current_time=now + timedelta(minutes=20),
    )
    assert eval_late.is_breached
    assert eval_late.timeliness == "late"
    assert completed_late > due_at

    # 5. Invalid data cases rejected by evaluate_target with ValueError
    # Case A: created in future
    with pytest.raises(ValueError, match="cannot be in the future"):
        evaluate_target(
            target_name="first_response",
            target_minutes=60,
            created_at=now + timedelta(minutes=5),
            completed_at=None,
            current_time=now,
        )

    # Case B: completed before created
    with pytest.raises(ValueError, match="cannot precede created_at"):
        evaluate_target(
            target_name="first_response",
            target_minutes=60,
            created_at=created_at,
            completed_at=created_at - timedelta(minutes=5),
            current_time=now,
        )

    # Case C: completed in future relative to current_time
    with pytest.raises(ValueError, match="cannot be in the future relative to current_time"):
        evaluate_target(
            target_name="first_response",
            target_minutes=60,
            created_at=created_at,
            completed_at=now + timedelta(minutes=5),
            current_time=now,
        )


def test_get_tenant_dashboard_metrics_repository_mock():
    """Verify repository logic with mocked database cursor returning aggregated row."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_conn.transaction.return_value.__enter__.return_value = mock_conn

    tenant_id = uuid.uuid4()
    evaluated_at = datetime(2026, 11, 8, 14, 0, 0, tzinfo=UTC)

    mock_cur.fetchone.return_value = {
        "policies_configured": 2,
        "total_tickets": 25,
        "count_new": 5,
        "count_open": 10,
        "count_pending": 3,
        "count_resolved": 5,
        "count_closed": 2,
        "unconfigured_tickets": 5,
        # First response
        "fr_eligible": 20,
        "fr_dq_issues": 1,
        "fr_completed_on_time": 10,
        "fr_completed_late": 2,
        "fr_incomplete_before": 5,
        "fr_incomplete_due_now": 1,
        "fr_incomplete_overdue": 1,
        # Resolution
        "res_eligible": 20,
        "res_dq_issues": 0,
        "res_completed_on_time": 8,
        "res_completed_late": 2,
        "res_incomplete_before": 7,
        "res_incomplete_due_now": 0,
        "res_incomplete_overdue": 3,
    }

    metrics = dashboard_repo.get_tenant_dashboard_metrics(
        conn=mock_conn,
        tenant_id=tenant_id,
        evaluated_at=evaluated_at,
    )

    assert metrics.tenant_id == tenant_id
    assert metrics.evaluated_at == evaluated_at
    assert metrics.schema_available is True
    assert metrics.status_counts.total == 25
    assert metrics.status_counts.new == 5
    assert metrics.status_counts.open == 10
    assert metrics.status_counts.pending == 3
    assert metrics.status_counts.resolved == 5
    assert metrics.status_counts.closed == 2
    # Open workload = new + open + pending = 5 + 10 + 3 = 18
    assert metrics.open_workload == 18
    # Completed workload = resolved + closed = 5 + 2 = 7
    assert metrics.completed_workload == 7

    assert metrics.sla_summary is not None
    assert metrics.sla_summary.policies_configured == 2
    assert metrics.sla_summary.unconfigured_tickets == 5

    # First response checks
    fr = metrics.sla_summary.first_response
    assert fr.eligible_total == 20
    assert fr.data_quality_issues == 1
    assert fr.valid_eligible_total == 19
    assert fr.completed_on_time == 10
    assert fr.completed_late == 2
    assert fr.completed_total == 12
    # 10 / 12 = 0.8333
    assert fr.completed_timeliness_rate == 0.8333
    assert fr.incomplete_before_deadline == 5
    assert fr.incomplete_due_now == 1
    assert fr.incomplete_overdue == 1
    assert fr.incomplete_total == 7
    # (5 + 1) / 7 = 6 / 7 = 0.8571
    assert fr.incomplete_not_overdue_rate == 0.8571


def test_get_tenant_dashboard_metrics_schema_unavailable():
    """Verify savepoint rollback and fallback execution when sla_policies table does not exist."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    tenant_id = uuid.uuid4()
    evaluated_at = datetime(2026, 11, 8, 14, 0, 0, tzinfo=UTC)

    # First execute fails with UndefinedTable on sla_policies
    exc = psycopg.errors.UndefinedTable("relation \"sla_policies\" does not exist")

    # Inner transaction raises UndefinedTable, fallback cursor succeeds
    mock_cur.execute.side_effect = [
        exc,
        None,  # Fallback query on tickets
    ]
    mock_cur.fetchone.return_value = {
        "total_tickets": 15,
        "count_new": 3,
        "count_open": 5,
        "count_pending": 2,
        "count_resolved": 4,
        "count_closed": 1,
    }

    metrics = dashboard_repo.get_tenant_dashboard_metrics(
        conn=mock_conn,
        tenant_id=tenant_id,
        evaluated_at=evaluated_at,
    )

    assert metrics.schema_available is False
    assert metrics.sla_summary is None
    assert metrics.message == "SLA policy schema not provisioned."
    assert metrics.status_counts.total == 15
    assert metrics.open_workload == 10
    assert metrics.completed_workload == 5


def test_dashboard_endpoint_requires_auth():
    """Verify GET /browser-api/dashboard rejects unauthenticated requests with 401."""
    client = TestClient(app)
    resp = client.get("/browser-api/dashboard")
    assert resp.status_code == 401


def test_dashboard_endpoint_derives_tenant_from_session():
    """Verify tenant isolation and response payload contract through authenticated endpoint."""
    test_tenant_id = uuid.uuid4()
    test_user_id = uuid.uuid4()
    evaluated_at = datetime(2026, 11, 8, 15, 0, 0, tzinfo=UTC)

    mock_metrics = dashboard_repo.DashboardMetrics(
        tenant_id=test_tenant_id,
        evaluated_at=evaluated_at,
        schema_available=True,
        status_counts=dashboard_repo.StatusCounts(
            new=2, open=4, pending=1, resolved=8, closed=5, total=20
        ),
        open_workload=7,
        completed_workload=13,
        sla_summary=dashboard_repo.DashboardSLASummary(
            policies_configured=1,
            unconfigured_tickets=2,
            first_response=dashboard_repo._calculate_target_metrics(
                eligible_total=18,
                dq_issues=0,
                completed_on_time=12,
                completed_late=1,
                incomplete_before=4,
                incomplete_due_now=1,
                incomplete_overdue=0,
            ),
            resolution=dashboard_repo._calculate_target_metrics(
                eligible_total=18,
                dq_issues=0,
                completed_on_time=10,
                completed_late=3,
                incomplete_before=3,
                incomplete_due_now=0,
                incomplete_overdue=2,
            ),
        ),
        message=None,
    )

    app_dependency_overrides = {}
    from app.deps_browser import get_browser_principal_json

    app_dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=test_user_id,
        role="agent",
    )

    with patch("app.routers.browser_api.get_connection") as mock_get_conn:
        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        with patch("app.repositories.dashboard.get_tenant_dashboard_metrics", return_value=mock_metrics) as mock_get:
            with patch.dict(app.dependency_overrides, app_dependency_overrides):
                client = TestClient(app)
                resp = client.get("/browser-api/dashboard")

                assert resp.status_code == 200
                assert resp.headers.get("Cache-Control") == "no-store"
                data = resp.json()

                # Verify tenant was derived from principal
                assert data["tenant_id"] == str(test_tenant_id)
                mock_get.assert_called_once()
                call_kwargs = mock_get.call_args.kwargs
                assert call_kwargs["tenant_id"] == test_tenant_id

                # Verify structure
                assert data["schema_available"] is True
                assert data["status_counts"]["total"] == 20
                assert data["workload"]["open"] == 7
                assert data["workload"]["completed"] == 13
                assert data["workload"]["scope_label"] == "All-time tenant workload"

                # SLA summary
                sla = data["sla_summary"]
                assert sla["policies_configured"] == 1
                assert sla["unconfigured_tickets"] == 2

                fr = sla["first_response"]
                assert fr["eligible_total"] == 18
                assert fr["completed_on_time"] == 12
                assert fr["completed_late"] == 1
                assert fr["completed_timeliness_rate"] == round(12 / 13, 4)
                assert fr["incomplete_due_now"] == 1
                assert fr["incomplete_not_overdue_rate"] == 1.0

                res = sla["resolution"]
                assert res["eligible_total"] == 18
                assert res["completed_timeliness_rate"] == round(10 / 13, 4)
                assert res["incomplete_not_overdue_rate"] == round(3 / 5, 4)

                # Disclaimer
                assert "Rules-based operational urgency" in data["urgency_notice"]


def test_get_tenant_dashboard_metrics_empty_tenant():
    """Verify repository correctly handles tenant with zero tickets but configured policies."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_conn.transaction.return_value.__enter__.return_value = mock_conn

    tenant_id = uuid.uuid4()
    evaluated_at = datetime(2026, 11, 8, 14, 0, 0, tzinfo=UTC)

    # Empty tenant SQL aggregate result
    mock_cur.fetchone.return_value = {
        "policies_configured": 2,
        "total_tickets": 0,
        "count_new": 0,
        "count_open": 0,
        "count_pending": 0,
        "count_resolved": 0,
        "count_closed": 0,
        "unconfigured_tickets": 0,
        "fr_eligible": 0,
        "fr_dq_issues": 0,
        "fr_completed_on_time": 0,
        "fr_completed_late": 0,
        "fr_incomplete_before": 0,
        "fr_incomplete_due_now": 0,
        "fr_incomplete_overdue": 0,
        "res_eligible": 0,
        "res_dq_issues": 0,
        "res_completed_on_time": 0,
        "res_completed_late": 0,
        "res_incomplete_before": 0,
        "res_incomplete_due_now": 0,
        "res_incomplete_overdue": 0,
    }

    metrics = dashboard_repo.get_tenant_dashboard_metrics(
        conn=mock_conn,
        tenant_id=tenant_id,
        evaluated_at=evaluated_at,
    )

    assert metrics.status_counts.total == 0
    assert metrics.open_workload == 0
    assert metrics.completed_workload == 0
    assert metrics.sla_summary is not None
    assert metrics.sla_summary.policies_configured == 2
    assert metrics.sla_summary.unconfigured_tickets == 0
    assert metrics.sla_summary.first_response.eligible_total == 0
    assert metrics.sla_summary.first_response.completed_timeliness_rate is None
    assert metrics.sla_summary.first_response.incomplete_not_overdue_rate is None


def test_get_tenant_dashboard_metrics_zero_policies():
    """Verify repository correctly handles tenant with tickets but zero configured policies."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_conn.transaction.return_value.__enter__.return_value = mock_conn

    tenant_id = uuid.uuid4()
    evaluated_at = datetime(2026, 11, 8, 14, 0, 0, tzinfo=UTC)

    mock_cur.fetchone.return_value = {
        "policies_configured": 0,
        "total_tickets": 10,
        "count_new": 4,
        "count_open": 6,
        "count_pending": 0,
        "count_resolved": 0,
        "count_closed": 0,
        "unconfigured_tickets": 10,  # All tickets unconfigured
        "fr_eligible": 0,
        "fr_dq_issues": 0,
        "fr_completed_on_time": 0,
        "fr_completed_late": 0,
        "fr_incomplete_before": 0,
        "fr_incomplete_due_now": 0,
        "fr_incomplete_overdue": 0,
        "res_eligible": 0,
        "res_dq_issues": 0,
        "res_completed_on_time": 0,
        "res_completed_late": 0,
        "res_incomplete_before": 0,
        "res_incomplete_due_now": 0,
        "res_incomplete_overdue": 0,
    }

    metrics = dashboard_repo.get_tenant_dashboard_metrics(
        conn=mock_conn,
        tenant_id=tenant_id,
        evaluated_at=evaluated_at,
    )

    assert metrics.status_counts.total == 10
    assert metrics.open_workload == 10
    assert metrics.sla_summary is not None
    assert metrics.sla_summary.policies_configured == 0
    assert metrics.sla_summary.unconfigured_tickets == 10
    assert metrics.sla_summary.first_response.eligible_total == 0
    assert metrics.sla_summary.first_response.completed_timeliness_rate is None
    assert metrics.sla_summary.first_response.incomplete_not_overdue_rate is None
