"""Artifact-free unit tests for deterministic SLA evaluation service.

Verifies:
1. Pure calculation with clock injection (never hides the clock).
2. Strict rejection of naive datetimes without timezone awareness.
3. Strict rejection of impossible event ordering.
4. Independent first-response and resolution target evaluation.
5. Exact deadline arithmetic (created_at + target_minutes).
6. Deadline boundary behavior: at exactly the deadline, incomplete target is due, NOT overdue.
7. Overdue threshold: strictly current_time > deadline.
8. Fulfilled vs breached completion: late completion remains breached/late permanently.
9. Rules-based urgency bands: low (<0.50), medium (0.50-0.80), high (>=0.80), overdue (>deadline).
10. Missing policy handling: returns is_configured=False and overall_status='not_configured'.
"""

from datetime import datetime, timedelta, timezone
import pytest

from app.services.sla import (
    SLAPolicyLike,
    SLATicketEvaluation,
    TargetEvaluation,
    evaluate_target,
    evaluate_ticket_sla,
)


class DummyPolicy:
    """Mock policy matching SLAPolicyLike protocol."""
    def __init__(self, priority: str, first_response_minutes: int, resolution_minutes: int):
        self.priority = priority
        self.first_response_minutes = first_response_minutes
        self.resolution_minutes = resolution_minutes


UTC = timezone.utc


def test_missing_policy_returns_not_configured():
    created = datetime(2026, 11, 8, 10, 0, tzinfo=UTC)
    now = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)

    result = evaluate_ticket_sla(
        created_at=created,
        priority="urgent",
        first_response_at=None,
        resolved_at=None,
        policy=None,
        current_time=now,
    )

    assert result.is_configured is False
    assert result.policy_priority is None
    assert result.overall_status == "not_configured"
    assert result.overall_urgency is None
    assert result.first_response is None
    assert result.resolution is None
    assert "Rules-based operational urgency" in result.urgency_notice


def test_rejection_of_naive_timestamps():
    naive_created = datetime(2026, 11, 8, 10, 0)
    aware_now = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
    policy = DummyPolicy("urgent", 60, 240)

    # Naive created_at
    with pytest.raises(ValueError, match="created_at.*must be timezone-aware"):
        evaluate_ticket_sla(
            created_at=naive_created,
            priority="urgent",
            first_response_at=None,
            resolved_at=None,
            policy=policy,
            current_time=aware_now,
        )

    aware_created = datetime(2026, 11, 8, 10, 0, tzinfo=UTC)
    naive_now = datetime(2026, 11, 8, 12, 0)

    # Naive current_time
    with pytest.raises(ValueError, match="current_time.*must be timezone-aware"):
        evaluate_ticket_sla(
            created_at=aware_created,
            priority="urgent",
            first_response_at=None,
            resolved_at=None,
            policy=policy,
            current_time=naive_now,
        )

    # Naive first_response_at
    with pytest.raises(ValueError, match="first_response_at.*must be timezone-aware"):
        evaluate_ticket_sla(
            created_at=aware_created,
            priority="urgent",
            first_response_at=datetime(2026, 11, 8, 10, 30),
            resolved_at=None,
            policy=policy,
            current_time=aware_now,
        )

    # Naive resolved_at
    with pytest.raises(ValueError, match="resolved_at.*must be timezone-aware"):
        evaluate_ticket_sla(
            created_at=aware_created,
            priority="urgent",
            first_response_at=None,
            resolved_at=datetime(2026, 11, 8, 11, 0),
            policy=policy,
            current_time=aware_now,
        )


def test_rejection_of_impossible_event_ordering():
    created = datetime(2026, 11, 8, 10, 0, tzinfo=UTC)
    past_time = datetime(2026, 11, 8, 9, 0, tzinfo=UTC)
    policy = DummyPolicy("urgent", 60, 240)

    # created_at in the future relative to current_time
    with pytest.raises(ValueError, match="created_at.*cannot be in the future"):
        evaluate_ticket_sla(
            created_at=created,
            priority="urgent",
            first_response_at=None,
            resolved_at=None,
            policy=policy,
            current_time=past_time,
        )

    # first_response_at precedes created_at
    with pytest.raises(ValueError, match="first_response_at.*cannot precede created_at"):
        evaluate_ticket_sla(
            created_at=created,
            priority="urgent",
            first_response_at=datetime(2026, 11, 8, 9, 30, tzinfo=UTC),
            resolved_at=None,
            policy=policy,
            current_time=datetime(2026, 11, 8, 12, 0, tzinfo=UTC),
        )

    # resolved_at precedes created_at
    with pytest.raises(ValueError, match="resolved_at.*cannot precede created_at"):
        evaluate_ticket_sla(
            created_at=created,
            priority="urgent",
            first_response_at=None,
            resolved_at=datetime(2026, 11, 8, 9, 45, tzinfo=UTC),
            policy=policy,
            current_time=datetime(2026, 11, 8, 12, 0, tzinfo=UTC),
        )

    # first_response_at in the future relative to current_time
    with pytest.raises(ValueError, match="first_response_at.*cannot be in the future"):
        evaluate_ticket_sla(
            created_at=created,
            priority="urgent",
            first_response_at=datetime(2026, 11, 8, 13, 0, tzinfo=UTC),
            resolved_at=None,
            policy=policy,
            current_time=datetime(2026, 11, 8, 12, 0, tzinfo=UTC),
        )

    # resolved_at in the future relative to current_time
    with pytest.raises(ValueError, match="resolved_at.*cannot be in the future"):
        evaluate_ticket_sla(
            created_at=created,
            priority="urgent",
            first_response_at=None,
            resolved_at=datetime(2026, 11, 8, 13, 0, tzinfo=UTC),
            policy=policy,
            current_time=datetime(2026, 11, 8, 12, 0, tzinfo=UTC),
        )


def test_exact_deadline_and_on_track_boundaries():
    created = datetime(2026, 11, 8, 10, 0, tzinfo=UTC)
    policy = DummyPolicy("high", 60, 120)  # FR 60m (due 11:00), RES 120m (due 12:00)

    # T = 10:15 (25% FR elapsed, 12.5% RES elapsed) -> both low urgency, active
    t1 = created + timedelta(minutes=15)
    res = evaluate_ticket_sla(created, "high", None, None, policy, t1)
    assert res.first_response.due_at == created + timedelta(minutes=60)
    assert res.resolution.due_at == created + timedelta(minutes=120)
    assert res.first_response.status == "active"
    assert res.first_response.timeliness == "on_track"
    assert res.first_response.urgency == "low"
    assert res.resolution.status == "active"
    assert res.resolution.urgency == "low"
    assert res.overall_status == "active"
    assert res.overall_urgency == "low"

    # T = 10:30 (50% FR elapsed) -> medium urgency for FR, low for RES -> overall medium
    t2 = created + timedelta(minutes=30)
    res = evaluate_ticket_sla(created, "high", None, None, policy, t2)
    assert res.first_response.urgency == "medium"
    assert res.resolution.urgency == "low"
    assert res.overall_urgency == "medium"

    # T = 10:48 (80% FR elapsed) -> high urgency for FR
    t3 = created + timedelta(minutes=48)
    res = evaluate_ticket_sla(created, "high", None, None, policy, t3)
    assert res.first_response.urgency == "high"
    assert res.first_response.timeliness == "on_track"
    assert res.overall_urgency == "high"

    # T = 11:00 (EXACTLY at deadline of FR: 100% elapsed)
    # Must be due, NOT overdue; active and on_track, NOT breached
    t_exact = created + timedelta(minutes=60)
    res = evaluate_ticket_sla(created, "high", None, None, policy, t_exact)
    assert res.first_response.is_breached is False
    assert res.first_response.status == "active"
    assert res.first_response.timeliness == "on_track"
    assert res.first_response.urgency == "high"
    assert res.overall_status == "active"

    # T = 11:00:01 (1 second past deadline) -> strictly overdue and breached!
    t_overdue = created + timedelta(minutes=60, seconds=1)
    res = evaluate_ticket_sla(created, "high", None, None, policy, t_overdue)
    assert res.first_response.is_breached is True
    assert res.first_response.status == "breached"
    assert res.first_response.timeliness == "overdue"
    assert res.first_response.urgency == "overdue"
    assert res.overall_status == "breached"
    assert res.overall_urgency == "overdue"


def test_fulfilled_and_late_completion_distinction():
    created = datetime(2026, 11, 8, 10, 0, tzinfo=UTC)
    policy = DummyPolicy("normal", 60, 240)  # FR due 11:00, RES due 14:00

    # 1. First response completed at 10:45 (before 11:00 deadline) -> fulfilled, on_time
    fr_time = created + timedelta(minutes=45)
    now = created + timedelta(minutes=90)  # current time is 11:30 (past FR deadline, but FR was fulfilled on time)
    res = evaluate_ticket_sla(created, "normal", fr_time, None, policy, now)

    assert res.first_response.is_completed is True
    assert res.first_response.is_breached is False
    assert res.first_response.status == "fulfilled"
    assert res.first_response.timeliness == "on_time"
    assert res.first_response.urgency == "completed"

    # Resolution is still incomplete and active at 11:30 (due at 14:00)
    assert res.resolution.is_completed is False
    assert res.resolution.is_breached is False
    assert res.resolution.status == "active"
    assert res.overall_status == "active"

    # 2. First response completed at 11:15 (LATE: after 11:00 deadline)
    fr_late = created + timedelta(minutes=75)
    res_late = evaluate_ticket_sla(created, "normal", fr_late, None, policy, now)

    assert res_late.first_response.is_completed is True
    assert res_late.first_response.is_breached is True
    assert res_late.first_response.status == "breached"
    assert res_late.first_response.timeliness == "late"
    assert res_late.first_response.urgency == "completed"
    # Even though FR is finished, because it breached, overall ticket status is breached!
    assert res_late.overall_status == "breached"

    # 3. Both completed on time:
    res_time = created + timedelta(minutes=180)  # 13:00 (before 14:00)
    res_both_on_time = evaluate_ticket_sla(created, "normal", fr_time, res_time, policy, now + timedelta(hours=3))
    assert res_both_on_time.first_response.status == "fulfilled"
    assert res_both_on_time.resolution.status == "fulfilled"
    assert res_both_on_time.overall_status == "fulfilled"
    assert res_both_on_time.overall_urgency == "completed"

    # 4. Both completed, but resolution was late:
    res_time_late = created + timedelta(minutes=300)  # 15:00 (after 14:00)
    res_both_late = evaluate_ticket_sla(created, "normal", fr_time, res_time_late, policy, now + timedelta(hours=5))
    assert res_both_late.first_response.status == "fulfilled"
    assert res_both_late.resolution.status == "breached"
    assert res_both_late.resolution.timeliness == "late"
    assert res_both_late.overall_status == "breached"
    assert res_both_late.overall_urgency == "completed"


def test_completed_at_exact_deadline_is_on_time():
    created = datetime(2026, 11, 8, 10, 0, tzinfo=UTC)
    policy = DummyPolicy("low", 120, 480)
    exact_fr = created + timedelta(minutes=120)
    exact_res = created + timedelta(minutes=480)

    res = evaluate_ticket_sla(created, "low", exact_fr, exact_res, policy, exact_res + timedelta(minutes=5))
    assert res.first_response.status == "fulfilled"
    assert res.first_response.timeliness == "on_time"
    assert res.resolution.status == "fulfilled"
    assert res.resolution.timeliness == "on_time"
    assert res.overall_status == "fulfilled"


def test_resolution_completed_without_first_response():
    """A ticket can be resolved directly without recording an explicit first response."""
    created = datetime(2026, 11, 8, 10, 0, tzinfo=UTC)
    policy = DummyPolicy("normal", 60, 240)
    res_time = created + timedelta(minutes=50)
    now = created + timedelta(minutes=70)  # past FR deadline

    res = evaluate_ticket_sla(created, "normal", None, res_time, policy, now)

    # Resolution was fulfilled on time
    assert res.resolution.is_completed is True
    assert res.resolution.status == "fulfilled"

    # First response was never recorded and deadline expired at 11:00 -> breached/overdue
    assert res.first_response.is_completed is False
    assert res.first_response.is_breached is True
    assert res.first_response.status == "breached"
    assert res.first_response.timeliness == "overdue"
    assert res.overall_status == "breached"
