"""Deterministic Service Level Agreement (SLA) evaluation service.

Core Principles:
1. Pure Mathematical Evaluation:
   - Zero database, network, or clock dependencies.
   - Evaluates strictly against injected `current_time`.
2. Strict Timezone & Boundary Enforcement:
   - Rejects naive datetimes without timezone awareness.
   - Rejects impossible event ordering (e.g. created_at in the future, responses before creation).
   - At exactly due_at (current_time == due_at), an incomplete target is due, NOT overdue.
3. Independent Target Lifecycles:
   - Evaluates first_response and resolution targets independently.
   - Keeps completion state (is_completed) and timeliness (on_time / late / on_track / overdue) distinct.
   - Completion after deadline is recorded as breached/late permanently.
4. Rules-Based Urgency Bands:
   - Low: elapsed fraction < 0.50
   - Medium: 0.50 <= fraction < 0.80
   - High: fraction >= 0.80 while not overdue
   - Overdue: current_time > deadline
   - Explicitly labeled as rules-based urgency; does NOT represent a trained predictive risk model.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol


class SLAPolicyLike(Protocol):
    """Protocol representing a policy with required duration attributes."""
    first_response_minutes: int
    resolution_minutes: int
    priority: str


@dataclass(frozen=True)
class TargetEvaluation:
    """Evaluation of an individual SLA target (first-response or resolution)."""
    target_name: str
    target_minutes: int
    due_at: datetime
    is_completed: bool
    completed_at: datetime | None
    is_breached: bool
    status: str  # "fulfilled", "breached", "active"
    timeliness: str  # "on_time", "late", "on_track", "overdue"
    elapsed_minutes: float
    elapsed_fraction: float
    urgency: str  # "low", "medium", "high", "overdue", "completed"


@dataclass(frozen=True)
class SLATicketEvaluation:
    """Complete SLA evaluation for a ticket across all applicable targets."""
    is_configured: bool
    policy_priority: str | None
    overall_status: str  # "not_configured", "active", "breached", "fulfilled"
    overall_urgency: str | None  # "low", "medium", "high", "overdue", "completed", or None if not configured
    evaluated_at: datetime
    first_response: TargetEvaluation | None
    resolution: TargetEvaluation | None
    urgency_notice: str = (
        "Rules-based operational urgency based on elapsed calendar time. "
        "Does not represent a trained statistical SLA risk model or calibrated breach probability."
    )


def _validate_timezone_aware(dt: datetime, name: str) -> None:
    """Ensure datetime has an explicit timezone (reject naive timestamps)."""
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(f"Timestamp '{name}' must be timezone-aware (received naive datetime: {dt})")


def evaluate_target(
    target_name: str,
    target_minutes: int,
    created_at: datetime,
    completed_at: datetime | None,
    current_time: datetime,
) -> TargetEvaluation:
    """Evaluate a single SLA target with explicit completion and timeliness semantics."""
    _validate_timezone_aware(created_at, "created_at")
    _validate_timezone_aware(current_time, "current_time")

    if created_at > current_time:
        raise ValueError(f"created_at ({created_at}) cannot be in the future relative to current_time ({current_time})")

    if completed_at is not None:
        _validate_timezone_aware(completed_at, "completed_at")
        if completed_at < created_at:
            raise ValueError(f"completed_at ({completed_at}) cannot precede created_at ({created_at})")
        if completed_at > current_time:
            raise ValueError(f"completed_at ({completed_at}) cannot be in the future relative to current_time ({current_time})")

    if target_minutes <= 0:
        raise ValueError(f"target_minutes for '{target_name}' must be > 0 (received {target_minutes})")

    due_at = created_at + timedelta(minutes=target_minutes)

    if completed_at is not None:
        is_completed = True
        elapsed_seconds = (completed_at - created_at).total_seconds()
        elapsed_minutes = max(0.0, elapsed_seconds / 60.0)
        elapsed_fraction = elapsed_minutes / float(target_minutes)

        if completed_at <= due_at:
            is_breached = False
            status = "fulfilled"
            timeliness = "on_time"
        else:
            is_breached = True
            status = "breached"
            timeliness = "late"

        urgency = "completed"
    else:
        is_completed = False
        elapsed_seconds = (current_time - created_at).total_seconds()
        elapsed_minutes = max(0.0, elapsed_seconds / 60.0)
        elapsed_fraction = elapsed_minutes / float(target_minutes)

        # Incomplete target is overdue only when strictly after due_at
        if current_time > due_at:
            is_breached = True
            status = "breached"
            timeliness = "overdue"
            urgency = "overdue"
        else:
            # At exactly due_at (or before), target is active and on_track
            is_breached = False
            status = "active"
            timeliness = "on_track"

            if elapsed_fraction >= 0.80:
                urgency = "high"
            elif elapsed_fraction >= 0.50:
                urgency = "medium"
            else:
                urgency = "low"

    return TargetEvaluation(
        target_name=target_name,
        target_minutes=target_minutes,
        due_at=due_at,
        is_completed=is_completed,
        completed_at=completed_at,
        is_breached=is_breached,
        status=status,
        timeliness=timeliness,
        elapsed_minutes=elapsed_minutes,
        elapsed_fraction=elapsed_fraction,
        urgency=urgency,
    )


def evaluate_ticket_sla(
    created_at: datetime,
    priority: str | None,
    first_response_at: datetime | None,
    resolved_at: datetime | None,
    policy: SLAPolicyLike | None,
    current_time: datetime,
) -> SLATicketEvaluation:
    """Deterministically evaluate all SLA targets for a ticket using currently configured policy.

    Guarantees:
    - Never hides the clock; current_time is strictly injected.
    - Rejects naive timestamps with ValueError.
    - Rejects impossible event ordering (e.g. response or resolution before creation).
    - If policy is None, returns is_configured=False and overall_status='not_configured'.
    - Completion at or before deadline is fulfilled/on_time.
    - Completion after deadline is breached/late, even after resolution.
    - Incomplete target is overdue only when current_time > deadline.
    - At exactly the deadline it is due, not overdue.
    - Urgency bands reflect rules-based elapsed calendar time, not predicted breach probability.
    """
    _validate_timezone_aware(created_at, "created_at")
    _validate_timezone_aware(current_time, "current_time")

    if created_at > current_time:
        raise ValueError(
            f"created_at ({created_at}) cannot be in the future relative to current_time ({current_time})"
        )

    if first_response_at is not None:
        _validate_timezone_aware(first_response_at, "first_response_at")
        if first_response_at < created_at:
            raise ValueError(
                f"first_response_at ({first_response_at}) cannot precede created_at ({created_at})"
            )
        if first_response_at > current_time:
            raise ValueError(
                f"first_response_at ({first_response_at}) cannot be in the future relative to current_time ({current_time})"
            )

    if resolved_at is not None:
        _validate_timezone_aware(resolved_at, "resolved_at")
        if resolved_at < created_at:
            raise ValueError(
                f"resolved_at ({resolved_at}) cannot precede created_at ({created_at})"
            )
        if resolved_at > current_time:
            raise ValueError(
                f"resolved_at ({resolved_at}) cannot be in the future relative to current_time ({current_time})"
            )

    if policy is None:
        return SLATicketEvaluation(
            is_configured=False,
            policy_priority=None,
            overall_status="not_configured",
            overall_urgency=None,
            evaluated_at=current_time,
            first_response=None,
            resolution=None,
        )

    # Evaluate independent targets
    fr_eval = evaluate_target(
        target_name="first_response",
        target_minutes=policy.first_response_minutes,
        created_at=created_at,
        completed_at=first_response_at,
        current_time=current_time,
    )

    res_eval = evaluate_target(
        target_name="resolution",
        target_minutes=policy.resolution_minutes,
        created_at=created_at,
        completed_at=resolved_at,
        current_time=current_time,
    )

    # Determine overall status:
    # Any breach (first response or resolution) results in an overall breached status
    if fr_eval.is_breached or res_eval.is_breached:
        overall_status = "breached"
    elif fr_eval.is_completed and res_eval.is_completed:
        overall_status = "fulfilled"
    else:
        overall_status = "active"

    # Determine overall urgency from incomplete targets:
    # Priority rank: overdue > high > medium > low > completed
    urgency_ranks = {"overdue": 4, "high": 3, "medium": 2, "low": 1, "completed": 0}
    incomplete_urgencies = [
        t.urgency for t in (fr_eval, res_eval) if not t.is_completed
    ]

    if not incomplete_urgencies:
        overall_urgency = "completed"
    else:
        overall_urgency = max(incomplete_urgencies, key=lambda u: urgency_ranks.get(u, 0))

    return SLATicketEvaluation(
        is_configured=True,
        policy_priority=policy.priority,
        overall_status=overall_status,
        overall_urgency=overall_urgency,
        evaluated_at=current_time,
        first_response=fr_eval,
        resolution=res_eval,
    )
