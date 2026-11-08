"""Operational dashboard repository layer for multi-tenant operational summaries.

Per Phase 5 Slice 3 requirements:
- Strictly tenant-isolated via tenant_id derived from authenticated principal.
- Set-based SQL aggregation across all tenant tickets using a single query.
- Injected UTC evaluation timestamp (evaluated_at) ensures deterministic snapshot.
- Evaluates first-response and resolution targets independently based on
  first_response_at and resolved_at timestamps (matching pure SLA evaluator semantics).
- Mutually exclusive and complete bucket partitioning:
  - completed_on_time: completed_at <= due_at
  - completed_late: completed_at > due_at
  - incomplete_before_deadline: completed_at IS NULL AND now < due_at
  - incomplete_due_now: completed_at IS NULL AND now = due_at
  - incomplete_overdue: completed_at IS NULL AND now > due_at
  - data_quality_issues: invalid timestamps (created in future, completed before created, or completed in future)
- Denominators are explicit; invalid data is excluded from timeliness rates.
- Rates return None when denominator is zero.
- Graceful recovery via savepoint when sla_policies schema is missing (migration 008 unapplied).
- Schema indexes and constraints defined in migrations:
  - tickets: idx_tickets_tenant_id_id, idx_tickets_tenant_status_created_at (migration 001)
  - sla_policies: idx_sla_policies_tenant_id_id, uq_sla_policies_tenant_priority (migration 008)
- Scale characteristic: all-time aggregation scans tenant tickets; aggregation cost grows with tenant volume.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import uuid

import psycopg
import psycopg.errors
from psycopg.rows import dict_row

from app.repositories.sla import _is_sla_policies_undefined, SLASchemaUnavailableError


@dataclass(frozen=True)
class StatusCounts:
    """Breakdown of tickets by lifecycle status."""
    new: int
    open: int
    pending: int
    resolved: int
    closed: int
    total: int


@dataclass(frozen=True)
class TargetMetrics:
    """Detailed metric breakdown for an SLA target (first-response or resolution)."""
    eligible_total: int
    data_quality_issues: int
    valid_eligible_total: int
    completed_on_time: int
    completed_late: int
    completed_total: int
    incomplete_before_deadline: int
    incomplete_due_now: int
    incomplete_overdue: int
    incomplete_total: int
    completed_timeliness_rate: float | None
    incomplete_not_overdue_rate: float | None


@dataclass(frozen=True)
class DashboardSLASummary:
    """Tenant-level SLA summary metrics."""
    policies_configured: int
    unconfigured_tickets: int
    first_response: TargetMetrics
    resolution: TargetMetrics


@dataclass(frozen=True)
class DashboardMetrics:
    """Complete tenant operational dashboard metrics snapshot."""
    tenant_id: uuid.UUID
    evaluated_at: datetime
    schema_available: bool
    status_counts: StatusCounts
    open_workload: int
    completed_workload: int
    sla_summary: DashboardSLASummary | None
    message: str | None = None


def _calculate_target_metrics(
    eligible_total: int,
    dq_issues: int,
    completed_on_time: int,
    completed_late: int,
    incomplete_before: int,
    incomplete_due_now: int,
    incomplete_overdue: int,
) -> TargetMetrics:
    """Calculate target metrics with exact mathematical validation."""
    valid_eligible = eligible_total - dq_issues
    completed_total = completed_on_time + completed_late
    incomplete_total = incomplete_before + incomplete_due_now + incomplete_overdue

    # Completed timeliness rate: completed_on_time / completed_total (if completed_total > 0)
    completed_rate = (
        round(completed_on_time / completed_total, 4)
        if completed_total > 0
        else None
    )

    # Incomplete not-overdue rate: (incomplete_before + incomplete_due_now) / incomplete_total
    not_overdue_rate = (
        round((incomplete_before + incomplete_due_now) / incomplete_total, 4)
        if incomplete_total > 0
        else None
    )

    return TargetMetrics(
        eligible_total=eligible_total,
        data_quality_issues=dq_issues,
        valid_eligible_total=valid_eligible,
        completed_on_time=completed_on_time,
        completed_late=completed_late,
        completed_total=completed_total,
        incomplete_before_deadline=incomplete_before,
        incomplete_due_now=incomplete_due_now,
        incomplete_overdue=incomplete_overdue,
        incomplete_total=incomplete_total,
        completed_timeliness_rate=completed_rate,
        incomplete_not_overdue_rate=not_overdue_rate,
    )


def get_tenant_dashboard_metrics(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    evaluated_at: datetime,
) -> DashboardMetrics:
    """Execute set-based tenant aggregation for the operational dashboard.

    Caller owns the database connection and transaction context.
    Evaluated_at must be a timezone-aware UTC datetime.
    """
    if evaluated_at.tzinfo is None or evaluated_at.tzinfo.utcoffset(evaluated_at) is None:
        raise ValueError("evaluated_at must be a timezone-aware datetime.")

    # Convert to UTC
    utc_evaluated_at = evaluated_at.astimezone(timezone.utc)

    # Attempt primary set-based aggregation using lateral policy join
    try:
        # We wrap in a subtransaction (savepoint in psycopg 3) so that if sla_policies
        # is missing, we can roll back cleanly to the savepoint and run fallback queries.
        with conn.transaction():
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    WITH tenant_pols AS (
                        SELECT id, priority, first_response_minutes, resolution_minutes
                        FROM sla_policies
                        WHERE tenant_id = %(tenant_id)s
                    ),
                    tenant_pols_count AS (
                        SELECT COUNT(*) AS count FROM tenant_pols
                    ),
                    ticket_matched AS (
                        SELECT
                            t.id,
                            t.status,
                            t.created_at,
                            t.first_response_at,
                            t.resolved_at,
                            pol.first_response_minutes,
                            pol.resolution_minutes
                        FROM tickets t
                        LEFT JOIN LATERAL (
                            SELECT tp.first_response_minutes, tp.resolution_minutes
                            FROM tenant_pols tp
                            WHERE (t.priority IS NOT NULL AND tp.priority = t.priority)
                               OR tp.priority = 'default'
                            ORDER BY (CASE WHEN t.priority IS NOT NULL AND tp.priority = t.priority THEN 0 ELSE 1 END) ASC
                            LIMIT 1
                        ) pol ON true
                        WHERE t.tenant_id = %(tenant_id)s
                    )
                    SELECT
                        (SELECT count FROM tenant_pols_count) AS policies_configured,
                        COUNT(tm.id) AS total_tickets,
                        COUNT(*) FILTER (WHERE tm.status = 'new') AS count_new,
                        COUNT(*) FILTER (WHERE tm.status = 'open') AS count_open,
                        COUNT(*) FILTER (WHERE tm.status = 'pending') AS count_pending,
                        COUNT(*) FILTER (WHERE tm.status = 'resolved') AS count_resolved,
                        COUNT(*) FILTER (WHERE tm.status = 'closed') AS count_closed,
                        COUNT(*) FILTER (WHERE tm.id IS NOT NULL AND tm.first_response_minutes IS NULL) AS unconfigured_tickets,

                        -- First Response Target
                        COUNT(*) FILTER (WHERE tm.id IS NOT NULL AND tm.first_response_minutes IS NOT NULL) AS fr_eligible,
                        COUNT(*) FILTER (
                            WHERE tm.id IS NOT NULL AND tm.first_response_minutes IS NOT NULL AND (
                                tm.created_at > %(now)s OR
                                (tm.first_response_at IS NOT NULL AND tm.first_response_at < tm.created_at) OR
                                (tm.first_response_at IS NOT NULL AND tm.first_response_at > %(now)s)
                            )
                        ) AS fr_dq_issues,
                        COUNT(*) FILTER (
                            WHERE tm.id IS NOT NULL AND tm.first_response_minutes IS NOT NULL
                              AND NOT (
                                tm.created_at > %(now)s OR
                                (tm.first_response_at IS NOT NULL AND tm.first_response_at < tm.created_at) OR
                                (tm.first_response_at IS NOT NULL AND tm.first_response_at > %(now)s)
                              )
                              AND tm.first_response_at IS NOT NULL
                              AND tm.first_response_at <= tm.created_at + (tm.first_response_minutes * INTERVAL '1 minute')
                        ) AS fr_completed_on_time,
                        COUNT(*) FILTER (
                            WHERE tm.id IS NOT NULL AND tm.first_response_minutes IS NOT NULL
                              AND NOT (
                                tm.created_at > %(now)s OR
                                (tm.first_response_at IS NOT NULL AND tm.first_response_at < tm.created_at) OR
                                (tm.first_response_at IS NOT NULL AND tm.first_response_at > %(now)s)
                              )
                              AND tm.first_response_at IS NOT NULL
                              AND tm.first_response_at > tm.created_at + (tm.first_response_minutes * INTERVAL '1 minute')
                        ) AS fr_completed_late,
                        COUNT(*) FILTER (
                            WHERE tm.id IS NOT NULL AND tm.first_response_minutes IS NOT NULL
                              AND NOT (
                                tm.created_at > %(now)s OR
                                (tm.first_response_at IS NOT NULL AND tm.first_response_at < tm.created_at) OR
                                (tm.first_response_at IS NOT NULL AND tm.first_response_at > %(now)s)
                              )
                              AND tm.first_response_at IS NULL
                              AND %(now)s < tm.created_at + (tm.first_response_minutes * INTERVAL '1 minute')
                        ) AS fr_incomplete_before,
                        COUNT(*) FILTER (
                            WHERE tm.id IS NOT NULL AND tm.first_response_minutes IS NOT NULL
                              AND NOT (
                                tm.created_at > %(now)s OR
                                (tm.first_response_at IS NOT NULL AND tm.first_response_at < tm.created_at) OR
                                (tm.first_response_at IS NOT NULL AND tm.first_response_at > %(now)s)
                              )
                              AND tm.first_response_at IS NULL
                              AND %(now)s = tm.created_at + (tm.first_response_minutes * INTERVAL '1 minute')
                        ) AS fr_incomplete_due_now,
                        COUNT(*) FILTER (
                            WHERE tm.id IS NOT NULL AND tm.first_response_minutes IS NOT NULL
                              AND NOT (
                                tm.created_at > %(now)s OR
                                (tm.first_response_at IS NOT NULL AND tm.first_response_at < tm.created_at) OR
                                (tm.first_response_at IS NOT NULL AND tm.first_response_at > %(now)s)
                              )
                              AND tm.first_response_at IS NULL
                              AND %(now)s > tm.created_at + (tm.first_response_minutes * INTERVAL '1 minute')
                        ) AS fr_incomplete_overdue,

                        -- Resolution Target
                        COUNT(*) FILTER (WHERE tm.id IS NOT NULL AND tm.resolution_minutes IS NOT NULL) AS res_eligible,
                        COUNT(*) FILTER (
                            WHERE tm.id IS NOT NULL AND tm.resolution_minutes IS NOT NULL AND (
                                tm.created_at > %(now)s OR
                                (tm.resolved_at IS NOT NULL AND tm.resolved_at < tm.created_at) OR
                                (tm.resolved_at IS NOT NULL AND tm.resolved_at > %(now)s)
                            )
                        ) AS res_dq_issues,
                        COUNT(*) FILTER (
                            WHERE tm.id IS NOT NULL AND tm.resolution_minutes IS NOT NULL
                              AND NOT (
                                tm.created_at > %(now)s OR
                                (tm.resolved_at IS NOT NULL AND tm.resolved_at < tm.created_at) OR
                                (tm.resolved_at IS NOT NULL AND tm.resolved_at > %(now)s)
                              )
                              AND tm.resolved_at IS NOT NULL
                              AND tm.resolved_at <= tm.created_at + (tm.resolution_minutes * INTERVAL '1 minute')
                        ) AS res_completed_on_time,
                        COUNT(*) FILTER (
                            WHERE tm.id IS NOT NULL AND tm.resolution_minutes IS NOT NULL
                              AND NOT (
                                tm.created_at > %(now)s OR
                                (tm.resolved_at IS NOT NULL AND tm.resolved_at < tm.created_at) OR
                                (tm.resolved_at IS NOT NULL AND tm.resolved_at > %(now)s)
                              )
                              AND tm.resolved_at IS NOT NULL
                              AND tm.resolved_at > tm.created_at + (tm.resolution_minutes * INTERVAL '1 minute')
                        ) AS res_completed_late,
                        COUNT(*) FILTER (
                            WHERE tm.id IS NOT NULL AND tm.resolution_minutes IS NOT NULL
                              AND NOT (
                                tm.created_at > %(now)s OR
                                (tm.resolved_at IS NOT NULL AND tm.resolved_at < tm.created_at) OR
                                (tm.resolved_at IS NOT NULL AND tm.resolved_at > %(now)s)
                              )
                              AND tm.resolved_at IS NULL
                              AND %(now)s < tm.created_at + (tm.resolution_minutes * INTERVAL '1 minute')
                        ) AS res_incomplete_before,
                        COUNT(*) FILTER (
                            WHERE tm.id IS NOT NULL AND tm.resolution_minutes IS NOT NULL
                              AND NOT (
                                tm.created_at > %(now)s OR
                                (tm.resolved_at IS NOT NULL AND tm.resolved_at < tm.created_at) OR
                                (tm.resolved_at IS NOT NULL AND tm.resolved_at > %(now)s)
                              )
                              AND tm.resolved_at IS NULL
                              AND %(now)s = tm.created_at + (tm.resolution_minutes * INTERVAL '1 minute')
                        ) AS res_incomplete_due_now,
                        COUNT(*) FILTER (
                            WHERE tm.id IS NOT NULL AND tm.resolution_minutes IS NOT NULL
                              AND NOT (
                                tm.created_at > %(now)s OR
                                (tm.resolved_at IS NOT NULL AND tm.resolved_at < tm.created_at) OR
                                (tm.resolved_at IS NOT NULL AND tm.resolved_at > %(now)s)
                              )
                              AND tm.resolved_at IS NULL
                              AND %(now)s > tm.created_at + (tm.resolution_minutes * INTERVAL '1 minute')
                        ) AS res_incomplete_overdue
                    FROM ticket_matched tm;
                    """,
                    {
                        "tenant_id": tenant_id,
                        "now": utc_evaluated_at,
                    },
                )
                row = cur.fetchone()
                if not row:
                    raise RuntimeError("Failed to compute operational dashboard metrics: no row returned.")

                status_counts = StatusCounts(
                    new=row["count_new"],
                    open=row["count_open"],
                    pending=row["count_pending"],
                    resolved=row["count_resolved"],
                    closed=row["count_closed"],
                    total=row["total_tickets"],
                )
                open_workload = status_counts.new + status_counts.open + status_counts.pending
                completed_workload = status_counts.resolved + status_counts.closed

                fr_metrics = _calculate_target_metrics(
                    eligible_total=row["fr_eligible"],
                    dq_issues=row["fr_dq_issues"],
                    completed_on_time=row["fr_completed_on_time"],
                    completed_late=row["fr_completed_late"],
                    incomplete_before=row["fr_incomplete_before"],
                    incomplete_due_now=row["fr_incomplete_due_now"],
                    incomplete_overdue=row["fr_incomplete_overdue"],
                )

                res_metrics = _calculate_target_metrics(
                    eligible_total=row["res_eligible"],
                    dq_issues=row["res_dq_issues"],
                    completed_on_time=row["res_completed_on_time"],
                    completed_late=row["res_completed_late"],
                    incomplete_before=row["res_incomplete_before"],
                    incomplete_due_now=row["res_incomplete_due_now"],
                    incomplete_overdue=row["res_incomplete_overdue"],
                )

                sla_summary = DashboardSLASummary(
                    policies_configured=row["policies_configured"] or 0,
                    unconfigured_tickets=row["unconfigured_tickets"],
                    first_response=fr_metrics,
                    resolution=res_metrics,
                )

                return DashboardMetrics(
                    tenant_id=tenant_id,
                    evaluated_at=utc_evaluated_at,
                    schema_available=True,
                    status_counts=status_counts,
                    open_workload=open_workload,
                    completed_workload=completed_workload,
                    sla_summary=sla_summary,
                    message=None,
                )

    except psycopg.errors.UndefinedTable as exc:
        if not _is_sla_policies_undefined(exc):
            raise

        # Savepoint rolled back. Run fallback status counts query on tickets table only
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total_tickets,
                    COUNT(*) FILTER (WHERE status = 'new') AS count_new,
                    COUNT(*) FILTER (WHERE status = 'open') AS count_open,
                    COUNT(*) FILTER (WHERE status = 'pending') AS count_pending,
                    COUNT(*) FILTER (WHERE status = 'resolved') AS count_resolved,
                    COUNT(*) FILTER (WHERE status = 'closed') AS count_closed
                FROM tickets
                WHERE tenant_id = %(tenant_id)s;
                """,
                {"tenant_id": tenant_id},
            )
            fallback_row = cur.fetchone()
            if not fallback_row:
                raise RuntimeError("Failed to compute fallback dashboard status counts: no row returned.")

            status_counts = StatusCounts(
                new=fallback_row["count_new"],
                open=fallback_row["count_open"],
                pending=fallback_row["count_pending"],
                resolved=fallback_row["count_resolved"],
                closed=fallback_row["count_closed"],
                total=fallback_row["total_tickets"],
            )
            open_workload = status_counts.new + status_counts.open + status_counts.pending
            completed_workload = status_counts.resolved + status_counts.closed

            return DashboardMetrics(
                tenant_id=tenant_id,
                evaluated_at=utc_evaluated_at,
                schema_available=False,
                status_counts=status_counts,
                open_workload=open_workload,
                completed_workload=completed_workload,
                sla_summary=None,
                message="SLA policy schema not provisioned.",
            )
