"""Guarded database integration tests for Phase 5 Slice 3: Tenant Operational Dashboard.

Strict safety constraints:
- Runs ONLY against an already established disposable local PostgreSQL
  instance (localhost or 127.0.0.1 with 'test' in dbname).
- Hosted databases (e.g. Supabase poolers) are strictly prohibited.
- If no established disposable local test database exists, tests are cleanly skipped.
- Tests verify actual aggregate SQL, lateral policy precedence, exact deadlines,
  evaluator parity, and two-tenant isolation.
"""

from datetime import datetime, timedelta, timezone
import uuid
import pytest
import psycopg

from app.repositories.auth import create_tenant_and_admin_user
import app.repositories.dashboard as dashboard_repo
import app.repositories.sla as sla_repo
import app.repositories.tickets as ticket_repo
from app.security import hash_password
from app.services.sla import evaluate_target
from tests.test_embedding_jobs_db import resolve_disposable_local_test_db

LOCAL_TEST_DB_URL, LOCAL_SAFETY_STATUS = resolve_disposable_local_test_db()
pytestmark = pytest.mark.skipif(
    LOCAL_TEST_DB_URL is None,
    reason=f"Guarded Dashboard DB integration tests require an established disposable local PostgreSQL instance: {LOCAL_SAFETY_STATUS}",
)

UTC = timezone.utc


def _create_test_tenant(company_name: str) -> uuid.UUID:
    """Helper to create an isolated test tenant."""
    assert LOCAL_TEST_DB_URL is not None
    suffix = uuid.uuid4().hex[:8]
    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        tenant_id, _ = create_tenant_and_admin_user(
            conn=conn,
            company_name=company_name,
            normalized_email=f"dash_user_{suffix}@test.com",
            password_hash=hash_password("Pass123!"),
        )
    return tenant_id


def test_dashboard_sql_aggregation_and_two_tenant_isolation():
    """Verify actual aggregate SQL, exact deadlines, policy resolution, and two-tenant isolation."""
    assert LOCAL_TEST_DB_URL is not None

    tenant_a = _create_test_tenant("Tenant A Operational Test")
    tenant_b = _create_test_tenant("Tenant B Operational Test")

    now = datetime(2026, 11, 8, 16, 0, 0, tzinfo=UTC)

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        # Check if sla_policies table exists in test DB
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'sla_policies'
                );
                """
            )
            sla_table_exists = cur.fetchone()[0]

        if not sla_table_exists:
            pytest.skip("sla_policies table not provisioned in local test database.")

        with conn.transaction():
            # Configure policies for Tenant A
            sla_repo.upsert_sla_policy(
                conn=conn,
                tenant_id=tenant_a,
                name="Tenant A Urgent",
                priority="urgent",
                first_response_minutes=30,
                resolution_minutes=120,
            )
            sla_repo.upsert_sla_policy(
                conn=conn,
                tenant_id=tenant_a,
                name="Tenant A Default",
                priority="default",
                first_response_minutes=60,
                resolution_minutes=240,
            )

            # Configure policies for Tenant B (different targets)
            sla_repo.upsert_sla_policy(
                conn=conn,
                tenant_id=tenant_b,
                name="Tenant B Default",
                priority="default",
                first_response_minutes=120,
                resolution_minutes=480,
            )

            # Insert Tenant A Tickets using supported arguments (returns (TicketRecord, is_created))
            # Ticket 1 (urgent): created 20m ago, incomplete -> due at +30m (in 10m) -> incomplete_before
            t1, _ = ticket_repo.create_or_get_ticket(
                conn=conn,
                tenant_id=tenant_a,
                customer_email="a1@test.com",
                subject="Tenant A - T1",
                body="Body",
            )
            # Ticket 2 (urgent): created 30m ago, incomplete -> due at +30m (now!) -> incomplete_due_now
            t2, _ = ticket_repo.create_or_get_ticket(
                conn=conn,
                tenant_id=tenant_a,
                customer_email="a2@test.com",
                subject="Tenant A - T2",
                body="Body",
            )
            # Ticket 3 (urgent): created 40m ago, response at +20m (on-time), resolved at +150m (late)
            t3, _ = ticket_repo.create_or_get_ticket(
                conn=conn,
                tenant_id=tenant_a,
                customer_email="a3@test.com",
                subject="Tenant A - T3",
                body="Body",
            )
            # Ticket 4 (normal -> default policy: 60m/240m): created 80m ago, response at +90m (late), incomplete resolution
            t4, _ = ticket_repo.create_or_get_ticket(
                conn=conn,
                tenant_id=tenant_a,
                customer_email="a4@test.com",
                subject="Tenant A - T4",
                body="Body",
            )
            # Ticket 5 (no priority -> default policy: 60m/240m): created 300m ago, no response, no resolution -> overdue
            t5, _ = ticket_repo.create_or_get_ticket(
                conn=conn,
                tenant_id=tenant_a,
                customer_email="a5@test.com",
                subject="Tenant A - T5",
                body="Body",
            )
            # Ticket 6 (urgent): created 50m ago, but has invalid timestamp (first_response_at before created_at)
            t6, _ = ticket_repo.create_or_get_ticket(
                conn=conn,
                tenant_id=tenant_a,
                customer_email="a6@test.com",
                subject="Tenant A - T6 (Data Quality Issue)",
                body="Body",
            )

            # Insert Tenant B Tickets:
            tb1, _ = ticket_repo.create_or_get_ticket(
                conn=conn,
                tenant_id=tenant_b,
                customer_email="b1@test.com",
                subject="Tenant B - T1",
                body="Body",
            )
            tb2, _ = ticket_repo.create_or_get_ticket(
                conn=conn,
                tenant_id=tenant_b,
                customer_email="b2@test.com",
                subject="Tenant B - T2",
                body="Body",
            )

            # Explicit test fixture state setup: establish required priority, status, and timestamps.
            # Strictly scoped by both tenant_id AND id to maintain multi-tenant isolation.
            with conn.cursor() as cur:
                # T1: created 20m ago (15:40), priority = 'urgent'
                cur.execute(
                    """
                    UPDATE tickets
                    SET priority = 'urgent', created_at = %s, status = 'open'
                    WHERE tenant_id = %s AND id = %s;
                    """,
                    (now - timedelta(minutes=20), tenant_a, t1.id),
                )
                # T2: created 30m ago (15:30) -> due at 16:00 (exact deadline), priority = 'urgent'
                cur.execute(
                    """
                    UPDATE tickets
                    SET priority = 'urgent', created_at = %s, status = 'open'
                    WHERE tenant_id = %s AND id = %s;
                    """,
                    (now - timedelta(minutes=30), tenant_a, t2.id),
                )
                # T3: created 40m ago (15:20), response at 15:40 (+20m), resolved at 17:50 (late, closed), priority = 'urgent'
                cur.execute(
                    """
                    UPDATE tickets
                    SET priority = 'urgent',
                        created_at = %s,
                        first_response_at = %s,
                        resolved_at = %s,
                        status = 'closed'
                    WHERE tenant_id = %s AND id = %s;
                    """,
                    (
                        now - timedelta(minutes=40),
                        now - timedelta(minutes=20),
                        now + timedelta(minutes=110),
                        tenant_a,
                        t3.id,
                    ),
                )
                # T4: created 80m ago (14:40), response at 16:10 (+90m, late), status = 'open', priority = 'normal'
                cur.execute(
                    """
                    UPDATE tickets
                    SET priority = 'normal',
                        created_at = %s,
                        first_response_at = %s,
                        status = 'open'
                    WHERE tenant_id = %s AND id = %s;
                    """,
                    (
                        now - timedelta(minutes=80),
                        now + timedelta(minutes=10),
                        tenant_a,
                        t4.id,
                    ),
                )
                # T5: created 300m ago (11:00), status = 'pending', priority = NULL (falls back to default policy)
                cur.execute(
                    """
                    UPDATE tickets
                    SET priority = NULL,
                        created_at = %s,
                        status = 'pending'
                    WHERE tenant_id = %s AND id = %s;
                    """,
                    (now - timedelta(minutes=300), tenant_a, t5.id),
                )
                # T6: created 50m ago, first_response_at 60m ago (before created_at -> invalid timestamp DQ issue), priority = 'urgent'
                cur.execute(
                    """
                    UPDATE tickets
                    SET priority = 'urgent',
                        created_at = %s,
                        first_response_at = %s,
                        status = 'open'
                    WHERE tenant_id = %s AND id = %s;
                    """,
                    (
                        now - timedelta(minutes=50),
                        now - timedelta(minutes=60),
                        tenant_a,
                        t6.id,
                    ),
                )

                # Tenant B timestamps:
                # TB1: created 10m ago, status = 'new', priority = NULL (falls back to Tenant B default policy)
                cur.execute(
                    """
                    UPDATE tickets
                    SET priority = NULL,
                        created_at = %s,
                        status = 'new'
                    WHERE tenant_id = %s AND id = %s;
                    """,
                    (now - timedelta(minutes=10), tenant_b, tb1.id),
                )
                # TB2: created 50m ago, status = 'open', priority = 'low' (falls back to Tenant B default policy)
                cur.execute(
                    """
                    UPDATE tickets
                    SET priority = 'low',
                        created_at = %s,
                        status = 'open'
                    WHERE tenant_id = %s AND id = %s;
                    """,
                    (now - timedelta(minutes=50), tenant_b, tb2.id),
                )

        # Execute dashboard aggregation for Tenant A
        metrics_a = dashboard_repo.get_tenant_dashboard_metrics(
            conn=conn,
            tenant_id=tenant_a,
            evaluated_at=now,
        )

        # Execute dashboard aggregation for Tenant B
        metrics_b = dashboard_repo.get_tenant_dashboard_metrics(
            conn=conn,
            tenant_id=tenant_b,
            evaluated_at=now,
        )

        # 1. Multi-Tenant Isolation Verification
        assert metrics_a.status_counts.total == 6
        assert metrics_b.status_counts.total == 2
        assert metrics_a.sla_summary is not None
        assert metrics_b.sla_summary is not None
        assert metrics_a.sla_summary.policies_configured == 2
        assert metrics_b.sla_summary.policies_configured == 1

        # 2. Tenant A Workload
        # new: 0, open: 4 (t1, t2, t4, t6), pending: 1 (t5), resolved: 0, closed: 1 (t3)
        assert metrics_a.status_counts.open == 4
        assert metrics_a.status_counts.pending == 1
        assert metrics_a.status_counts.closed == 1
        assert metrics_a.open_workload == 5  # open + pending
        assert metrics_a.completed_workload == 1  # closed

        # 3. Tenant A First Response Target Verification
        fr_a = metrics_a.sla_summary.first_response
        assert fr_a.eligible_total == 6
        assert fr_a.data_quality_issues == 1  # T6 has invalid timestamp!
        assert fr_a.valid_eligible_total == 5
        # Completed: T3 on time, T4 late
        assert fr_a.completed_on_time == 1
        assert fr_a.completed_late == 1
        assert fr_a.completed_total == 2
        assert fr_a.completed_timeliness_rate == 0.5
        # Incomplete: T1 before deadline, T2 due now, T5 overdue
        assert fr_a.incomplete_before_deadline == 1
        assert fr_a.incomplete_due_now == 1
        assert fr_a.incomplete_overdue == 1
        assert fr_a.incomplete_total == 3
        # Incomplete not overdue rate: (1 + 1) / 3 = 0.6667
        assert fr_a.incomplete_not_overdue_rate == 0.6667
        # Full partition identity
        assert fr_a.valid_eligible_total == fr_a.completed_total + fr_a.incomplete_total
        assert fr_a.eligible_total == fr_a.valid_eligible_total + fr_a.data_quality_issues

        # Evaluator parity checks for valid tickets:
        # T1 (urgent 30m, created 20m ago): on track
        ev_t1 = evaluate_target("first_response", 30, now - timedelta(minutes=20), None, now)
        assert not ev_t1.is_breached and ev_t1.timeliness == "on_track"

        # T2 (urgent 30m, created 30m ago): due at exact deadline
        ev_t2 = evaluate_target("first_response", 30, now - timedelta(minutes=30), None, now)
        assert not ev_t2.is_breached and ev_t2.timeliness == "on_track"

        # T3 (urgent 30m, created 40m ago, response at +20m): fulfilled on time
        ev_t3 = evaluate_target("first_response", 30, now - timedelta(minutes=40), now - timedelta(minutes=20), now)
        assert not ev_t3.is_breached and ev_t3.timeliness == "on_time"

        # T4 (default 60m, created 80m ago, response at +90m): breached late
        ev_t4 = evaluate_target("first_response", 60, now - timedelta(minutes=80), now + timedelta(minutes=10), now + timedelta(minutes=15))
        assert ev_t4.is_breached and ev_t4.timeliness == "late"

        # T5 (default 60m, created 300m ago, no response): overdue
        ev_t5 = evaluate_target("first_response", 60, now - timedelta(minutes=300), None, now)
        assert ev_t5.is_breached and ev_t5.timeliness == "overdue"

        # 4. Tenant B Verification
        assert metrics_b.open_workload == 2  # new + open
        assert metrics_b.completed_workload == 0
        fr_b = metrics_b.sla_summary.first_response
        assert fr_b.eligible_total == 2
        assert fr_b.completed_total == 0
        assert fr_b.completed_timeliness_rate is None  # Zero denominator
        assert fr_b.incomplete_total == 2
        assert fr_b.incomplete_not_overdue_rate == 1.0  # Both before deadline
