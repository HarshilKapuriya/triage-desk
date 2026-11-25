"""Guarded database integration tests for SLA policies and ticket lifecycle primitives.

Strict safety constraints:
- Runs ONLY against an already established disposable local PostgreSQL
  instance (localhost or 127.0.0.1 with 'test' in dbname).
- Hosted databases (e.g. Supabase poolers) are strictly prohibited.
- If no established disposable local test database exists, tests are cleanly skipped.
"""

from datetime import datetime, timezone
import uuid
import pytest
import psycopg

from app.repositories.auth import create_tenant_and_admin_user
import app.repositories.sla as sla_repo
import app.repositories.tickets as ticket_repo
from app.repositories.tickets import InvalidStatusTransitionError
from app.security import hash_password
from tests.test_embedding_jobs_db import resolve_disposable_local_test_db

LOCAL_TEST_DB_URL, LOCAL_SAFETY_STATUS = resolve_disposable_local_test_db()
pytestmark = pytest.mark.skipif(
    LOCAL_TEST_DB_URL is None,
    reason=f"Guarded SLA DB integration tests require an established disposable local PostgreSQL instance: {LOCAL_SAFETY_STATUS}",
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
            normalized_email=f"sla_user_{suffix}@test.com",
            password_hash=hash_password("Pass123!"),
        )
    return tenant_id


def test_sla_policies_table_and_tenant_isolation():
    """Verify SLA policy persistence and strict tenant isolation."""
    assert LOCAL_TEST_DB_URL is not None

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'sla_policies'
                );
                """
            )
            table_exists = cur.fetchone()[0]

    if not table_exists:
        pytest.skip("sla_policies table does not exist in local test DB (migration 008 not applied)")

    tenant_a = _create_test_tenant("SLA Tenant A")
    tenant_b = _create_test_tenant("SLA Tenant B")

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        # 1. Tenant A configures an urgent policy
        pol_a = sla_repo.upsert_sla_policy(
            conn=conn,
            tenant_id=tenant_a,
            name="Urgent Tier A",
            priority="urgent",
            first_response_minutes=30,
            resolution_minutes=120,
        )
        conn.commit()

        assert pol_a.priority == "urgent"
        assert pol_a.first_response_minutes == 30

        # 2. Tenant B must NOT see Tenant A's policy
        pol_b_lookup = sla_repo.get_sla_policy_by_priority(conn, tenant_b, "urgent")
        assert pol_b_lookup is None

        eff_b = sla_repo.resolve_effective_policy(conn, tenant_b, "urgent")
        assert eff_b is None

        # 3. Tenant A can resolve its policy
        eff_a = sla_repo.resolve_effective_policy(conn, tenant_a, "urgent")
        assert eff_a is not None
        assert eff_a.id == pol_a.id

        # 4. Upsert idempotency: updating urgent policy alters existing record
        pol_a_updated = sla_repo.upsert_sla_policy(
            conn=conn,
            tenant_id=tenant_a,
            name="Urgent Tier A Revised",
            priority="urgent",
            first_response_minutes=45,
            resolution_minutes=180,
        )
        conn.commit()
        assert pol_a_updated.id == pol_a.id
        assert pol_a_updated.first_response_minutes == 45


def test_ticket_lifecycle_db_primitives():
    """Verify first response recording, resolution, and closure against local database."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Lifecycle Tenant")

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        # Create a test ticket using supported arguments
        ticket, _ = ticket_repo.create_or_get_ticket(
            conn=conn,
            tenant_id=tenant_id,
            subject="Billing issue",
            body="Incorrect fee charged",
            customer_email="customer@example.com",
        )
        t_created = datetime(2026, 11, 8, 10, 0, 0, tzinfo=UTC)
        # Test fixture state setup: set category, priority, and deterministic created_at
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE tickets
                SET category = %s, priority = %s, created_at = %s
                WHERE tenant_id = %s AND id = %s;
                """,
                ("billing", "urgent", t_created, tenant_id, ticket.id),
            )
        conn.commit()

        # Re-fetch ticket to verify updated fixture state
        ticket = ticket_repo.get_ticket_by_id(conn, tenant_id, ticket.id)
        assert ticket is not None
        assert ticket.status == "new"
        assert ticket.first_response_at is None
        assert ticket.resolved_at is None
        assert ticket.category == "billing"
        assert ticket.priority == "urgent"
        assert ticket.created_at == t_created

        # 1. Record first response
        t_resp1 = datetime(2026, 11, 8, 10, 30, tzinfo=UTC)
        updated1 = ticket_repo.record_first_response(conn, tenant_id, ticket.id, t_resp1)
        conn.commit()
        assert updated1 is not None
        assert updated1.first_response_at == t_resp1
        assert updated1.status == "new"  # status unchanged

        # Idempotent repeated first response: preserves original timestamp
        t_resp2 = datetime(2026, 11, 8, 11, 0, tzinfo=UTC)
        updated2 = ticket_repo.record_first_response(conn, tenant_id, ticket.id, t_resp2)
        conn.commit()
        assert updated2.first_response_at == t_resp1

        # 2. Cannot close before resolution
        with pytest.raises(InvalidStatusTransitionError, match="must be resolved first"):
            ticket_repo.close_ticket(conn, tenant_id, ticket.id)

        # 3. Resolve ticket
        t_res1 = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
        resolved = ticket_repo.resolve_ticket(conn, tenant_id, ticket.id, t_res1)
        conn.commit()
        assert resolved is not None
        assert resolved.status == "resolved"
        assert resolved.resolved_at == t_res1

        # Repeated resolution preserves original resolved_at
        t_res2 = datetime(2026, 11, 8, 12, 30, tzinfo=UTC)
        resolved2 = ticket_repo.resolve_ticket(conn, tenant_id, ticket.id, t_res2)
        conn.commit()
        assert resolved2.resolved_at == t_res1

        # 4. Close resolved ticket
        closed = ticket_repo.close_ticket(conn, tenant_id, ticket.id)
        conn.commit()
        assert closed is not None
        assert closed.status == "closed"

        # Cannot resolve a closed ticket (no reopening)
        with pytest.raises(InvalidStatusTransitionError, match="already closed"):
            ticket_repo.resolve_ticket(conn, tenant_id, ticket.id, t_res2)
