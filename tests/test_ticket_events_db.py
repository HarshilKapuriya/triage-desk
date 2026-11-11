"""Database integration tests for ticket lifecycle events and SLA operational telemetry.

Target Resource:
- Disposable local PostgreSQL container on 127.0.0.1:5433 (dbname: triage_events_disposable).
- Never targets hosted Supabase databases or default DATABASE_URL.
- Strict multi-attribute target safety guard enforces loopback host and port 5433.

Verifies:
1. Migration 009 execution, PostgreSQL catalog, composite foreign key, and partial unique indexes.
2. Cross-tenant foreign-key rejection at PostgreSQL engine level.
3. One-time event uniqueness enforcement via partial unique indexes.
4. State mutation and event rollback atomicity upon transaction failure.
5. Sequential and concurrent duplicate lifecycle requests (application idempotency).
6. Creation deduplication via external_id and event atomicity.
7. Semantic parity between browser router and repository mutation paths.
8. Worker completion atomicity, transition to 'open', and retry idempotency.
9. Configured vs unconfigured SLA policy snapshots at intake.
10. Legacy tickets receiving genuine future events without fabricated history.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
import json
import os
from typing import Any, Generator
import uuid

import pytest
import psycopg
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app
import app.db as app_db
from app.deps import Principal, get_api_key_principal
from app.deps_browser import get_browser_principal_json
from app.repositories import (
    ticket_events as ticket_events_repo,
    tickets as ticket_repo,
    jobs as job_repo,
    sla as sla_repo,
)
from app.security import (
    hash_password,
    generate_api_key,
    create_access_token,
    derive_session_identifier,
    generate_csrf_token,
)
from tests.target_validator import (
    validate_isolated_target,
    TargetSafetyError,
    REQUIRED_TARGET,
    AUTHORIZED_DISPOSABLE_URL,
)


TARGET_DB_URL = os.getenv("TEST_DISPOSABLE_DATABASE_URL", AUTHORIZED_DISPOSABLE_URL)

# Validate configured target URL immediately on module load (fails closed if invalid)
validate_isolated_target(TARGET_DB_URL)

RUN_DISPOSABLE_DB_TESTS = os.getenv("RUN_DISPOSABLE_DB_TESTS", "").lower() == "true"
skip_no_disposable_db = pytest.mark.skipif(
    not RUN_DISPOSABLE_DB_TESTS,
    reason="Disposable database tests require RUN_DISPOSABLE_DB_TESTS=true and active container on 127.0.0.1:5433",
)
pytestmark = skip_no_disposable_db


@asynccontextmanager
async def _disposable_test_lifespan(test_app: FastAPI):
    """Bypasses app.db.lifespan to prevent reading ambient .env settings or re-opening pools."""
    yield


@pytest.fixture(scope="session")
def db_pool() -> Generator[ConnectionPool | None, None, None]:
    """Session-scoped connection pool strictly connected to the verified disposable database."""
    if not RUN_DISPOSABLE_DB_TESTS:
        yield None
        return

    validate_isolated_target(TARGET_DB_URL)
    pool = ConnectionPool(
        conninfo=TARGET_DB_URL,
        min_size=1,
        max_size=10,
        open=False,
    )
    # Bounded wait timeout prevents indefinite hangs if disposable container is not reachable
    pool.open(wait=True, timeout=10)
    app_db.set_pool(pool)

    # Override application lifespan to prevent TestClient from reading ambient .env settings
    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _disposable_test_lifespan

    try:
        yield pool
    finally:
        app.dependency_overrides.clear()
        app.router.lifespan_context = original_lifespan
        pool.close()
        app_db.set_pool(None)


@pytest.fixture
def conn(db_pool: ConnectionPool | None) -> Generator[psycopg.Connection[Any], None, None]:
    """Dedicated connection per test function."""
    if db_pool is None:
        raise RuntimeError("Disposable test database pool is not initialized.")
    with db_pool.connection() as connection:
        try:
            yield connection
        finally:
            app.dependency_overrides.clear()


@pytest.fixture
def synthetic_tenant(conn: psycopg.Connection[Any]) -> Generator[tuple[uuid.UUID, uuid.UUID], None, None]:
    """Fixture providing a fresh synthetic tenant and admin user, cleanly cascading on teardown."""
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tenants (id, name, plan) VALUES (%s, %s, 'free');",
                (tenant_id, f"Tenant-{tenant_id.hex[:8]}"),
            )
            cur.execute(
                """
                INSERT INTO users (id, tenant_id, email, password_hash, role)
                VALUES (%s, %s, %s, %s, 'admin');
                """,
                (user_id, tenant_id, f"admin-{user_id.hex[:8]}@example.com", hash_password("test-pass-123")),
            )
    try:
        yield tenant_id, user_id
    finally:
        try:
            if not conn.closed:
                if conn.info.transaction_status in (
                    psycopg.pq.TransactionStatus.INERROR,
                    psycopg.pq.TransactionStatus.INTRANS,
                ):
                    conn.rollback()
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM tenants WHERE id = %s;", (tenant_id,))
        except Exception:
            pass


# 1. Migration Execution and Catalog Inspection

def test_01_migration_catalog_inspection(conn: psycopg.Connection[Any]) -> None:
    """Verify migration 009 created ticket_events table, composite FK, and 5 partial unique indexes."""
    with conn.cursor(row_factory=dict_row) as cur:
        # 1. Verify composite foreign key on (tenant_id, ticket_id) references tickets(tenant_id, id) ON DELETE CASCADE
        cur.execute(
            """
            SELECT conname, confdeltype
            FROM pg_constraint
            WHERE conname = 'fk_ticket_events_tickets';
            """
        )
        fk_row = cur.fetchone()
        assert fk_row is not None, "Foreign key 'fk_ticket_events_tickets' must exist."
        assert fk_row["confdeltype"] == "c", "Foreign key must specify CASCADE on delete."

        # 2. Verify SPEC multi-tenant unique constraint
        cur.execute(
            """
            SELECT conname
            FROM pg_constraint
            WHERE conname = 'uq_ticket_events_tenant_id_id';
            """
        )
        assert cur.fetchone() is not None, "Unique constraint 'uq_ticket_events_tenant_id_id' must exist."

        # 3. Verify all 5 partial unique indexes exist in PostgreSQL catalog
        cur.execute(
            """
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'ticket_events';
            """
        )
        indexes = {row["indexname"]: row["indexdef"] for row in cur.fetchall()}

        expected_partial_indexes = [
            "uq_ticket_events_ticket_created",
            "uq_ticket_events_ticket_processed",
            "uq_ticket_events_first_response",
            "uq_ticket_events_resolved",
            "uq_ticket_events_closed",
        ]
        for idx_name in expected_partial_indexes:
            assert idx_name in indexes, f"Partial index '{idx_name}' must exist."
            assert "WHERE" in indexes[idx_name], f"Index '{idx_name}' must have a WHERE clause."

        # 4. Verify chronological lookup index
        assert "idx_ticket_events_lookup" in indexes, "Lookup index 'idx_ticket_events_lookup' must exist."


# 2. Cross-Tenant Foreign-Key Rejection

def test_02_cross_tenant_foreign_key_rejection(
    conn: psycopg.Connection[Any],
    synthetic_tenant: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """PostgreSQL rejects inserting a ticket_event where tenant_id does not match the ticket's tenant_id."""
    tenant_a, _ = synthetic_tenant
    tenant_b = uuid.uuid4()

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("INSERT INTO tenants (id, name, plan) VALUES (%s, 'Tenant B', 'free');", (tenant_b,))

    # Create Ticket under Tenant A
    ticket_a, is_created = ticket_repo.create_or_get_ticket(
        conn=conn,
        tenant_id=tenant_a,
        subject="Tenant A Issue",
        body="Body content",
    )
    assert is_created is True

    # Attempt to insert event for Ticket A using Tenant B's ID -> must fail FK constraint
    now = datetime.now(timezone.utc)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with conn.transaction():
            ticket_events_repo.append_ticket_event(
                conn_or_cur=conn,
                tenant_id=tenant_b,
                ticket_id=ticket_a.id,
                event_type="first_response_recorded",
                occurred_at=now,
                actor_type="agent",
            )

    # Clean up Tenant B
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tenants WHERE id = %s;", (tenant_b,))


# 3. One-Time Event Uniqueness

def test_03_one_time_event_uniqueness(
    conn: psycopg.Connection[Any],
    synthetic_tenant: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """PostgreSQL partial unique index rejects multiple identical transition events per ticket."""
    tenant_id, _ = synthetic_tenant

    ticket, _ = ticket_repo.create_or_get_ticket(
        conn=conn,
        tenant_id=tenant_id,
        subject="Subject",
        body="Body",
    )

    # ticket_created event already exists from creation
    # Attempting a second ticket_created event directly must violate uq_ticket_events_ticket_created
    now = datetime.now(timezone.utc)
    with pytest.raises(psycopg.errors.UniqueViolation) as excinfo:
        with conn.transaction():
            ticket_events_repo.append_ticket_event(
                conn_or_cur=conn,
                tenant_id=tenant_id,
                ticket_id=ticket.id,
                event_type="ticket_created",
                occurred_at=now,
                actor_type="system",
            )
    assert "uq_ticket_events_ticket_created" in str(excinfo.value)


# 4. State Mutation and Event Rollback Atomicity

def test_04_mutation_event_rollback_atomicity(
    conn: psycopg.Connection[Any],
    synthetic_tenant: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """If an error occurs before transaction commit, both state update and ticket event roll back cleanly."""
    tenant_id, _ = synthetic_tenant
    ticket, _ = ticket_repo.create_or_get_ticket(
        conn=conn,
        tenant_id=tenant_id,
        subject="Rollback Subject",
        body="Rollback Body",
    )

    now = datetime.now(timezone.utc)
    try:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE tickets SET status = 'resolved', resolved_at = %s WHERE tenant_id = %s AND id = %s;",
                    (now, tenant_id, ticket.id),
                )
                ticket_events_repo.append_ticket_event(
                    conn_or_cur=cur,
                    tenant_id=tenant_id,
                    ticket_id=ticket.id,
                    event_type="ticket_resolved",
                    occurred_at=now,
                    actor_type="agent",
                )
                # Deliberate failure inside transaction
                raise RuntimeError("Simulated transaction abort")
    except RuntimeError:
        pass

    # Verify ticket state is untouched and zero resolved events exist
    reloaded = ticket_repo.get_ticket_by_id(conn, tenant_id, ticket.id)
    assert reloaded is not None
    assert reloaded.status == "new"
    assert reloaded.resolved_at is None

    events = ticket_events_repo.list_ticket_events(conn, tenant_id, ticket.id)
    event_types = [e.event_type for e in events]
    assert "ticket_resolved" not in event_types


# 5. Sequential and Concurrent Duplicate Lifecycle Requests (Application Idempotency)

def test_05_sequential_and_concurrent_duplicate_lifecycle_requests(
    conn: psycopg.Connection[Any],
    synthetic_tenant: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """Repeated and concurrent first-response calls return 200 and produce exactly 1 event."""
    tenant_id, user_id = synthetic_tenant

    ticket, _ = ticket_repo.create_or_get_ticket(
        conn=conn,
        tenant_id=tenant_id,
        subject="Concurrent First Response",
        body="Body",
    )

    # Transition to open
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("UPDATE tickets SET status = 'open' WHERE id = %s;", (ticket.id,))
    conn.commit()

    session_token = create_access_token(
        user_id=user_id,
        tenant_id=tenant_id,
        role="agent",
        expires_delta=timedelta(minutes=60),
    )
    session_id = derive_session_identifier(session_token)
    csrf_token = generate_csrf_token(session_id)
    cookies = {"triage_session": session_token}
    headers = {"X-CSRF-Token": csrf_token}

    client = TestClient(app, cookies=cookies)

    # Sequential calls
    res1 = client.post(f"/browser-api/tickets/{ticket.id}/first-response", headers=headers)
    assert res1.status_code == 200
    assert res1.json()["ok"] is True

    res2 = client.post(f"/browser-api/tickets/{ticket.id}/first-response", headers=headers)
    assert res2.status_code == 200
    assert res2.json()["ok"] is True

    t1_after = ticket_repo.get_ticket_by_id(conn, tenant_id, ticket.id)
    assert t1_after is not None
    assert t1_after.first_response_at is not None

    events = ticket_events_repo.list_ticket_events(conn, tenant_id, ticket.id)
    fr_events = [e for e in events if e.event_type == "first_response_recorded"]
    assert len(fr_events) == 1, "Sequential duplicate calls must result in exactly 1 first-response event."

    # Concurrent calls on a fresh ticket
    t2, _ = ticket_repo.create_or_get_ticket(conn, tenant_id, "Concurrent 2", "Body 2")
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("UPDATE tickets SET status = 'open' WHERE id = %s;", (t2.id,))
    conn.commit()

    def post_fr():
        c = TestClient(app, cookies=cookies)
        return c.post(f"/browser-api/tickets/{t2.id}/first-response", headers=headers)

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut1 = executor.submit(post_fr)
        fut2 = executor.submit(post_fr)
        r1, r2 = fut1.result(timeout=10), fut2.result(timeout=10)

    assert r1.status_code == 200
    assert r2.status_code == 200

    t2_after = ticket_repo.get_ticket_by_id(conn, tenant_id, t2.id)
    assert t2_after is not None
    assert t2_after.first_response_at is not None

    events2 = ticket_events_repo.list_ticket_events(conn, tenant_id, t2.id)
    fr_events2 = [e for e in events2 if e.event_type == "first_response_recorded"]
    assert len(fr_events2) == 1, "Concurrent duplicate calls must result in exactly 1 first-response event."


# 6. Creation Deduplication via external_id and Event Atomicity

def test_06_creation_deduplication_via_external_id(
    conn: psycopg.Connection[Any],
    synthetic_tenant: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """Calling create_ticket with identical external_id yields 1 ticket and 1 creation event."""
    tenant_id, _ = synthetic_tenant
    ext_id = f"ext-dedup-{uuid.uuid4().hex[:6]}"

    t1, is_created1 = ticket_repo.create_or_get_ticket(
        conn=conn,
        tenant_id=tenant_id,
        subject="Ext Title",
        body="Ext Body",
        external_id=ext_id,
    )
    assert is_created1 is True

    t2, is_created2 = ticket_repo.create_or_get_ticket(
        conn=conn,
        tenant_id=tenant_id,
        subject="Ext Title",
        body="Ext Body",
        external_id=ext_id,
    )
    assert is_created2 is False
    assert t1.id == t2.id

    events = ticket_events_repo.list_ticket_events(conn, tenant_id, t1.id)
    creation_events = [e for e in events if e.event_type == "ticket_created"]
    assert len(creation_events) == 1


# 7. Browser and Repository Lifecycle Parity

def test_07_browser_and_repository_lifecycle_parity(
    conn: psycopg.Connection[Any],
    synthetic_tenant: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """Verify semantic equivalence of event fields across browser router and repository paths."""
    tenant_id, user_id = synthetic_tenant

    t_repo, _ = ticket_repo.create_or_get_ticket(conn, tenant_id, "Repo Path", "Body")
    t_browser, _ = ticket_repo.create_or_get_ticket(conn, tenant_id, "Browser Path", "Body")

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("UPDATE tickets SET status = 'open' WHERE id IN (%s, %s);", (t_repo.id, t_browser.id))
    conn.commit()

    # Path A: Repository resolve
    now = datetime.now(timezone.utc)
    with conn.transaction():
        rec_repo = ticket_repo.resolve_ticket(conn, tenant_id, t_repo.id, resolved_at=now, actor_id=str(user_id))
    conn.commit()
    assert rec_repo is not None
    assert rec_repo.status == "resolved"

    # Path B: Browser endpoint resolve
    session_token = create_access_token(
        user_id=user_id,
        tenant_id=tenant_id,
        role="agent",
        expires_delta=timedelta(minutes=60),
    )
    session_id = derive_session_identifier(session_token)
    csrf_token = generate_csrf_token(session_id)
    cookies = {"triage_session": session_token}
    headers = {"X-CSRF-Token": csrf_token}

    client = TestClient(app, cookies=cookies)
    res = client.post(f"/browser-api/tickets/{t_browser.id}/resolve", headers=headers)
    assert res.status_code == 200

    # Compare event payloads
    events_repo = ticket_events_repo.list_ticket_events(conn, tenant_id, t_repo.id)
    events_browser = ticket_events_repo.list_ticket_events(conn, tenant_id, t_browser.id)

    ev_repo = next(e for e in events_repo if e.event_type == "ticket_resolved")
    ev_browser = next(e for e in events_browser if e.event_type == "ticket_resolved")

    assert ev_repo.schema_version == ev_browser.schema_version == 1
    assert ev_repo.payload["from_status"] == ev_browser.payload["from_status"] == "open"
    assert ev_repo.payload["to_status"] == ev_browser.payload["to_status"] == "resolved"
    assert "resolved_at" in ev_repo.payload and "resolved_at" in ev_browser.payload


# 8. Worker Completion Atomicity and Retry Behavior

def test_08_worker_completion_atomicity_and_retry(
    conn: psycopg.Connection[Any],
    synthetic_tenant: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """Worker transitions new to open with ticket_processed; retry skips and does not duplicate."""
    tenant_id, _ = synthetic_tenant
    ticket, _ = ticket_repo.create_or_get_ticket(conn, tenant_id, "Worker Subject", "Body")

    # Enqueue job
    with conn.transaction():
        enqueued_job = job_repo.enqueue_job(conn, tenant_id, ticket.id, "ticket_ingest")

    # Claim job
    claim = job_repo.claim_next_job(conn, worker_id="worker-test", lease_seconds=60, tenant_id=tenant_id)
    assert claim is not None
    assert claim.id == enqueued_job.id

    # 1. First completion -> transitions ticket to open and emits ticket_processed
    with conn.transaction():
        ok = job_repo.complete_job_with_ticket_transition(
            conn=conn,
            tenant_id=tenant_id,
            job_id=enqueued_job.id,
            ticket_id=ticket.id,
            worker_id="worker-test",
            generation=claim.attempts,
            predicted_label="Credit reporting",
        )
        assert ok is True

    t_after = ticket_repo.get_ticket_by_id(conn, tenant_id, ticket.id)
    assert t_after is not None
    assert t_after.status == "open"

    events = ticket_events_repo.list_ticket_events(conn, tenant_id, ticket.id)
    proc_events = [e for e in events if e.event_type == "ticket_processed"]
    assert len(proc_events) == 1
    assert proc_events[0].payload["predicted_category"] == "Credit reporting"
    assert proc_events[0].payload["from_status"] == "new"
    assert proc_events[0].payload["to_status"] == "open"

    # 2. Simulated retry of already progressed ticket -> skips event
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status = 'running', locked_by = 'worker-test', locked_at = NOW() WHERE id = %s;",
                (enqueued_job.id,),
            )
        ok_retry = job_repo.complete_job_with_ticket_transition(
            conn=conn,
            tenant_id=tenant_id,
            job_id=enqueued_job.id,
            ticket_id=ticket.id,
            worker_id="worker-test",
            generation=claim.attempts,
        )
        assert ok_retry is True

    events_after = ticket_events_repo.list_ticket_events(conn, tenant_id, ticket.id)
    proc_events_after = [e for e in events_after if e.event_type == "ticket_processed"]
    assert len(proc_events_after) == 1, "Retry must not duplicate ticket_processed event."


# 9. Configured vs Missing Policy Snapshots

def test_09_configured_vs_missing_policy_snapshots(
    conn: psycopg.Connection[Any],
    synthetic_tenant: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """Ticket created with active policy records configured snapshot; without policy records unconfigured."""
    tenant_id, _ = synthetic_tenant

    # Case A: Tenant with configured default SLA policy
    sla_repo.upsert_sla_policy(
        conn=conn,
        tenant_id=tenant_id,
        name="Default Policy",
        priority="default",
        first_response_minutes=45,
        resolution_minutes=180,
    )
    t_configured, _ = ticket_repo.create_or_get_ticket(conn, tenant_id, "With Policy", "Body")
    events_conf = ticket_events_repo.list_ticket_events(conn, tenant_id, t_configured.id)
    ev_conf = next(e for e in events_conf if e.event_type == "ticket_created")
    snap_conf = ev_conf.payload["policy_snapshot"]

    assert snap_conf["is_configured"] is True
    assert snap_conf["first_response_minutes"] == 45
    assert snap_conf["resolution_minutes"] == 180

    # Case B: Delete policy, next ticket created records unconfigured
    sla_repo.delete_sla_policy(conn, tenant_id, "default")
    t_unconf, _ = ticket_repo.create_or_get_ticket(conn, tenant_id, "Without Policy", "Body")
    events_unconf = ticket_events_repo.list_ticket_events(conn, tenant_id, t_unconf.id)
    ev_unconf = next(e for e in events_unconf if e.event_type == "ticket_created")
    snap_unconf = ev_unconf.payload["policy_snapshot"]

    assert snap_unconf["is_configured"] is False
    assert snap_unconf["policy_id"] is None
    assert snap_unconf["first_response_minutes"] is None


# 10. Legacy Tickets Without Fabricated History

def test_10_legacy_tickets_without_fabricated_history(
    conn: psycopg.Connection[Any],
    synthetic_tenant: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """Legacy ticket inserted without events receives genuine resolve event without fabricated creation."""
    tenant_id, _ = synthetic_tenant
    legacy_id = uuid.uuid4()
    now = datetime.now(timezone.utc) - timedelta(days=5)

    # Seed ticket directly without ticket_events
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tickets (id, tenant_id, subject, body, status, created_at)
                VALUES (%s, %s, 'Legacy Ticket', 'Legacy Body', 'open', %s);
                """,
                (legacy_id, tenant_id, now),
            )

    events_initial = ticket_events_repo.list_ticket_events(conn, tenant_id, legacy_id)
    assert events_initial == [], "Legacy ticket must have empty initial event history."

    # Genuine lifecycle resolution
    resolve_time = datetime.now(timezone.utc)
    ticket_repo.resolve_ticket(conn, tenant_id, legacy_id, resolved_at=resolve_time)

    events_after = ticket_events_repo.list_ticket_events(conn, tenant_id, legacy_id)
    assert len(events_after) == 1
    assert events_after[0].event_type == "ticket_resolved"
    assert events_after[0].occurred_at == resolve_time

