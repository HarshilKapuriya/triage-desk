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


# 11. Browser Lifecycle Timeline Endpoint: Tenant Isolation, Privacy, and Policy Summary


def test_11_browser_events_endpoint_lifecycle_and_privacy(
    conn: psycopg.Connection[Any],
    synthetic_tenant: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """Real DB integration check for GET /browser-api/tickets/{id}/events auth, isolation, privacy, and intake policy."""
    tenant_a, user_a = synthetic_tenant
    tenant_b = uuid.uuid4()
    user_b = uuid.uuid4()

    # Configure default SLA policy on Tenant A and create ticket_conf
    sla_repo.upsert_sla_policy(
        conn=conn,
        tenant_id=tenant_a,
        name="Default SLA",
        priority="default",
        first_response_minutes=30,
        resolution_minutes=240,
    )
    conn.commit()

    secret_subject = "CONFIDENTIAL_SUBJECT_XYZ"
    secret_body = "CONFIDENTIAL_BODY_CONTAINING_PII_SSN_123"
    secret_email = "private.customer.pii@example.com"
    secret_ext_id = "EXT-PRIVATE-98765"
    secret_worker_id = "worker-private-pod-77"
    secret_model_ver_id = uuid.uuid4()

    ticket_conf, _ = ticket_repo.create_or_get_ticket(
        conn=conn,
        tenant_id=tenant_a,
        subject=secret_subject,
        body=secret_body,
        customer_email=secret_email,
        external_id=secret_ext_id,
    )

    # Remove default policy and create ticket_unconf
    sla_repo.delete_sla_policy(conn, tenant_a, "default")
    ticket_unconf, _ = ticket_repo.create_or_get_ticket(
        conn=conn,
        tenant_id=tenant_a,
        subject="Unconfigured Intake",
        body="No policy active",
    )

    # Record worker processed + full agent lifecycle on ticket_conf without starting worker/ML
    t_proc = ticket_conf.created_at + timedelta(seconds=5)
    t_fr = ticket_conf.created_at + timedelta(minutes=10)
    t_res = ticket_conf.created_at + timedelta(minutes=25)
    t_cls = ticket_conf.created_at + timedelta(minutes=40)

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("UPDATE tickets SET status = 'open' WHERE tenant_id = %s AND id = %s;", (tenant_a, ticket_conf.id))
            ticket_events_repo.append_ticket_event(
                conn_or_cur=cur,
                tenant_id=tenant_a,
                ticket_id=ticket_conf.id,
                event_type="ticket_processed",
                occurred_at=t_proc,
                actor_type="worker",
                actor_id=secret_worker_id,
                payload=ticket_events_repo.build_ticket_processed_payload(
                    from_status="new",
                    to_status="open",
                    predicted_category="Credit card",
                    model_version_id=str(secret_model_ver_id),
                ),
            )

        ticket_repo.record_first_response(
            conn=conn,
            tenant_id=tenant_a,
            ticket_id=ticket_conf.id,
            responded_at=t_fr,
            actor_id=str(user_a),
        )
        ticket_repo.resolve_ticket(
            conn=conn,
            tenant_id=tenant_a,
            ticket_id=ticket_conf.id,
            resolved_at=t_res,
            actor_id=str(user_a),
        )
        ticket_repo.close_ticket(
            conn=conn,
            tenant_id=tenant_a,
            ticket_id=ticket_conf.id,
            closed_at=t_cls,
            actor_id=str(user_a),
        )

    # Create Tenant B for cross-tenant isolation check
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("INSERT INTO tenants (id, name, plan) VALUES (%s, %s, 'free');", (tenant_b, f"TenantB-{tenant_b.hex[:8]}"))
            cur.execute(
                "INSERT INTO users (id, tenant_id, email, password_hash, role) VALUES (%s, %s, %s, %s, 'agent');",
                (user_b, tenant_b, f"agentb-{user_b.hex[:8]}@example.com", hash_password("test-pass-123")),
            )
    conn.commit()

    try:
        # 1. Unauthenticated access rejected with 401 + no-store
        unauth_client = TestClient(app)
        res_unauth = unauth_client.get(f"/browser-api/tickets/{ticket_conf.id}/events")
        assert res_unauth.status_code == 401
        assert res_unauth.json()["error"]["code"] == "unauthenticated"
        assert res_unauth.headers.get("cache-control") == "no-store"

        # 2. Owning tenant (Tenant A) receives all 5 allowlisted events without PII/internal IDs
        token_a = create_access_token(user_id=user_a, tenant_id=tenant_a, role="admin", expires_delta=timedelta(minutes=60))
        client_a = TestClient(app, cookies={"triage_session": token_a})

        res_a = client_a.get(f"/browser-api/tickets/{ticket_conf.id}/events")
        assert res_a.status_code == 200
        assert res_a.headers.get("cache-control") == "no-store"

        raw_a = res_a.text
        for forbidden in (
            str(user_a),
            secret_worker_id,
            str(secret_model_ver_id),
            secret_subject,
            secret_body,
            secret_email,
            secret_ext_id,
            '"actor_id"',
            '"policy_id"',
            '"model_version_id"',
            '"tenant_id"',
        ):
            assert forbidden not in raw_a

        data_a = res_a.json()
        assert data_a["schema_available"] is True
        assert data_a["ticket_id"] == str(ticket_conf.id)
        assert [e["event_type"] for e in data_a["events"]] == [
            "ticket_created",
            "ticket_processed",
            "first_response_recorded",
            "ticket_resolved",
            "ticket_closed",
        ]
        assert data_a["events"][0]["details"]["intake_policy"] == {
            "is_configured": True,
            "policy_priority": "default",
            "first_response_minutes": 30,
            "resolution_minutes": 240,
        }
        assert all(e["schema_supported"] and e["payload_valid"] for e in data_a["events"])

        # Unconfigured policy ticket preserves honest unconfigured snapshot
        res_unconf = client_a.get(f"/browser-api/tickets/{ticket_unconf.id}/events")
        assert res_unconf.status_code == 200
        assert res_unconf.json()["events"][0]["details"]["intake_policy"] == {
            "is_configured": False,
            "policy_priority": None,
            "first_response_minutes": None,
            "resolution_minutes": None,
        }

        # 3. Tenant B receives exact same 404 for Tenant A's ticket as for a non-existent ticket
        token_b = create_access_token(user_id=user_b, tenant_id=tenant_b, role="agent", expires_delta=timedelta(minutes=60))
        client_b = TestClient(app, cookies={"triage_session": token_b})

        res_foreign = client_b.get(f"/browser-api/tickets/{ticket_conf.id}/events")
        res_missing = client_b.get(f"/browser-api/tickets/{uuid.uuid4()}/events")
        assert res_foreign.status_code == 404
        assert res_missing.status_code == 404
        assert res_foreign.json() == res_missing.json() == {
            "error": {"code": "not_found", "message": "Ticket not found."}
        }
    finally:
        if conn.info.transaction_status in (
            psycopg.pq.TransactionStatus.INERROR,
            psycopg.pq.TransactionStatus.INTRANS,
        ):
            conn.rollback()
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("DELETE FROM tenants WHERE id = %s;", (tenant_b,))
        conn.commit()


# 12. Browser Lifecycle Timeline Endpoint: Legacy Empty History & Backdated Ordering


def test_12_browser_events_legacy_empty_and_backdated_ordering(
    conn: psycopg.Connection[Any],
    synthetic_tenant: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """Legacy tickets return empty events; synthetic event-read fixture verifies occurred_at ASC, recorded_at ASC, id ASC.

    Note: Case B directly seeds a historical synthetic ticket and explicit event ordering keys
    (occurred_at, recorded_at, id) to deterministically exercise event-read sorting and tie-breaking;
    it is an event-read ordering fixture, not verification of the complete lifecycle-write path.
    """
    tenant_id, user_id = synthetic_tenant
    historical_t0 = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(days=30)
    recorded_base = historical_t0 + timedelta(days=15)

    # Case A: Legacy ticket with zero event rows
    legacy_id = uuid.uuid4()
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tickets (id, tenant_id, subject, body, status, created_at)
                VALUES (%s, %s, 'Legacy No-Events Ticket', 'Body', 'open', %s);
                """,
                (legacy_id, tenant_id, historical_t0 - timedelta(days=3)),
            )

    # Case B: Synthetic event-read ordering fixture with historical business baseline & explicit tie-break keys.
    # Deterministic UUID suffixes share a per-test unique 96-bit prefix so ordering does not rely on uuid4 luck.
    ord_ticket_id = uuid.uuid4()
    uid_prefix = uuid.uuid4().hex[:24]
    uid_resolved = uuid.UUID(f"{uid_prefix}000000000001")
    uid_closed = uuid.UUID(f"{uid_prefix}000000000002")
    uid_first_resp = uuid.UUID(f"{uid_prefix}000000000003")
    uid_processed = uuid.UUID(f"{uid_prefix}000000000004")
    uid_created = uuid.UUID(f"{uid_prefix}000000000005")

    t_fr = historical_t0 + timedelta(minutes=10)
    t_res = historical_t0 + timedelta(minutes=10)
    t_cls = historical_t0 + timedelta(minutes=10)
    t_proc_backdated = historical_t0 + timedelta(minutes=2)

    fixture_rows_in_scrambled_insert_order = [
        # 1. ticket_closed (+10m, rec +20m, uid ...0002): inserted before ticket_resolved (...0001) to prove id ASC beats insert order
        (
            uid_closed,
            "ticket_closed",
            t_cls,
            recorded_base + timedelta(minutes=20),
            "agent",
            str(user_id),
            json.dumps(ticket_events_repo.build_ticket_closed_payload(closed_at=t_cls)),
        ),
        # 2. first_response_recorded (+10m, rec +10m, uid ...0003): earlier recorded_at beats lower UUIDs (...0001, ...0002)
        (
            uid_first_resp,
            "first_response_recorded",
            t_fr,
            recorded_base + timedelta(minutes=10),
            "agent",
            str(user_id),
            json.dumps(
                ticket_events_repo.build_first_response_payload(
                    responded_at=t_fr,
                    elapsed_calendar_seconds=(t_fr - historical_t0).total_seconds(),
                )
            ),
        ),
        # 3. ticket_resolved (+10m, rec +20m, uid ...0001): ties occurred_at and recorded_at with ticket_closed, wins on id ASC
        (
            uid_resolved,
            "ticket_resolved",
            t_res,
            recorded_base + timedelta(minutes=20),
            "agent",
            str(user_id),
            json.dumps(
                ticket_events_repo.build_ticket_resolved_payload(
                    resolved_at=t_res,
                    from_status="open",
                    to_status="resolved",
                    elapsed_calendar_seconds=(t_res - historical_t0).total_seconds(),
                )
            ),
        ),
        # 4. ticket_processed (+2m, rec +40m, uid ...0004): genuine historical backdated occurred_at, recorded LAST
        (
            uid_processed,
            "ticket_processed",
            t_proc_backdated,
            recorded_base + timedelta(minutes=40),
            "worker",
            None,
            json.dumps(ticket_events_repo.build_ticket_processed_payload(from_status="new", to_status="open")),
        ),
        # 5. ticket_created (+0m, rec +1m, uid ...0005): earliest occurred_at despite highest UUID and last insert
        (
            uid_created,
            "ticket_created",
            historical_t0,
            recorded_base + timedelta(minutes=1),
            "customer",
            None,
            json.dumps(ticket_events_repo.build_ticket_created_payload()),
        ),
    ]

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tickets (id, tenant_id, subject, body, status, created_at, first_response_at, resolved_at)
                VALUES (%s, %s, 'Ordering Fixture Ticket', 'Body', 'closed', %s, %s, %s);
                """,
                (ord_ticket_id, tenant_id, historical_t0, t_fr, t_res),
            )
            for ev_id, ev_type, occ_at, rec_at, actor_type, actor_id, payload_json in fixture_rows_in_scrambled_insert_order:
                cur.execute(
                    """
                    INSERT INTO ticket_events (
                        id, tenant_id, ticket_id, event_type, occurred_at, recorded_at,
                        actor_type, actor_id, schema_version, payload
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, 1, %s::jsonb
                    );
                    """,
                    (ev_id, tenant_id, ord_ticket_id, ev_type, occ_at, rec_at, actor_type, actor_id, payload_json),
                )

    # Derive expected order from the fixture's stored PostgreSQL ordering keys independently of the endpoint result
    with conn.transaction():
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, event_type, occurred_at, recorded_at
                FROM ticket_events
                WHERE tenant_id = %s AND ticket_id = %s;
                """,
                (tenant_id, ord_ticket_id),
            )
            stored_rows = cur.fetchall()

    expected_rows = sorted(
        stored_rows,
        key=lambda r: (
            r["occurred_at"].astimezone(timezone.utc),
            r["recorded_at"].astimezone(timezone.utc),
            uuid.UUID(str(r["id"])),
        ),
    )
    assert [uuid.UUID(str(r["id"])) for r in expected_rows] == [
        uid_created,
        uid_processed,
        uid_first_resp,
        uid_resolved,
        uid_closed,
    ]

    token = create_access_token(user_id=user_id, tenant_id=tenant_id, role="agent", expires_delta=timedelta(minutes=60))
    client = TestClient(app, cookies={"triage_session": token})

    res_legacy = client.get(f"/browser-api/tickets/{legacy_id}/events")
    assert res_legacy.status_code == 200
    assert res_legacy.headers.get("cache-control") == "no-store"
    assert res_legacy.json()["schema_available"] is True
    assert res_legacy.json()["events"] == []

    res_ord = client.get(f"/browser-api/tickets/{ord_ticket_id}/events")
    assert res_ord.status_code == 200
    assert res_ord.headers.get("cache-control") == "no-store"
    events = res_ord.json()["events"]

    # 1. Exact count, identities, and PostgreSQL ordering contract (occurred_at ASC, recorded_at ASC, id ASC)
    assert len(events) == len(expected_rows)
    assert [uuid.UUID(e["id"]) for e in events] == [uuid.UUID(str(r["id"])) for r in expected_rows]
    assert [e["event_type"] for e in events] == [
        "ticket_created",
        "ticket_processed",
        "first_response_recorded",
        "ticket_resolved",
        "ticket_closed",
    ]
    for ev, exp in zip(events, expected_rows):
        assert uuid.UUID(ev["id"]) == uuid.UUID(str(exp["id"]))
        assert datetime.fromisoformat(ev["occurred_at"]).astimezone(timezone.utc) == exp["occurred_at"].astimezone(timezone.utc)
        assert datetime.fromisoformat(ev["recorded_at"]).astimezone(timezone.utc) == exp["recorded_at"].astimezone(timezone.utc)

    # 2. Genuinely backdated ticket_processed sorts before later business-time events despite latest recorded_at
    proc_ev, fr_ev, res_ev, cls_ev = events[1], events[2], events[3], events[4]
    proc_occ = datetime.fromisoformat(proc_ev["occurred_at"]).astimezone(timezone.utc)
    proc_rec = datetime.fromisoformat(proc_ev["recorded_at"]).astimezone(timezone.utc)
    fr_occ = datetime.fromisoformat(fr_ev["occurred_at"]).astimezone(timezone.utc)
    fr_rec = datetime.fromisoformat(fr_ev["recorded_at"]).astimezone(timezone.utc)
    res_occ = datetime.fromisoformat(res_ev["occurred_at"]).astimezone(timezone.utc)
    res_rec = datetime.fromisoformat(res_ev["recorded_at"]).astimezone(timezone.utc)
    cls_occ = datetime.fromisoformat(cls_ev["occurred_at"]).astimezone(timezone.utc)
    cls_rec = datetime.fromisoformat(cls_ev["recorded_at"]).astimezone(timezone.utc)

    assert proc_occ < fr_occ
    assert proc_rec > fr_rec
    assert proc_occ < proc_rec < datetime.now(timezone.utc)

    # 3. Equal-occurred_at tie-breaking: recorded_at ASC precedes higher-to-lower id, then id ASC breaks ties
    assert fr_occ == res_occ == cls_occ
    assert fr_rec < res_rec
    assert uuid.UUID(fr_ev["id"]) > uuid.UUID(res_ev["id"])
    assert res_rec == cls_rec
    assert uuid.UUID(res_ev["id"]) < uuid.UUID(cls_ev["id"])


# 13. Non-Destructive Missing-Relation Savepoint Recovery & HTTP Fallback Mapping


def test_13_missing_relation_savepoint_recovery_and_http_fallback(
    conn: psycopg.Connection[Any],
    synthetic_tenant: tuple[uuid.UUID, uuid.UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify real PostgreSQL UndefinedTable + SAVEPOINT recovery via SET LOCAL search_path = '' without DDL."""
    tenant_id, user_id = synthetic_tenant
    ticket, _ = ticket_repo.create_or_get_ticket(conn, tenant_id, "Fallback Check", "Body")
    conn.commit()

    # Part A: Dedicated connection proves PostgreSQL raises real UndefinedTable and repo savepoint keeps outer txn alive
    dedicated_conn = psycopg.connect(TARGET_DB_URL)
    try:
        with dedicated_conn.transaction():
            with dedicated_conn.cursor() as cur:
                cur.execute("SET LOCAL search_path = '';")

            with pytest.raises(ticket_events_repo.TicketEventsSchemaUnavailableError) as exc_info:
                ticket_events_repo.list_ticket_events(
                    dedicated_conn,
                    tenant_id,
                    ticket.id,
                    limit=ticket_events_repo.BOUNDED_EVENT_READ_LIMIT,
                )
            assert isinstance(exc_info.value.__cause__, psycopg.errors.UndefinedTable)

            # Outer transaction must remain usable (not INERROR / InFailedSqlTransaction)
            with dedicated_conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT 1 AS alive;")
                row = cur.fetchone()
                assert row is not None and row["alive"] == 1
        dedicated_conn.rollback()
    finally:
        dedicated_conn.close()

    # Part B: Endpoint mapping using test-local search_path wrapper around real repository read (restored cleanly)
    real_list_ticket_events = ticket_events_repo.list_ticket_events

    def _scoped_empty_search_path_list(conn_or_cur: Any, tid: uuid.UUID, tk_id: uuid.UUID, *, limit: int | None = None):
        with conn_or_cur.transaction():
            with conn_or_cur.cursor() as cur:
                cur.execute("SET LOCAL search_path = '';")
            return real_list_ticket_events(conn_or_cur, tid, tk_id, limit=limit)

    token = create_access_token(user_id=user_id, tenant_id=tenant_id, role="agent", expires_delta=timedelta(minutes=60))
    client = TestClient(app, cookies={"triage_session": token})

    with monkeypatch.context() as mp:
        mp.setattr(ticket_events_repo, "list_ticket_events", _scoped_empty_search_path_list)
        res_fallback = client.get(f"/browser-api/tickets/{ticket.id}/events")

    assert res_fallback.status_code == 200
    assert res_fallback.headers.get("cache-control") == "no-store"
    body = res_fallback.json()
    assert body["schema_available"] is False
    assert body["events"] == []
    assert "009" in body["message"]

    # After wrapper removal, pool connection remains healthy and returns real events normally
    res_ok = client.get(f"/browser-api/tickets/{ticket.id}/events")
    assert res_ok.status_code == 200
    assert res_ok.json()["schema_available"] is True
    assert len(res_ok.json()["events"]) == 1
