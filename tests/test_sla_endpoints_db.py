"""Guarded database integration tests for browser SLA endpoints and audit records.

Safety constraints:
- Runs ONLY against an established disposable local PostgreSQL instance.
- Hosted Supabase poolers are strictly skipped.
"""

from datetime import datetime, timezone
import uuid
import pytest
import psycopg

from app.repositories.auth import create_tenant_and_admin_user
import app.repositories.sla as sla_repo
import app.repositories.tickets as ticket_repo
from app.security import hash_password
from tests.test_embedding_jobs_db import resolve_disposable_local_test_db

LOCAL_TEST_DB_URL, LOCAL_SAFETY_STATUS = resolve_disposable_local_test_db()
pytestmark = pytest.mark.skipif(
    LOCAL_TEST_DB_URL is None,
    reason=f"Guarded SLA endpoints DB integration tests require an established disposable local PostgreSQL instance: {LOCAL_SAFETY_STATUS}",
)

UTC = timezone.utc


def test_audit_log_schema_conformance():
    """Verify audit_log table adheres to the four-column schema without sensitive payload fields."""
    assert LOCAL_TEST_DB_URL is not None

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'audit_log'
                ORDER BY ordinal_position;
                """
            )
            cols = [r[0] for r in cur.fetchall()]
            assert "tenant_id" in cols
            assert "actor" in cols
            assert "action" in cols
            assert "at" in cols
            # Verify no sensitive payload / body columns were added
            assert "body" not in cols
            assert "payload" not in cols
            assert "details" not in cols
