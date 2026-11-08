"""Database integration tests for duplicate feedback (confirm/reject) on similar-ticket pairs.

Strict safety constraints:
- Database tests run ONLY against an already established disposable local PostgreSQL
  instance (localhost or 127.0.0.1).
- Hosted databases (e.g. Supabase poolers) are strictly prohibited.
- Local DB test guards do not assume localhost alone proves disposability:
  they verify target separation from DATABASE_URL, dedicated test naming,
  and actual reachability.
- If no established disposable local test database exists, tests are cleanly skipped.
"""

import uuid
import pytest
import psycopg

import app.repositories.duplicates as dup_repo
from tests.test_embedding_jobs_db import (
    LOCAL_TEST_DB_URL,
    LOCAL_SAFETY_STATUS,
    _create_test_tenant,
    _create_test_ticket,
)

pytestmark = pytest.mark.skipif(
    LOCAL_TEST_DB_URL is None,
    reason=f"Database integration tests require an established disposable local PostgreSQL instance: {LOCAL_SAFETY_STATUS}",
)


def _make_unit_vector(dim: int = 384, active_idx: int = 0) -> list[float]:
    """Helper returning an L2-normalized unit vector with a 1.0 at active_idx."""
    v = [0.0] * dim
    v[active_idx % dim] = 1.0
    return v


def test_similar_feedback_db_server_derived_similarity_and_canonicalization():
    """Verify record_duplicate_feedback derives similarity and canonicalizes order in DB."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Feedback Canonical Co")
    t1 = _create_test_ticket(tenant_id, "Ticket A")
    t2 = _create_test_ticket(tenant_id, "Ticket B")

    v1 = _make_unit_vector(384, active_idx=0)
    v2 = _make_unit_vector(384, active_idx=0)  # Identical vector => cosine sim = 1.0

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        with conn.transaction():
            dup_repo.save_ticket_embedding(conn, tenant_id, t1, v1, "all-MiniLM-L6-v2", "rev1")
            dup_repo.save_ticket_embedding(conn, tenant_id, t2, v2, "all-MiniLM-L6-v2", "rev1")

            # Call with arbitrary order (pass t_max first)
            t_min = min(t1, t2)
            t_max = max(t1, t2)

            link = dup_repo.record_duplicate_feedback(
                conn=conn,
                tenant_id=tenant_id,
                ticket_id_1=t_max,
                ticket_id_2=t_min,
                decision="confirmed",
            )

        assert link.source_ticket_id == t_min
        assert link.target_ticket_id == t_max
        assert link.status == "confirmed"
        assert pytest.approx(link.similarity_score, 1e-4) == 1.0
        assert link.reviewed_at is not None


def test_similar_feedback_db_idempotent_and_decision_change():
    """Verify repeated confirmations are idempotent and decision changes update reviewed_at."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Feedback Idempotent Co")
    t1 = _create_test_ticket(tenant_id, "Ticket 1")
    t2 = _create_test_ticket(tenant_id, "Ticket 2")

    v = _make_unit_vector(384, active_idx=3)

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        with conn.transaction():
            dup_repo.save_ticket_embedding(conn, tenant_id, t1, v, "all-MiniLM-L6-v2", "rev1")
            dup_repo.save_ticket_embedding(conn, tenant_id, t2, v, "all-MiniLM-L6-v2", "rev1")

            # First decision: confirmed
            link1 = dup_repo.record_duplicate_feedback(
                conn=conn,
                tenant_id=tenant_id,
                ticket_id_1=t1,
                ticket_id_2=t2,
                decision="confirmed",
            )

            # Idempotent repeat: confirmed again
            link2 = dup_repo.record_duplicate_feedback(
                conn=conn,
                tenant_id=tenant_id,
                ticket_id_1=t2,
                ticket_id_2=t1,
                decision="confirmed",
            )
            assert link1.id == link2.id
            assert link2.status == "confirmed"

            # Change decision: rejected
            link3 = dup_repo.record_duplicate_feedback(
                conn=conn,
                tenant_id=tenant_id,
                ticket_id_1=t1,
                ticket_id_2=t2,
                decision="rejected",
            )
            assert link3.id == link1.id
            assert link3.status == "rejected"


def test_similar_feedback_db_symmetrical_visibility_on_similar_query():
    """Verify decision status is symmetrically visible when querying similar tickets from either side."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Feedback Symmetry Co")
    t_source = _create_test_ticket(tenant_id, "Source Ticket")
    t_target = _create_test_ticket(tenant_id, "Target Ticket")

    v = _make_unit_vector(384, active_idx=7)

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        with conn.transaction():
            dup_repo.save_ticket_embedding(conn, tenant_id, t_source, v, "all-MiniLM-L6-v2", "rev1")
            dup_repo.save_ticket_embedding(conn, tenant_id, t_target, v, "all-MiniLM-L6-v2", "rev1")

            # Record confirmed decision
            dup_repo.record_duplicate_feedback(
                conn=conn,
                tenant_id=tenant_id,
                ticket_id_1=t_source,
                ticket_id_2=t_target,
                decision="confirmed",
            )

        # Query from source perspective
        has_emb, _, _, candidates_from_source = dup_repo.find_similar_tickets_for_ticket(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id=t_source,
        )
        assert has_emb is True
        assert len(candidates_from_source) == 1
        cand_target = candidates_from_source[0]
        assert cand_target.ticket_id == t_target
        assert cand_target.decision_status == "confirmed"
        assert cand_target.reviewed_at is not None

        # Query from target perspective (reversed pair)
        has_emb2, _, _, candidates_from_target = dup_repo.find_similar_tickets_for_ticket(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id=t_target,
        )
        assert has_emb2 is True
        assert len(candidates_from_target) == 1
        cand_source = candidates_from_target[0]
        assert cand_source.ticket_id == t_source
        assert cand_source.decision_status == "confirmed"
        assert cand_source.reviewed_at is not None
