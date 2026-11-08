"""Database integration tests for tenant-scoped similar-ticket suggestions.

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
import app.repositories.tickets as ticket_repo
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


def test_similar_tickets_exact_cosine_retrieval_and_ordering():
    """Verify exact cosine similarity calculation and deterministic descending ordering."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Cosine Retrieval Co")
    src_ticket_id = _create_test_ticket(tenant_id, "Source Ticket")
    match_high_id = _create_test_ticket(tenant_id, "High Similarity Ticket")
    match_med_id = _create_test_ticket(tenant_id, "Medium Similarity Ticket")

    # High match has identical vector (cosine sim = 1.0)
    # Med match has orthogonal vector (cosine sim = 0.0)
    src_vec = _make_unit_vector(384, active_idx=0)
    high_vec = _make_unit_vector(384, active_idx=0)
    med_vec = _make_unit_vector(384, active_idx=1)

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        with conn.transaction():
            dup_repo.save_ticket_embedding(conn, tenant_id, src_ticket_id, src_vec, "all-MiniLM-L6-v2", "rev1")
            dup_repo.save_ticket_embedding(conn, tenant_id, match_high_id, high_vec, "all-MiniLM-L6-v2", "rev1")
            dup_repo.save_ticket_embedding(conn, tenant_id, match_med_id, med_vec, "all-MiniLM-L6-v2", "rev1")

        has_emb, m_name, m_ver, candidates = dup_repo.find_similar_tickets_for_ticket(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id=src_ticket_id,
            limit=5,
        )

        assert has_emb is True
        assert m_name == "all-MiniLM-L6-v2"
        assert m_ver == "rev1"
        assert len(candidates) == 2

        # High match should be first with similarity approx 1.0
        assert candidates[0].ticket_id == match_high_id
        assert pytest.approx(candidates[0].similarity_score, 1e-4) == 1.0

        # Med match should be second with similarity approx 0.0
        assert candidates[1].ticket_id == match_med_id
        assert pytest.approx(candidates[1].similarity_score, 1e-4) == 0.0


def test_similar_tickets_tenant_scoping():
    """Verify candidate search strictly isolates tickets by tenant."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_a = _create_test_tenant("Tenant A Similar Co")
    tenant_b = _create_test_tenant("Tenant B Similar Co")

    t_a1 = _create_test_ticket(tenant_a, "Ticket A1")
    t_a2 = _create_test_ticket(tenant_a, "Ticket A2")
    t_b1 = _create_test_ticket(tenant_b, "Ticket B1")

    vec = _make_unit_vector(384, active_idx=5)

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        with conn.transaction():
            dup_repo.save_ticket_embedding(conn, tenant_a, t_a1, vec, "all-MiniLM-L6-v2", "rev1")
            dup_repo.save_ticket_embedding(conn, tenant_a, t_a2, vec, "all-MiniLM-L6-v2", "rev1")
            # Tenant B has identical vector
            dup_repo.save_ticket_embedding(conn, tenant_b, t_b1, vec, "all-MiniLM-L6-v2", "rev1")

        has_emb, _, _, candidates = dup_repo.find_similar_tickets_for_ticket(
            conn=conn,
            tenant_id=tenant_a,
            ticket_id=t_a1,
            limit=10,
        )

        assert has_emb is True
        returned_ids = {c.ticket_id for c in candidates}
        assert t_a2 in returned_ids
        assert t_b1 not in returned_ids, "Cross-tenant ticket must NEVER appear in similar suggestions"


def test_similar_tickets_model_scoping():
    """Verify search only compares embeddings with the exact same model_name and model_version."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Model Scoping Co")
    t1 = _create_test_ticket(tenant_id, "Ticket 1")
    t2 = _create_test_ticket(tenant_id, "Ticket 2 Same Model")
    t3 = _create_test_ticket(tenant_id, "Ticket 3 Different Version")
    t4 = _create_test_ticket(tenant_id, "Ticket 4 Different Name")

    vec = _make_unit_vector(384, active_idx=10)

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        with conn.transaction():
            dup_repo.save_ticket_embedding(conn, tenant_id, t1, vec, "all-MiniLM-L6-v2", "revA")
            dup_repo.save_ticket_embedding(conn, tenant_id, t2, vec, "all-MiniLM-L6-v2", "revA")
            dup_repo.save_ticket_embedding(conn, tenant_id, t3, vec, "all-MiniLM-L6-v2", "revB")
            dup_repo.save_ticket_embedding(conn, tenant_id, t4, vec, "different-encoder", "revA")

        has_emb, m_name, m_ver, candidates = dup_repo.find_similar_tickets_for_ticket(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id=t1,
            limit=10,
        )

        assert has_emb is True
        assert m_name == "all-MiniLM-L6-v2"
        assert m_ver == "revA"

        returned_ids = {c.ticket_id for c in candidates}
        assert t2 in returned_ids
        assert t3 not in returned_ids, "Different model_version must not be compared"
        assert t4 not in returned_ids, "Different model_name must not be compared"


def test_similar_tickets_source_exclusion_and_limit():
    """Verify the source ticket is excluded from results and limits are enforced."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Exclusion Limit Co")
    src_ticket = _create_test_ticket(tenant_id, "Source Ticket")

    other_tickets = [_create_test_ticket(tenant_id, f"Candidate {i}") for i in range(5)]
    vec = _make_unit_vector(384, active_idx=2)

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        with conn.transaction():
            dup_repo.save_ticket_embedding(conn, tenant_id, src_ticket, vec, "all-MiniLM-L6-v2", "rev1")
            for t_id in other_tickets:
                dup_repo.save_ticket_embedding(conn, tenant_id, t_id, vec, "all-MiniLM-L6-v2", "rev1")

        has_emb, _, _, candidates = dup_repo.find_similar_tickets_for_ticket(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id=src_ticket,
            limit=2,
        )

        assert has_emb is True
        assert len(candidates) == 2, "Limit of 2 must be respected"
        returned_ids = {c.ticket_id for c in candidates}
        assert src_ticket not in returned_ids, "Source ticket must be excluded from results"


def test_similar_tickets_missing_embedding_db():
    """Verify that a ticket with no embedding returns has_embedding=False in DB."""
    assert LOCAL_TEST_DB_URL is not None
    tenant_id = _create_test_tenant("No Emb Co")
    ticket_id = _create_test_ticket(tenant_id, "Unembedded Ticket")

    with psycopg.connect(LOCAL_TEST_DB_URL) as conn:
        has_emb, m_name, m_ver, candidates = dup_repo.find_similar_tickets_for_ticket(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id=ticket_id,
        )

        assert has_emb is False
        assert m_name is None
        assert m_ver is None
        assert candidates == []
