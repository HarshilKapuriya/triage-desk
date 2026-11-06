"""Database integration tests for Phase 4 Ticket Embeddings and Duplicate Links.

Per SPEC.md Section 5, 6 and Phase 4 requirements:
1. Migration application and safe rerun through migration runner.
2. Valid same-tenant embedding and duplicate link writes.
3. Database constraint rejection:
   - Cross-tenant ticket_embeddings foreign key rejection
   - Cross-tenant duplicate_links foreign key rejection (both source and target)
   - Self-link rejection (source == target)
   - Canonical ordering check (source < target) rejecting inverted/reversed pairs
   - Cosine similarity range constraint [-1.0, 1.0]
   - Vector dimension constraint vector(384)
4. Tenant isolation:
   - Embeddings read/delete scoped strictly to tenant_id
   - Duplicate links read/update/delete scoped strictly to tenant_id
5. Uniqueness and idempotency:
   - Embedding uniqueness on (tenant_id, ticket_id, model_name, model_version) with upsert
   - Duplicate link uniqueness on (tenant_id, source_ticket_id, target_ticket_id)
6. Cascade deletion:
   - Deleting a ticket cascades to its embeddings and any duplicate links referencing it.
"""

from datetime import datetime, timezone
from pathlib import Path
import uuid
import pytest
import psycopg
from psycopg.rows import dict_row

from app.repositories.auth import create_tenant_and_admin_user
import app.repositories.duplicates as dup_repo
from app.repositories.duplicates import (
    SelfLinkError,
    InvalidStatusError,
    InvalidSimilarityScoreError,
    ContradictoryReviewStateError,
)
import app.repositories.tickets as ticket_repo
from app.security import hash_password
from scripts.migrate import run_migrations, get_applied_migrations
from tests.test_auth_db import TEST_DB_URL, SAFETY_STATUS

skip_no_test_db = pytest.mark.skipif(TEST_DB_URL is None, reason=SAFETY_STATUS)


def _create_test_tenant(company_name: str) -> uuid.UUID:
    """Helper to create an isolated test tenant."""
    assert TEST_DB_URL is not None
    suffix = uuid.uuid4().hex[:8]
    with psycopg.connect(TEST_DB_URL) as conn:
        tenant_id, _ = create_tenant_and_admin_user(
            conn=conn,
            company_name=company_name,
            normalized_email=f"user_{suffix}@test.com",
            password_hash=hash_password("Pass123!"),
        )
    return tenant_id


def _create_test_ticket(tenant_id: uuid.UUID, subject: str = "Test Subject") -> uuid.UUID:
    """Helper to create a ticket for a tenant."""
    assert TEST_DB_URL is not None
    with psycopg.connect(TEST_DB_URL) as conn:
        ticket, _ = ticket_repo.create_or_get_ticket(
            conn=conn,
            tenant_id=tenant_id,
            subject=subject,
            body="Test complaint body text.",
        )
        return ticket.id


def _dummy_vector(fill_val: float = 0.1, dim: int = 384) -> list[float]:
    """Helper to generate a mock vector of specified dimension."""
    return [fill_val] * dim


# ==============================================================================
# 1. Migration Application & Safe Rerun
# ==============================================================================

@skip_no_test_db
def test_migration_004_application_and_safe_rerun() -> None:
    """Migration 004 applies successfully and safe rerun leaves schema unchanged."""
    assert TEST_DB_URL is not None
    migrations_dir = Path(__file__).resolve().parent.parent / "migrations"

    # 1. Apply migrations using the migration runner
    run_migrations(TEST_DB_URL, migrations_dir)

    # 2. Verify 004 is recorded in schema_migrations
    with psycopg.connect(TEST_DB_URL) as conn:
        applied = get_applied_migrations(conn)
        assert "004_embeddings_duplicates.sql" in applied

    # 3. Rerun migrations: must succeed idempotently with 0 pending
    run_migrations(TEST_DB_URL, migrations_dir)

    # 4. Safe direct SQL execution of 004 to verify raw DDL idempotency
    migration_file = migrations_dir / "004_embeddings_duplicates.sql"
    sql_text = migration_file.read_text(encoding="utf-8")
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql_text)
        conn.commit()


# ==============================================================================
# 2. Ticket Embeddings Tests
# ==============================================================================

@skip_no_test_db
def test_valid_same_tenant_embedding_write_and_read() -> None:
    """Valid same-tenant embedding writes and reads with 384-dim vector."""
    assert TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Embedding Corp")
    ticket_id = _create_test_ticket(tenant_id, "Card payment declined")

    vec = _dummy_vector(0.05, 384)
    model_name = "all-MiniLM-L6-v2"
    model_version = "v1.0"

    with psycopg.connect(TEST_DB_URL) as conn:
        saved = dup_repo.save_ticket_embedding(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id=ticket_id,
            embedding=vec,
            model_name=model_name,
            model_version=model_version,
        )
        assert saved.tenant_id == tenant_id
        assert saved.ticket_id == ticket_id
        assert saved.model_name == model_name
        assert saved.model_version == model_version
        assert len(saved.embedding) == 384
        assert pytest.approx(saved.embedding[0], 1e-4) == 0.05

        # Read back
        fetched = dup_repo.get_ticket_embedding(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id=ticket_id,
            model_name=model_name,
            model_version=model_version,
        )
        assert fetched is not None
        assert fetched.id == saved.id
        assert len(fetched.embedding) == 384
        assert pytest.approx(fetched.embedding[0], 1e-4) == 0.05


@skip_no_test_db
def test_embedding_uniqueness_and_upsert() -> None:
    """Repeated writes update the existing embedding rather than creating duplicates."""
    assert TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Embedding Upsert Corp")
    ticket_id = _create_test_ticket(tenant_id, "Billing dispute")

    vec_1 = _dummy_vector(0.1, 384)
    vec_2 = _dummy_vector(0.9, 384)
    model_name = "all-MiniLM-L6-v2"
    model_version = "v1.0"

    with psycopg.connect(TEST_DB_URL) as conn:
        record_1 = dup_repo.save_ticket_embedding(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id=ticket_id,
            embedding=vec_1,
            model_name=model_name,
            model_version=model_version,
        )

        record_2 = dup_repo.save_ticket_embedding(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id=ticket_id,
            embedding=vec_2,
            model_name=model_name,
            model_version=model_version,
        )

        # Same record id, updated content
        assert record_1.id == record_2.id
        assert pytest.approx(record_2.embedding[0], 1e-4) == 0.9

        # Direct INSERT without ON CONFLICT raises UniqueViolation
        with pytest.raises(psycopg.errors.UniqueViolation):
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ticket_embeddings (tenant_id, ticket_id, embedding, model_name, model_version)
                    VALUES (%s, %s, %s::vector, %s, %s);
                    """,
                    (tenant_id, ticket_id, dup_repo._format_vector(vec_1), model_name, model_version),
                )
        conn.rollback()


@skip_no_test_db
def test_embedding_invalid_dimension_rejected_by_db() -> None:
    """PostgreSQL rejects vectors that do not match the declared vector(384) dimension."""
    assert TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Embedding Dim Corp")
    ticket_id = _create_test_ticket(tenant_id, "Dimension check ticket")

    # 3-dimensional vector instead of 384
    short_vec = [0.1, 0.2, 0.3]
    with psycopg.connect(TEST_DB_URL) as conn:
        with pytest.raises(psycopg.errors.DataError) as exc_info:
            dup_repo.save_ticket_embedding(
                conn=conn,
                tenant_id=tenant_id,
                ticket_id=ticket_id,
                embedding=short_vec,
                model_name="all-MiniLM-L6-v2",
                model_version="v1.0",
            )
        assert "expected 384 dimensions" in str(exc_info.value)
        conn.rollback()


@skip_no_test_db
def test_embedding_cross_tenant_foreign_key_rejection() -> None:
    """Database rejects embedding attaching Tenant A to Tenant B's ticket."""
    assert TEST_DB_URL is not None
    tenant_a = _create_test_tenant("Tenant A Embedding")
    tenant_b = _create_test_tenant("Tenant B Embedding")
    ticket_b = _create_test_ticket(tenant_b, "Tenant B ticket")

    vec = _dummy_vector(0.1, 384)
    with psycopg.connect(TEST_DB_URL) as conn:
        with pytest.raises(psycopg.errors.ForeignKeyViolation) as exc_info:
            # Tenant A tries to store embedding referencing Tenant B's ticket
            dup_repo.save_ticket_embedding(
                conn=conn,
                tenant_id=tenant_a,
                ticket_id=ticket_b,
                embedding=vec,
                model_name="all-MiniLM-L6-v2",
                model_version="v1.0",
            )
        assert "fk_ticket_embeddings_tenant_ticket" in str(exc_info.value)
        conn.rollback()


@skip_no_test_db
def test_embedding_tenant_isolation_reads_and_deletes() -> None:
    """Tenant B cannot read or delete Tenant A's ticket embeddings."""
    assert TEST_DB_URL is not None
    tenant_a = _create_test_tenant("Tenant A Isolation")
    tenant_b = _create_test_tenant("Tenant B Isolation")
    ticket_a = _create_test_ticket(tenant_a, "Tenant A ticket")

    vec = _dummy_vector(0.2, 384)
    with psycopg.connect(TEST_DB_URL) as conn:
        dup_repo.save_ticket_embedding(
            conn=conn,
            tenant_id=tenant_a,
            ticket_id=ticket_a,
            embedding=vec,
            model_name="all-MiniLM-L6-v2",
            model_version="v1.0",
        )

        # Tenant B cannot read Tenant A's embedding
        b_read = dup_repo.get_ticket_embedding(
            conn=conn,
            tenant_id=tenant_b,
            ticket_id=ticket_a,
            model_name="all-MiniLM-L6-v2",
            model_version="v1.0",
        )
        assert b_read is None

        # Tenant B delete affects 0 rows
        deleted = dup_repo.delete_ticket_embedding(
            conn=conn,
            tenant_id=tenant_b,
            ticket_id=ticket_a,
        )
        assert deleted == 0

        # Tenant A's embedding still intact
        a_read = dup_repo.get_ticket_embedding(
            conn=conn,
            tenant_id=tenant_a,
            ticket_id=ticket_a,
            model_name="all-MiniLM-L6-v2",
            model_version="v1.0",
        )
        assert a_read is not None


# ==============================================================================
# 3. Duplicate Links Tests
# ==============================================================================

@skip_no_test_db
def test_valid_same_tenant_duplicate_link_write_and_read() -> None:
    """Valid same-tenant duplicate link creation and symmetric retrieval."""
    assert TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Dup Links Corp")
    t1 = _create_test_ticket(tenant_id, "Cannot login with password")
    t2 = _create_test_ticket(tenant_id, "Login error password rejected")

    with psycopg.connect(TEST_DB_URL) as conn:
        link = dup_repo.create_duplicate_link(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id_1=t1,
            ticket_id_2=t2,
            similarity_score=0.895,
            model_name="all-MiniLM-L6-v2",
            model_version="v1.0",
            status="suggested",
        )
        assert link.tenant_id == tenant_id
        assert link.status == "suggested"
        assert pytest.approx(link.similarity_score, 1e-4) == 0.895
        assert link.source_ticket_id < link.target_ticket_id

        # Lookup by link id
        fetched = dup_repo.get_duplicate_link(conn=conn, tenant_id=tenant_id, link_id=link.id)
        assert fetched is not None
        assert fetched.id == link.id

        # Undirected retrieval: t1 sees the link
        links_t1 = dup_repo.get_duplicate_links_for_ticket(conn=conn, tenant_id=tenant_id, ticket_id=t1)
        assert len(links_t1) == 1
        assert links_t1[0].id == link.id

        # Undirected retrieval: t2 also sees the link
        links_t2 = dup_repo.get_duplicate_links_for_ticket(conn=conn, tenant_id=tenant_id, ticket_id=t2)
        assert len(links_t2) == 1
        assert links_t2[0].id == link.id


@skip_no_test_db
def test_duplicate_link_self_link_rejection() -> None:
    """Self-links (linking a ticket to itself) are rejected by repo and DB constraint."""
    assert TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Self Link Corp")
    t1 = _create_test_ticket(tenant_id, "Ticket A")

    with psycopg.connect(TEST_DB_URL) as conn:
        # Repository check
        with pytest.raises(SelfLinkError):
            dup_repo.create_duplicate_link(
                conn=conn,
                tenant_id=tenant_id,
                ticket_id_1=t1,
                ticket_id_2=t1,
                similarity_score=1.0,
                model_name="all-MiniLM-L6-v2",
                model_version="v1.0",
            )

        # Database check constraint enforcement
        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO duplicate_links (
                        tenant_id, source_ticket_id, target_ticket_id, similarity_score, model_name, model_version
                    ) VALUES (%s, %s, %s, 1.0, 'all-MiniLM-L6-v2', 'v1.0');
                    """,
                    (tenant_id, t1, t1),
                )
        conn.rollback()


@skip_no_test_db
def test_duplicate_link_cross_tenant_rejection() -> None:
    """Database rejects duplicate link between tickets of different tenants."""
    assert TEST_DB_URL is not None
    tenant_a = _create_test_tenant("Tenant A Cross Link")
    tenant_b = _create_test_tenant("Tenant B Cross Link")
    ticket_a = _create_test_ticket(tenant_a, "Ticket in Tenant A")
    ticket_b = _create_test_ticket(tenant_b, "Ticket in Tenant B")

    with psycopg.connect(TEST_DB_URL) as conn:
        # 1. Attempt link claiming Tenant A ownership
        with pytest.raises(psycopg.errors.ForeignKeyViolation) as exc_info_a:
            dup_repo.create_duplicate_link(
                conn=conn,
                tenant_id=tenant_a,
                ticket_id_1=ticket_a,
                ticket_id_2=ticket_b,
                similarity_score=0.85,
                model_name="all-MiniLM-L6-v2",
                model_version="v1.0",
            )
        # Foreign key for whichever ticket was placed into target or source fails
        assert "fk_duplicate_links_tenant" in str(exc_info_a.value)
        conn.rollback()

        # 2. Attempt link claiming Tenant B ownership
        with pytest.raises(psycopg.errors.ForeignKeyViolation) as exc_info_b:
            dup_repo.create_duplicate_link(
                conn=conn,
                tenant_id=tenant_b,
                ticket_id_1=ticket_a,
                ticket_id_2=ticket_b,
                similarity_score=0.85,
                model_name="all-MiniLM-L6-v2",
                model_version="v1.0",
            )
        assert "fk_duplicate_links_tenant" in str(exc_info_b.value)
        conn.rollback()


@skip_no_test_db
def test_duplicate_link_canonical_ordering_and_reversed_pair() -> None:
    """Undirected canonical ordering enforces unique pair semantics and rejects reversed insertions."""
    assert TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Canonical Corp")
    t1 = _create_test_ticket(tenant_id, "Ticket 1")
    t2 = _create_test_ticket(tenant_id, "Ticket 2")

    min_t = t1 if t1 < t2 else t2
    max_t = t2 if t1 < t2 else t1

    with psycopg.connect(TEST_DB_URL) as conn:
        # Create canonical link (min_t, max_t)
        link = dup_repo.create_duplicate_link(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id_1=min_t,
            ticket_id_2=max_t,
            similarity_score=0.75,
            model_name="all-MiniLM-L6-v2",
            model_version="v1.0",
        )
        assert link.source_ticket_id == min_t
        assert link.target_ticket_id == max_t

        # Attempting to insert repeated link with reversed arguments:
        # Auto-canonicalize sorts them to (min_t, max_t), which hits UniqueViolation
        with pytest.raises(psycopg.errors.UniqueViolation):
            dup_repo.create_duplicate_link(
                conn=conn,
                tenant_id=tenant_id,
                ticket_id_1=max_t,
                ticket_id_2=min_t,
                similarity_score=0.78,
                model_name="all-MiniLM-L6-v2",
                model_version="v1.0",
                auto_canonicalize=True,
            )
        conn.rollback()

        # Attempting raw insert of reversed pair (max_t, min_t):
        # Database check constraint `chk_duplicate_links_canonical_order` rejects it
        with pytest.raises(psycopg.errors.CheckViolation) as exc_info:
            dup_repo.create_duplicate_link(
                conn=conn,
                tenant_id=tenant_id,
                ticket_id_1=max_t,
                ticket_id_2=min_t,
                similarity_score=0.78,
                model_name="all-MiniLM-L6-v2",
                model_version="v1.0",
                auto_canonicalize=False,
            )
        assert "chk_duplicate_links_canonical_order" in str(exc_info.value)
        conn.rollback()


@skip_no_test_db
def test_duplicate_link_similarity_score_contract_and_bounds() -> None:
    """Cosine similarity score contract enforces range [-1.0, 1.0] and rejects non-finite floats."""
    assert TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Bounds Corp")
    t1 = _create_test_ticket(tenant_id, "Ticket Bound 1")
    t2 = _create_test_ticket(tenant_id, "Ticket Bound 2")
    t3 = _create_test_ticket(tenant_id, "Ticket Bound 3")
    t4 = _create_test_ticket(tenant_id, "Ticket Bound 4")

    with psycopg.connect(TEST_DB_URL) as conn:
        # Boundary cases: -1.0 and 1.0 are valid
        link_neg = dup_repo.create_duplicate_link(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id_1=t1,
            ticket_id_2=t2,
            similarity_score=-1.0,
            model_name="all-MiniLM-L6-v2",
            model_version="v1.0",
        )
        assert pytest.approx(link_neg.similarity_score) == -1.0

        link_pos = dup_repo.create_duplicate_link(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id_1=t1,
            ticket_id_2=t3,
            similarity_score=1.0,
            model_name="all-MiniLM-L6-v2",
            model_version="v1.0",
        )
        assert pytest.approx(link_pos.similarity_score) == 1.0

        # Non-finite values: NaN, +inf, -inf must raise InvalidSimilarityScoreError before SQL execution
        for non_finite in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(InvalidSimilarityScoreError):
                dup_repo.create_duplicate_link(
                    conn=conn,
                    tenant_id=tenant_id,
                    ticket_id_1=t1,
                    ticket_id_2=t4,
                    similarity_score=non_finite,
                    model_name="all-MiniLM-L6-v2",
                    model_version="v1.0",
                )

        # Out-of-bounds: > 1.0
        with pytest.raises(InvalidSimilarityScoreError):
            dup_repo.create_duplicate_link(
                conn=conn,
                tenant_id=tenant_id,
                ticket_id_1=t2,
                ticket_id_2=t3,
                similarity_score=1.01,
                model_name="all-MiniLM-L6-v2",
                model_version="v1.0",
            )

        # Out-of-bounds: < -1.0
        with pytest.raises(InvalidSimilarityScoreError):
            dup_repo.create_duplicate_link(
                conn=conn,
                tenant_id=tenant_id,
                ticket_id_1=t2,
                ticket_id_2=t3,
                similarity_score=-1.01,
                model_name="all-MiniLM-L6-v2",
                model_version="v1.0",
            )

        # Direct SQL violating score constraint fails at DB level
        with pytest.raises(psycopg.errors.CheckViolation):
            min_t = t2 if t2 < t3 else t3
            max_t = t3 if t2 < t3 else t2
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO duplicate_links (
                        tenant_id, source_ticket_id, target_ticket_id, similarity_score, model_name, model_version
                    ) VALUES (%s, %s, %s, 10.5, 'all-MiniLM-L6-v2', 'v1.0');
                    """,
                    (tenant_id, min_t, max_t),
                )
        conn.rollback()


@skip_no_test_db
def test_duplicate_link_allowed_statuses_and_transitions() -> None:
    """Duplicate link enforces review-state contract: suggested => NULL, confirmed/rejected => timestamp."""
    assert TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Status Corp")
    t1 = _create_test_ticket(tenant_id, "Ticket S1")
    t2 = _create_test_ticket(tenant_id, "Ticket S2")
    t3 = _create_test_ticket(tenant_id, "Ticket S3")
    t4 = _create_test_ticket(tenant_id, "Ticket S4")

    now_ts = datetime.now(timezone.utc)

    with psycopg.connect(TEST_DB_URL) as conn:
        # 1. Default creation: status is 'suggested' => reviewed_at must be None
        link = dup_repo.create_duplicate_link(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id_1=t1,
            ticket_id_2=t2,
            similarity_score=0.82,
            model_name="all-MiniLM-L6-v2",
            model_version="v1.0",
        )
        assert link.status == "suggested"
        assert link.reviewed_at is None

        # 2. Contradictory creation: suggested with explicit reviewed_at must be rejected
        with pytest.raises(ContradictoryReviewStateError):
            dup_repo.create_duplicate_link(
                conn=conn,
                tenant_id=tenant_id,
                ticket_id_1=t1,
                ticket_id_2=t3,
                similarity_score=0.80,
                model_name="all-MiniLM-L6-v2",
                model_version="v1.0",
                status="suggested",
                reviewed_at=now_ts,
            )

        # 3. Creation directly with 'confirmed' status => reviewed_at is populated
        link_confirmed_init = dup_repo.create_duplicate_link(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id_1=t1,
            ticket_id_2=t3,
            similarity_score=0.80,
            model_name="all-MiniLM-L6-v2",
            model_version="v1.0",
            status="confirmed",
        )
        assert link_confirmed_init.status == "confirmed"
        assert link_confirmed_init.reviewed_at is not None

        # 4. Creation directly with 'rejected' status => reviewed_at is populated
        link_rejected_init = dup_repo.create_duplicate_link(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id_1=t1,
            ticket_id_2=t4,
            similarity_score=0.70,
            model_name="all-MiniLM-L6-v2",
            model_version="v1.0",
            status="rejected",
        )
        assert link_rejected_init.status == "rejected"
        assert link_rejected_init.reviewed_at is not None

        # 5. Transition to confirmed updates reviewed_at
        confirmed = dup_repo.update_duplicate_link_status(
            conn=conn,
            tenant_id=tenant_id,
            link_id=link.id,
            status="confirmed",
        )
        assert confirmed is not None
        assert confirmed.status == "confirmed"
        assert confirmed.reviewed_at is not None

        # 6. Transition confirmed -> suggested resets reviewed_at to None
        reset_to_suggested = dup_repo.update_duplicate_link_status(
            conn=conn,
            tenant_id=tenant_id,
            link_id=link.id,
            status="suggested",
        )
        assert reset_to_suggested is not None
        assert reset_to_suggested.status == "suggested"
        assert reset_to_suggested.reviewed_at is None

        # 7. Transition to rejected populates reviewed_at
        rejected = dup_repo.update_duplicate_link_status(
            conn=conn,
            tenant_id=tenant_id,
            link_id=link.id,
            status="rejected",
        )
        assert rejected is not None
        assert rejected.status == "rejected"
        assert rejected.reviewed_at is not None

        # 8. Transition rejected -> suggested resets reviewed_at to None
        reset_from_rejected = dup_repo.update_duplicate_link_status(
            conn=conn,
            tenant_id=tenant_id,
            link_id=link.id,
            status="suggested",
        )
        assert reset_from_rejected is not None
        assert reset_from_rejected.status == "suggested"
        assert reset_from_rejected.reviewed_at is None

        # 9. Updating to 'suggested' with explicit reviewed_at raises ContradictoryReviewStateError
        with pytest.raises(ContradictoryReviewStateError):
            dup_repo.update_duplicate_link_status(
                conn=conn,
                tenant_id=tenant_id,
                link_id=link.id,
                status="suggested",
                reviewed_at=now_ts,
            )

        # 10. Invalid status is rejected by repository
        with pytest.raises(InvalidStatusError):
            dup_repo.update_duplicate_link_status(
                conn=conn,
                tenant_id=tenant_id,
                link_id=link.id,
                status="merged",
            )

        # 11. Invalid status is rejected by DB constraint on raw update
        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE duplicate_links SET status = 'unapproved' WHERE id = %s;",
                    (link.id,),
                )
        conn.rollback()


@skip_no_test_db
def test_duplicate_link_tenant_isolation() -> None:
    """Tenant B cannot read, list, update, or delete Tenant A's duplicate links."""
    assert TEST_DB_URL is not None
    tenant_a = _create_test_tenant("Tenant A Link Isolation")
    tenant_b = _create_test_tenant("Tenant B Link Isolation")
    t1_a = _create_test_ticket(tenant_a, "A Ticket 1")
    t2_a = _create_test_ticket(tenant_a, "A Ticket 2")

    with psycopg.connect(TEST_DB_URL) as conn:
        link_a = dup_repo.create_duplicate_link(
            conn=conn,
            tenant_id=tenant_a,
            ticket_id_1=t1_a,
            ticket_id_2=t2_a,
            similarity_score=0.91,
            model_name="all-MiniLM-L6-v2",
            model_version="v1.0",
        )

        # Tenant B lookup returns None
        assert dup_repo.get_duplicate_link(conn, tenant_b, link_a.id) is None

        # Tenant B query by ticket returns empty list
        assert dup_repo.get_duplicate_links_for_ticket(conn, tenant_b, t1_a) == []

        # Tenant B list returns empty list
        assert dup_repo.list_duplicate_links(conn, tenant_b) == []

        # Tenant B update returns None and doesn't modify Tenant A's link
        assert dup_repo.update_duplicate_link_status(conn, tenant_b, link_a.id, "confirmed") is None
        check_a = dup_repo.get_duplicate_link(conn, tenant_a, link_a.id)
        assert check_a is not None
        assert check_a.status == "suggested"

        # Tenant B delete returns False and doesn't remove Tenant A's link
        assert dup_repo.delete_duplicate_link(conn, tenant_b, link_a.id) is False
        assert dup_repo.get_duplicate_link(conn, tenant_a, link_a.id) is not None


@skip_no_test_db
def test_ticket_deletion_cascades_to_embeddings_and_links() -> None:
    """Deleting a ticket cascades deletion to its embeddings and all associated duplicate links."""
    assert TEST_DB_URL is not None
    tenant_id = _create_test_tenant("Cascade Corp")
    t1 = _create_test_ticket(tenant_id, "Ticket to delete")
    t2 = _create_test_ticket(tenant_id, "Ticket to remain")

    with psycopg.connect(TEST_DB_URL) as conn:
        # Create embedding for t1
        emb = dup_repo.save_ticket_embedding(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id=t1,
            embedding=_dummy_vector(0.3, 384),
            model_name="all-MiniLM-L6-v2",
            model_version="v1.0",
        )

        # Create link between t1 and t2
        link = dup_repo.create_duplicate_link(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id_1=t1,
            ticket_id_2=t2,
            similarity_score=0.88,
            model_name="all-MiniLM-L6-v2",
            model_version="v1.0",
        )

        # Delete ticket t1
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tickets WHERE tenant_id = %s AND id = %s;", (tenant_id, t1))
        conn.commit()

        # Embedding for t1 is cascaded
        assert dup_repo.get_ticket_embedding(conn, tenant_id, t1, "all-MiniLM-L6-v2", "v1.0") is None

        # Link between t1 and t2 is cascaded
        assert dup_repo.get_duplicate_link(conn, tenant_id, link.id) is None
        assert dup_repo.get_duplicate_links_for_ticket(conn, tenant_id, t2) == []

        # Ticket t2 remains intact
        t2_record = ticket_repo.get_ticket_by_id(conn, tenant_id, t2)
        assert t2_record is not None
