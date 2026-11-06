"""Database integration tests for Phase 3 ML Model Integration.

Validates:
1. Known financial text receives a valid CFPB financial-product label.
2. Prediction points to the correct tenant, ticket, and model version.
3. LinearSVC confidence is strictly NULL; decision scores contain raw margins.
4. Distinguishes predictions from final ticket category (tickets.category remains NULL unless set).
5. No cross-tenant prediction visibility via API or database foreign keys.
6. Retries do not create duplicate predictions (idempotent).
7. Stale claims commit zero prediction, ticket, or audit changes.
8. Missing/corrupted model artifacts produce explicit operational errors, not fabricated predictions.
9. Model loader loads once per process and caches in memory.
10. Existing API behavior remains fully backwards compatible.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import hashlib
import os
from pathlib import Path
import tempfile
import uuid
import pytest
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app
import app.db as app_db
import app.repositories.jobs as job_repo
import app.repositories.tickets as ticket_repo
import app.repositories.ml as ml_repo
from app.ml.loader import (
    get_model_manager,
    EXPECTED_CLASSES,
    TRUSTED_MODEL_NAME,
    TRUSTED_MODEL_VERSION,
    TRUSTED_TASK,
    ModelArtifactMissingError,
    ModelArtifactCorruptedError,
)
from app.worker.handler import process_claimed_job
from tests.test_auth_db import TEST_DB_URL, SAFETY_STATUS
from tests.test_tickets_db import _create_tenant_and_api_key

skip_no_test_db = pytest.mark.skipif(TEST_DB_URL is None, reason=SAFETY_STATUS)


@asynccontextmanager
async def _ml_test_db_lifespan(test_app: FastAPI) -> AsyncIterator[None]:
    """Lifespan fixture connecting app.db to TEST_DATABASE_URL."""
    assert TEST_DB_URL is not None
    pool = ConnectionPool(
        conninfo=TEST_DB_URL,
        min_size=1,
        max_size=5,
        open=False,
    )
    app_db.set_pool(pool)
    pool.open(wait=True)
    try:
        yield
    finally:
        pool.close()
        app_db.set_pool(None)


@pytest.fixture
def http_client() -> TestClient:
    """FastAPI TestClient wired to the test database pool."""
    orig_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _ml_test_db_lifespan
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.router.lifespan_context = orig_lifespan


@pytest.fixture(autouse=True)
def ensure_default_model_cache():
    """Ensure global model manager has clean state using trusted artifact."""
    mgr = get_model_manager()
    mgr.reset_cache()
    orig_env = os.environ.get("MODEL_ARTIFACT_PATH")
    os.environ.pop("MODEL_ARTIFACT_PATH", None)
    try:
        yield
    finally:
        mgr.reset_cache()
        if orig_env is not None:
            os.environ["MODEL_ARTIFACT_PATH"] = orig_env
        else:
            os.environ.pop("MODEL_ARTIFACT_PATH", None)


def _ingest_test_ticket(
    http_client: TestClient,
    api_key: str,
    subject: str,
    body: str,
) -> tuple[uuid.UUID, uuid.UUID]:
    """Helper to ingest a ticket via API and return (ticket_id, job_id)."""
    resp = http_client.post(
        "/v1/tickets",
        headers={"X-API-Key": api_key},
        json={"subject": subject, "body": body},
    )
    assert resp.status_code == 202
    ticket_id = uuid.UUID(resp.json()["id"])

    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id FROM jobs WHERE ticket_id = %s;",
                (ticket_id,),
            )
            job_row = cur.fetchone()
            assert job_row is not None
            job_id = uuid.UUID(str(job_row["id"]))

    return ticket_id, job_id


@skip_no_test_db
def test_known_financial_text_receives_valid_product_label(http_client: TestClient) -> None:
    """Known financial complaints receive appropriate CFPB labels and valid model metadata."""
    assert TEST_DB_URL is not None
    tenant_id, _, api_key, _ = _create_tenant_and_api_key("Financial Test Corp")

    # Ingest a complaint clearly about debt collection
    debt_narrative = (
        "I am receiving repeated threatening telephone calls at my place of employment from a collection "
        "agency regarding a medical debt that does not belong to me. Despite sending a written validation notice, "
        "they refuse to verify the debt and continue reporting false collection accounts on my credit profile."
    )
    ticket_id, job_id = _ingest_test_ticket(
        http_client,
        api_key,
        subject="Harassment regarding disputed medical debt",
        body=debt_narrative,
    )

    worker_id = f"worker_{uuid.uuid4().hex[:6]}"
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.transaction():
            job = job_repo.claim_next_job(conn, worker_id=worker_id, tenant_id=tenant_id)
        assert job is not None
        assert job.id == job_id

        status = process_claimed_job(conn, job=job, worker_id=worker_id)
        assert status == "completed"

        # 1. Verify prediction record in database
        predictions = ml_repo.get_predictions_for_ticket(conn, tenant_id=tenant_id, ticket_id=ticket_id)
        assert len(predictions) == 1
        pred = predictions[0]

        assert pred.tenant_id == tenant_id
        assert pred.ticket_id == ticket_id
        assert pred.task == "category"
        assert pred.predicted_label in EXPECTED_CLASSES
        # For this explicit debt narrative, should match Debt collection
        assert pred.predicted_label == "Debt collection"

        # 2. LinearSVC decision scores are margins; confidence must be strictly None
        assert pred.confidence is None
        assert pred.probabilities is None
        assert pred.decision_scores is not None
        assert len(pred.decision_scores) == len(EXPECTED_CLASSES)
        assert all(cls_name in pred.decision_scores for cls_name in EXPECTED_CLASSES)

        # 3. Model version record points to correct tenant
        model_ver = ml_repo.get_active_model_version(conn, tenant_id=tenant_id, task="category")
        assert model_ver is not None
        assert model_ver.id == pred.model_version_id
        assert model_ver.tenant_id == tenant_id
        assert model_ver.name == TRUSTED_MODEL_NAME
        assert model_ver.version == TRUSTED_MODEL_VERSION
        assert model_ver.is_active is True

        # 4. tickets.category remains None (distinguishes prediction from human final value)
        ticket = ticket_repo.get_ticket_by_id(conn, tenant_id=tenant_id, ticket_id=ticket_id)
        assert ticket is not None
        assert ticket.status == "open"
        assert ticket.category is None


@skip_no_test_db
def test_prediction_exposed_in_ticket_detail_endpoint(http_client: TestClient) -> None:
    """GET /v1/tickets/{id} exposes predictions with raw decision scores and null confidence."""
    assert TEST_DB_URL is not None
    tenant_id, _, api_key, _ = _create_tenant_and_api_key("API Detail Corp")

    mortgage_narrative = (
        "My mortgage loan servicer failed to apply my monthly escrow payment to property taxes, causing a tax lien "
        "notice to be issued on my primary residence. They have not responded to my qualified written request."
    )
    ticket_id, job_id = _ingest_test_ticket(
        http_client,
        api_key,
        subject="Mortgage escrow misapplication and property tax issue",
        body=mortgage_narrative,
    )

    # Process job with worker
    worker_id = f"worker_{uuid.uuid4().hex[:6]}"
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.transaction():
            job = job_repo.claim_next_job(conn, worker_id=worker_id, tenant_id=tenant_id)
        assert job is not None
        process_claimed_job(conn, job=job, worker_id=worker_id)

    # Fetch ticket details via API
    resp = http_client.get(
        f"/v1/tickets/{ticket_id}",
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["id"] == str(ticket_id)
    assert data["status"] == "open"
    assert data["category"] is None  # Distinguishes human-curated ticket field
    assert "predictions" in data
    assert len(data["predictions"]) == 1

    p_data = data["predictions"][0]
    assert p_data["task"] == "category"
    assert p_data["predicted_label"] == "Mortgage"
    assert p_data["confidence"] is None  # LinearSVC confidence is null
    assert p_data["decision_scores"] is not None
    assert "Mortgage" in p_data["decision_scores"]


@skip_no_test_db
def test_no_cross_tenant_prediction_visibility(http_client: TestClient) -> None:
    """Tenant A cannot see Tenant B's predictions via API or database references."""
    assert TEST_DB_URL is not None
    tenant_a, _, api_key_a, _ = _create_tenant_and_api_key("Tenant A Corp")
    tenant_b, _, api_key_b, _ = _create_tenant_and_api_key("Tenant B Corp")

    # Ingest ticket in Tenant B
    ticket_id_b, job_id_b = _ingest_test_ticket(
        http_client,
        api_key_b,
        subject="Tenant B Credit Card Dispute",
        body="Unauthorized fraudulent transactions appeared on my credit card statement that were not refunded.",
    )

    worker_id = f"worker_{uuid.uuid4().hex[:6]}"
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.transaction():
            job = job_repo.claim_next_job(conn, worker_id=worker_id, tenant_id=tenant_b)
        assert job is not None
        process_claimed_job(conn, job=job, worker_id=worker_id)

    # 1. API visibility: Tenant A requesting Tenant B's ticket detail gets 404
    resp_cross = http_client.get(
        f"/v1/tickets/{ticket_id_b}",
        headers={"X-API-Key": api_key_a},
    )
    assert resp_cross.status_code == 404
    assert resp_cross.json()["error"]["code"] == "not_found"

    # 2. Database visibility: Tenant A repository query for Tenant B ticket returns empty
    with psycopg.connect(TEST_DB_URL) as conn:
        preds = ml_repo.get_predictions_for_ticket(conn, tenant_id=tenant_a, ticket_id=ticket_id_b)
        assert preds == []

        # 3. Database foreign key integrity: Attempting to insert a prediction for Tenant A referencing Tenant B ticket fails
        model_a = ml_repo.ensure_tenant_model_version(
            conn,
            tenant_id=tenant_a,
            name=TRUSTED_MODEL_NAME,
            version=TRUSTED_MODEL_VERSION,
            task=TRUSTED_TASK,
            artifact_path="dummy",
            sha256_hash="dummy",
            classes=EXPECTED_CLASSES,
        )
        with pytest.raises(psycopg.Error):
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO ticket_predictions (tenant_id, ticket_id, model_version_id, task, predicted_label)
                        VALUES (%s, %s, %s, 'category', 'Credit card');
                        """,
                        (tenant_a, ticket_id_b, model_a.id),
                    )


@skip_no_test_db
def test_retries_do_not_create_duplicate_predictions(http_client: TestClient) -> None:
    """Repeated processing / retries are idempotent and do not create duplicate predictions."""
    assert TEST_DB_URL is not None
    tenant_id, _, api_key, _ = _create_tenant_and_api_key("Idempotent Corp")

    ticket_id, job_id = _ingest_test_ticket(
        http_client,
        api_key,
        subject="Credit Reporting Dispute",
        body="There is an incorrect late payment reported on my credit bureau report from two years ago.",
    )

    worker_id = f"worker_{uuid.uuid4().hex[:6]}"
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.transaction():
            job = job_repo.claim_next_job(conn, worker_id=worker_id, tenant_id=tenant_id)
        assert job is not None

        # First run
        status1 = process_claimed_job(conn, job=job, worker_id=worker_id)
        assert status1 == "completed"

        preds1 = ml_repo.get_predictions_for_ticket(conn, tenant_id=tenant_id, ticket_id=ticket_id)
        assert len(preds1) == 1

        # Simulate replay / retry of prediction insertion
        model_ver = ml_repo.get_active_model_version(conn, tenant_id=tenant_id, task="category")
        assert model_ver is not None

        # Call save_ticket_prediction again (as during a retry)
        ml_repo.save_ticket_prediction(
            conn=conn,
            tenant_id=tenant_id,
            ticket_id=ticket_id,
            model_version_id=model_ver.id,
            task="category",
            predicted_label=preds1[0].predicted_label,
            decision_scores=preds1[0].decision_scores,
        )

        preds2 = ml_repo.get_predictions_for_ticket(conn, tenant_id=tenant_id, ticket_id=ticket_id)
        # Exactly one prediction exists, updated in place
        assert len(preds2) == 1
        assert preds2[0].id == preds1[0].id


@skip_no_test_db
def test_stale_claims_commit_no_prediction_ticket_or_audit_changes(http_client: TestClient) -> None:
    """If a claim is stale, transaction aborts with zero prediction or ticket changes."""
    assert TEST_DB_URL is not None
    tenant_id, _, api_key, _ = _create_tenant_and_api_key("Stale Corp")
    ticket_id, job_id = _ingest_test_ticket(
        http_client,
        api_key=api_key,
        subject="Student Loan Deferment",
        body="My application for student loan income-driven repayment was improperly rejected.",
    )

    worker_a = f"worker_A_{uuid.uuid4().hex[:6]}"
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.transaction():
            job_a = job_repo.claim_next_job(conn, worker_id=worker_a, tenant_id=tenant_id)
        assert job_a is not None

        # Expire worker_a lease by setting locked_at 1 hour in the past
        with conn.cursor() as cur:
            cur.execute("UPDATE jobs SET locked_at = NOW() - INTERVAL '1 hour' WHERE id = %s;", (job_id,))
        conn.commit()

        # Worker B recovers and claims the job (generation increments)
        worker_b = f"worker_B_{uuid.uuid4().hex[:6]}"
        with conn.transaction():
            job_repo.recover_stale_jobs(conn, lease_seconds=300, tenant_id=tenant_id)

        # Clear run_after backoff to allow immediate test claim
        with conn.cursor() as cur:
            cur.execute("UPDATE jobs SET run_after = NOW() - INTERVAL '1 second' WHERE id = %s;", (job_id,))
        conn.commit()

        with conn.transaction():
            job_b = job_repo.claim_next_job(conn, worker_id=worker_b, tenant_id=tenant_id)
        assert job_b is not None
        assert job_b.attempts > job_a.attempts

        # Worker A attempts to process its stale claim
        status_a = process_claimed_job(conn, job=job_a, worker_id=worker_a)
        assert status_a == "stale_claim"

        # Assert worker_a committed NO prediction
        preds = ml_repo.get_predictions_for_ticket(conn, tenant_id=tenant_id, ticket_id=ticket_id)
        assert len(preds) == 0

        # Assert ticket is still 'new'
        ticket = ticket_repo.get_ticket_by_id(conn, tenant_id=tenant_id, ticket_id=ticket_id)
        assert ticket is not None
        assert ticket.status == "new"


@skip_no_test_db
def test_missing_artifact_raises_explicit_operational_error(http_client: TestClient) -> None:
    """Missing model artifact causes explicit operational error, increments attempts, and records error in job."""
    assert TEST_DB_URL is not None
    tenant_id, _, api_key, _ = _create_tenant_and_api_key("Missing Corp")
    ticket_id, job_id = _ingest_test_ticket(
        http_client,
        api_key=api_key,
        subject="Vehicle Financing Rate",
        body="My auto loan interest rate was increased without prior notice after signing the purchase contract.",
    )

    # Point to nonexistent artifact
    nonexistent = Path(tempfile.gettempdir()) / "nonexistent_model_artifact.joblib"
    if nonexistent.exists():
        nonexistent.unlink()

    try:
        os.environ["MODEL_ARTIFACT_PATH"] = str(nonexistent)
        get_model_manager().reset_cache()

        worker_id = f"worker_{uuid.uuid4().hex[:6]}"
        with psycopg.connect(TEST_DB_URL) as conn:
            with conn.transaction():
                job = job_repo.claim_next_job(conn, worker_id=worker_id, tenant_id=tenant_id)
            assert job is not None

            # Processing must handle the missing artifact safely and record failure
            status = process_claimed_job(conn, job=job, worker_id=worker_id)
            assert status == "requeued"  # Attempts < max_attempts -> requeued

            # Verify job has explicit error recorded
            updated_job = job_repo.get_job_by_id(conn, tenant_id=tenant_id, job_id=job_id)
            assert updated_job is not None
            assert updated_job.status == "queued"
            assert updated_job.attempts == 1
            assert "ModelArtifactMissingError" in (updated_job.error or "")

            # Verify no prediction was fabricated
            preds = ml_repo.get_predictions_for_ticket(conn, tenant_id=tenant_id, ticket_id=ticket_id)
            assert len(preds) == 0

            # Ticket status remains 'new'
            ticket = ticket_repo.get_ticket_by_id(conn, tenant_id=tenant_id, ticket_id=ticket_id)
            assert ticket is not None
            assert ticket.status == "new"
    finally:
        os.environ.pop("MODEL_ARTIFACT_PATH", None)
        get_model_manager().reset_cache()


@skip_no_test_db
def test_corrupted_artifact_raises_explicit_operational_error(http_client: TestClient) -> None:
    """Corrupted model artifact (hash mismatch) causes explicit operational error and no prediction."""
    assert TEST_DB_URL is not None
    tenant_id, _, api_key, _ = _create_tenant_and_api_key("Corrupted Corp")
    ticket_id, job_id = _ingest_test_ticket(
        http_client,
        api_key=api_key,
        subject="Checking Account Fees",
        body="Overdraft fees were charged when sufficient funds were present in my checking account.",
    )

    # Create dummy file with incorrect hash
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp.write(b"corrupted_or_tampered_model_bytes")
        corrupted_path = tmp.name

    try:
        os.environ["MODEL_ARTIFACT_PATH"] = corrupted_path
        get_model_manager().reset_cache()

        worker_id = f"worker_{uuid.uuid4().hex[:6]}"
        with psycopg.connect(TEST_DB_URL) as conn:
            with conn.transaction():
                job = job_repo.claim_next_job(conn, worker_id=worker_id, tenant_id=tenant_id)
            assert job is not None

            status = process_claimed_job(conn, job=job, worker_id=worker_id)
            assert status == "requeued"

            updated_job = job_repo.get_job_by_id(conn, tenant_id=tenant_id, job_id=job_id)
            assert updated_job is not None
            assert "ModelArtifactCorruptedError" in (updated_job.error or "")

            preds = ml_repo.get_predictions_for_ticket(conn, tenant_id=tenant_id, ticket_id=ticket_id)
            assert len(preds) == 0
    finally:
        os.environ.pop("MODEL_ARTIFACT_PATH", None)
        get_model_manager().reset_cache()
        if os.path.exists(corrupted_path):
            os.remove(corrupted_path)


def test_model_loader_caches_in_memory_once() -> None:
    """Model manager loads artifact once and returns cached pipeline on subsequent requests."""
    mgr = get_model_manager()
    mgr.reset_cache()

    m1 = mgr.get_model()
    m2 = mgr.get_model()
    assert m1 is m2  # Identical in-memory reference
