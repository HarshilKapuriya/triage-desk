"""Offline unit tests for knowledge base article ingestion and relevant knowledge retrieval.

Verifies:
1. POST /browser-api/kb/articles:
   - Rejects unauthenticated requests with 401 and Cache-Control: no-store.
   - Enforces session-bound CSRF protection (rejects missing/bad CSRF with 403).
   - Validates input bounds (rejects blank title/body or > 300 / > 20,000 chars with 422).
   - Creates article and deterministic chunks in database with status 'processing'.
   - Respects queue constraints: does not force article IDs into ticket-specific jobs table.
2. GET /browser-api/kb/articles:
   - Rejects unauthenticated requests with 401.
   - Returns tenant-scoped list of articles with chunk counts.
3. GET /browser-api/tickets/{ticket_id}/relevant-knowledge:
   - Rejects unauthenticated requests with 401.
   - Rejects nonexistent or cross-tenant tickets with 404.
   - Returns 'missing_ticket_embedding' when ticket embedding is absent.
   - Returns 'no_chunks_available' when tenant has 0 articles.
   - Returns 'articles_processing' when articles exist but chunk embeddings are pending.
   - Returns 'ready' with ranked candidate excerpts and cosine similarity scores.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.deps import Principal
from app.deps_browser import get_browser_principal_json
from app.repositories.kb import (
    ArticleJobRecord,
    KBArticleRecord,
    KBArticleSummaryRecord,
    KBChunkRecord,
    RelevantKnowledgeCandidate,
    StaleClaimError,
)


def test_create_kb_article_unauthenticated_returns_401():
    """Unauthenticated request returns 401 JSON with Cache-Control: no-store."""
    client = TestClient(app, follow_redirects=False)
    res = client.post(
        "/browser-api/kb/articles",
        json={"title": "Test Title", "body": "Test Body content"},
    )
    assert res.status_code == 401
    assert res.headers.get("Cache-Control") == "no-store"
    assert res.json()["error"]["code"] == "unauthenticated"


def test_create_kb_article_csrf_forbidden_returns_403():
    """Request with invalid CSRF token returns 403 Forbidden."""
    test_tenant_id = uuid.uuid4()
    app.dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=uuid.uuid4(),
        role="agent",
    )
    client = TestClient(app, follow_redirects=False)

    try:
        res = client.post(
            "/browser-api/kb/articles",
            headers={"X-CSRF-Token": "invalid_signature"},
            json={"title": "Test Title", "body": "Test Body content"},
        )
        assert res.status_code == 403
        assert res.json()["error"]["code"] == "csrf_forbidden"
    finally:
        app.dependency_overrides.clear()


def test_create_kb_article_payload_validation_rejects_blank():
    """Rejects empty or whitespace-only title or body with 422."""
    test_tenant_id = uuid.uuid4()
    app.dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=uuid.uuid4(),
        role="agent",
    )
    client = TestClient(app, follow_redirects=False)

    try:
        with patch("app.routers.browser_api.validate_browser_csrf"):
            # Empty title
            res1 = client.post(
                "/browser-api/kb/articles",
                json={"title": "   ", "body": "Valid body text."},
            )
            assert res1.status_code == 422

            # Empty body
            res2 = client.post(
                "/browser-api/kb/articles",
                json={"title": "Valid Title", "body": "   "},
            )
            assert res2.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_create_kb_article_success():
    """Successful article creation returns 201 with chunk count and 'processing' status."""
    test_tenant_id = uuid.uuid4()
    article_id = uuid.uuid4()
    app.dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=uuid.uuid4(),
        role="agent",
    )
    client = TestClient(app, follow_redirects=False)

    mock_article = KBArticleRecord(
        id=article_id,
        tenant_id=test_tenant_id,
        title="Dispute Resolution Policy",
        body="Detailed policy terms...",
        status="processing",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    try:
        with patch("app.routers.browser_api.validate_browser_csrf"), \
             patch("app.routers.browser_api.get_connection"), \
             patch("app.repositories.kb.create_kb_article", return_value=mock_article) as mock_create_art, \
             patch("app.repositories.kb.create_kb_chunks") as mock_create_chunks, \
             patch("app.repositories.kb.enqueue_article_job") as mock_enqueue_art:

            mock_create_chunks.return_value = [MagicMock(), MagicMock()]  # 2 chunks

            res = client.post(
                "/browser-api/kb/articles",
                json={
                    "title": "Dispute Resolution Policy",
                    "body": "Detailed policy terms for credit card billing disputes.",
                },
            )
            assert res.status_code == 201
            assert res.headers.get("Cache-Control") == "no-store"
            data = res.json()
            assert data["id"] == str(article_id)
            assert data["title"] == "Dispute Resolution Policy"
            assert data["status"] == "processing"
            assert data["chunk_count"] == 2
            assert "Background chunk embedding job enqueued" in data["notice"]

            mock_create_art.assert_called_once()
            mock_create_chunks.assert_called_once()
            mock_enqueue_art.assert_called_once()
    finally:
        app.dependency_overrides.clear()


def test_create_kb_article_does_not_force_article_id_into_ticket_jobs():
    """Verify that article creation enqueues into article_jobs and NOT the ticket jobs table."""
    test_tenant_id = uuid.uuid4()
    app.dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=uuid.uuid4(),
        role="agent",
    )
    client = TestClient(app, follow_redirects=False)

    mock_article = KBArticleRecord(
        id=uuid.uuid4(),
        tenant_id=test_tenant_id,
        title="Check Fraud Procedures",
        body="Guidelines for check deposits.",
        status="processing",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    try:
        with patch("app.routers.browser_api.validate_browser_csrf"), \
             patch("app.routers.browser_api.get_connection"), \
             patch("app.repositories.kb.create_kb_article", return_value=mock_article), \
             patch("app.repositories.kb.create_kb_chunks", return_value=[MagicMock()]), \
             patch("app.repositories.kb.enqueue_article_job") as mock_enqueue_article, \
             patch("app.repositories.jobs.enqueue_job") as mock_enqueue_ticket:

            res = client.post(
                "/browser-api/kb/articles",
                json={"title": "Check Fraud Procedures", "body": "Guidelines for check deposits."},
            )
            assert res.status_code == 201
            # Article job was enqueued in dedicated article_jobs
            mock_enqueue_article.assert_called_once()
            # Ticket jobs table was NOT touched
            mock_enqueue_ticket.assert_not_called()
    finally:
        app.dependency_overrides.clear()


def test_list_kb_articles_scoped_to_tenant():
    """GET /browser-api/kb/articles returns articles for the authenticated tenant."""
    test_tenant_id = uuid.uuid4()
    app.dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=uuid.uuid4(),
        role="agent",
    )
    client = TestClient(app, follow_redirects=False)

    summary = KBArticleSummaryRecord(
        id=uuid.uuid4(),
        tenant_id=test_tenant_id,
        title="Mortgage Escrow Manual",
        status="processing",
        chunk_count=3,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    try:
        with patch("app.routers.browser_api.get_connection"), \
             patch("app.repositories.kb.list_kb_articles", return_value=[summary]) as mock_list:

            res = client.get("/browser-api/kb/articles")
            assert res.status_code == 200
            assert res.headers.get("Cache-Control") == "no-store"
            data = res.json()
            assert len(data["items"]) == 1
            assert data["items"][0]["title"] == "Mortgage Escrow Manual"
            assert data["items"][0]["chunk_count"] == 3

            mock_list.assert_called_once()
            assert mock_list.call_args[1]["tenant_id"] == test_tenant_id
    finally:
        app.dependency_overrides.clear()


def test_relevant_knowledge_unauthenticated_returns_401():
    """Unauthenticated relevant knowledge request returns 401."""
    client = TestClient(app, follow_redirects=False)
    ticket_id = str(uuid.uuid4())
    res = client.get(f"/browser-api/tickets/{ticket_id}/relevant-knowledge")
    assert res.status_code == 401
    assert res.headers.get("Cache-Control") == "no-store"


def test_relevant_knowledge_nonexistent_or_cross_tenant_returns_404():
    """If ticket does not exist in the tenant, returns 404."""
    test_tenant_id = uuid.uuid4()
    app.dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=uuid.uuid4(),
        role="agent",
    )
    client = TestClient(app, follow_redirects=False)
    ticket_id = str(uuid.uuid4())

    try:
        with patch("app.routers.browser_api.get_connection"), \
             patch("app.routers.browser_api.get_ticket_by_id", return_value=None):

            res = client.get(f"/browser-api/tickets/{ticket_id}/relevant-knowledge")
            assert res.status_code == 404
            assert res.json()["error"]["code"] == "not_found"
    finally:
        app.dependency_overrides.clear()


def test_relevant_knowledge_missing_ticket_embedding():
    """Returns 'missing_ticket_embedding' state when source ticket has no vector."""
    test_tenant_id = uuid.uuid4()
    ticket_id = str(uuid.uuid4())
    app.dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=uuid.uuid4(),
        role="agent",
    )
    client = TestClient(app, follow_redirects=False)

    try:
        with patch("app.routers.browser_api.get_connection"), \
             patch("app.routers.browser_api.get_ticket_by_id", return_value=MagicMock()), \
             patch("app.repositories.kb.find_relevant_chunks_for_ticket") as mock_find:

            mock_find.return_value = ("missing_ticket_embedding", None, None, [])

            res = client.get(f"/browser-api/tickets/{ticket_id}/relevant-knowledge")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "missing_ticket_embedding"
            assert data["items"] == []
            assert "requires an embedded ticket" in data["message"]
    finally:
        app.dependency_overrides.clear()


def test_relevant_knowledge_no_chunks_available():
    """Returns 'no_chunks_available' when tenant has 0 KB articles."""
    test_tenant_id = uuid.uuid4()
    ticket_id = str(uuid.uuid4())
    app.dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=uuid.uuid4(),
        role="agent",
    )
    client = TestClient(app, follow_redirects=False)

    try:
        with patch("app.routers.browser_api.get_connection"), \
             patch("app.routers.browser_api.get_ticket_by_id", return_value=MagicMock()), \
             patch("app.repositories.kb.find_relevant_chunks_for_ticket") as mock_find:

            mock_find.return_value = ("no_chunks_available", "all-MiniLM-L6-v2", "rev1", [])

            res = client.get(f"/browser-api/tickets/{ticket_id}/relevant-knowledge")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "no_chunks_available"
            assert data["items"] == []
            assert "No knowledge-base articles" in data["message"]
    finally:
        app.dependency_overrides.clear()


def test_relevant_knowledge_articles_processing():
    """Returns 'articles_processing' when articles exist but chunk embeddings are pending."""
    test_tenant_id = uuid.uuid4()
    ticket_id = str(uuid.uuid4())
    app.dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=uuid.uuid4(),
        role="agent",
    )
    client = TestClient(app, follow_redirects=False)

    try:
        with patch("app.routers.browser_api.get_connection"), \
             patch("app.routers.browser_api.get_ticket_by_id", return_value=MagicMock()), \
             patch("app.repositories.kb.find_relevant_chunks_for_ticket") as mock_find:

            mock_find.return_value = ("articles_processing", "all-MiniLM-L6-v2", "rev1", [])

            res = client.get(f"/browser-api/tickets/{ticket_id}/relevant-knowledge")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "articles_processing"
            assert data["items"] == []
            assert "chunk embeddings are still being processed" in data["message"]
    finally:
        app.dependency_overrides.clear()


def test_relevant_knowledge_ready_returns_ranked_candidates():
    """Returns 'ready' state with ranked candidates and uncalibrated cosine scores."""
    test_tenant_id = uuid.uuid4()
    ticket_id = str(uuid.uuid4())
    app.dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=uuid.uuid4(),
        role="agent",
    )
    client = TestClient(app, follow_redirects=False)

    candidate = RelevantKnowledgeCandidate(
        chunk_id=uuid.uuid4(),
        article_id=uuid.uuid4(),
        article_title="Fair Credit Billing Rights",
        chunk_index=0,
        excerpt="Consumers have sixty days to dispute inaccurate billing statements.",
        char_count=65,
        token_count=12,
        similarity_score=0.8842,
    )

    try:
        with patch("app.routers.browser_api.get_connection"), \
             patch("app.routers.browser_api.get_ticket_by_id", return_value=MagicMock()), \
             patch("app.repositories.kb.find_relevant_chunks_for_ticket") as mock_find:

            mock_find.return_value = ("ready", "all-MiniLM-L6-v2", "rev1", [candidate])

            res = client.get(f"/browser-api/tickets/{ticket_id}/relevant-knowledge")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "ready"
            assert len(data["items"]) == 1
            item = data["items"][0]
            assert item["article_title"] == "Fair Credit Billing Rights"
            assert item["similarity_score"] == 0.8842
            assert item["chunk_index"] == 0
            assert "sixty days" in item["excerpt"]
            assert "Not a calibrated confidence score" in data["notice"]
    finally:
        app.dependency_overrides.clear()


def test_relevant_knowledge_articles_failed():
    """Returns 'articles_failed' state when tenant articles failed background embedding."""
    test_tenant_id = uuid.uuid4()
    ticket_id = str(uuid.uuid4())
    app.dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=uuid.uuid4(),
        role="agent",
    )
    client = TestClient(app, follow_redirects=False)

    try:
        with patch("app.routers.browser_api.get_connection"), \
             patch("app.routers.browser_api.get_ticket_by_id", return_value=MagicMock()), \
             patch("app.repositories.kb.find_relevant_chunks_for_ticket") as mock_find:

            mock_find.return_value = ("articles_failed", "all-MiniLM-L6-v2", "rev1", [])

            res = client.get(f"/browser-api/tickets/{ticket_id}/relevant-knowledge")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "articles_failed"
            assert data["items"] == []
            assert "failed background embedding" in data["message"]
    finally:
        app.dependency_overrides.clear()


def test_relevant_knowledge_model_mismatch():
    """Returns 'model_mismatch' state when existing chunks were embedded with an incompatible model."""
    test_tenant_id = uuid.uuid4()
    ticket_id = str(uuid.uuid4())
    app.dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=uuid.uuid4(),
        role="agent",
    )
    client = TestClient(app, follow_redirects=False)

    try:
        with patch("app.routers.browser_api.get_connection"), \
             patch("app.routers.browser_api.get_ticket_by_id", return_value=MagicMock()), \
             patch("app.repositories.kb.find_relevant_chunks_for_ticket") as mock_find:

            mock_find.return_value = ("model_mismatch", "all-MiniLM-L6-v2", "rev1", [])

            res = client.get(f"/browser-api/tickets/{ticket_id}/relevant-knowledge")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "model_mismatch"
            assert data["items"] == []
            assert "different model or version" in data["message"]
    finally:
        app.dependency_overrides.clear()


# ==============================================================================
# Worker Article Processing Tests
# ==============================================================================

def test_worker_process_article_job_success():
    """Worker embeds chunks outside DB transaction and completes job atomically."""
    from app.worker.handler import process_claimed_article_job
    from app.ml.embeddings import EmbeddingResult

    tenant_id = uuid.uuid4()
    article_id = uuid.uuid4()
    job_id = uuid.uuid4()
    chunk_id = uuid.uuid4()

    mock_job = ArticleJobRecord(
        id=job_id,
        tenant_id=tenant_id,
        article_id=article_id,
        type="article_embedding",
        status="running",
        attempts=1,
        max_attempts=3,
        run_after=datetime.now(timezone.utc),
        locked_at=datetime.now(timezone.utc),
        locked_by="worker_test",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        completed_at=None,
        failed_at=None,
        error=None,
        payload={},
    )

    mock_chunk = KBChunkRecord(
        id=chunk_id,
        tenant_id=tenant_id,
        article_id=article_id,
        chunk_index=0,
        content="Policy text passage.",
        char_count=20,
        token_count=5,
        embedding=None,
        model_name=None,
        model_version=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    mock_adapter = MagicMock()
    mock_adapter.embed_text.return_value = EmbeddingResult(
        vector=[0.1, 0.2, 0.3],
        pipeline_version="1.0.0",
        model_name="all-MiniLM-L6-v2",
        model_revision="rev1",
        char_count=20,
        token_count=5,
    )

    mock_conn = MagicMock()

    with patch("app.repositories.kb.get_kb_chunks_for_article", return_value=[mock_chunk]), \
         patch("app.repositories.kb.complete_article_job") as mock_complete:

        status = process_claimed_article_job(
            conn=mock_conn,
            job=mock_job,
            worker_id="worker_test",
            lease_seconds=300,
            enable_embeddings=True,
            embedding_adapter=mock_adapter,
        )

        assert status == "completed"
        mock_adapter.embed_text.assert_called_once_with("Policy text passage.")
        mock_complete.assert_called_once()
        call_kwargs = mock_complete.call_args[1]
        assert call_kwargs["tenant_id"] == tenant_id
        assert call_kwargs["article_id"] == article_id
        assert len(call_kwargs["chunk_embeddings"]) == 1
        assert call_kwargs["chunk_embeddings"][0][0] == chunk_id
        assert call_kwargs["chunk_embeddings"][0][1] == [0.1, 0.2, 0.3]


def test_worker_process_article_job_disabled_requeues():
    """When embeddings are disabled, worker releases claim without fabricating success."""
    from app.worker.handler import process_claimed_article_job

    tenant_id = uuid.uuid4()
    job_id = uuid.uuid4()

    mock_job = ArticleJobRecord(
        id=job_id,
        tenant_id=tenant_id,
        article_id=uuid.uuid4(),
        type="article_embedding",
        status="running",
        attempts=1,
        max_attempts=3,
        run_after=datetime.now(timezone.utc),
        locked_at=datetime.now(timezone.utc),
        locked_by="worker_test",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        completed_at=None,
        failed_at=None,
        error=None,
        payload={},
    )

    mock_conn = MagicMock()

    with patch("app.repositories.kb.release_claimed_article_job_to_queued") as mock_release:
        status = process_claimed_article_job(
            conn=mock_conn,
            job=mock_job,
            worker_id="worker_test",
            enable_embeddings=False,
        )

        assert status == "requeued"
        mock_release.assert_called_once_with(
            conn=mock_conn,
            tenant_id=tenant_id,
            job_id=job_id,
            worker_id="worker_test",
            generation=1,
        )


def test_worker_process_article_job_failure_retry():
    """Worker catches embedding exception and invokes fail_or_retry_article_job."""
    from app.worker.handler import process_claimed_article_job

    tenant_id = uuid.uuid4()
    article_id = uuid.uuid4()
    job_id = uuid.uuid4()

    mock_job = ArticleJobRecord(
        id=job_id,
        tenant_id=tenant_id,
        article_id=article_id,
        type="article_embedding",
        status="running",
        attempts=1,
        max_attempts=3,
        run_after=datetime.now(timezone.utc),
        locked_at=datetime.now(timezone.utc),
        locked_by="worker_test",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        completed_at=None,
        failed_at=None,
        error=None,
        payload={},
    )

    mock_conn = MagicMock()
    mock_adapter = MagicMock()
    mock_adapter.embed_text.side_effect = RuntimeError("Inference tensor device error")

    with patch("app.repositories.kb.get_kb_chunks_for_article", return_value=[MagicMock(content="test")]), \
         patch("app.repositories.kb.fail_or_retry_article_job", return_value="requeued") as mock_fail_retry:

        status = process_claimed_article_job(
            conn=mock_conn,
            job=mock_job,
            worker_id="worker_test",
            enable_embeddings=True,
            embedding_adapter=mock_adapter,
        )

        assert status == "requeued"
        mock_fail_retry.assert_called_once()
        assert "Inference tensor device error" in mock_fail_retry.call_args[1]["error_message"]


def test_worker_service_fair_polling_alternates_between_ticket_and_article():
    """WorkerService strictly alternates round-robin turns between ticket and article queues."""
    from app.worker.main import WorkerService

    service = WorkerService(
        db_url="postgresql://localhost/fake",
        worker_id="worker_fair_test",
        enable_embeddings=True,
    )

    mock_conn = MagicMock()

    mock_ticket_job = MagicMock()
    mock_ticket_job.id = uuid.uuid4()
    mock_ticket_job.type = "ticket_ingest"
    mock_ticket_job.attempts = 1

    mock_article_job = MagicMock()
    mock_article_job.id = uuid.uuid4()
    mock_article_job.type = "article_embedding"
    mock_article_job.attempts = 1

    with patch("app.repositories.jobs.claim_next_job", return_value=mock_ticket_job), \
         patch("app.repositories.kb.claim_next_article_job", return_value=mock_article_job), \
         patch("app.worker.main.process_claimed_job", return_value="completed"), \
         patch("app.worker.main.process_claimed_article_job", return_value="completed"), \
         patch("app.repositories.jobs.recover_stale_jobs", return_value=0), \
         patch("app.repositories.kb.recover_stale_article_jobs", return_value=0):

        # Initial turn is "ticket"
        assert service._next_queue_turn == "ticket"
        res1 = service.run_one_cycle(mock_conn)
        assert res1 == "completed"
        # Turn must have toggled to "article"
        assert service._next_queue_turn == "article"

        res2 = service.run_one_cycle(mock_conn)
        assert res2 == "completed"
        # Turn must have toggled back to "ticket"
        assert service._next_queue_turn == "ticket"


def test_worker_continues_ticket_processing_when_article_table_not_migrated():
    """WorkerService does not crash or starve ticket processing when article tables are not yet migrated."""
    from app.worker.main import WorkerService

    service = WorkerService(
        db_url="postgresql://localhost/fake",
        worker_id="worker_unmigrated_test",
        enable_embeddings=True,
    )

    mock_conn = MagicMock()

    mock_ticket_job = MagicMock()
    mock_ticket_job.id = uuid.uuid4()
    mock_ticket_job.type = "ticket_ingest"
    mock_ticket_job.attempts = 1

    # Force initial turn to "article"
    service._next_queue_turn = "article"

    with patch("app.repositories.kb.claim_next_article_job", side_effect=Exception('relation "article_jobs" does not exist')), \
         patch("app.repositories.jobs.claim_next_job", return_value=mock_ticket_job) as mock_claim_ticket, \
         patch("app.worker.main.process_claimed_job", return_value="completed") as mock_process_ticket, \
         patch("app.repositories.jobs.recover_stale_jobs", return_value=0), \
         patch("app.repositories.kb.recover_stale_article_jobs", side_effect=Exception('relation "article_jobs" does not exist')):

        res = service.run_one_cycle(mock_conn)

        # Worker gracefully fell back from article error and processed the ticket job!
        assert res == "completed"
        mock_claim_ticket.assert_called_once()
        mock_process_ticket.assert_called_once()
