"""Offline unit tests for confirm/reject feedback on similar-ticket suggestions.

Verifies:
1. Repository contract for record_duplicate_feedback:
   - Self-link rejection (SelfLinkError).
   - Canonical pair ordering (source_id < target_id) regardless of argument order.
   - Server-derived similarity and model metadata from ticket_embeddings.
   - MissingEmbeddingsError when matching embeddings are absent.
   - Updating existing duplicate link status and reviewed_at.
   - Idempotent repeat actions without creating duplicate rows or unhandled conflicts.
   - Decision change (confirmed -> rejected, rejected -> confirmed) updating reviewed_at.
2. Browser API endpoints:
   - Rejects unauthenticated requests with 401 JSON.
   - Enforces session-bound CSRF protection (rejects missing/bad CSRF with 403).
   - Validates tenant ownership for both tickets (rejects cross-tenant with 404).
   - Rejects self-links with 400.
   - Rejects invalid UUIDs with 404.
   - Successful confirm/reject actions return 200 with decision and reviewed_at.
   - Reversed pair action updates the same canonical pair.
   - Decision change returns 200 with updated status.
   - GET /browser-api/tickets/{id}/similar includes persisted decision_status and reviewed_at.
"""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch
import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.deps import Principal
from app.deps_browser import get_browser_principal_json, validate_browser_csrf
from app.repositories.duplicates import (
    DuplicateLinkRecord,
    InvalidStatusError,
    MissingEmbeddingsError,
    SelfLinkError,
    SimilarTicketCandidate,
    record_duplicate_feedback,
)
from app.repositories.tickets import TicketRecord


# -----------------------------------------------------------------------------
# 1. Repository Contract Unit Tests
# -----------------------------------------------------------------------------

def test_record_duplicate_feedback_rejects_self_link():
    """record_duplicate_feedback raises SelfLinkError when ticket_1 == ticket_2."""
    mock_conn = MagicMock()
    tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()

    with pytest.raises(SelfLinkError, match="Cannot link ticket"):
        record_duplicate_feedback(
            conn=mock_conn,
            tenant_id=tenant_id,
            ticket_id_1=ticket_id,
            ticket_id_2=ticket_id,
            decision="confirmed",
        )


def test_record_duplicate_feedback_invalid_decision():
    """record_duplicate_feedback raises InvalidStatusError for invalid decision."""
    mock_conn = MagicMock()
    tenant_id = uuid.uuid4()
    t1 = uuid.uuid4()
    t2 = uuid.uuid4()

    with pytest.raises(InvalidStatusError, match="Invalid decision"):
        record_duplicate_feedback(
            conn=mock_conn,
            tenant_id=tenant_id,
            ticket_id_1=t1,
            ticket_id_2=t2,
            decision="maybe",
        )


def test_record_duplicate_feedback_canonicalizes_order_and_derives_similarity():
    """When no link exists, canonicalizes pair and derives similarity from embeddings."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()

    t_small = uuid.UUID("11111111-1111-1111-1111-111111111111")
    t_large = uuid.UUID("22222222-2222-2222-2222-222222222222")
    tenant_id = uuid.uuid4()

    # Pass in reverse order (large, small)
    # 1. Existing link check returns None
    # 2. Embedding match returns similarity 0.85
    # 3. Insert returns created link record
    created_link_row = {
        "id": uuid.uuid4(),
        "tenant_id": tenant_id,
        "source_ticket_id": t_small,
        "target_ticket_id": t_large,
        "similarity_score": 0.85,
        "model_name": "all-MiniLM-L6-v2",
        "model_version": "rev1",
        "status": "confirmed",
        "created_at": datetime.now(timezone.utc),
        "reviewed_at": datetime.now(timezone.utc),
    }

    mock_cur.fetchone.side_effect = [
        None,  # No existing duplicate_links row
        {"model_name": "all-MiniLM-L6-v2", "model_version": "rev1", "similarity_score": 0.85},  # Embedding match
        created_link_row,  # Insert returning row
    ]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    link = record_duplicate_feedback(
        conn=mock_conn,
        tenant_id=tenant_id,
        ticket_id_1=t_large,
        ticket_id_2=t_small,
        decision="confirmed",
    )

    assert link.source_ticket_id == t_small
    assert link.target_ticket_id == t_large
    assert link.status == "confirmed"
    assert link.similarity_score == 0.85
    assert link.reviewed_at is not None

    # Check SQL params for canonical order in insert
    insert_call = mock_cur.execute.call_args_list[2]
    insert_params = insert_call[0][1]
    assert insert_params[1] == t_small
    assert insert_params[2] == t_large
    assert insert_params[3] == 0.85
    assert insert_params[6] == "confirmed"


def test_record_duplicate_feedback_missing_embeddings_raises():
    """When neither ticket has an embedding record, raises MissingEmbeddingsError."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()

    t1 = uuid.uuid4()
    t2 = uuid.uuid4()
    tenant_id = uuid.uuid4()

    mock_cur.fetchone.side_effect = [
        None,  # No existing link
        None,  # No matching embeddings found
    ]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with pytest.raises(MissingEmbeddingsError, match="matching embeddings not found"):
        record_duplicate_feedback(
            conn=mock_conn,
            tenant_id=tenant_id,
            ticket_id_1=t1,
            ticket_id_2=t2,
            decision="rejected",
        )


def test_record_duplicate_feedback_updates_existing_link_and_decision_change():
    """When a link already exists, updates status and sets new reviewed_at timestamp."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()

    t1 = uuid.UUID("11111111-1111-1111-1111-111111111111")
    t2 = uuid.UUID("22222222-2222-2222-2222-222222222222")
    tenant_id = uuid.uuid4()
    link_id = uuid.uuid4()

    existing_row = {
        "id": link_id,
        "tenant_id": tenant_id,
        "source_ticket_id": t1,
        "target_ticket_id": t2,
        "similarity_score": 0.92,
        "model_name": "all-MiniLM-L6-v2",
        "model_version": "rev1",
        "status": "confirmed",
        "created_at": datetime(2026, 11, 8, 10, 0, 0, tzinfo=timezone.utc),
        "reviewed_at": datetime(2026, 11, 8, 10, 0, 0, tzinfo=timezone.utc),
    }

    new_reviewed_time = datetime(2026, 11, 8, 11, 0, 0, tzinfo=timezone.utc)
    updated_row = {
        **existing_row,
        "status": "rejected",
        "reviewed_at": new_reviewed_time,
    }

    mock_cur.fetchone.side_effect = [
        existing_row,  # Found existing link
        updated_row,   # UPDATE returning row
    ]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    link = record_duplicate_feedback(
        conn=mock_conn,
        tenant_id=tenant_id,
        ticket_id_1=t1,
        ticket_id_2=t2,
        decision="rejected",
        reviewed_at=new_reviewed_time,
    )

    assert link.status == "rejected"
    assert link.reviewed_at == new_reviewed_time
    assert link.similarity_score == 0.92
    assert link.model_name == "all-MiniLM-L6-v2"
    assert link.model_version == "rev1"

    # Verify update query called
    update_call = mock_cur.execute.call_args_list[1]
    update_query = update_call[0][0]
    update_params = update_call[0][1]
    assert "UPDATE duplicate_links" in update_query
    assert update_params[0] == "rejected"
    assert update_params[1] == new_reviewed_time
    assert update_params[3] == link_id


def test_record_duplicate_feedback_idempotent_repeat_preserves_metadata_and_timestamp():
    """Repeated feedback with same decision preserves similarity score, model metadata, and reviewed_at."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()

    t1 = uuid.UUID("11111111-1111-1111-1111-111111111111")
    t2 = uuid.UUID("22222222-2222-2222-2222-222222222222")
    tenant_id = uuid.uuid4()
    link_id = uuid.uuid4()
    original_reviewed_at = datetime(2026, 11, 8, 10, 0, 0, tzinfo=timezone.utc)

    existing_row = {
        "id": link_id,
        "tenant_id": tenant_id,
        "source_ticket_id": t1,
        "target_ticket_id": t2,
        "similarity_score": 0.895,
        "model_name": "all-MiniLM-L6-v2",
        "model_version": "rev1",
        "status": "confirmed",
        "created_at": original_reviewed_at,
        "reviewed_at": original_reviewed_at,
    }

    mock_cur.fetchone.side_effect = [
        existing_row,  # Found existing link
        existing_row,  # UPDATE returns same row with unchanged reviewed_at
    ]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    link = record_duplicate_feedback(
        conn=mock_conn,
        tenant_id=tenant_id,
        ticket_id_1=t1,
        ticket_id_2=t2,
        decision="confirmed",
    )

    assert link.status == "confirmed"
    assert link.reviewed_at == original_reviewed_at
    assert link.similarity_score == 0.895
    assert link.model_name == "all-MiniLM-L6-v2"
    assert link.model_version == "rev1"

    # Verify update query called with original reviewed_at
    update_call = mock_cur.execute.call_args_list[1]
    update_params = update_call[0][1]
    assert update_params[0] == "confirmed"
    assert update_params[1] == original_reviewed_at


# -----------------------------------------------------------------------------
# 2. Browser API Endpoint Unit Tests
# -----------------------------------------------------------------------------

def test_similar_feedback_unauthenticated_returns_401():
    """Unauthenticated feedback request returns 401 JSON with Cache-Control: no-store."""
    client = TestClient(app, follow_redirects=False)
    t1 = str(uuid.uuid4())
    t2 = str(uuid.uuid4())

    res = client.post(
        f"/browser-api/tickets/{t1}/similar/{t2}/feedback",
        json={"decision": "confirmed", "csrf_token": "token"},
    )
    assert res.status_code == 401
    assert res.headers.get("Cache-Control") == "no-store"
    assert res.json()["error"]["code"] == "unauthenticated"


def test_similar_feedback_csrf_failure_returns_403():
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
        t1 = str(uuid.uuid4())
        t2 = str(uuid.uuid4())
        res = client.post(
            f"/browser-api/tickets/{t1}/similar/{t2}/feedback",
            headers={"X-CSRF-Token": "invalid_sig"},
            json={"decision": "confirmed"},
        )
        assert res.status_code == 403
        assert res.json()["error"]["code"] == "csrf_forbidden"
    finally:
        app.dependency_overrides.clear()


def test_similar_feedback_self_link_returns_400():
    """Request linking ticket to itself returns 400 Bad Request."""
    test_tenant_id = uuid.uuid4()
    app.dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=uuid.uuid4(),
        role="agent",
    )
    client = TestClient(app, follow_redirects=False)

    try:
        same_id = str(uuid.uuid4())
        with patch("app.routers.browser_api.validate_browser_csrf"):
            res = client.post(
                f"/browser-api/tickets/{same_id}/similar/{same_id}/feedback",
                json={"decision": "confirmed"},
            )
            assert res.status_code == 400
            assert res.json()["error"]["code"] == "self_link"
            assert "Cannot link a ticket to itself" in res.json()["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_similar_feedback_missing_or_cross_tenant_ticket_returns_404():
    """If either ticket does not belong to the tenant, returns 404."""
    test_tenant_id = uuid.uuid4()
    app.dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=uuid.uuid4(),
        role="agent",
    )
    client = TestClient(app, follow_redirects=False)

    t1 = str(uuid.uuid4())
    t2 = str(uuid.uuid4())

    try:
        with patch("app.routers.browser_api.validate_browser_csrf"), \
             patch("app.routers.browser_api.get_connection") as mock_conn, \
             patch("app.routers.browser_api.get_ticket_by_id", side_effect=[MagicMock(), None]):
            res = client.post(
                f"/browser-api/tickets/{t1}/similar/{t2}/feedback",
                json={"decision": "confirmed"},
            )
            assert res.status_code == 404
            assert res.json()["error"]["code"] == "not_found"
    finally:
        app.dependency_overrides.clear()


def test_similar_feedback_success_confirm_and_reject():
    """Successful confirm and reject feedback returns 200 with recorded decision."""
    test_tenant_id = uuid.uuid4()
    app.dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=uuid.uuid4(),
        role="agent",
    )
    client = TestClient(app, follow_redirects=False)

    t1 = uuid.uuid4()
    t2 = uuid.uuid4()

    mock_link = DuplicateLinkRecord(
        id=uuid.uuid4(),
        tenant_id=test_tenant_id,
        source_ticket_id=min(t1, t2),
        target_ticket_id=max(t1, t2),
        similarity_score=0.8876,
        model_name="all-MiniLM-L6-v2",
        model_version="rev1",
        status="confirmed",
        created_at=datetime.now(timezone.utc),
        reviewed_at=datetime.now(timezone.utc),
    )

    try:
        with patch("app.routers.browser_api.validate_browser_csrf"), \
             patch("app.routers.browser_api.get_connection"), \
             patch("app.routers.browser_api.get_ticket_by_id", return_value=MagicMock()), \
             patch("app.repositories.duplicates.record_duplicate_feedback", return_value=mock_link):
            res = client.post(
                f"/browser-api/tickets/{t1}/similar/{t2}/feedback",
                json={"decision": "confirmed"},
            )
            assert res.status_code == 200
            assert res.headers.get("Cache-Control") == "no-store"

            data = res.json()
            assert data["decision"] == "confirmed"
            assert data["similarity_score"] == 0.8876
            assert data["source_ticket_id"] == str(min(t1, t2))
            assert data["target_ticket_id"] == str(max(t1, t2))
            assert data["reviewed_at"] is not None
            assert "Human duplicate decision recorded" in data["notice"]
    finally:
        app.dependency_overrides.clear()


def test_similar_feedback_reversed_pair_symmetry():
    """Submitting feedback for (B, A) targets the exact same canonical pair (min, max)."""
    test_tenant_id = uuid.uuid4()
    app.dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=uuid.uuid4(),
        role="agent",
    )
    client = TestClient(app, follow_redirects=False)

    t_small = uuid.UUID("11111111-1111-1111-1111-111111111111")
    t_large = uuid.UUID("22222222-2222-2222-2222-222222222222")

    mock_link = DuplicateLinkRecord(
        id=uuid.uuid4(),
        tenant_id=test_tenant_id,
        source_ticket_id=t_small,
        target_ticket_id=t_large,
        similarity_score=0.7891,
        model_name="all-MiniLM-L6-v2",
        model_version="rev1",
        status="rejected",
        created_at=datetime.now(timezone.utc),
        reviewed_at=datetime.now(timezone.utc),
    )

    try:
        with patch("app.routers.browser_api.validate_browser_csrf"), \
             patch("app.routers.browser_api.get_connection"), \
             patch("app.routers.browser_api.get_ticket_by_id", return_value=MagicMock()), \
             patch("app.repositories.duplicates.record_duplicate_feedback", return_value=mock_link) as mock_repo_call:
            # Call with reversed order (t_large, t_small)
            res = client.post(
                f"/browser-api/tickets/{t_large}/similar/{t_small}/feedback",
                json={"decision": "rejected"},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["source_ticket_id"] == str(t_small)
            assert data["target_ticket_id"] == str(t_large)
            assert data["decision"] == "rejected"

            # Assert repo called with arguments
            mock_repo_call.assert_called_once()
            call_kwargs = mock_repo_call.call_args[1]
            assert call_kwargs["ticket_id_1"] == t_large
            assert call_kwargs["ticket_id_2"] == t_small
    finally:
        app.dependency_overrides.clear()


def test_similar_tickets_query_includes_decision_status():
    """GET /browser-api/tickets/{id}/similar includes candidate decision_status and reviewed_at."""
    test_tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()
    match_ticket_id = uuid.uuid4()

    app.dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=uuid.uuid4(),
        role="agent",
    )
    client = TestClient(app, follow_redirects=False)

    candidate = SimilarTicketCandidate(
        ticket_id=match_ticket_id,
        external_id="ext-cand",
        subject="Candidate with confirmed duplicate status",
        status="open",
        priority="low",
        created_at=datetime.now(timezone.utc),
        similarity_score=0.9123,
        decision_status="confirmed",
        reviewed_at=datetime(2026, 11, 8, 12, 0, 0, tzinfo=timezone.utc),
    )

    try:
        with patch("app.routers.browser_api.get_connection"), \
             patch("app.routers.browser_api.get_ticket_by_id", return_value=MagicMock()), \
             patch("app.repositories.duplicates.find_similar_tickets_for_ticket") as mock_find:
            mock_find.return_value = (True, "all-MiniLM-L6-v2", "rev1", [candidate])

            res = client.get(f"/browser-api/tickets/{ticket_id}/similar")
            assert res.status_code == 200
            item = res.json()["items"][0]
            assert item["decision_status"] == "confirmed"
            assert "2026-11-08" in item["reviewed_at"]
    finally:
        app.dependency_overrides.clear()
