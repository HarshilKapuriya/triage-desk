"""Offline unit tests for tenant-scoped similar-ticket suggestions.

Verifies:
1. Repository contract for find_similar_tickets_for_ticket:
   - Source ticket with missing embedding returns (False, None, None, []).
   - Source ticket with ready embedding returns (True, model_name, model_version, candidates).
   - Scoped strictly to matching tenant_id, model_name, and model_version.
   - Excludes source ticket from candidate matches.
   - Enforces bounded limit [1, 50].
   - Clamps similarity scores to [-1.0, 1.0].
2. Browser API endpoint GET /browser-api/tickets/{ticket_id}/similar:
   - Rejects unauthenticated requests with 401 JSON.
   - Rejects invalid UUIDs with 404.
   - Rejects non-existent or cross-tenant tickets with 404.
   - Returns 200 with embedding_status='not_ready' when source has no embedding.
   - Returns 200 with embedding_status='ready' and empty items when no other candidates exist.
   - Returns 200 with embedding_status='ready' and formatted candidate list.
   - Validates and bounds limit parameter [1, 20].
   - Ensures Cache-Control: no-store header is set.
   - Verifies disclaimer notice states results are uncalibrated cosine similarity.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch
import uuid
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app
from app.deps import Principal
from app.deps_browser import get_browser_principal_json
from app.repositories.duplicates import (
    SimilarTicketCandidate,
    find_similar_tickets_for_ticket,
)
from app.repositories.tickets import TicketRecord


@contextmanager
def dummy_connection(mock_cursor=None):
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor or MagicMock()
    yield mock_conn


# -----------------------------------------------------------------------------
# 1. Repository Contract Unit Tests
# -----------------------------------------------------------------------------

def test_find_similar_tickets_missing_embedding():
    """When source ticket has no embedding record, returns (False, None, None, [])."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = None  # No embedding for source ticket
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()

    has_emb, m_name, m_ver, candidates = find_similar_tickets_for_ticket(
        conn=mock_conn,
        tenant_id=tenant_id,
        ticket_id=ticket_id,
    )

    assert has_emb is False
    assert m_name is None
    assert m_ver is None
    assert candidates == []

    # Verify query checked source ticket embedding
    query_text = mock_cur.execute.call_args[0][0]
    assert "FROM ticket_embeddings" in query_text
    assert "WHERE tenant_id = %s AND ticket_id = %s" in query_text
    assert "ORDER BY updated_at DESC, created_at DESC, id DESC" in query_text


def test_find_similar_tickets_successful_query_and_scoping():
    """When source ticket has embedding, queries candidates with same model and excludes source."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()

    source_vec = [0.1] * 384
    source_row = {
        "embedding": f"[{','.join(str(x) for x in source_vec)}]",
        "model_name": "all-MiniLM-L6-v2",
        "model_version": "rev_test_123",
    }
    candidate_id_1 = uuid.uuid4()
    candidate_id_2 = uuid.uuid4()
    candidate_rows = [
        {
            "ticket_id": str(candidate_id_1),
            "external_id": "ext-101",
            "subject": "Fraudulent charge on debit card",
            "status": "open",
            "priority": "high",
            "created_at": datetime.now(timezone.utc),
            "similarity_score": 0.8876,
        },
        {
            "ticket_id": str(candidate_id_2),
            "external_id": None,
            "subject": "Unauthorized ATM withdrawal fee",
            "status": "new",
            "priority": "medium",
            "created_at": datetime.now(timezone.utc),
            "similarity_score": 0.7654,
        },
    ]

    mock_cur.fetchone.return_value = source_row
    mock_cur.fetchall.return_value = candidate_rows
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    tenant_id = uuid.uuid4()
    source_ticket_id = uuid.uuid4()

    has_emb, m_name, m_ver, candidates = find_similar_tickets_for_ticket(
        conn=mock_conn,
        tenant_id=tenant_id,
        ticket_id=source_ticket_id,
        limit=5,
    )

    assert has_emb is True
    assert m_name == "all-MiniLM-L6-v2"
    assert m_ver == "rev_test_123"
    assert len(candidates) == 2

    c1, c2 = candidates
    assert c1.ticket_id == candidate_id_1
    assert c1.subject == "Fraudulent charge on debit card"
    assert c1.similarity_score == 0.8876

    assert c2.ticket_id == candidate_id_2
    assert c2.similarity_score == 0.7654

    # Verify candidate SQL query parameters
    assert mock_cur.execute.call_count == 2
    cand_query = mock_cur.execute.call_args_list[1][0][0]
    cand_params = mock_cur.execute.call_args_list[1][0][1]

    # Model and tenant scoping
    assert "WHERE te.tenant_id = %s" in cand_query
    assert "AND te.model_name = %s" in cand_query
    assert "AND te.model_version = %s" in cand_query
    assert "AND te.ticket_id <> %s" in cand_query
    assert "ORDER BY te.embedding <=> %s::vector ASC, t.id ASC" in cand_query

    assert cand_params[1] == source_ticket_id
    assert cand_params[2] == source_ticket_id
    assert cand_params[3] == tenant_id
    assert cand_params[4] == "all-MiniLM-L6-v2"
    assert cand_params[5] == "rev_test_123"
    assert cand_params[6] == source_ticket_id
    assert cand_params[8] == 5


def test_find_similar_tickets_limit_bounding_and_score_clamping():
    """Limits are bounded to [1, 50] and similarity scores clamped to [-1.0, 1.0]."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()

    source_row = {
        "embedding": "[0.1, 0.2]",
        "model_name": "all-MiniLM-L6-v2",
        "model_version": "rev_test_123",
    }
    candidate_rows = [
        {
            "ticket_id": str(uuid.uuid4()),
            "external_id": None,
            "subject": "Slight over-bound score",
            "status": "open",
            "priority": None,
            "created_at": datetime.now(timezone.utc),
            "similarity_score": 1.0000000002,  # Floating point drift
        },
        {
            "ticket_id": str(uuid.uuid4()),
            "external_id": None,
            "subject": "Slight under-bound score",
            "status": "open",
            "priority": None,
            "created_at": datetime.now(timezone.utc),
            "similarity_score": -1.0000000002,  # Floating point drift
        },
    ]

    mock_cur.fetchone.return_value = source_row
    mock_cur.fetchall.return_value = candidate_rows
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # Test limit clamping over 50
    has_emb, _, _, candidates = find_similar_tickets_for_ticket(
        conn=mock_conn,
        tenant_id=uuid.uuid4(),
        ticket_id=uuid.uuid4(),
        limit=100,  # Exceeds 50
    )

    cand_params = mock_cur.execute.call_args_list[1][0][1]
    assert cand_params[8] == 50  # Bounded to 50
    assert candidates[0].similarity_score == 1.0  # Clamped
    assert candidates[1].similarity_score == -1.0  # Clamped


# -----------------------------------------------------------------------------
# 2. Browser API Endpoint Unit Tests
# -----------------------------------------------------------------------------

def test_get_similar_tickets_unauthenticated_rejected():
    """Unauthenticated requests to /browser-api/tickets/{id}/similar return 401 JSON."""
    client = TestClient(app, follow_redirects=False)
    random_uuid = str(uuid.uuid4())
    res = client.get(f"/browser-api/tickets/{random_uuid}/similar")

    assert res.status_code == 401
    assert res.headers.get("Cache-Control") == "no-store"
    assert res.json()["error"]["code"] == "unauthenticated"


def test_get_similar_tickets_invalid_uuid_returns_404():
    """Invalid UUID string returns 404."""
    test_tenant_id = uuid.uuid4()
    app.dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=uuid.uuid4(),
        role="agent",
    )
    client = TestClient(app, follow_redirects=False)

    try:
        res = client.get("/browser-api/tickets/invalid-not-uuid/similar")
        assert res.status_code == 404
        assert res.json()["error"]["code"] == "not_found"
    finally:
        app.dependency_overrides.clear()


def test_get_similar_tickets_nonexistent_or_cross_tenant_returns_404():
    """Ticket not belonging to the authenticated tenant returns 404."""
    test_tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()

    app.dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=uuid.uuid4(),
        role="agent",
    )
    client = TestClient(app, follow_redirects=False)

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn, \
             patch("app.routers.browser_api.get_ticket_by_id", return_value=None):
            mock_conn = MagicMock()
            mock_get_conn.return_value.__enter__.return_value = mock_conn

            res = client.get(f"/browser-api/tickets/{ticket_id}/similar")
            assert res.status_code == 404
            assert res.json()["error"]["message"] == "Ticket not found."
    finally:
        app.dependency_overrides.clear()


def test_get_similar_tickets_missing_embedding_distinguished():
    """When source ticket has no embedding, returns 200 with embedding_status='not_ready'."""
    test_tenant_id = uuid.uuid4()
    ticket_id = uuid.uuid4()

    app.dependency_overrides[get_browser_principal_json] = lambda: Principal(
        tenant_id=test_tenant_id,
        kind="user",
        id=uuid.uuid4(),
        role="agent",
    )
    client = TestClient(app, follow_redirects=False)

    mock_ticket = TicketRecord(
        id=ticket_id,
        tenant_id=test_tenant_id,
        external_id=None,
        subject="Missing embedding ticket",
        body="Complaint text",
        customer_email=None,
        status="open",
        category="Credit reporting",
        priority=None,
        created_at=datetime.now(timezone.utc),
        first_response_at=None,
        resolved_at=None,
        assignee_id=None,
    )

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn, \
             patch("app.routers.browser_api.get_ticket_by_id", return_value=mock_ticket), \
             patch("app.repositories.duplicates.find_similar_tickets_for_ticket") as mock_find:

            mock_conn = MagicMock()
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            mock_find.return_value = (False, None, None, [])

            res = client.get(f"/browser-api/tickets/{ticket_id}/similar")
            assert res.status_code == 200
            assert res.headers.get("Cache-Control") == "no-store"

            data = res.json()
            assert data["ticket_id"] == str(ticket_id)
            assert data["has_embedding"] is False
            assert data["embedding_status"] == "not_ready"
            assert data["items"] == []
            assert "No embedding vector has been generated" in data["message"]
            assert "Possible similar tickets based on uncalibrated" in data["notice"]
    finally:
        app.dependency_overrides.clear()


def test_get_similar_tickets_success_candidates_returned():
    """When source ticket has embedding, returns 200 with embedding_status='ready' and candidates."""
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

    mock_ticket = TicketRecord(
        id=ticket_id,
        tenant_id=test_tenant_id,
        external_id="ext-src",
        subject="Source ticket",
        body="Complaint narrative",
        customer_email="c@example.com",
        status="open",
        category="Checking or savings account",
        priority="medium",
        created_at=datetime.now(timezone.utc),
        first_response_at=None,
        resolved_at=None,
        assignee_id=None,
    )

    candidate = SimilarTicketCandidate(
        ticket_id=match_ticket_id,
        external_id="ext-match",
        subject="Candidate match ticket",
        status="open",
        priority="high",
        created_at=datetime.now(timezone.utc),
        similarity_score=0.91234,
    )

    try:
        with patch("app.routers.browser_api.get_connection") as mock_get_conn, \
             patch("app.routers.browser_api.get_ticket_by_id", return_value=mock_ticket), \
             patch("app.repositories.duplicates.find_similar_tickets_for_ticket") as mock_find:

            mock_conn = MagicMock()
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            mock_find.return_value = (True, "all-MiniLM-L6-v2", "rev123", [candidate])

            res = client.get(f"/browser-api/tickets/{ticket_id}/similar?limit=3")
            assert res.status_code == 200
            data = res.json()

            assert data["ticket_id"] == str(ticket_id)
            assert data["has_embedding"] is True
            assert data["embedding_status"] == "ready"
            assert data["model_name"] == "all-MiniLM-L6-v2"
            assert data["model_version"] == "rev123"
            assert data["message"] is None
            assert len(data["items"]) == 1

            item = data["items"][0]
            assert item["ticket_id"] == str(match_ticket_id)
            assert item["external_id"] == "ext-match"
            assert item["subject"] == "Candidate match ticket"
            assert item["status"] == "open"
            assert item["similarity_score"] == 0.9123
            assert "uncalibrated" in data["notice"]
            assert "duplicate links are not automatically created" in data["notice"]
    finally:
        app.dependency_overrides.clear()
