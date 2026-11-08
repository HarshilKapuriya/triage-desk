"""Dedicated JSON browser API router for React frontend (/browser-api/*).

Preserves:
- Strict tenant isolation: tenant_id exclusively resolved from verified session claims.
- Cryptographic session-bound CSRF protection (returning 403 on validation failure).
- Never leaks the JWT access token in response bodies or storage.
- Cache-Control: no-store on all browser API responses.
- Explicit JSON 401 unauthenticated errors (never HTML redirects).
"""

from datetime import datetime, timezone
import secrets
from typing import Any
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.config import get_settings
from app.db import get_connection
from app.deps import Principal
from app.deps_browser import (
    BrowserJsonAuthError,
    CsrfValidationError,
    PayloadTooLargeError,
    get_browser_principal_json,
    read_bounded_json_body,
    validate_browser_csrf,
)
from app.repositories.auth import get_user_by_email
import app.repositories.duplicates as dup_repo
import app.repositories.kb as kb_repo
from app.ml.chunking import chunk_article_text, ChunkingError
from app.repositories.jobs import enqueue_job, get_job_by_ticket_id
from app.repositories.ml import get_prediction_with_model_for_ticket
from app.repositories.tickets import (
    count_tickets,
    create_or_get_ticket,
    get_ticket_by_id,
    list_tickets_with_prediction,
)
from app.security import (
    create_access_token,
    decode_access_token,
    derive_session_identifier,
    generate_csrf_token,
    verify_password,
)

router = APIRouter(prefix="/browser-api", tags=["Browser API"])


def derive_processing_status(
    status: str | None,
    attempts: int | None = None,
    max_attempts: int | None = None,
    run_after: datetime | None = None,
) -> dict[str, Any]:
    """Derive user-facing processing state, label, and generic details.

    Maps real job queue statuses strictly to authorized user-facing states:
    - Queued: job enqueued, attempts == 0.
    - Processing: actively running in background worker.
    - Retry Scheduled: failed prior attempt, queued with attempts > 0 and future run_after.
    - Completed: worker finished classification and ticket updates.
    - Failed: permanent failure after retries exhausted (attempts >= max_attempts).
    - No Job Found: no job record exists.

    Security note: Strictly omits raw database error messages, stack traces,
    worker hostnames, and credentials.
    """
    if status is None:
        return {
            "state": "not_found",
            "label": "No Job Found",
            "details": "No background processing job was found for this ticket.",
        }

    if status == "running":
        return {
            "state": "processing",
            "label": "Processing",
            "details": "Worker is currently analyzing the complaint narrative.",
        }

    if status == "completed":
        return {
            "state": "completed",
            "label": "Completed",
            "details": None,
        }

    if status == "failed":
        return {
            "state": "failed",
            "label": "Failed",
            "details": "Processing encountered an error and could not complete automated triage.",
        }

    if status == "queued":
        if attempts is not None and attempts > 0:
            now = datetime.now(timezone.utc)
            is_future = False
            if run_after is not None:
                if run_after.tzinfo is None:
                    is_future = run_after > datetime.utcnow()
                else:
                    is_future = run_after > now

            if is_future:
                return {
                    "state": "retry_scheduled",
                    "label": "Retry Scheduled",
                    "details": "Temporary processing issue encountered; retry scheduled.",
                }
            # When run_after is due or in the past, the job is actively queued waiting for next worker pickup
            return {
                "state": "queued",
                "label": "Queued",
                "details": "Prior attempt encountered an issue; waiting in queue for worker.",
            }
        return {
            "state": "queued",
            "label": "Queued",
            "details": "Waiting in queue for worker classification.",
        }

    return {
        "state": "unknown",
        "label": status.capitalize(),
        "details": None,
    }


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------

class LoginPayload(BaseModel):
    email: str = Field(..., min_length=1, max_length=255, description="Registered user email.")
    password: str = Field(..., min_length=1, max_length=255, description="Plaintext password.")
    csrf_token: str | None = Field(default=None, max_length=255, description="Optional CSRF token in body if not in header.")


class LogoutPayload(BaseModel):
    csrf_token: str | None = Field(default=None, max_length=255, description="Optional CSRF token in body if not in header.")


class CreateBrowserTicketPayload(BaseModel):
    subject: str = Field(..., min_length=1, max_length=300, description="Ticket subject (1-300 chars).")
    body: str = Field(..., min_length=1, max_length=20000, description="Complaint narrative body (1-20,000 chars).")
    csrf_token: str | None = Field(default=None, max_length=255, description="Optional CSRF token in body if not in header.")

    @field_validator("subject", "body")
    @classmethod
    def validate_non_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field must not be blank.")
        return cleaned


class SimilarFeedbackPayload(BaseModel):
    decision: str = Field(..., description="Operator duplicate decision: 'confirmed' or 'rejected'.")
    csrf_token: str | None = Field(default=None, max_length=255, description="Optional CSRF token in body if not in header.")

    @field_validator("decision")
    @classmethod
    def validate_decision(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if cleaned not in ("confirmed", "rejected"):
            raise ValueError("Decision must be either 'confirmed' or 'rejected'.")
        return cleaned


class SimilarFeedbackBodyPayload(SimilarFeedbackPayload):
    similar_ticket_id: str = Field(..., description="Target similar ticket ID.")


class CreateKBArticlePayload(BaseModel):
    title: str = Field(..., min_length=1, max_length=300, description="Article title (1-300 chars).")
    body: str = Field(..., min_length=1, max_length=20000, description="Article body content (1-20,000 chars).")
    csrf_token: str | None = Field(default=None, max_length=255, description="Optional CSRF token in body if not in header.")

    @field_validator("title", "body")
    @classmethod
    def validate_non_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field must not be blank.")
        return cleaned


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/csrf-token")
def get_csrf_token(request: Request, response: Response) -> dict[str, str]:
    """Provide a signed session-bound CSRF token for React.

    Cryptographically checks triage_session before binding to it; if expired or
    corrupted, clears triage_session and seamlessly falls back to pre-login context.
    """
    response.headers["Cache-Control"] = "no-store"
    settings = get_settings()

    # Check if triage_session exists and is cryptographically valid
    triage_token = request.cookies.get("triage_session")
    if triage_token:
        try:
            decode_access_token(triage_token)
            session_id = derive_session_identifier(triage_token)
            token = generate_csrf_token(session_id)
            return {"csrf_token": token}
        except ValueError:
            # Stale / expired cookie: delete on this response and fall through to pre-login
            response.delete_cookie(key="triage_session", path="/")

    # Anonymous pre-login session context
    pre_session = request.cookies.get("pre_login_session")
    if not pre_session:
        pre_session = secrets.token_urlsafe(32)
        response.set_cookie(
            key="pre_login_session",
            value=pre_session,
            httponly=True,
            samesite="lax",
            path="/",
            secure=settings.cookie_secure,
            max_age=1800,
        )

    session_id = derive_session_identifier(pre_session)
    token = generate_csrf_token(session_id)
    return {"csrf_token": token}


@router.post("/auth/login")
async def login(request: Request, response: Response) -> dict[str, Any]:
    """Authenticate browser user with pre-login CSRF validation and issue HttpOnly cookie."""
    response.headers["Cache-Control"] = "no-store"

    raw_data = await read_bounded_json_body(request)
    try:
        payload = LoginPayload.model_validate(raw_data)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc

    # Validate CSRF against pre_login_session cookie
    validate_browser_csrf(request, "pre_login_session", payload.csrf_token)

    email_clean = payload.email.strip().lower()
    with get_connection() as conn:
        user = get_user_by_email(conn, email_clean)

    if not user or not verify_password(user.password_hash, payload.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "invalid_credentials",
                "message": "Invalid email or password.",
            },
        )

    settings = get_settings()
    token = create_access_token(
        user_id=user.id,
        tenant_id=user.tenant_id,
        role=user.role,
    )

    # Derive cookie Max-Age from configured JWT expiry (in minutes)
    max_age_seconds = settings.jwt_expire_minutes * 60

    response.set_cookie(
        key="triage_session",
        value=token,
        httponly=True,
        samesite="lax",
        path="/",
        secure=settings.cookie_secure,
        max_age=max_age_seconds,
    )
    # Clear anonymous pre-login session context
    response.delete_cookie(key="pre_login_session", path="/")

    # Issue fresh authenticated CSRF token bound to the new session
    session_id = derive_session_identifier(token)
    new_csrf_token = generate_csrf_token(session_id)

    # Note: Authentication JWT is strictly excluded from JSON response
    return {
        "user": {
            "id": str(user.id),
            "tenant_id": str(user.tenant_id),
            "role": user.role,
            "email": user.email,
        },
        "csrf_token": new_csrf_token,
    }


@router.post("/auth/logout")
async def logout(
    request: Request,
    response: Response,
    principal: Principal = Depends(get_browser_principal_json),
) -> dict[str, str]:
    """Log out browser session after verifying session-bound CSRF token."""
    response.headers["Cache-Control"] = "no-store"

    raw_data = await read_bounded_json_body(request)
    try:
        payload = LogoutPayload.model_validate(raw_data)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc

    validate_browser_csrf(request, "triage_session", payload.csrf_token)

    response.delete_cookie(key="triage_session", path="/")
    return {"status": "ok"}


@router.get("/auth/session")
def get_session(
    request: Request,
    response: Response,
    principal: Principal = Depends(get_browser_principal_json),
) -> dict[str, Any]:
    """Return active session metadata and refreshed CSRF token for React startup."""
    response.headers["Cache-Control"] = "no-store"

    token = request.cookies.get("triage_session", "")
    session_id = derive_session_identifier(token)
    csrf_token = generate_csrf_token(session_id)

    return {
        "authenticated": True,
        "user": {
            "id": str(principal.id),
            "tenant_id": str(principal.tenant_id),
            "role": principal.role,
        },
        "csrf_token": csrf_token,
    }


@router.get("/tickets")
def list_tickets_json(
    request: Request,
    response: Response,
    page: int = Query(default=1, ge=1, description="Page number (>=1)"),
    per_page: int = Query(default=20, ge=1, le=100, description="Items per page (1-100)"),
    principal: Principal = Depends(get_browser_principal_json),
) -> dict[str, Any]:
    """Retrieve tenant-isolated ticket list with deterministic pagination."""
    response.headers["Cache-Control"] = "no-store"
    offset = (page - 1) * per_page

    with get_connection() as conn:
        total_count = count_tickets(conn, principal.tenant_id)
        items = list_tickets_with_prediction(
            conn,
            principal.tenant_id,
            limit=per_page,
            offset=offset,
        )

    total_pages = max(1, (total_count + per_page - 1) // per_page)

    return {
        "tickets": [
            {
                "id": str(t.id),
                "external_id": t.external_id,
                "subject": t.subject,
                "customer_email": t.customer_email,
                "status": t.status,
                "category": t.category,
                "priority": t.priority,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "predicted_label": t.predicted_label,
                "model_name": t.model_name,
                "model_version": t.model_version,
                "processing_status": derive_processing_status(
                    t.job_status,
                    t.job_attempts,
                    t.job_max_attempts,
                    t.job_run_after,
                ),
            }
            for t in items
        ],
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total_count": total_count,
            "total_pages": total_pages,
        },
    }


@router.post("/tickets", status_code=status.HTTP_201_CREATED)
async def create_ticket_json(
    request: Request,
    response: Response,
    principal: Principal = Depends(get_browser_principal_json),
) -> dict[str, Any]:
    """Create a new support ticket and enqueue background classification job for browser user.

    Strictly derives tenant identity from authenticated browser session principal.
    Enforces session-bound CSRF validation and bounded raw JSON body limit (32 KiB).
    """
    response.headers["Cache-Control"] = "no-store"

    raw_data = await read_bounded_json_body(request)
    try:
        payload = CreateBrowserTicketPayload.model_validate(raw_data)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc

    validate_browser_csrf(request, "triage_session", payload.csrf_token)

    with get_connection() as conn:
        with conn.transaction():
            ticket_record, is_created = create_or_get_ticket(
                conn=conn,
                tenant_id=principal.tenant_id,
                subject=payload.subject,
                body=payload.body,
            )
            if is_created:
                enqueue_job(
                    conn=conn,
                    tenant_id=principal.tenant_id,
                    ticket_id=ticket_record.id,
                    job_type="ticket_ingest",
                )

    return {
        "id": str(ticket_record.id),
        "ticket": {
            "id": str(ticket_record.id),
            "status": ticket_record.status,
            "subject": ticket_record.subject,
            "created_at": ticket_record.created_at.isoformat() if ticket_record.created_at else None,
        },
    }


@router.get("/tickets/{ticket_id}")
def get_ticket_detail_json(
    ticket_id: str,
    request: Request,
    response: Response,
    principal: Principal = Depends(get_browser_principal_json),
) -> dict[str, Any]:
    """Retrieve single ticket detail with model prediction and domain notices."""
    response.headers["Cache-Control"] = "no-store"

    try:
        ticket_uuid = uuid.UUID(ticket_id)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": "Ticket not found."},
        )

    with get_connection() as conn:
        ticket = get_ticket_by_id(conn, principal.tenant_id, ticket_uuid)
        if not ticket:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "not_found", "message": "Ticket not found."},
            )
        prediction = get_prediction_with_model_for_ticket(
            conn,
            principal.tenant_id,
            ticket_uuid,
            task="category",
        )
        job = get_job_by_ticket_id(
            conn,
            principal.tenant_id,
            ticket_uuid,
            job_type="ticket_ingest",
        )

    pred_data = None
    if prediction:
        pred_data = {
            "predicted_label": prediction.predicted_label,
            "model_name": prediction.model_name,
            "model_version": prediction.model_version,
            "confidence": None,
            "confidence_display": "Confidence unavailable",
            "decision_scores": prediction.decision_scores,
            "created_at": prediction.created_at.isoformat() if prediction.created_at else None,
        }

    processing_status = derive_processing_status(
        job.status if job else None,
        job.attempts if job else None,
        job.max_attempts if job else None,
        job.run_after if job else None,
    )

    return {
        "ticket": {
            "id": str(ticket.id),
            "external_id": ticket.external_id,
            "subject": ticket.subject,
            "body": ticket.body,
            "customer_email": ticket.customer_email,
            "status": ticket.status,
            "category": ticket.category,
            "priority": ticket.priority,
            "created_at": ticket.created_at.isoformat() if ticket.created_at else None,
        },
        "prediction": pred_data,
        "processing": processing_status,
        "notices": {
            "domain_notice": "CFPB financial product classifier (run_004 baseline). Does not detect out-of-domain inquiries.",
            "confidence_notice": "LinearSVC margins represent uncalibrated hyperplane distances, not probabilities.",
            "priority_notice": "Priority prediction deferred.",
        },
    }


@router.get("/tickets/{ticket_id}/similar")
def get_similar_tickets_json(
    ticket_id: str,
    request: Request,
    response: Response,
    limit: int = Query(default=5, ge=1, le=20),
    principal: Principal = Depends(get_browser_principal_json),
) -> dict[str, Any]:
    """Retrieve top-k similar ticket suggestions for a ticket within the authenticated tenant.

    - Exact cosine-similarity retrieval using existing ticket_embeddings.
    - Scoped strictly to authenticated tenant and same model_name and model_version.
    - Excludes the source ticket.
    - Returns bounded, deterministically ordered list with ticket summaries and similarity scores.
    - Distinguishes missing/not-ready embeddings vs zero candidates.
    - Strictly describes results as possible similar tickets, not confirmed duplicates.
    """
    response.headers["Cache-Control"] = "no-store"

    try:
        ticket_uuid = uuid.UUID(ticket_id)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": "Ticket not found."},
        )

    with get_connection() as conn:
        # Verify ticket exists within this tenant
        ticket = get_ticket_by_id(conn, principal.tenant_id, ticket_uuid)
        if not ticket:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "not_found", "message": "Ticket not found."},
            )

        has_embedding, model_name, model_version, candidates = dup_repo.find_similar_tickets_for_ticket(
            conn=conn,
            tenant_id=principal.tenant_id,
            ticket_id=ticket_uuid,
            limit=limit,
        )

    items = [
        {
            "ticket_id": str(c.ticket_id),
            "external_id": c.external_id,
            "subject": c.subject,
            "status": c.status,
            "priority": c.priority,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "similarity_score": round(c.similarity_score, 4),
            "decision_status": c.decision_status,
            "reviewed_at": c.reviewed_at.isoformat() if c.reviewed_at else None,
        }
        for c in candidates
    ]

    return {
        "ticket_id": str(ticket_uuid),
        "has_embedding": has_embedding,
        "embedding_status": "ready" if has_embedding else "not_ready",
        "model_name": model_name,
        "model_version": model_version,
        "items": items,
        "notice": "Possible similar tickets based on uncalibrated cosine text similarity. Relevance quality is not evaluated; duplicate links are not automatically created.",
        "message": None if has_embedding else "No embedding vector has been generated for this ticket yet. Similar ticket suggestions will become available once background embedding processing completes.",
    }


@router.post("/tickets/{ticket_id}/similar/{similar_ticket_id}/feedback")
async def record_similar_feedback_json(
    ticket_id: str,
    similar_ticket_id: str,
    request: Request,
    response: Response,
    principal: Principal = Depends(get_browser_principal_json),
) -> dict[str, Any]:
    """Record operator confirm or reject duplicate feedback for a suggested ticket pair.

    Rules:
    - Session-bound CSRF validation.
    - Tenant ownership validation for both tickets.
    - Self-link rejection (returns 400).
    - Canonical pair ordering in duplicate_links (source < target).
    - Server-derived similarity and embedding model metadata (never trust client scores).
    - Idempotent repeat actions without duplicate rows.
    - Updates reviewed_at when decision changes.
    - Human decision only: does not merge, delete, or alter tickets.
    """
    response.headers["Cache-Control"] = "no-store"

    try:
        t1_uuid = uuid.UUID(ticket_id)
        t2_uuid = uuid.UUID(similar_ticket_id)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": "Ticket not found."},
        )

    if t1_uuid == t2_uuid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "self_link", "message": "Cannot link a ticket to itself."},
        )

    raw_data = await read_bounded_json_body(request)
    try:
        payload = SimilarFeedbackPayload.model_validate(raw_data)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc

    validate_browser_csrf(request, "triage_session", payload.csrf_token)

    with get_connection() as conn:
        # Validate tenant ownership for both tickets
        t1 = get_ticket_by_id(conn, principal.tenant_id, t1_uuid)
        t2 = get_ticket_by_id(conn, principal.tenant_id, t2_uuid)
        if not t1 or not t2:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "not_found", "message": "Ticket not found."},
            )

        try:
            with conn.transaction():
                link = dup_repo.record_duplicate_feedback(
                    conn=conn,
                    tenant_id=principal.tenant_id,
                    ticket_id_1=t1_uuid,
                    ticket_id_2=t2_uuid,
                    decision=payload.decision,
                )
        except dup_repo.MissingEmbeddingsError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "missing_embeddings", "message": str(err)},
            )
        except dup_repo.SelfLinkError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "self_link", "message": str(err)},
            )
        except dup_repo.InvalidStatusError as err:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "invalid_input", "message": str(err)},
            )

    return {
        "link_id": str(link.id),
        "source_ticket_id": str(link.source_ticket_id),
        "target_ticket_id": str(link.target_ticket_id),
        "similarity_score": round(link.similarity_score, 4),
        "model_name": link.model_name,
        "model_version": link.model_version,
        "decision": link.status,
        "reviewed_at": link.reviewed_at.isoformat() if link.reviewed_at else None,
        "notice": "Human duplicate decision recorded. This does not alter ticket contents or automatically retrain models.",
    }


@router.post("/tickets/{ticket_id}/similar-feedback")
async def record_similar_feedback_body_json(
    ticket_id: str,
    request: Request,
    response: Response,
    principal: Principal = Depends(get_browser_principal_json),
) -> dict[str, Any]:
    """Alternative endpoint accepting similar_ticket_id in JSON payload."""
    response.headers["Cache-Control"] = "no-store"

    raw_data = await read_bounded_json_body(request)
    try:
        payload = SimilarFeedbackBodyPayload.model_validate(raw_data)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc

    try:
        t1_uuid = uuid.UUID(ticket_id)
        t2_uuid = uuid.UUID(payload.similar_ticket_id)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": "Ticket not found."},
        )

    if t1_uuid == t2_uuid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "self_link", "message": "Cannot link a ticket to itself."},
        )

    validate_browser_csrf(request, "triage_session", payload.csrf_token)

    with get_connection() as conn:
        t1 = get_ticket_by_id(conn, principal.tenant_id, t1_uuid)
        t2 = get_ticket_by_id(conn, principal.tenant_id, t2_uuid)
        if not t1 or not t2:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "not_found", "message": "Ticket not found."},
            )

        try:
            with conn.transaction():
                link = dup_repo.record_duplicate_feedback(
                    conn=conn,
                    tenant_id=principal.tenant_id,
                    ticket_id_1=t1_uuid,
                    ticket_id_2=t2_uuid,
                    decision=payload.decision,
                )
        except dup_repo.MissingEmbeddingsError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "missing_embeddings", "message": str(err)},
            )
        except dup_repo.SelfLinkError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "self_link", "message": str(err)},
            )
        except dup_repo.InvalidStatusError as err:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "invalid_input", "message": str(err)},
            )

    return {
        "link_id": str(link.id),
        "source_ticket_id": str(link.source_ticket_id),
        "target_ticket_id": str(link.target_ticket_id),
        "similarity_score": round(link.similarity_score, 4),
        "model_name": link.model_name,
        "model_version": link.model_version,
        "decision": link.status,
        "reviewed_at": link.reviewed_at.isoformat() if link.reviewed_at else None,
        "notice": "Human duplicate decision recorded. This does not alter ticket contents or automatically retrain models.",
    }


# ---------------------------------------------------------------------------
# Knowledge Base Articles & Retrieval Endpoints
# ---------------------------------------------------------------------------

@router.post("/kb/articles", status_code=status.HTTP_201_CREATED)
async def create_kb_article_json(
    request: Request,
    response: Response,
    principal: Principal = Depends(get_browser_principal_json),
) -> dict[str, Any]:
    """Create a tenant-owned knowledge-base article and partition into deterministic chunks.

    Guarantees:
    - Authenticated session principal.
    - CSRF protection validated against active session.
    - 32 KiB raw request body bound enforced via stream byte counting.
    - Title (1-300 chars) and body (1-20,000 chars) validated.
    - Deterministic chunking respecting model tokenizer length (TOKENIZER_MAX_LENGTH=256).
    - Composite foreign key (tenant_id, article_id) enforced on chunks.
    - Queue constraint: Does not force article IDs into ticket-specific jobs queue.
    """
    response.headers["Cache-Control"] = "no-store"

    raw_data = await read_bounded_json_body(request)
    try:
        payload = CreateKBArticlePayload.model_validate(raw_data)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc

    validate_browser_csrf(request, "triage_session", payload.csrf_token)

    try:
        chunks = chunk_article_text(title=payload.title, body=payload.body)
    except ChunkingError as err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_article", "message": str(err)},
        )

    with get_connection() as conn:
        with conn.transaction():
            article = kb_repo.create_kb_article(
                conn=conn,
                tenant_id=principal.tenant_id,
                title=payload.title,
                body=payload.body,
                status="processing",
            )
            created_chunks = kb_repo.create_kb_chunks(
                conn=conn,
                tenant_id=principal.tenant_id,
                article_id=article.id,
                chunks=chunks,
            )
            job = kb_repo.enqueue_article_job(
                conn=conn,
                tenant_id=principal.tenant_id,
                article_id=article.id,
                job_type="article_embedding",
            )

    return {
        "id": str(article.id),
        "title": article.title,
        "status": article.status,
        "chunk_count": len(created_chunks),
        "created_at": article.created_at.isoformat(),
        "notice": "Article ingested and partitioned into deterministic chunks. Background chunk embedding job enqueued.",
    }


@router.get("/kb/articles")
def list_kb_articles_json(
    request: Request,
    response: Response,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    principal: Principal = Depends(get_browser_principal_json),
) -> dict[str, Any]:
    """List tenant-owned knowledge-base articles with chunk counts and processing states."""
    response.headers["Cache-Control"] = "no-store"

    with get_connection() as conn:
        articles = kb_repo.list_kb_articles(
            conn=conn,
            tenant_id=principal.tenant_id,
            limit=limit,
            offset=offset,
        )

    return {
        "items": [
            {
                "id": str(a.id),
                "title": a.title,
                "status": a.status,
                "chunk_count": a.chunk_count,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "updated_at": a.updated_at.isoformat() if a.updated_at else None,
            }
            for a in articles
        ]
    }


@router.get("/tickets/{ticket_id}/relevant-knowledge")
def get_relevant_knowledge_json(
    ticket_id: str,
    request: Request,
    response: Response,
    limit: int = Query(default=5, ge=1, le=20),
    principal: Principal = Depends(get_browser_principal_json),
) -> dict[str, Any]:
    """Retrieve top-k relevant knowledge-base excerpts for a ticket using exact cosine similarity.

    Distinct States:
    - 'missing_ticket_embedding': Ticket has no embedding vector in ticket_embeddings.
    - 'no_chunks_available': Tenant has no knowledge-base articles or chunks.
    - 'articles_processing': Articles/chunks exist, but embeddings are still pending/processing.
    - 'ready': Retrieval succeeded (may return 0 items if none qualify).
    """
    response.headers["Cache-Control"] = "no-store"

    try:
        ticket_uuid = uuid.UUID(ticket_id)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": "Ticket not found."},
        )

    with get_connection() as conn:
        ticket = get_ticket_by_id(conn, principal.tenant_id, ticket_uuid)
        if not ticket:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "not_found", "message": "Ticket not found."},
            )

        retrieval_status, model_name, model_version, candidates = kb_repo.find_relevant_chunks_for_ticket(
            conn=conn,
            tenant_id=principal.tenant_id,
            ticket_id=ticket_uuid,
            limit=limit,
        )

    messages = {
        "missing_ticket_embedding": "No embedding vector has been generated for this ticket yet. Knowledge retrieval requires an embedded ticket.",
        "no_chunks_available": "No knowledge-base articles or passages exist for this tenant.",
        "articles_processing": "Knowledge-base articles have been ingested, but chunk embeddings are still being processed or pending queue activation.",
        "articles_failed": "Tenant knowledge-base articles failed background embedding. Please check the Knowledge Base view.",
        "model_mismatch": "Existing knowledge-base chunks were generated using a different model or version than this ticket.",
        "ready": None,
    }

    items = [
        {
            "chunk_id": str(c.chunk_id),
            "article_id": str(c.article_id),
            "article_title": c.article_title,
            "chunk_index": c.chunk_index,
            "excerpt": c.excerpt,
            "char_count": c.char_count,
            "token_count": c.token_count,
            "similarity_score": round(c.similarity_score, 4),
        }
        for c in candidates
    ]

    return {
        "ticket_id": str(ticket_uuid),
        "status": retrieval_status,
        "model_name": model_name,
        "model_version": model_version,
        "items": items,
        "notice": "Candidate knowledge-base excerpts based on exact cosine similarity against this ticket's embedding vector. Not a calibrated confidence score.",
        "message": messages.get(retrieval_status),
    }
