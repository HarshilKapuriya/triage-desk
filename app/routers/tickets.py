"""Ticket router: POST /v1/tickets, POST /v1/tickets/bulk, GET /v1/tickets, and GET /v1/tickets/{id}.

Per SPEC.md Section 7:
- POST /v1/tickets: API key only.
  New ticket: 202 with id and status: "new".
  Existing external_id for same tenant: 200 with existing ticket.
- POST /v1/tickets/bulk: API key only.
  CSV bulk upload. Atomically creates tickets and enqueues ingest jobs.
  Returns 202 with received, created, replayed counts and ticket IDs.
- GET /v1/tickets: JWT or API key.
  Returns list of tickets for caller's tenant, newest first.
  Filters: status, limit (default 50, max 200), offset (>= 0).
- GET /v1/tickets/{id}: JWT or API key.
  Returns ticket with 200, or 404 if missing or another tenant's ticket.
"""

import csv
from datetime import datetime
import io
from typing import Annotated
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field, field_validator

from app.db import get_connection
from app.deps import Principal, get_api_key_principal, get_any_principal
import app.repositories.tickets as ticket_repo
import app.repositories.jobs as job_repo

router = APIRouter(prefix="/v1/tickets", tags=["Tickets"])

MAX_CSV_BYTES = 5 * 1024 * 1024  # 5 MB upload limit
MAX_CSV_ROWS = 500  # Max 500 rows per batch


class CreateTicketRequest(BaseModel):
    subject: str = Field(..., min_length=1, max_length=300, description="Ticket subject (max 300 chars).")
    body: str = Field(..., min_length=1, max_length=20000, description="Ticket body content (max 20,000 chars).")
    external_id: str | None = Field(default=None, max_length=255, description="Client-provided unique ticket reference.")
    customer_email: str | None = Field(default=None, max_length=255, description="Customer email address.")

    @field_validator("subject", "body")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be blank.")
        return v


class TicketResponse(BaseModel):
    id: str
    tenant_id: str
    external_id: str | None = None
    subject: str
    body: str
    customer_email: str | None = None
    status: str
    category: str | None = None
    priority: str | None = None
    created_at: datetime
    first_response_at: datetime | None = None
    resolved_at: datetime | None = None
    assignee_id: str | None = None


def _to_response_model(record: ticket_repo.TicketRecord) -> TicketResponse:
    """Helper to convert TicketRecord to TicketResponse."""
    return TicketResponse(
        id=str(record.id),
        tenant_id=str(record.tenant_id),
        external_id=record.external_id,
        subject=record.subject,
        body=record.body,
        customer_email=record.customer_email,
        status=record.status,
        category=record.category,
        priority=record.priority,
        created_at=record.created_at,
        first_response_at=record.first_response_at,
        resolved_at=record.resolved_at,
        assignee_id=str(record.assignee_id) if record.assignee_id else None,
    )


@router.post(
    "",
    response_model=TicketResponse,
    summary="Ingest a new support ticket (API key only)",
)
def create_ticket(
    payload: CreateTicketRequest,
    response: Response,
    principal: Annotated[Principal, Depends(get_api_key_principal)],
) -> TicketResponse:
    """Ingest a support ticket.

    Returns HTTP 202 Accepted for newly created tickets.
    Returns HTTP 200 OK with the existing ticket if external_id already exists for this tenant.
    """
    with get_connection() as conn:
        with conn.transaction():
            ticket_record, is_created = ticket_repo.create_or_get_ticket(
                conn=conn,
                tenant_id=principal.tenant_id,
                subject=payload.subject,
                body=payload.body,
                external_id=payload.external_id,
                customer_email=payload.customer_email,
            )
            if is_created:
                job_repo.enqueue_job(
                    conn=conn,
                    tenant_id=principal.tenant_id,
                    ticket_id=ticket_record.id,
                    job_type="ticket_ingest",
                )

    response.status_code = status.HTTP_202_ACCEPTED if is_created else status.HTTP_200_OK
    return _to_response_model(ticket_record)


class BulkTicketResponse(BaseModel):
    received: int
    created: int
    replayed: int
    ticket_ids: list[str]


@router.post(
    "/bulk",
    response_model=BulkTicketResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest tickets via CSV upload (API key only)",
)
async def bulk_create_tickets(
    request: Request,
    principal: Annotated[Principal, Depends(get_api_key_principal)],
) -> BulkTicketResponse:
    """Bulk ingest support tickets from a CSV payload.

    API key authentication only.
    Atomic: validates whole file upfront, then creates tickets and enqueues jobs in a transaction.
    Replays existing external_id without creating duplicate tickets or duplicate jobs.
    """
    byte_chunks: list[bytes] = []
    total_bytes = 0

    async for chunk in request.stream():
        total_bytes += len(chunk)
        if total_bytes > MAX_CSV_BYTES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": "invalid_input",
                    "message": f"CSV payload exceeds maximum size limit of {MAX_CSV_BYTES // (1024 * 1024)} MB.",
                },
            )
        byte_chunks.append(chunk)

    raw_bytes = b"".join(byte_chunks)

    try:
        content = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "invalid_input",
                "message": "CSV content must be valid UTF-8.",
            },
        )

    if not content.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "invalid_input",
                "message": "CSV body is empty.",
            },
        )

    try:
        reader = csv.DictReader(io.StringIO(content))
        headers = reader.fieldnames
        if not headers or "subject" not in headers or "body" not in headers:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": "invalid_input",
                    "message": "CSV missing required column headers ('subject', 'body').",
                },
            )

        validated_rows: list[dict[str, str | None]] = []
        for row_idx, row in enumerate(reader, start=1):
            if row_idx > MAX_CSV_ROWS:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail={
                        "code": "invalid_input",
                        "message": f"CSV row count exceeds maximum limit of {MAX_CSV_ROWS} rows.",
                    },
                )

            subject_raw = row.get("subject")
            subject = subject_raw.strip() if subject_raw else ""
            if not subject:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail={
                        "code": "invalid_input",
                        "message": f"Row {row_idx}: subject is required and cannot be blank.",
                    },
                )
            if len(subject) > 300:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail={
                        "code": "invalid_input",
                        "message": f"Row {row_idx}: subject exceeds maximum length of 300 characters.",
                    },
                )

            body_raw = row.get("body")
            body = body_raw.strip() if body_raw else ""
            if not body:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail={
                        "code": "invalid_input",
                        "message": f"Row {row_idx}: body is required and cannot be blank.",
                    },
                )
            if len(body) > 20000:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail={
                        "code": "invalid_input",
                        "message": f"Row {row_idx}: body exceeds maximum length of 20,000 characters.",
                    },
                )

            ext_id_raw = row.get("external_id")
            ext_id = ext_id_raw.strip() if ext_id_raw else None
            if ext_id:
                if len(ext_id) > 255:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        detail={
                            "code": "invalid_input",
                            "message": f"Row {row_idx}: external_id exceeds maximum length of 255 characters.",
                        },
                    )
            else:
                ext_id = None

            email_raw = row.get("customer_email")
            email = email_raw.strip() if email_raw else None
            if email:
                if len(email) > 255:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        detail={
                            "code": "invalid_input",
                            "message": f"Row {row_idx}: customer_email exceeds maximum length of 255 characters.",
                        },
                    )
            else:
                email = None

            validated_rows.append({
                "subject": subject,
                "body": body,
                "external_id": ext_id,
                "customer_email": email,
            })

    except csv.Error as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "invalid_input",
                "message": f"Malformed CSV syntax: {str(e)}",
            },
        )

    if not validated_rows:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "invalid_input",
                "message": "CSV contains no data rows.",
            },
        )

    ticket_ids: list[str] = []
    created_count = 0
    replayed_count = 0

    with get_connection() as conn:
        with conn.transaction():
            for item in validated_rows:
                ticket_record, is_created = ticket_repo.create_or_get_ticket(
                    conn=conn,
                    tenant_id=principal.tenant_id,
                    subject=item["subject"],  # type: ignore[arg-type]
                    body=item["body"],  # type: ignore[arg-type]
                    external_id=item["external_id"],
                    customer_email=item["customer_email"],
                )
                if is_created:
                    job_repo.enqueue_job(
                        conn=conn,
                        tenant_id=principal.tenant_id,
                        ticket_id=ticket_record.id,
                        job_type="ticket_ingest",
                    )
                    created_count += 1
                else:
                    replayed_count += 1
                ticket_ids.append(str(ticket_record.id))

    return BulkTicketResponse(
        received=len(validated_rows),
        created=created_count,
        replayed=replayed_count,
        ticket_ids=ticket_ids,
    )


@router.get(
    "",
    response_model=list[TicketResponse],
    status_code=status.HTTP_200_OK,
    summary="List tickets (JWT or API key)",
)
def list_tickets(
    principal: Annotated[Principal, Depends(get_any_principal)],
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[TicketResponse]:
    """List tickets belonging to caller's tenant, ordered newest first."""
    with get_connection() as conn:
        records = ticket_repo.list_tickets(
            conn=conn,
            tenant_id=principal.tenant_id,
            status_filter=status_filter,
            limit=limit,
            offset=offset,
        )

    return [_to_response_model(r) for r in records]


@router.get(
    "/{ticket_id}",
    response_model=TicketResponse,
    status_code=status.HTTP_200_OK,
    summary="Get ticket details (JWT or API key)",
)
def get_ticket(
    ticket_id: str,
    principal: Annotated[Principal, Depends(get_any_principal)],
) -> TicketResponse:
    """Retrieve ticket details.

    Returns 404 if ticket does not exist or belongs to another tenant.
    """
    try:
        parsed_id = uuid.UUID(ticket_id)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "not_found",
                "message": "Ticket not found.",
            },
        )

    with get_connection() as conn:
        record = ticket_repo.get_ticket_by_id(
            conn=conn,
            tenant_id=principal.tenant_id,
            ticket_id=parsed_id,
        )

    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "not_found",
                "message": "Ticket not found.",
            },
        )

    return _to_response_model(record)

