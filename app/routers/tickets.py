"""Ticket router: POST /v1/tickets, GET /v1/tickets, and GET /v1/tickets/{id}.

Per SPEC.md Section 7:
- POST /v1/tickets: API key only.
  New ticket: 202 with id and status: "new".
  Existing external_id for same tenant: 200 with existing ticket.
- GET /v1/tickets: JWT or API key.
  Returns list of tickets for caller's tenant, newest first.
  Filters: status, limit (default 50, max 200), offset (>= 0).
- GET /v1/tickets/{id}: JWT or API key.
  Returns ticket with 200, or 404 if missing or another tenant's ticket.
"""

from datetime import datetime
from typing import Annotated
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field, field_validator

from app.db import get_connection
from app.deps import Principal, get_api_key_principal, get_any_principal
import app.repositories.tickets as ticket_repo

router = APIRouter(prefix="/v1/tickets", tags=["Tickets"])


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

    response.status_code = status.HTTP_202_ACCEPTED if is_created else status.HTTP_200_OK
    return _to_response_model(ticket_record)


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

