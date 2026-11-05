"""API Key router endpoints: POST /v1/api-keys and DELETE /v1/api-keys/{id}.

Per SPEC.md Section 7:
- POST /v1/api-keys: JWT admin only. Accepts label. Returns 201 with {id, label, key} (full key shown once).
- DELETE /v1/api-keys/{id}: JWT admin only. Revokes key (sets revoked_at). Returns 204.
- Cross-tenant requests return 404.
"""

from typing import Annotated
import uuid
from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field, field_validator

from app.db import get_connection
from app.deps import Principal, require_admin
import app.repositories.api_keys as api_key_repo
from app.security import generate_api_key

router = APIRouter(prefix="/v1/api-keys", tags=["API Keys"])


class CreateApiKeyRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=100, description="Descriptive label for the API key.")

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Label must not be empty.")
        return cleaned


class CreateApiKeyResponse(BaseModel):
    id: str
    label: str
    key: str


@router.post(
    "",
    response_model=CreateApiKeyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an API key (Admin only)",
)
def create_api_key(
    payload: CreateApiKeyRequest,
    principal: Annotated[Principal, Depends(require_admin)],
) -> CreateApiKeyResponse:
    """Generate and store an API key for the authenticated admin's tenant.

    The plaintext key is returned once in this response and is never stored or displayed again.
    """
    full_key, key_prefix, key_hash = generate_api_key()

    with get_connection() as conn:
        with conn.transaction():
            key_id = api_key_repo.create_api_key(
                conn=conn,
                tenant_id=principal.tenant_id,
                key_prefix=key_prefix,
                key_hash=key_hash,
                label=payload.label,
            )

    return CreateApiKeyResponse(
        id=str(key_id),
        label=payload.label,
        key=full_key,
    )


@router.delete(
    "/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke an API key (Admin only)",
)
def revoke_api_key(
    key_id: str,
    principal: Annotated[Principal, Depends(require_admin)],
) -> Response:
    """Revoke an API key by ID.

    Sets revoked_at. If the key belongs to another tenant or does not exist,
    returns 404 to avoid leaking existence.
    """
    try:
        parsed_id = uuid.UUID(key_id)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "not_found",
                "message": "API key not found.",
            },
        )

    with get_connection() as conn:
        with conn.transaction():
            revoked = api_key_repo.revoke_api_key(
                conn=conn,
                tenant_id=principal.tenant_id,
                key_id=parsed_id,
            )

    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "not_found",
                "message": "API key not found.",
            },
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)

