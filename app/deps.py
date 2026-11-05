"""Request authentication dependencies resolving the authenticated Principal.

Per SPEC.md Section 6 & 7:
- Principal holds (tenant_id, kind, id, role).
- tenant_id comes exclusively from verified credentials, never from request input.
- Missing, malformed, expired, or invalid credentials return 401 with standard error envelope.
- Non-admin callers trying to access admin-only routes return 403 with standard error envelope.
- Supports both JWT Bearer tokens and API Keys (via X-API-Key or Bearer tk_...).
"""

from dataclasses import dataclass
from typing import Annotated, Literal
import uuid

from fastapi import Depends, Header, HTTPException, status
from app.db import get_connection
import app.repositories.api_keys as api_key_repo
from app.security import decode_access_token, hash_api_key


@dataclass(frozen=True)
class Principal:
    """Resolved authenticated identity."""
    tenant_id: uuid.UUID
    kind: Literal["user", "api_key"]
    id: uuid.UUID
    role: str  # 'admin', 'agent', or 'ingestion' for API keys


def get_jwt_principal(
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    """Extract and validate the Bearer JWT token from the Authorization header.

    Returns:
        Principal: Resolved authenticated user identity.

    Raises:
        HTTPException: 401 if missing, invalid format, expired, or invalid claims.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "missing_authorization_header",
                "message": "Authorization header is required.",
            },
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "invalid_authorization_header",
                "message": "Authorization header must be in format 'Bearer <token>'.",
            },
        )

    token = parts[1]
    try:
        payload = decode_access_token(token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "invalid_token",
                "message": str(exc),
            },
        ) from exc

    return Principal(
        tenant_id=uuid.UUID(payload["tenant_id"]),
        kind="user",
        id=uuid.UUID(payload["sub"]),
        role=payload["role"],
    )


def require_admin(
    principal: Annotated[Principal, Depends(get_jwt_principal)],
) -> Principal:
    """Enforce that the authenticated Principal is a user with the 'admin' role.

    Raises:
        HTTPException: 403 if the principal's role is not 'admin'.
    """
    if principal.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "forbidden",
                "message": "Admin privileges are required for this action.",
            },
        )
    return principal


def get_api_key_principal(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    """Extract and authenticate caller via API key.

    Accepts key from 'X-API-Key: <key>' or 'Authorization: Bearer <key>' where key starts with 'tk_'.
    """
    raw_key: str | None = None

    if x_api_key and x_api_key.strip():
        raw_key = x_api_key.strip()
    elif authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].startswith("tk_"):
            raw_key = parts[1].strip()

    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "missing_api_key",
                "message": "Valid API key is required via 'X-API-Key' or 'Authorization: Bearer tk_...'.",
            },
        )

    if not raw_key.startswith("tk_") or len(raw_key) < 16:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "invalid_api_key",
                "message": "Invalid API key format.",
            },
        )

    key_hash = hash_api_key(raw_key)

    with get_connection() as conn:
        with conn.transaction():
            key_record = api_key_repo.get_active_api_key_by_hash(conn, key_hash)
            if not key_record or key_record.revoked_at is not None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={
                        "code": "invalid_api_key",
                        "message": "Invalid or revoked API key.",
                    },
                )

            # Update last_used_at on successful authentication
            api_key_repo.update_api_key_last_used(conn, key_record.id)

    return Principal(
        tenant_id=key_record.tenant_id,
        kind="api_key",
        id=key_record.id,
        role="ingestion",
    )


def get_any_principal(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    """Resolve caller using either JWT Bearer token or API Key.

    Used by endpoints accepting both credential types (e.g. GET /v1/tickets).
    """
    # 1. Check if X-API-Key header is present
    if x_api_key and x_api_key.strip():
        return get_api_key_principal(x_api_key=x_api_key)

    # 2. Check Authorization header
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            # If Bearer token is an API key (starts with tk_)
            if parts[1].startswith("tk_"):
                return get_api_key_principal(authorization=authorization)
            # Otherwise treat as JWT access token
            return get_jwt_principal(authorization=authorization)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "code": "missing_credentials",
            "message": "Authentication required via JWT access token or API key.",
        },
    )
