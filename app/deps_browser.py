"""Browser authentication and CSRF dependencies for /browser-api/*.

Enforces:
- Strict JSON error responses (never redirects fetch to HTML).
- Cookie clearance on authentication failure.
- CSRF validation returning 403 Forbidden on failure.
- Extraction of tenant_id strictly from verified JWT claims.
- Bounded input validation.
"""

import json
from typing import Annotated, Any
import uuid

from fastapi import HTTPException, Request, Response, status
from app.deps import Principal
from app.security import decode_access_token, derive_session_identifier, verify_csrf_token


MAX_BROWSER_REQUEST_BYTES = 32 * 1024  # 32 KiB


class BrowserJsonAuthError(Exception):
    """Raised when browser session cookie is missing, expired, or invalid."""
    def __init__(self, message: str = "Authentication required via session cookie."):
        self.message = message
        super().__init__(message)


class CsrfValidationError(Exception):
    """Raised when CSRF token verification fails."""
    def __init__(self, message: str = "Invalid or expired CSRF token."):
        self.message = message
        super().__init__(message)


class PayloadTooLargeError(Exception):
    """Raised when incoming raw request body exceeds the maximum permitted byte size."""
    def __init__(self, message: str = f"Request body exceeds maximum allowed size of {MAX_BROWSER_REQUEST_BYTES} bytes."):
        self.message = message
        super().__init__(message)


def get_browser_principal_json(request: Request) -> Principal:
    """Resolve authenticated user principal from HttpOnly triage_session cookie.

    Raises:
        BrowserJsonAuthError: If cookie is missing, expired, or invalid.
    """
    token = request.cookies.get("triage_session")
    if not token:
        raise BrowserJsonAuthError("Authentication required via session cookie.")

    try:
        payload = decode_access_token(token)
    except ValueError as exc:
        raise BrowserJsonAuthError("Session expired or invalid.") from exc

    return Principal(
        tenant_id=uuid.UUID(payload["tenant_id"]),
        kind="user",
        id=uuid.UUID(payload["sub"]),
        role=payload["role"],
    )


def validate_browser_csrf(
    request: Request,
    session_cookie_name: str,
    csrf_token_header_or_body: str | None = None,
) -> None:
    """Validate signed session-bound CSRF token against the given session cookie.

    Checks X-CSRF-Token header first, then the optional token argument.
    Raises:
        CsrfValidationError: If session cookie is missing or token is invalid/expired.
    """
    cookie_value = request.cookies.get(session_cookie_name)
    if not cookie_value:
        raise CsrfValidationError("Missing required session context for CSRF validation.")

    token = request.headers.get("X-CSRF-Token") or csrf_token_header_or_body
    if not token:
        raise CsrfValidationError("Missing CSRF token.")

    session_id = derive_session_identifier(cookie_value)
    if not verify_csrf_token(token, session_id):
        raise CsrfValidationError("Invalid or expired CSRF token.")


async def read_bounded_json_body(
    request: Request,
    max_bytes: int = MAX_BROWSER_REQUEST_BYTES,
) -> dict[str, Any]:
    """Read and validate raw request body chunks up to max_bytes before JSON deserialization.

    Counts actual streamed bytes and does not rely solely on Content-Length.
    Raises:
        PayloadTooLargeError: If Content-Length or actual streamed bytes exceed max_bytes.
        HTTPException(400): If Content-Length header is malformed.
        HTTPException(422): If body is malformed JSON or not a JSON object.
    """
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            cl = int(content_length)
            if cl > max_bytes:
                raise PayloadTooLargeError(f"Request body exceeds maximum allowed size of {max_bytes} bytes.")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "invalid_input", "message": "Invalid Content-Length header."},
            )

    body_chunks: list[bytes] = []
    total_bytes = 0
    async for chunk in request.stream():
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise PayloadTooLargeError(f"Request body exceeds maximum allowed size of {max_bytes} bytes.")
        body_chunks.append(chunk)

    raw_bytes = b"".join(body_chunks)
    if not raw_bytes:
        return {}

    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_input", "message": "Malformed JSON payload."},
        ) from exc

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_input", "message": "JSON body must be an object."},
        )

    return data
