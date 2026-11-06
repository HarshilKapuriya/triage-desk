"""Security utilities for password hashing, JWT operations, and API-key management.

Never log passwords, hashes, tokens, or plaintext API keys.
"""

from datetime import datetime, timezone, timedelta
import hashlib
import hmac
import secrets
import time
from typing import Any
import uuid

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHashError, VerificationError
import jwt

from app.config import get_settings

# Initialize Argon2 password hasher
_ph = PasswordHasher()

# Allowed roles for token claims
SUPPORTED_ROLES = frozenset({"admin", "agent"})

# Enforced algorithm
JWT_ALGORITHM = "HS256"


# ---------------------------------------------------------------------------
# 1. Password Utilities
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Hash a plaintext password using Argon2.

    Returns the salted hash string. Does not log passwords or hashes.
    """
    if not isinstance(password, str) or not password:
        raise ValueError("Password must be a non-empty string.")
    return _ph.hash(password)


def verify_password(hash_value: str, password: str) -> bool:
    """Verify a password against an Argon2 hash.

    Returns True if the password matches, False if incorrect or if the hash
    is malformed / invalid. Never raises on invalid credentials.
    """
    if not isinstance(hash_value, str) or not isinstance(password, str):
        return False
    try:
        return _ph.verify(hash_value, password)
    except (VerifyMismatchError, InvalidHashError, VerificationError):
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 2. JWT Utilities
# ---------------------------------------------------------------------------

def create_access_token(
    user_id: str | uuid.UUID,
    tenant_id: str | uuid.UUID,
    role: str,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed HS256 JWT access token.

    Claims:
        sub: user ID as string UUID
        tenant_id: tenant ID as string UUID
        role: user role ('admin' or 'agent')
        iat: issued-at timestamp (UTC)
        exp: expiration timestamp (UTC)
    """
    # Validate UUID formats
    sub_str = str(user_id)
    tenant_id_str = str(tenant_id)
    try:
        uuid.UUID(sub_str)
        uuid.UUID(tenant_id_str)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("user_id and tenant_id must be valid UUIDs.") from exc

    # Validate role
    if role not in SUPPORTED_ROLES:
        raise ValueError(f"Unsupported role '{role}'. Supported roles: {sorted(SUPPORTED_ROLES)}.")

    settings = get_settings()
    secret = settings.jwt_secret.get_secret_value()

    now = datetime.now(timezone.utc)
    if expires_delta is not None:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.jwt_expire_minutes)

    payload: dict[str, Any] = {
        "sub": sub_str,
        "tenant_id": tenant_id_str,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }

    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and strictly validate an HS256 JWT access token.

    Validates:
        - Signature using configured JWT_SECRET
        - Explicitly allows only ["HS256"]
        - Token expiration (exp)
        - Presence and valid UUID format of 'sub' and 'tenant_id'
        - Presence and validity of 'role'

    Raises:
        ValueError: If the token is invalid, expired, tampered, or missing claims.
    """
    if not isinstance(token, str) or not token:
        raise ValueError("Token must be a non-empty string.")

    settings = get_settings()
    secret = settings.jwt_secret.get_secret_value()

    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[JWT_ALGORITHM],
            options={
                "require": ["sub", "tenant_id", "role", "iat", "exp"],
                "verify_exp": True,
                "verify_iat": True,
            },
        )
    except jwt.ExpiredSignatureError as exc:
        raise ValueError("Token has expired.") from exc
    except jwt.InvalidSignatureError as exc:
        raise ValueError("Invalid token signature.") from exc
    except jwt.InvalidAlgorithmError as exc:
        raise ValueError("Unsupported token algorithm.") from exc
    except jwt.DecodeError as exc:
        raise ValueError("Malformed token.") from exc
    except jwt.PyJWTError as exc:
        raise ValueError(f"Invalid token: {exc}") from exc

    # Validate UUID format of 'sub' and 'tenant_id'
    try:
        uuid.UUID(str(payload.get("sub")))
        uuid.UUID(str(payload.get("tenant_id")))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Token contains invalid UUID claims.") from exc

    # Validate role
    role = payload.get("role")
    if role not in SUPPORTED_ROLES:
        raise ValueError(f"Token contains unsupported role '{role}'.")

    return payload


# ---------------------------------------------------------------------------
# 3. API-Key Utilities
# ---------------------------------------------------------------------------

def generate_api_key() -> tuple[str, str, str]:
    """Generate a new secure API key.

    Returns:
        tuple of (full_key, key_prefix, key_hash):
            full_key: 'tk_' + secrets.token_urlsafe(32) (shown once to user)
            key_prefix: first 8 characters of full_key (for display)
            key_hash: SHA-256 hexadecimal digest of full_key (for DB storage)
    """
    full_key = f"tk_{secrets.token_urlsafe(32)}"
    key_prefix = full_key[:8]
    key_hash = hash_api_key(full_key)
    return full_key, key_prefix, key_hash


def hash_api_key(key: str) -> str:
    """Compute the SHA-256 hex digest of an API key for secure storage / lookup.

    Does not log plaintext keys.
    """
    if not isinstance(key, str) or not key:
        raise ValueError("API key must be a non-empty string.")
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 4. Browser CSRF & Session Binding Utilities
# ---------------------------------------------------------------------------

def derive_session_identifier(raw_session_token: str) -> str:
    """Derive an opaque session identifier from a session token or pre-login token.

    Uses HMAC-SHA256 with the server's secret key so the raw token (such as a JWT)
    is never exposed in CSRF tokens or logs.
    """
    if not isinstance(raw_session_token, str) or not raw_session_token:
        raise ValueError("Session token must be a non-empty string.")

    settings = get_settings()
    secret = settings.jwt_secret.get_secret_value().encode("utf-8")
    msg = f"session_binding:{raw_session_token}".encode("utf-8")
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


def generate_csrf_token(session_id: str) -> str:
    """Generate a signed, session-bound CSRF token.

    Format: {timestamp}.{nonce}.{signature}
    """
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string.")

    settings = get_settings()
    secret = settings.jwt_secret.get_secret_value().encode("utf-8")
    ts = int(time.time())
    nonce = secrets.token_hex(16)
    msg = f"{session_id}:{ts}:{nonce}".encode("utf-8")
    sig = hmac.new(secret, msg, hashlib.sha256).hexdigest()
    return f"{ts}.{nonce}.{sig}"


def verify_csrf_token(csrf_token: str, session_id: str, max_age_seconds: int = 3600) -> bool:
    """Verify a signed, session-bound CSRF token in constant time.

    Returns True if valid, False if invalid, expired, malformed, or mismatched.
    """
    if not isinstance(csrf_token, str) or not isinstance(session_id, str):
        return False
    if not csrf_token or not session_id:
        return False

    parts = csrf_token.split(".")
    if len(parts) != 3:
        return False

    ts_str, nonce, sig = parts
    try:
        ts = int(ts_str)
    except (ValueError, TypeError):
        return False

    now = time.time()
    # Reject expired tokens or tokens with timestamps more than 60s in the future
    if (now - ts) > max_age_seconds or (ts - now) > 60:
        return False

    if not nonce or not sig:
        return False

    settings = get_settings()
    secret = settings.jwt_secret.get_secret_value().encode("utf-8")
    msg = f"{session_id}:{ts_str}:{nonce}".encode("utf-8")
    expected_sig = hmac.new(secret, msg, hashlib.sha256).hexdigest()

    return hmac.compare_digest(sig, expected_sig)

