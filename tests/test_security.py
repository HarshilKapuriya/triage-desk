"""Focused unit tests for app/security.py.

Tests:
1. Password hash/verify success and failure.
2. Distinct salts producing different hashes for identical passwords.
3. Safe rejection of malformed password hashes without crashing.
4. JWT token creation and decoding round trip with expected claims.
5. Rejection of expired JWTs without sleeping.
6. Rejection of wrong signing secret and tampered tokens.
7. Rejection of missing/invalid claims and unsupported algorithms.
8. API-key format (tk_ prefix, length), prefix derivation, SHA-256 consistency,
   and uniqueness across generations.
"""

from datetime import datetime, timezone, timedelta
import uuid
import pytest
import jwt

from app.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    generate_api_key,
    hash_api_key,
    SUPPORTED_ROLES,
    JWT_ALGORITHM,
)
from app.config import get_settings


# ---------------------------------------------------------------------------
# Password Tests
# ---------------------------------------------------------------------------

def test_password_hash_and_verify_success() -> None:
    """Correct password verifies successfully against generated Argon2 hash."""
    raw_password = "CorrectHorseBatteryStaple123!"
    pw_hash = hash_password(raw_password)

    assert pw_hash.startswith("$argon2")
    assert verify_password(pw_hash, raw_password) is True


def test_password_verify_failure_wrong_password() -> None:
    """Incorrect password fails verification safely."""
    raw_password = "SuperSecretPassword1"
    pw_hash = hash_password(raw_password)

    assert verify_password(pw_hash, "WrongPassword2") is False


def test_password_salts_produce_distinct_hashes() -> None:
    """Hashing the same password twice produces different hashes due to salting."""
    password = "ConsistentPassword123"
    hash1 = hash_password(password)
    hash2 = hash_password(password)

    assert hash1 != hash2
    assert verify_password(hash1, password) is True
    assert verify_password(hash2, password) is True


def test_password_verify_malformed_hashes_fail_safely() -> None:
    """Malformed or invalid hashes return False without raising exceptions."""
    assert verify_password("not-a-valid-argon2-hash", "password") is False
    assert verify_password("$argon2id$v=19$m=65536,t=3,p=4$malformed", "password") is False
    assert verify_password("", "password") is False
    assert verify_password(None, "password") is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# JWT Tests
# ---------------------------------------------------------------------------

def test_jwt_round_trip_and_expected_claims() -> None:
    """Valid JWT round trip returns all expected claims."""
    user_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    role = "admin"

    token = create_access_token(user_id=user_id, tenant_id=tenant_id, role=role)
    payload = decode_access_token(token)

    assert payload["sub"] == str(user_id)
    assert payload["tenant_id"] == str(tenant_id)
    assert payload["role"] == role
    assert "iat" in payload
    assert "exp" in payload
    assert payload["exp"] > payload["iat"]


def test_jwt_expired_token_rejected_without_sleep() -> None:
    """Expired token is rejected immediately by creating it with negative delta."""
    user_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    expired_token = create_access_token(
        user_id=user_id,
        tenant_id=tenant_id,
        role="agent",
        expires_delta=timedelta(seconds=-10),
    )

    with pytest.raises(ValueError, match="Token has expired"):
        decode_access_token(expired_token)


def test_jwt_wrong_secret_rejected() -> None:
    """Token signed with a different secret is rejected."""
    user_id = str(uuid.uuid4())
    tenant_id = str(uuid.uuid4())
    foreign_secret = "different_secret_key_used_to_tamper_token_12345"

    now = datetime.now(timezone.utc)
    tampered_payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "role": "admin",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=10)).timestamp()),
    }
    foreign_token = jwt.encode(tampered_payload, foreign_secret, algorithm="HS256")

    with pytest.raises(ValueError, match="Invalid token signature"):
        decode_access_token(foreign_token)


def test_jwt_tampered_payload_rejected() -> None:
    """Tampering with token payload invalidates signature."""
    user_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    valid_token = create_access_token(user_id=user_id, tenant_id=tenant_id, role="agent")

    parts = valid_token.split(".")
    # Tamper with the middle part (payload base64)
    tampered_token = f"{parts[0]}.eyJhZG1pbiI6dHJ1ZX0.{parts[2]}"

    with pytest.raises(ValueError):
        decode_access_token(tampered_token)


def test_jwt_missing_or_invalid_claims_rejected() -> None:
    """Tokens missing required claims or with invalid UUIDs are rejected."""
    settings = get_settings()
    secret = settings.jwt_secret.get_secret_value()
    now = datetime.now(timezone.utc)

    # Missing 'role'
    payload_missing_role = {
        "sub": str(uuid.uuid4()),
        "tenant_id": str(uuid.uuid4()),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=10)).timestamp()),
    }
    token_no_role = jwt.encode(payload_missing_role, secret, algorithm="HS256")
    with pytest.raises(ValueError):
        decode_access_token(token_no_role)

    # Invalid UUID for sub
    payload_bad_uuid = {
        "sub": "not-a-valid-uuid",
        "tenant_id": str(uuid.uuid4()),
        "role": "admin",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=10)).timestamp()),
    }
    token_bad_uuid = jwt.encode(payload_bad_uuid, secret, algorithm="HS256")
    with pytest.raises(ValueError, match="invalid UUID"):
        decode_access_token(token_bad_uuid)

    # Unsupported role
    payload_bad_role = {
        "sub": str(uuid.uuid4()),
        "tenant_id": str(uuid.uuid4()),
        "role": "superadmin",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=10)).timestamp()),
    }
    token_bad_role = jwt.encode(payload_bad_role, secret, algorithm="HS256")
    with pytest.raises(ValueError, match="unsupported role"):
        decode_access_token(token_bad_role)


def test_jwt_unsupported_algorithm_rejected() -> None:
    """Tokens signed with 'none' or unapproved algorithms are rejected."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(uuid.uuid4()),
        "tenant_id": str(uuid.uuid4()),
        "role": "admin",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=10)).timestamp()),
    }
    # Unsecured 'none' algorithm token
    insecure_token = jwt.encode(payload, key="", algorithm="none")

    with pytest.raises(ValueError):
        decode_access_token(insecure_token)


# ---------------------------------------------------------------------------
# API-Key Tests
# ---------------------------------------------------------------------------

def test_api_key_format_and_prefix() -> None:
    """Generated API keys adhere to prefix format, length, and derivation rules."""
    full_key, key_prefix, key_hash = generate_api_key()

    # Starts with 'tk_'
    assert full_key.startswith("tk_")
    # Prefix is first 8 characters
    assert key_prefix == full_key[:8]
    assert len(key_prefix) == 8
    # Hash is 64 hex characters (SHA-256)
    assert len(key_hash) == 64
    assert all(c in "0123456789abcdef" for c in key_hash)


def test_api_key_hash_consistency() -> None:
    """Hashing an API key produces reproducible SHA-256 digest."""
    full_key, _, key_hash = generate_api_key()

    assert hash_api_key(full_key) == key_hash


def test_api_key_distinct_generations() -> None:
    """Consecutive API key generations yield unique keys."""
    keys = {generate_api_key()[0] for _ in range(50)}
    assert len(keys) == 50

