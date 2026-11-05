"""API Key repository layer for managing tenant-scoped API keys and pre-auth lookup.

All tenant-owned functions require tenant_id and filter by it in every query.
Approved pre-authentication exception:
- get_active_api_key_by_hash: lookup by unique SHA-256 key hash to authenticate callers.
Never logs or stores plaintext keys.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
import uuid
import psycopg
from psycopg.rows import dict_row


@dataclass(frozen=True)
class ApiKeyRecord:
    """Read-only view of an API key record retrieved from database."""
    id: uuid.UUID
    tenant_id: uuid.UUID
    key_prefix: str
    label: str
    revoked_at: datetime | None
    last_used_at: datetime | None


def get_active_api_key_by_hash(
    conn: psycopg.Connection[Any],
    key_hash: str,
) -> ApiKeyRecord | None:
    """Pre-authentication lookup of an API key by its unique SHA-256 hash.

    Returns the record if found, regardless of revoked status (caller handles revoked check).
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, tenant_id, key_prefix, label, revoked_at, last_used_at
            FROM api_keys
            WHERE key_hash = %s;
            """,
            (key_hash,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return ApiKeyRecord(
            id=uuid.UUID(str(row["id"])),
            tenant_id=uuid.UUID(str(row["tenant_id"])),
            key_prefix=row["key_prefix"],
            label=row["label"],
            revoked_at=row["revoked_at"],
            last_used_at=row["last_used_at"],
        )


def update_api_key_last_used(
    conn: psycopg.Connection[Any],
    key_id: uuid.UUID,
) -> None:
    """Update last_used_at timestamp on successful API key authentication."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE api_keys
            SET last_used_at = NOW()
            WHERE id = %s;
            """,
            (key_id,),
        )


def create_api_key(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    key_prefix: str,
    key_hash: str,
    label: str,
) -> uuid.UUID:
    """Insert a new API key record for a tenant.

    Only key_prefix and key_hash are stored. Plaintext key is never written to DB.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO api_keys (tenant_id, key_prefix, key_hash, label)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
            """,
            (tenant_id, key_prefix, key_hash, label.strip()),
        )
        row = cur.fetchone()
        return uuid.UUID(str(row["id"]))


def revoke_api_key(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    key_id: uuid.UUID,
) -> bool:
    """Revoke an API key belonging to tenant_id by setting revoked_at.

    Repeat-revocation behavior:
    If the key exists and belongs to tenant_id, updating an already-revoked key
    succeeds idempotently (sets revoked_at = COALESCE(revoked_at, NOW())) and returns True.
    If the key does not exist or belongs to another tenant, 0 rows match and returns False (resulting in 404).

    Returns:
        bool: True if the key was found and belongs to tenant_id, False otherwise.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE api_keys
            SET revoked_at = COALESCE(revoked_at, NOW())
            WHERE tenant_id = %s AND id = %s;
            """,
            (tenant_id, key_id),
        )
        return cur.rowcount > 0
