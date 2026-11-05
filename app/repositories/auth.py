"""Pre-authentication repository functions for signup and login.

All SQL queries remain strictly encapsulated in this repository layer.
These functions represent the approved pre-authentication exceptions:
1. create_tenant_and_admin_user: Transactionally creates a tenant, admin user, and audit entry.
2. get_user_by_email: Looks up a user by globally unique lowercase email.
"""

from dataclasses import dataclass
from typing import Any
import uuid
import psycopg
from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row


class EmailAlreadyExistsError(Exception):
    """Raised when an account already exists with the requested email."""


@dataclass(frozen=True)
class UserRecord:
    """Read-only view of a user record retrieved from database."""
    id: uuid.UUID
    tenant_id: uuid.UUID
    email: str
    password_hash: str
    role: str


def get_user_by_email(conn: psycopg.Connection[Any], email: str) -> UserRecord | None:
    """Find a user by normalized lowercase email.

    Pre-authentication query strictly for login resolution.
    """
    normalized_email = email.strip().lower()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, tenant_id, email, password_hash, role
            FROM users
            WHERE email = %s;
            """,
            (normalized_email,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return UserRecord(
            id=uuid.UUID(str(row["id"])),
            tenant_id=uuid.UUID(str(row["tenant_id"])),
            email=row["email"],
            password_hash=row["password_hash"],
            role=row["role"],
        )


def _record_signup_audit(cur: psycopg.Cursor[Any], tenant_id: uuid.UUID, user_id: uuid.UUID) -> None:
    """Internal helper to insert the signup audit log entry within an open cursor."""
    cur.execute(
        """
        INSERT INTO audit_log (tenant_id, actor, action)
        VALUES (%s, %s, 'tenant_signup');
        """,
        (tenant_id, str(user_id)),
    )


def create_tenant_and_admin_user(
    conn: psycopg.Connection[Any],
    company_name: str,
    normalized_email: str,
    password_hash: str,
) -> tuple[uuid.UUID, uuid.UUID]:
    """Atomically create tenant, first admin user, and signup audit entry in one transaction.

    Args:
        conn: Active psycopg connection.
        company_name: Name of the tenant organization.
        normalized_email: Pre-normalized lowercase email address.
        password_hash: Argon2 hash of the admin password.

    Returns:
        tuple[uuid.UUID, uuid.UUID]: (tenant_id, user_id)

    Raises:
        EmailAlreadyExistsError: If normalized_email is already registered.
    """
    try:
        # Enforce atomic transaction
        with conn.transaction():
            with conn.cursor(row_factory=dict_row) as cur:
                # 1. Early check if email already registered
                cur.execute("SELECT id FROM users WHERE email = %s;", (normalized_email,))
                if cur.fetchone():
                    raise EmailAlreadyExistsError(f"Email '{normalized_email}' is already registered.")

                # 2. Create tenant
                cur.execute(
                    """
                    INSERT INTO tenants (name, plan)
                    VALUES (%s, 'free')
                    RETURNING id;
                    """,
                    (company_name.strip(),),
                )
                tenant_id = uuid.UUID(str(cur.fetchone()["id"]))

                # 3. Create initial admin user
                cur.execute(
                    """
                    INSERT INTO users (tenant_id, email, password_hash, role)
                    VALUES (%s, %s, %s, 'admin')
                    RETURNING id;
                    """,
                    (tenant_id, normalized_email, password_hash),
                )
                user_id = uuid.UUID(str(cur.fetchone()["id"]))

                # 4. Create audit log entry via helper
                _record_signup_audit(cur, tenant_id, user_id)

        return tenant_id, user_id

    except UniqueViolation as exc:
        # Strictly match users.email constraint violations (constraint_name or table/column)
        constraint_name = getattr(exc.diag, "constraint_name", None)
        table_name = getattr(exc.diag, "table_name", None)
        column_name = getattr(exc.diag, "column_name", None)

        is_email_violation = (
            constraint_name in ("users_email_key", "uq_users_email")
            or (table_name == "users" and (column_name == "email" or "email" in str(exc.diag.message_detail or "")))
            or "Key (email)=" in str(exc)
        )

        if is_email_violation:
            raise EmailAlreadyExistsError(f"Email '{normalized_email}' is already registered.") from exc

        # Do not disguise other unique constraint failures
        raise
