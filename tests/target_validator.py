"""Pure test-support module for validating isolated disposable database targets.

Enforces strict containment constraints before any database connections, migrations,
or application initialization can occur.

Independent of application code, models, repositories, and database pools.
Does not perform network I/O or initialize ambient connections at import time.
"""

from __future__ import annotations

import os
import sys
from typing import Mapping

from psycopg.conninfo import conninfo_to_dict


class TargetSafetyError(ValueError):
    """Raised when a database target violates isolated disposable constraints."""
    pass


REQUIRED_TARGET: dict[str, str] = {
    "host": "127.0.0.1",
    "port": "5433",
    "dbname": "triage_events_disposable",
    "user": "test_user",
}

AUTHORIZED_DISPOSABLE_URL = (
    "postgresql://test_user:test_password@127.0.0.1:5433/triage_events_disposable"
)

ALLOWED_CONNECTION_PARAMS = frozenset({
    "user",
    "password",
    "pass",
    "dbname",
    "host",
    "port",
    "hostaddr",
    "sslmode",
    "client_encoding",
    "application_name",
    "connect_timeout",
})

CLOUD_INFRASTRUCTURE_KEYWORDS = (
    "supabase",
    "pooler",
    "aws",
    "neon",
    "rds",
    "azure",
    "gcp",
)


def validate_isolated_target(
    url: str | None,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Strict safety check: reject any connection not explicitly bound to authorized disposable target.

    Enforces:
    - Host must be exactly '127.0.0.1' (rejects 'localhost', domain names, multiple hosts).
    - Port must be exactly '5433' (rejects '5432' or other ports).
    - dbname must be exactly 'triage_events_disposable'.
    - user must be exactly 'test_user'.
    - Rejects 'hostaddr' overrides or mismatches.
    - Rejects 'service' and 'servicefile' configuration (pg_service.conf).
    - Rejects multi-host strings (commas in host, port, or hostaddr).
    - Rejects external cloud keywords (supabase, neon, rds, aws, pooler, azure, gcp).
    - Inspects ambient libpq environment variables (PGSERVICE, PGSERVICEFILE, PGSYSCONFDIR,
      PGHOST, PGHOSTADDR, PGPORT, PGDATABASE, PGUSER) to prevent ambient fallback.
    - Rejects unexpected connection options or overrides.
    - Never echoes passwords or raw connection URLs in exception messages.
    """
    env_source: Mapping[str, str] = os.environ if env is None else env

    # Libpq ambient environment checks to prevent fallback to ambient settings or service files
    if env_source.get("PGSERVICE"):
        raise TargetSafetyError("Safety Violation: Ambient environment variable PGSERVICE is set.")
    if env_source.get("PGSERVICEFILE"):
        raise TargetSafetyError("Safety Violation: Ambient environment variable PGSERVICEFILE is set.")
    if env_source.get("PGSYSCONFDIR"):
        raise TargetSafetyError("Safety Violation: Ambient environment variable PGSYSCONFDIR is set.")

    pg_host = env_source.get("PGHOST")
    if pg_host and pg_host != REQUIRED_TARGET["host"]:
        raise TargetSafetyError(
            f"Safety Violation: Ambient environment variable PGHOST conflicts with required host '{REQUIRED_TARGET['host']}'."
        )
    pg_hostaddr = env_source.get("PGHOSTADDR")
    if pg_hostaddr and pg_hostaddr != REQUIRED_TARGET["host"]:
        raise TargetSafetyError(
            f"Safety Violation: Ambient environment variable PGHOSTADDR conflicts with required host '{REQUIRED_TARGET['host']}'."
        )
    pg_port = env_source.get("PGPORT")
    if pg_port and str(pg_port) != REQUIRED_TARGET["port"]:
        raise TargetSafetyError(
            f"Safety Violation: Ambient environment variable PGPORT conflicts with required port '{REQUIRED_TARGET['port']}'."
        )
    pg_database = env_source.get("PGDATABASE")
    if pg_database and pg_database != REQUIRED_TARGET["dbname"]:
        raise TargetSafetyError(
            f"Safety Violation: Ambient environment variable PGDATABASE conflicts with required database '{REQUIRED_TARGET['dbname']}'."
        )
    pg_user = env_source.get("PGUSER")
    if pg_user and pg_user != REQUIRED_TARGET["user"]:
        raise TargetSafetyError(
            f"Safety Violation: Ambient environment variable PGUSER conflicts with required user '{REQUIRED_TARGET['user']}'."
        )

    if not url or not isinstance(url, str) or not url.strip():
        raise TargetSafetyError("Safety Violation: Database connection URL is empty or None.")

    if "servicefile" in url.lower():
        raise TargetSafetyError("Safety Violation: Servicefile connection configuration is rejected.")

    try:
        params = conninfo_to_dict(url)
    except Exception:
        raise TargetSafetyError("Safety Violation: Malformed connection URL.") from None

    # 1. Reject service configuration
    if params.get("service"):
        raise TargetSafetyError("Safety Violation: Service-based connection configuration is rejected.")

    # 2. Host validation (exact match 127.0.0.1 required; reject localhost or host lists)
    host = params.get("host")
    if not host:
        raise TargetSafetyError("Safety Violation: Missing required host parameter.")
    port = str(params.get("port", ""))
    if not port:
        raise TargetSafetyError("Safety Violation: Missing required port parameter.")

    if "," in host or "," in port:
        raise TargetSafetyError("Safety Violation: Multiple host or port targets are rejected.")

    if host != REQUIRED_TARGET["host"]:
        raise TargetSafetyError(
            f"Safety Violation: Host must be exactly '{REQUIRED_TARGET['host']}', received: '{host}'."
        )

    # 3. Hostaddr validation (reject bypass/override)
    hostaddr = params.get("hostaddr")
    if hostaddr is not None:
        if "," in hostaddr:
            raise TargetSafetyError("Safety Violation: Multiple hostaddr targets are rejected.")
        if hostaddr != REQUIRED_TARGET["host"]:
            raise TargetSafetyError(
                f"Safety Violation: Alternate hostaddr override '{hostaddr}' is rejected."
            )

    # 4. Port validation (exact match 5433 required)
    if port != REQUIRED_TARGET["port"]:
        raise TargetSafetyError(
            f"Safety Violation: Port must be isolated port '{REQUIRED_TARGET['port']}', received: '{port}'."
        )

    # 5. Database name validation
    dbname = params.get("dbname")
    if not dbname:
        raise TargetSafetyError("Safety Violation: Missing required database name parameter.")
    if dbname != REQUIRED_TARGET["dbname"]:
        raise TargetSafetyError(
            f"Safety Violation: Database name must be '{REQUIRED_TARGET['dbname']}', received: '{dbname}'."
        )

    # 6. User validation
    user = params.get("user")
    if not user:
        raise TargetSafetyError("Safety Violation: Missing required user parameter.")
    if user != REQUIRED_TARGET["user"]:
        raise TargetSafetyError(
            f"Safety Violation: Database user must be '{REQUIRED_TARGET['user']}', received: '{user}'."
        )

    # 7. Allowed parameters whitelist to reject unexpected connection overrides
    for k in params:
        if k not in ALLOWED_CONNECTION_PARAMS:
            raise TargetSafetyError(f"Safety Violation: Unexpected connection parameter '{k}' is rejected.")

    # 8. Reject hosted cloud keywords in any parameter (excluding password)
    for k, v in params.items():
        if k in ("password", "pass"):
            continue
        v_str = str(v).lower()
        if any(cloud_kw in v_str for cloud_kw in CLOUD_INFRASTRUCTURE_KEYWORDS):
            raise TargetSafetyError(
                f"Safety Violation: Parameter '{k}' references non-disposable infrastructure."
            )

    return params


def main() -> None:
    """CLI entrypoint for fail-closed pre-flight validation in CI and scripts."""
    target_url = sys.argv[1] if len(sys.argv) > 1 else os.getenv("TEST_DISPOSABLE_DATABASE_URL")
    try:
        validate_isolated_target(target_url)
        # Redacted, secret-free confirmation output
        print(
            "Target validation passed: authorized disposable PostgreSQL service confirmed "
            f"({REQUIRED_TARGET['host']}:{REQUIRED_TARGET['port']}/{REQUIRED_TARGET['dbname']})."
        )
    except TargetSafetyError as exc:
        print(f"Target validation failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        print("Target validation failed: Malformed target configuration.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

