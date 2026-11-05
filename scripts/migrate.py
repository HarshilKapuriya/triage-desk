#!/usr/bin/env python3
"""Database migration runner for Ticket Intelligence Platform.

Applies ordered .sql migrations from the migrations/ directory to the database
configured via DATABASE_URL (or --database-url argument) and tracks applied
versions in the `schema_migrations` table.
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg


def get_connection_url(cli_url: str | None = None) -> str:
    """Resolve database connection URL prioritizing CLI argument, then environment."""
    load_dotenv()
    url = cli_url or os.getenv("DATABASE_URL")
    if not url:
        print("Error: DATABASE_URL not set in environment or provided via arguments.", file=sys.stderr)
        sys.exit(1)
    return url


def ensure_migration_table(conn: psycopg.Connection) -> None:
    """Ensure the schema_migrations tracking table exists."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
    conn.commit()


def get_applied_migrations(conn: psycopg.Connection) -> set[str]:
    """Retrieve set of migration versions already applied."""
    with conn.cursor() as cur:
        cur.execute("SELECT version FROM schema_migrations;")
        return {row[0] for row in cur.fetchall()}


def run_migrations(db_url: str, migrations_dir: Path) -> None:
    """Scan and apply pending migrations in alphabetical/numerical order."""
    if not migrations_dir.exists():
        print(f"Error: Migrations directory '{migrations_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    migration_files = sorted(migrations_dir.glob("*.sql"))
    if not migration_files:
        print(f"No migration files found in {migrations_dir}.")
        return

    print(f"Connecting to database...")
    try:
        with psycopg.connect(db_url, autocommit=False) as conn:
            ensure_migration_table(conn)
            applied = get_applied_migrations(conn)

            pending = [f for f in migration_files if f.name not in applied]
            if not pending:
                print(f"Database is up to date ({len(applied)} migration(s) applied).")
                return

            print(f"Found {len(pending)} pending migration(s) out of {len(migration_files)} total.")

            for migration_file in pending:
                version_name = migration_file.name
                print(f"Applying {version_name}...", end=" ", flush=True)

                sql_content = migration_file.read_text(encoding="utf-8")
                with conn.cursor() as cur:
                    cur.execute(sql_content)
                    cur.execute(
                        "INSERT INTO schema_migrations (version) VALUES (%s);",
                        (version_name,)
                    )
                conn.commit()
                print("DONE")

            print("\nAll migrations applied successfully.")

    except Exception as e:
        print(f"\nMigration failed: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply SQL migrations to database.")
    parser.add_argument(
        "--database-url",
        help="Postgres connection string. Defaults to DATABASE_URL from environment.",
    )
    parser.add_argument(
        "--dir",
        default=str(Path(__file__).resolve().parent.parent / "migrations"),
        help="Path to migrations directory.",
    )

    args = parser.parse_args()
    db_url = get_connection_url(args.database_url)
    migrations_dir = Path(args.dir)

    run_migrations(db_url, migrations_dir)


if __name__ == "__main__":
    main()

