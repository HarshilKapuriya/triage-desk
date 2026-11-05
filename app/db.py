"""Database connection pool management and transaction helpers.

Uses psycopg_pool.ConnectionPool with open=False initially.
The pool is opened during the FastAPI lifespan context manager and closed cleanly
on application shutdown. Connections are not acquired at module import time.
"""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
import logging
from typing import Any
from fastapi import FastAPI
import psycopg
from psycopg_pool import ConnectionPool

from app.config import Settings, get_settings

logger = logging.getLogger("app.db")

# Global connection pool instance. Opened and closed via lifespan.
_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    """Return the global ConnectionPool instance.

    Raises:
        RuntimeError: If called before the pool has been initialized in lifespan.
    """
    global _pool
    if _pool is None:
        raise RuntimeError("Database connection pool is not initialized.")
    return _pool


def set_pool(pool: ConnectionPool | None) -> None:
    """Set or clear the global pool (useful for tests and lifespan management)."""
    global _pool
    _pool = pool


def create_pool(settings: Settings | None = None) -> ConnectionPool:
    """Create an un-opened ConnectionPool configured from settings."""
    cfg = settings or get_settings()
    conninfo = cfg.database_url.get_secret_value()

    # Create pool with open=False so no network connections occur at creation
    return ConnectionPool(
        conninfo=conninfo,
        min_size=cfg.db_pool_min_size,
        max_size=cfg.db_pool_max_size,
        timeout=cfg.db_pool_timeout_seconds,
        open=False,
        name="triage_desk_pool",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan context manager managing pool startup and shutdown.

    Startup Policy:
        - The pool is created with open=False.
        - The pool is opened via pool.open().
        - If db_connect_on_startup is True, the application attempts a quick
          'SELECT 1' check with a bounded timeout (db_startup_timeout_seconds, default 5s).
          If it fails or times out, an informative error is raised rather than
          letting the application hang indefinitely.
    Shutdown Policy:
        - The pool is cleanly closed via pool.close().
    """
    settings = get_settings()
    pool = create_pool(settings)
    set_pool(pool)

    # Open pool
    pool.open(wait=False)

    if settings.db_connect_on_startup:
        logger.info("Verifying database connectivity on startup...")
        try:
            with pool.connection(timeout=settings.db_startup_timeout_seconds) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
            logger.info("Database connection verified successfully.")
        except Exception as exc:
            logger.error("Failed to connect to database during startup.")
            pool.close()
            set_pool(None)
            raise RuntimeError(
                f"Database connectivity check failed on startup (timeout: {settings.db_startup_timeout_seconds}s)."
            ) from exc

    try:
        yield
    finally:
        logger.info("Closing database connection pool...")
        current_pool = get_pool() if _pool is not None else None
        if current_pool is not None:
            current_pool.close()
        set_pool(None)
        logger.info("Database connection pool closed.")


@contextmanager
def get_connection() -> Iterator[psycopg.Connection[Any]]:
    """Context manager for acquiring a connection from the pool.

    Yields a connection that is automatically returned to the pool when done.
    """
    pool = get_pool()
    with pool.connection() as conn:
        yield conn

