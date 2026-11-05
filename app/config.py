"""Application configuration module using pydantic-settings.

Reads settings from environment variables and an optional .env file.
Sensitive values (database URLs, JWT secrets) use SecretStr to prevent
accidental exposure in logs, stdout, or exception traces.
"""

from functools import lru_cache
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Core settings for Ticket Intelligence Platform."""

    # Supabase / PostgreSQL connection string
    database_url: SecretStr = Field(
        ...,
        alias="DATABASE_URL",
        description="Postgres connection string for the primary application database.",
    )

    # Separate throwaway database connection string strictly for tests
    test_database_url: SecretStr | None = Field(
        default=None,
        alias="TEST_DATABASE_URL",
        description="Postgres connection string strictly for running tests.",
    )

    # Cryptographic secret for signing HS256 JWT tokens
    jwt_secret: SecretStr = Field(
        ...,
        alias="JWT_SECRET",
        description="Random secret key for signing JWT tokens.",
    )

    # Token expiration in minutes
    jwt_expire_minutes: int = Field(
        default=60,
        alias="JWT_EXPIRE_MINUTES",
        description="Lifetime of dashboard JWT access tokens in minutes.",
    )

    # Database connection pool settings
    db_pool_min_size: int = Field(
        default=1,
        alias="DB_POOL_MIN_SIZE",
        description="Minimum number of connections in the connection pool.",
    )
    db_pool_max_size: int = Field(
        default=5,
        alias="DB_POOL_MAX_SIZE",
        description="Maximum number of connections in the connection pool.",
    )
    db_pool_timeout_seconds: float = Field(
        default=10.0,
        alias="DB_POOL_TIMEOUT_SECONDS",
        description="Connection acquisition timeout in seconds before failing.",
    )

    # Startup policy: whether the app verifies DB connectivity on startup
    # and the maximum timeout in seconds allowed for the check.
    db_connect_on_startup: bool = Field(
        default=True,
        alias="DB_CONNECT_ON_STARTUP",
        description="Check database connectivity during application startup.",
    )
    db_startup_timeout_seconds: float = Field(
        default=5.0,
        alias="DB_STARTUP_TIMEOUT_SECONDS",
        description="Maximum seconds to wait for initial DB connectivity check.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve cached application settings instance."""
    return Settings()

