# Triage Desk - Ticket Intelligence Platform

A multi-tenant customer support ticket intelligence platform built with FastAPI, PostgreSQL (Supabase), and CPU-only ML models.

---

## Getting Started (Windows / PowerShell)

### 1. Prerequisites
- Python 3.11 or newer (tested on Python 3.14)
- Git

### 2. Environment Setup

Clone the repository and set up a local virtual environment:

```powershell
# Navigate to repository
cd triage-desk

# Create a local virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install approved dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configuration

Copy `.env.example` to `.env` and fill in your values privately:

```powershell
Copy-Item .env.example .env
```

Required environment variables in `.env`:
- `DATABASE_URL`: Your primary Postgres connection string (e.g. Supabase connection pooler URL).
- `TEST_DATABASE_URL`: A dedicated test database connection string used only by tests.
- `JWT_SECRET`: Random 32+ character string for token signing.
- `JWT_EXPIRE_MINUTES`: JWT token validity lifetime (default: 60).
- `DB_CONNECT_ON_STARTUP`: Set to `true` (default) to test database reachability upon startup.
- `DB_STARTUP_TIMEOUT_SECONDS`: Maximum seconds to wait for initial DB connectivity (default: 5.0).

> **Important**: Never commit `.env` or print secret connection strings to logs or chat.

### 4. Running Migrations

Apply database migrations:

```powershell
python scripts/migrate.py
```

### 5. Running the API Server

Start the FastAPI application with Uvicorn:

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

Once running:
- **Liveness probe**: `GET http://localhost:8000/healthz` -> `{"status": "ok"}`
- **Interactive API Docs (Swagger)**: `http://localhost:8000/docs`

---

## API Authentication Conventions

Triage Desk enforces credential isolation depending on the operation:

1. **User JWT Authentication (Admin Endpoints Remain Strictly JWT-Only)**:
   - Admin management routes (e.g. creating and revoking API keys under `/v1/api-keys`) strictly require an authenticated User JWT with the `admin` role.
   - API keys cannot authenticate to admin endpoints under any header; supplying an API key to admin routes returns `401 Unauthorized` (`missing_authorization_header`).
   - Header: `Authorization: Bearer <jwt_token>`
   - Acquired via `POST /v1/auth/login`.

2. **API Key Authentication** (Ingestion and programmatic access):
   - Ticket ingestion (`POST /v1/tickets`) strictly requires an API key. User JWTs return `401 Unauthorized`.
   - Header: `X-API-Key: tk_<random_string>`
   - Supported alternative: `Authorization: Bearer tk_<random_string>`
   - Keys are created by tenant admins via `POST /v1/api-keys`.

3. **Dual Credential Endpoints** (Ticket list & detail queries):
   - Endpoints supporting dual authentication (`GET /v1/tickets` and `GET /v1/tickets/{id}`) accept either User JWT (`Authorization: Bearer <jwt>`) or API Key (`X-API-Key: tk_...`).

4. **Dual-Header Precedence on Endpoints Supporting API Keys**:
   - On endpoints that accept API keys (ticket ingestion and dual-credential ticket queries), if both `X-API-Key` and `Authorization` headers are supplied simultaneously:
     - `X-API-Key` takes strict precedence and is evaluated exclusively.
     - The resolved `Principal` derives all attributes (`tenant_id`, `kind="api_key"`, `id`, `role="ingestion"`) strictly from the verified API key; claims or identities from `Authorization` are never mixed or blended.
     - If `X-API-Key` is invalid, malformed, or revoked, authentication immediately fails with HTTP 401 without falling back to `Authorization`.
   - On JWT-only admin endpoints (`/v1/api-keys`), this precedence rule does not apply: admin routes ignore API keys and strictly validate the User JWT in the `Authorization` header.

---

## Endpoints Summary

### Authentication (`/v1/auth`)
- `POST /v1/auth/signup`: Create tenant and initial admin user.
- `POST /v1/auth/login`: Discover tenant and authenticate admin user, returns JWT.

### API Keys (`/v1/api-keys`) - Admin JWT Only
- `POST /v1/api-keys`: Create an API key with label. Returns plaintext key once (`tk_...`).
- `DELETE /v1/api-keys/{id}`: Idempotently revoke an API key.

### Tickets (`/v1/tickets`)
- `POST /v1/tickets`: Ingest ticket (API Key only).
  - Body: `{"subject": "...", "body": "...", "external_id": "optional", "customer_email": "optional"}`
  - Returns `202 Accepted` on new ticket creation with `{"id": "...", "status": "received", "external_id": "..."}`.
  - Returns `200 OK` if a ticket with the same `external_id` already exists for this tenant (idempotent replay).
- `GET /v1/tickets`: List tickets for tenant (JWT or API Key).
  - Query parameters: `status` (optional), `limit` (1-200, default 50), `offset` (>= 0, default 0).
  - Returns a plain JSON array of tickets `[...]` ordered newest first (`created_at DESC, id DESC`).
- `GET /v1/tickets/{id}`: Get ticket detail by UUID (JWT or API Key).
  - Returns `200 OK` with full ticket details or `404 Not Found` if missing or belonging to another tenant.

---

## Running Tests

Run the test suite inside your activated virtual environment:

```powershell
# Run all unit tests and integration tests
python -m pytest -v

# Run only authentication unit tests
python -m pytest tests/test_auth.py -v

# Run only API key unit tests
python -m pytest tests/test_api_keys.py -v

# Run only ticket unit tests
python -m pytest tests/test_tickets.py -v

# Run only security unit tests
python -m pytest tests/test_security.py -v

# Run only health unit tests
python -m pytest tests/test_health.py -v

# Run database integration tests (requires configured TEST_DATABASE_URL)
python -m pytest tests/test_auth_db.py tests/test_api_keys_db.py tests/test_tickets_db.py -v

# Run the dedicated two-tenant isolation test (SPEC Section 10)
python -m pytest tests/test_isolation.py -v
```

> **Note on Integration Tests & CI**:
> Database integration tests require a configured `TEST_DATABASE_URL` that resolves to a different database target than `DATABASE_URL`. They do NOT run against `DATABASE_URL`. If `TEST_DATABASE_URL` is empty, integration tests will be skipped locally.
>
> In GitHub Actions CI (`.github/workflows/ci.yml`), a dedicated PostgreSQL 16 service container is configured to provision automatically, apply migrations using `scripts/migrate.py`, and run the full test suite against the throwaway CI database.
>
> **Current CI Status**: GitHub CI is currently blocked before execution due to an account-level billing lock ("The job was not started because your account is locked due to a billing issue"). Phase 1 has been thoroughly verified locally against a dedicated PostgreSQL test database (52 passed, 0 failed, 0 skipped).

