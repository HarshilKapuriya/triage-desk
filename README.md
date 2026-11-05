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

## Running Tests

Run the test suite inside your activated virtual environment:

```powershell
# Run all unit tests (isolated, fast, no DB required)
python -m pytest tests/ -v

# Run only authentication unit tests
python -m pytest tests/test_auth.py -v

# Run only security unit tests
python -m pytest tests/test_security.py -v

# Run only health unit tests
python -m pytest tests/test_health.py -v

# Run database integration tests (requires configured TEST_DATABASE_URL)
python -m pytest tests/test_auth_db.py -v
```

> **Note on Integration Tests & CI**:
> Database integration tests require a configured `TEST_DATABASE_URL` that resolves to a different database target than `DATABASE_URL`. They do NOT run against `DATABASE_URL`. If `TEST_DATABASE_URL` is empty, integration tests will be skipped locally.
>
> In GitHub Actions CI (`.github/workflows/ci.yml`), a dedicated PostgreSQL 16 service container is automatically provisioned, migrations are applied using `scripts/migrate.py`, and the full test suite (including all database integration tests) runs against the throwaway CI database.

