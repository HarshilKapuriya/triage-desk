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

### 4. Database Migrations

#### Standard Migration Procedure (Fresh Databases & CI)
The repository uses sequential numbered SQL migrations under `migrations/` (`001_init.sql`, `002_jobs.sql`, `003_ml_predictions.sql`, `004_embeddings_duplicates.sql`), managed by `scripts/migrate.py`.

To apply all pending migrations on a new database:
```powershell
python scripts/migrate.py
```
`scripts/migrate.py` automatically provisions the `schema_migrations` tracking table if missing, scans `migrations/*.sql` in alphanumeric order, executes each pending migration within a transaction, and records the applied version.

#### Historical Development Database Reconciliation Note
*Distinction from standard migrations*: In the existing development database (`DATABASE_URL`), tables for `001_init.sql`, `002_jobs.sql`, and `003_ml_predictions.sql` had previously been introduced prior to establishing the migration runner. A one-time SQL reconciliation was executed directly on `DATABASE_URL` to create `schema_migrations` and record `001_init.sql`, `002_jobs.sql`, and `003_ml_predictions.sql` as applied without re-running DDL. This was a one-time environment reconciliation, distinct from repository migration files and distinct from the standard fresh-database migration procedure above.

### 5. Three-Terminal Startup Procedure

To run the complete platform locally with live background inference and the React browser interface, open three separate PowerShell terminal windows:

#### Terminal 1: FastAPI API Server
Start the Uvicorn server bound to `127.0.0.1:8000`:
```powershell
cd triage-desk
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- **Liveness probe**: `GET http://127.0.0.1:8000/healthz` -> `{"status": "ok"}`
- **Interactive API Docs (Swagger)**: `http://127.0.0.1:8000/docs`
- **React SPA (built static bundle)**: `http://127.0.0.1:8000/`

#### Terminal 2: Background Queue Worker Process
Start the standalone queue worker process:
```powershell
cd triage-desk
.\.venv\Scripts\python.exe -m app.worker.main
```
- Standalone queue worker polling `jobs` using `SELECT ... FOR UPDATE SKIP LOCKED` (default 1.0s interval).
- Loads and verifies trusted LinearSVC model (`run_004`) once on process startup.
- Claims queued tickets, executes narrative inference, registers model version, stores prediction in `ticket_predictions`, and transitions ticket from `new` to `open`.
- Graceful shutdown on `Ctrl+C` (SIGINT) or SIGTERM. Optional flags: `--poll-interval <sec>`, `--lease-seconds <sec>`.

#### Terminal 3: React Frontend (Vite Development Server)
Start the Vite development server with strict port binding:
```powershell
cd triage-desk\frontend
npm run dev -- --port 5173 --strictPort
```
- Vite dev server runs at `http://127.0.0.1:5173`.
- **Proxy Configuration**: As defined in `frontend/vite.config.ts`, Vite proxies `/browser-api` requests directly to `http://127.0.0.1:8000` with `changeOrigin: true`.
- Alternatively, production assets compiled via `npm run build` into `frontend/dist` are served directly by FastAPI at `http://127.0.0.1:8000` without requiring a Node process in production.

---

### 6. Authentication & Demo-Ticket Submission

Use a separate, temporary PowerShell window to authenticate, create an ingestion API key, and submit demo tickets.

> **Security Note on Shell Credentials**: Setting variables to `$null` does not guarantee memory sanitization in PowerShell because intermediate response/pipeline objects retain credential data. **Close the temporary ingestion PowerShell window when finished** to guarantee all credentials and in-memory variables are fully purged.

---

#### Option A: Existing User Login (Recommended)
If your demo account (`demo.agent@example.com`) is already registered, authenticate via `POST /v1/auth/login` to obtain your admin JWT:

```powershell
# Retrieve password securely via masked prompt
$plainPwd = [System.Net.NetworkCredential]::new("", (Read-Host "Enter password for demo.agent@example.com" -AsSecureString)).Password

$loginPayload = @{
    email    = 'demo.agent@example.com'
    password = $plainPwd
} | ConvertTo-Json

$loginResp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/auth/login" `
    -Method Post `
    -ContentType "application/json" `
    -Body $loginPayload

$jwtToken = $loginResp.access_token
Write-Host "Authenticated successfully with existing account."
```

---

#### Option B: First-Time Tenant Signup (New Companies Only)
> **Warning**: Do not run this if your demo account already exists. Re-registering an existing email will fail with HTTP 409 Conflict. Run this only when onboarding a brand-new tenant:

```powershell
$plainPwd = [System.Net.NetworkCredential]::new("", (Read-Host "Enter admin password (min 8 characters)" -AsSecureString)).Password

$signupPayload = @{
    company_name = 'Demo Bank Corporation'
    email        = 'new.tenant.admin@example.com'
    password     = $plainPwd
} | ConvertTo-Json

$signupResp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/auth/signup" `
    -Method Post `
    -ContentType "application/json" `
    -Body $signupPayload

$jwtToken = $signupResp.access_token
Write-Host "Registered Tenant ID: $($signupResp.tenant_id), User ID: $($signupResp.user_id)"
```

---

#### Provision Ingestion API Key (Admin JWT Required)
With your `$jwtToken` from Option A or B, provision an API key for ticket ingestion:

```powershell
$keyPayload = @{ label = 'demo-ingest-key' } | ConvertTo-Json
$keyResp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/api-keys" `
    -Method Post `
    -Headers @{ Authorization = "Bearer $jwtToken" } `
    -ContentType "application/json" `
    -Body $keyPayload

$apiKey = $keyResp.key
Write-Host "Provisioned API Key in memory."
```

---

#### Ingest Synthetic Financial Complaint Ticket
> **Important**: In PowerShell, literal strings containing dollar signs (such as `$1,200` and `$105`) must be enclosed in single quotes (`'...'`). Double quotes will cause PowerShell to evaluate `$1` and `$105` as variables, corrupting the complaint text.

This sample narrative is a **synthetic financial complaint** constructed to exercise the checking account overdraft triage path:

```powershell
# Use single quotes for text containing literal currency signs ($1,200, $105)
$complaint = @{
    subject     = 'Dispute over uncredited checking account deposit and overdraft fees'
    body        = 'I deposited a check for $1,200 into my personal checking account at the branch ATM on Monday. The teller stated funds would be available immediately, but the bank placed an extended hold without providing an exception hold notice. As a result, three routine household utility bills bounced, and the bank assessed $105 in non-sufficient funds (NSF) overdraft fees against my account.'
    external_id = 'synthetic-cfpb-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
} | ConvertTo-Json

$ticketResp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/tickets" `
    -Method Post `
    -Headers @{ "X-API-Key" = $apiKey } `
    -ContentType "application/json" `
    -Body $complaint

Write-Host "Ingested Ticket: $($ticketResp.id) [Status: $($ticketResp.status)]"
Write-Host "Close this PowerShell window when finished to purge credentials from memory."
```

---

#### View in React Frontend
1. Open browser to `http://127.0.0.1:5173/login` (or `http://127.0.0.1:8000/login`).
2. Log in with registered email (`demo.agent@example.com`) and password.
3. Observe ticket transitioning from `new` to `open` as the worker completes inference.
4. Click ticket row to inspect predicted category (`Checking or savings account`), model version (`run_004`), decision margins, and domain limitations.
5. Close the temporary ingestion PowerShell window.

---

## Phase 3 ML Model Integration (Financial Complaint Demo)

Triage Desk incorporates a CPU-only text classification pipeline for automated ticket triage. In Phase 3, this is demonstrated using a real-world financial consumer complaint classifier trained on official CFPB narratives:

- **Target Domain**: Financial product classification across 11 CFPB classes (`Checking or savings account`, `Credit card`, `Credit reporting or other personal consumer reports`, `Debt collection`, `Debt or credit management`, `Money transfer, virtual currency, or money service`, `Mortgage`, `Payday loan, title loan, personal loan, or advance loan`, `Prepaid card`, `Student loan`, `Vehicle loan or lease`).
- **Domain Limitation Notice**: The model is trained exclusively on financial consumer complaint narratives. It has no out-of-domain detection mechanism; non-financial support tickets (e.g., IT hardware, password resets) will receive an arbitrary financial product label.
- **Model Architecture**: Frozen TF-IDF (1-2 ngrams, sublinear TF, 25k features) + LinearSVC ($C=1.0$, `class_weight=None`) from `run_004`.
- **Confidence & Decision Scores**: LinearSVC decision scores are hyperplane margin distances, not probabilities. Margin distances and softmax transformations of margins are not calibrated probabilities. Therefore, `confidence` is left strictly `NULL` in `ticket_predictions` and API responses, while raw margins are preserved under `decision_scores`.
- **Priority Status**: Priority prediction remains explicitly deferred; priority is not populated with heuristics or uncalibrated predictions.
- **Multi-Tenant Isolation**: Every tenant maintains its own explicit record in `model_versions` (`tenant_id NOT NULL`), and `ticket_predictions` enforces composite foreign keys `(tenant_id, ticket_id)` and `(tenant_id, model_version_id)` preventing cross-tenant visibility or tampering. Predictions are distinguished from human-curated ticket attributes (`tickets.category` remains untouched).
- **Artifact Provisioning & Integrity**:
  - Expected SHA-256 Digest: `7e833e8992a3150bdaf2489b9b9117d700f7875e9dde045e4f638bd1e656b360`.
  - Required Environment & Dependencies: Python 3.11+, `scikit-learn==1.9.1`, `numpy==2.5.4`, `scipy==1.18.1`, `joblib==1.6.0` (as specified in `requirements.txt`).
  - Default Location: `experiments/runs/run_004_cfpb_imbalance_tuning/models/selected_cw_None_C_1.0.joblib`.
  - Configurable Override: `MODEL_ARTIFACT_PATH` environment variable.
  - Provisioning Steps: Provision the verified, trusted artifact matching the expected SHA-256 digest at the default path or at the path designated by `MODEL_ARTIFACT_PATH`.
  - Reproducibility & Binary Hash Notice: Running `python scripts/tune_cfpb_imbalance.py` reproduces the training pipeline, hyperparameters, and split configuration, but retraining across different OS environments, CPU architectures, or library builds is not guaranteed to reproduce a bit-for-bit identical serialized `.joblib` file hash. Production/worker runtimes should provision the exact verified artifact matching the SHA-256 digest above.
  - Missing or corrupted artifacts raise explicit operational errors (`ModelArtifactMissingError`, `ModelArtifactCorruptedError`) and requeue/fail the worker job without fabricating predictions.

---

## API Authentication Conventions

Triage Desk enforces credential isolation depending on the operation:

1. **User JWT Authentication (Admin Endpoints Remain Strictly JWT-Only)**:
   - Admin management routes (e.g. creating and revoking API keys under `/v1/api-keys`) strictly require an authenticated User JWT with the `admin` role.
   - API keys cannot authenticate to admin endpoints under any header; supplying an API key to admin routes returns `401 Unauthorized` (`missing_authorization_header`).
   - Header: `Authorization: Bearer <jwt_token>`
   - Acquired via `POST /v1/auth/login`.

2. **API Key Authentication** (Ingestion and programmatic access):
   - Ticket ingestion (`POST /v1/tickets` and `POST /v1/tickets/bulk`) strictly requires an API key. User JWTs return `401 Unauthorized`.
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
  - Atomically creates the ticket and enqueues a `ticket_ingest` job in `jobs` table within a single transaction.
  - Returns `202 Accepted` on new ticket creation with `{"id": "...", "status": "received", "external_id": "..."}`.
  - Returns `200 OK` if a ticket with the same `external_id` already exists for this tenant (idempotent replay, no duplicate job).
- `POST /v1/tickets/bulk`: Bulk ingest tickets via CSV (API Key only).
  - Format: Raw CSV body with headers (`subject,body,external_id,customer_email`).
  - Limits: Maximum 500 rows per request, maximum 5 MB payload size.
  - Streaming: Reads request body in chunks via `request.stream()`, counting actual incoming bytes and aborting with `422 Unprocessable Content` immediately if size exceeds 5 MB without relying solely on `Content-Length`.
  - Validation: Whole-file pre-validation. Any malformed row, missing required header, or length violation rejects the entire request with `422 Unprocessable Content` (no partial commits).
  - Atomicity: Newly created tickets and their matching `ticket_ingest` jobs commit together in a single transaction.
  - Idempotency: Existing `external_id` references within the tenant replay without creating duplicate tickets or jobs.
  - Returns `202 Accepted` with `{"received": N, "created": C, "replayed": R, "ticket_ids": [...]}`.
- `GET /v1/tickets`: List tickets for tenant (JWT or API Key).
  - Query parameters: `status` (optional), `limit` (1-200, default 50), `offset` (>= 0, default 0).
  - Returns a plain JSON array of tickets `[...]` ordered newest first (`created_at DESC, id DESC`).
- `GET /v1/tickets/{id}`: Get ticket detail by UUID (JWT or API Key).
  - Returns `200 OK` with full ticket details or `404 Not Found` if missing or belonging to another tenant.

### Browser API (`/browser-api`) - React Frontend
- `GET /browser-api/csrf-token`: Retrieve signed, session-bound CSRF token.
- `POST /browser-api/auth/login`: Authenticate browser user; issues HttpOnly `triage_session` cookie.
- `POST /browser-api/auth/logout`: Log out browser user and clear cookie.
- `GET /browser-api/auth/session`: Return active user session metadata and refreshed CSRF token.
- `GET /browser-api/tickets`: Paginated ticket list with ML predictions and accurate processing status (`queued`, `processing`, `retry_scheduled`, `completed`, `failed`, `not_found`).
- `POST /browser-api/tickets`: Submit new ticket from browser and enqueue background worker job.
- `GET /browser-api/tickets/{id}`: Ticket detail with sanitized processing status, refresh capability, and ML prediction diagnostics.

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
python -m pytest tests/test_auth_db.py tests/test_api_keys_db.py tests/test_tickets_db.py tests/test_jobs_db.py tests/test_worker_db.py tests/test_duplicates_db.py -v

# Run the dedicated two-tenant isolation test (SPEC Section 10)
python -m pytest tests/test_isolation.py -v
```

> **Note on Integration Tests & CI**:
> Database integration tests require a configured `TEST_DATABASE_URL` that resolves to a different database target than `DATABASE_URL`. They do NOT run against `DATABASE_URL`. If `TEST_DATABASE_URL` is empty, integration tests will be skipped locally.
>
> *Safety Check Limitation*: The `resolve_safe_test_db_url()` safety helper compares the parsed connection tuple `(host, port, dbname, user)`. In Supabase, connection poolers embed the project reference in the username (`postgres.<project_ref>`), which validates distinct projects when both URLs use poolers. However, a mixed configuration (e.g. direct `db.<ref>.supabase.co` vs pooler `postgres.<ref>@aws-...pooler.supabase.com`) would present different connection tuples despite pointing to the same underlying project. Distinct project references must always be maintained.
>
> In GitHub Actions CI (`.github/workflows/ci.yml`), a dedicated PostgreSQL service container using `pgvector/pgvector:pg16` is configured as a prerequisite for migration 004 to provision automatically, apply migrations using `scripts/migrate.py`, and run the test suite against the throwaway CI database.
>
> **Current CI Status & Temporary Phase-Gate Exception**: GitHub Actions CI status remains unverified on GitHub remote runners due to an observed account billing block preventing workflow execution ("The job was not started because your account is locked due to a billing issue"). The CI container image update to `pgvector/pgvector:pg16` represents a CI compatibility prerequisite correction, not a verified remote run. Under an approved temporary phase-gate exception, passing all local PostgreSQL tests against the separate `TEST_DATABASE_URL` gates development, while remote GitHub Actions verification remains explicitly unverified and outstanding.

---

## Verification & Testing Record

### 1. Automated Test Suite (FastAPI, Pytest & TypeScript)
- **Historical Full Pytest Regression Suite (`pytest -q`, prior to narrow corrections)**: 147 passed, 0 failed, 1 Starlette deprecation warning (`Using 'httpx' with 'starlette.testclient' is deprecated; install 'httpx2' instead.`, recorded separately). Note: This historical baseline reflects 133 pre-existing tests (including the 9th processing status edge-case test) plus 14 initial Phase 4 tests.
- **Duplicate Detection Focused Integration Tests (`pytest tests/test_duplicates_db.py -v`)**: 14 passed, 0 failed (verifying Phase 4 schema and repository foundations—not working automatic duplicate detection—including migration 004 rerun idempotency, vector(384) dimension constraints, `math.isfinite` cosine similarity bounds [-1.0, 1.0] rejecting NaN/inf, canonical undirected link ordering, review-state consistency enforced via repository validation [suggested => NULL, confirmed/rejected => timestamp, rejecting contradictory inputs; migration 004 defines nullable reviewed_at without status-tying DB check constraint], self-link & cross-tenant DB rejection, tenant isolation, idempotency/upsert, and cascade deletion).
- **Processing Status Focused Integration Tests (`pytest tests/test_browser_ui.py -k "test_browser_processing_status" -v`)**: 9 passed, 0 failed (verifying all 6 queue pipeline states, retry delay handling, and raw error/worker ID sanitization).
- **TypeScript Static Verification (`npx tsc --noEmit`)**: 0 type errors.
- **Production Bundle Build (`npm run build`)**: Built cleanly into `frontend/dist/` in 1.03s.

### 2. Manual Browser Testing Record (2026-11-06)
- **Ticket Submission Smoke Test (Conducted by User)**:
  *Conducted by the user in a live browser session against origin `http://localhost:5173`, clearly separated from automated test runs:*
  - **Login Flow**: User authenticated at `http://localhost:5173/login` with `demo.agent@example.com`; received HttpOnly session cookie and routed to ticket queue.
  - **Ticket Submission**: User submitted a new synthetic complaint from the browser UI form.
  - **Navigation & Processing**: Navigated immediately to ticket detail; background worker claimed and classified complaint.
  - **Prediction Display**: Detail view displayed predicted label (`Checking or savings account`), model version (`run_004`), and decision margins.
- **Processing Status & Refresh Verification (Conducted by User)**:
  - **Agent Role**: The AI agent did **not** perform live browser testing for this feature; all agent verification was conducted via automated API and TypeScript integration suites.
  - **User Verification**: The user performed manual verification in a live browser session against `http://localhost:5173` and specifically confirmed:
    1. The Processing column and "Completed" state badge display correctly in the ticket list view.
    2. The detail-page "Processing Status" card displays correctly with queue status details.
    3. The manual "↻ Refresh" button works properly and the saved prediction remains visible.
    - *Scope Note: Failed and retry states were verified via automated integration assertions; the user did not manually test failed or retry states.*

