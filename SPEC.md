# Ticket Intelligence Platform: SPEC

This is the single source of truth for the coding agent. Read the whole file at the start of every session. Work only on the current phase (section 11). If a request conflicts with this file, stop and ask before changing anything.

## 1. What we are building

A multi-tenant platform that receives customer support tickets through an API and uses small CPU-only ML models to classify, prioritize, deduplicate, suggest replies for, and predict SLA breaches on them. Each company (tenant) sees only its own data.

MVP features: tenant signup and login, API keys, ticket ingestion (API and CSV), category and priority prediction, duplicate detection, knowledge-base reply suggestions (RAG), SLA breach risk, a dashboard, and agent feedback buttons.

## 2. Hard constraints

- The target machine has 4 to 8 GB of RAM and no GPU. Everything must run on it.
- Everything must be free: free tiers only, no paid services, no credit card.
- No Docker for local development (Docker is allowed in CI only). No local LLMs, no Redis, no Celery, no Kubernetes.
- No new dependency without asking first. Prefer the standard library and the packages listed in section 3.
- Anything heavier than scikit-learn or LightGBM is trained on Kaggle or Colab, never locally.
- Never commit secrets, real data, or large model files. Download models at runtime.
- Never use real tickets from the owner's employer. Use public datasets and synthetic tickets only.

## 3. Stack (fixed)

| Layer | Choice |
| --- | --- |
| Language | Python 3.11 or newer |
| API | FastAPI with Uvicorn |
| Database | Supabase hosted PostgreSQL, accessed with psycopg 3 and a connection pool. Hand-written SQL lives only in the repository layer |
| Migrations | Numbered `.sql` files in `migrations/`, applied by `scripts/migrate.py`, which records applied versions in a `schema_migrations` table |
| Vector search | pgvector, `vector(384)` columns in the same database (phase 4) |
| Job queue | A `jobs` table claimed with `SELECT ... FOR UPDATE SKIP LOCKED` by one worker process (phase 2) |
| Auth | Own implementation: argon2-cffi password hashes, PyJWT (HS256) tokens for dashboard users, hashed API keys for ingestion |
| ML | scikit-learn, LightGBM, sentence-transformers `all-MiniLM-L6-v2` (phase 3 onward) |
| Reply drafting | Hosted LLM API (Gemini or Groq free tier) behind a provider interface. Optional (phase 4) |
| Frontend | Server-rendered Jinja2 templates with HTMX and Chart.js (phase 5). No UI in phase 1 |
| Tests and CI | pytest, GitHub Actions |

## 4. Repository layout (proposed, change it if you prefer)

```
app/
  main.py            FastAPI app and router registration
  config.py          settings read from environment variables
  db.py              connection pool and transaction helper
  security.py        password hashing, JWT, API key generation and hashing
  deps.py            request dependencies that resolve the Principal
  repositories/      ALL SQL lives here; every function takes tenant_id
  routers/           auth.py, api_keys.py, tickets.py, health.py
  worker/            (phase 2)
  ml/                (phase 3 onward)
  templates/         (phase 5)
migrations/          001_init.sql, 002_...
scripts/migrate.py
tests/
.github/workflows/ci.yml
.env.example
.gitignore
SPEC.md
DECISIONS.md
README.md
```

## 5. Data model

Every table except `tenants` has a `tenant_id uuid not null references tenants(id)` column and an index on `(tenant_id, id)`.

### Created in phase 1

**tenants**: `id uuid primary key default gen_random_uuid()`, `name text not null`, `plan text not null default 'free'`, `created_at timestamptz not null default now()`.

**users**: `id`, `tenant_id`, `email text not null unique` (stored lowercase, one account belongs to one tenant), `password_hash text not null`, `role text not null check (role in ('admin','agent'))`, `created_at`.

**api_keys**: `id`, `tenant_id`, `key_prefix text not null` (first 8 characters, for display), `key_hash text not null unique`, `label text not null`, `created_at`, `last_used_at timestamptz null`, `revoked_at timestamptz null`.

**tickets**: `id`, `tenant_id`, `external_id text null`, `subject text not null`, `body text not null`, `customer_email text null`, `status text not null default 'new' check (status in ('new','open','pending','resolved','closed'))`, `category text null`, `priority text null check (priority in ('low','normal','high','urgent'))`, `created_at`, `first_response_at timestamptz null`, `resolved_at timestamptz null`, `assignee_id uuid null references users(id)`. Add a partial unique index on `(tenant_id, external_id) where external_id is not null` and an index on `(tenant_id, status, created_at desc)`.

**audit_log**: `id bigint generated always as identity primary key`, `tenant_id`, `actor text not null` (a user id or `api_key:<id>`), `action text not null`, `at timestamptz not null default now()`.

### Created in later phases (do not create yet)

`ticket_predictions`, `ticket_embeddings`, `duplicate_links`, `kb_articles`, `kb_chunks`, `reply_suggestions`, `sla_policies`, `jobs`, `model_versions`. Column lists are in the design document.

## 6. Tenant isolation rules (non-negotiable)

1. The tenant always comes from the credential: an API key is hashed and looked up, and a JWT carries a `tenant_id` claim. Never read the tenant from the request body, query string or URL.
2. All database access goes through `app/repositories/`. Every tenant-owned repository function takes `tenant_id` as a required argument and uses it in every query. No SQL anywhere else.
   - Clarification: The only pre-authentication exceptions are:
     a) Creating a new tenant during signup.
     b) Looking up a user by globally unique normalized email for login.
     c) Looking up an API key by its unique hash for authentication.
   - These exceptions must live in dedicated, clearly named repository functions and are not general-purpose cross-tenant query interfaces. Once authentication resolves the tenant, all subsequent tenant-owned operations must strictly use that resolved tenant_id. Never accept a tenant_id from the signup/login request to bypass this.
3. A resource that belongs to another tenant returns `404`, never `403`, so its existence is not revealed.
4. API keys are stored only as SHA-256 hashes, shown in full once at creation, and revocable.
5. The two-tenant isolation test (section 10) must stay green. A change that breaks it is not allowed.

## 7. API contract for phase 1

All paths are under `/v1` except `/healthz`. All errors use the body `{"error": {"code": "<machine_code>", "message": "<text>"}}`.

| Endpoint | Auth | Request | Success response | Errors |
| --- | --- | --- | --- | --- |
| `POST /v1/auth/signup` | None | `company_name`, `email`, `password` (at least 8 characters) | `201` with `tenant_id`, `user_id`, `access_token`, `token_type` | `409` email already registered, `422` invalid input |
| `POST /v1/auth/login` | None | `email`, `password` | `200` with `access_token`, `token_type` | `401` with the same message for unknown email or wrong password |
| `POST /v1/api-keys` | JWT, admin | `label` | `201` with `id`, `label`, `key` (full key, shown once) | `401`, `403` not an admin |
| `DELETE /v1/api-keys/{id}` | JWT, admin | none | `204`, sets `revoked_at` | `401`, `403`, `404` |
| `POST /v1/tickets` | API key | `subject` (up to 300 characters), `body` (up to 20,000), optional `external_id`, optional `customer_email` | `202` with `id` and `status: "new"`. If `external_id` already exists for this tenant, return the existing ticket with `200` | `401`, `422` |
| `GET /v1/tickets` | JWT or API key | optional `status`, `limit` (default 50, maximum 200), `offset` | `200` with a list, newest first | `401` |
| `GET /v1/tickets/{id}` | JWT or API key | none | `200` with the ticket | `401`, `404` |
| `GET /healthz` | None | none | `200` with `{"status": "ok"}` | none |

Details:

- **Signup** creates the tenant and its first admin user in one database transaction, and writes an `audit_log` row.
- **JWT**: HS256, expires after `JWT_EXPIRE_MINUTES` (default 60), claims `sub` (user id), `tenant_id`, `role`.
- **API key format**: `tk_` followed by `secrets.token_urlsafe(32)`. Store `key_prefix` (first 8 characters) and the SHA-256 hex digest. Passwords use argon2, not SHA-256.
- **A revoked key** returns `401`. Update `last_used_at` when a key is used.
- **Ticket creation in phase 1** only stores the ticket. The `202` status is kept now so the contract does not change when phase 2 adds the queue and worker.
- A small dependency in `deps.py` resolves every request to a `Principal(tenant_id, kind, id, role)` where `kind` is `user` or `api_key`. Routers use only `principal.tenant_id`.

## 8. Configuration

Read all settings from environment variables. Commit `.env.example` with placeholders only.

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | Supabase Postgres connection string |
| `TEST_DATABASE_URL` | A separate database used only by tests |
| `JWT_SECRET` | Long random string for signing tokens |
| `JWT_EXPIRE_MINUTES` | Token lifetime, default 60 |

Supabase note: the direct connection string may be IPv6 only. If it fails to connect from your network, use the session pooler connection string from the Supabase dashboard instead.

Tests need a throwaway database. In CI use a PostgreSQL service container. Locally, point `TEST_DATABASE_URL` at a second free Supabase project, never at the project that holds real development data.

## 9. Phase 1 deliverables

- [ ] Repository with the layout in section 4, `.gitignore`, `.env.example` and a README that explains how to run the API and the tests
- [ ] Supabase project created, and `scripts/migrate.py` applies `001_init.sql` for the five phase 1 tables
- [ ] Signup, login, API key creation and revocation working as in section 7
- [ ] `POST /v1/tickets`, `GET /v1/tickets` and `GET /v1/tickets/{id}` working with both credential types where allowed
- [ ] Repository layer in place with `tenant_id` required on every function
- [ ] Tests for: signup, duplicate email, login failure, admin-only key creation, revoked key rejected, ticket idempotency by `external_id`
- [ ] The two-tenant isolation test (section 10)
- [ ] GitHub Actions workflow that runs the tests on every push

Phase 1 is done when all boxes are ticked and CI is green.

*Temporary Phase-Gate Exception (Approved 2026-11-05)*: Passing local PostgreSQL tests against the separate `TEST_DATABASE_URL` may gate Phase 2 development while GitHub CI is blocked due to the repository account billing issue. GitHub CI verification remains an explicitly outstanding requirement and is not waived.

## 10. The two-tenant isolation test

Write this as one test module, `tests/test_isolation.py`, and run it in CI.

1. Sign up tenant A and tenant B. Create one API key for each.
2. Using A's key, create three tickets. Using B's key, create two tickets.
3. List tickets with A's credentials and assert that exactly A's three tickets come back and none of B's ids appear. Repeat with B's credentials.
4. Request each of B's ticket ids with A's credentials and assert `404`. Repeat in the other direction.
5. Using A's admin token, try to revoke B's API key id and assert `404`, then confirm B's key still works.
6. Assert that the same `external_id` can exist in both tenants without conflict.

## 11. Phases (reference, do not start a phase early)

| Phase | Weeks | Deliverable | Done when |
| --- | --- | --- | --- |
| 1. Foundations | 1 to 2 | This file's phase 1 deliverables | Isolation test passes in CI |
| 2. Ingestion and queue | 3 to 4 | CSV bulk upload, `jobs` table, worker process with retries | 1,000 tickets processed with no lost jobs |
| 3. Core ML | 5 to 7 | Category and priority models compared, `model_versions` records | Metrics beat the baseline and are saved in the repo |
| 4. Duplicates and knowledge base | 8 to 10 | Embeddings, duplicate links, article upload, retrieval, cited draft replies, feedback buttons | Retrieval recall@k measured on a hand-made question set |
| 5. SLA and dashboard | 11 to 12 | SLA risk model with rescoring, dashboard pages | Dashboard shows live numbers for two tenants |
| 6. Hardening and deploy | 13 to 14 | Rate limits, audit log coverage, row-level security, load test, hosted demo | Demo online, load-test numbers recorded |
| 7. Evaluation and report | 15 to 16 | Final metrics, user feedback, report, viva rehearsal | Guide has reviewed a full draft |

## 12. Rules for the agent

- Build one feature per session, with tests, then stop and summarize what changed.
- Do not swap the database, framework or ML libraries without asking.
- Append one line to `DECISIONS.md` for every non-trivial design choice, with the reason.
- Training and evaluation scripts write their metrics to a file and never overwrite earlier results.
- Do not add features, endpoints, tables or dependencies that are not in this file.
- Never expose the Supabase service role key, and never print secrets in logs or test output.
- If a free-tier limit blocks the work, say so and propose an option. Do not suggest a paid plan.

## 13. Out of scope

Live chat, voice, payments and billing, a mobile app, running an email server, local LLMs, training transformer models on the laptop, Docker Desktop, Kubernetes. Email ingestion, per-tenant retraining and usage metering are stretch items after phase 5.
