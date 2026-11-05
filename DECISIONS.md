# Decisions

One line per decision: what was chosen, why, and what was rejected. Rewrite the reasons in your own words as you learn them, because these become your report and your viva answers. Add the date for every new entry.

- 2026-11-05: Build a multi-tenant support ticket intelligence platform. Reason: it is a real product category, mixes software engineering with ML, and matches my work in technical support. Rejected: crop disease detection, scam message detector, and other single-model projects, because they are not platforms.
- 2026-11-05: One shared database with a `tenant_id` on every row. Reason: simplest design that is still multi-tenant, and cheap on a free tier. Rejected: a database or schema per tenant, which is heavier to run and migrate.
- 2026-11-05: Supabase hosted PostgreSQL as the database (my choice). Reason: free, uses no RAM on my low-spec PC, and supports pgvector. Trade-offs: needs an internet connection, free projects pause after 1 week of inactivity, 500 MB size limit, no automatic backups.
- 2026-11-05: Job queue is a `jobs` table claimed with `FOR UPDATE SKIP LOCKED`. Reason: no extra service to run. Rejected: Redis and Celery, which cost memory my machine does not have.
- 2026-11-05: Own login and API-key system instead of Supabase Auth. Reason: I can explain every part in the viva. Rejected: Supabase Auth, which is faster to build but hides the interesting parts.
- 2026-11-05: Reply drafting uses a hosted LLM (Gemini or Groq free tier) behind a provider interface, and it is optional. Reason: no local LLM fits my PC, and free limits can change. Rejected: running a local model with Ollama.
- 2026-11-05: Server-rendered pages with Jinja2 and HTMX for the dashboard. Reason: avoids a Node dev server eating RAM. Rejected: a React single-page app.
- 2026-11-05: Phase 1 includes a minimal ticket create and list. Reason: the isolation test needs tenant-owned data before the queue exists. Ticket creation returns `202` already so the contract does not change in phase 2.
