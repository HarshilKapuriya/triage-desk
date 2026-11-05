# Phase 2 1,000-Ticket Local Verification Report

> **DISCLAIMER:**
> This report documents a **local correctness and throughput demonstration** executed against an isolated PostgreSQL test database (`TEST_DATABASE_URL`).
> It is **NOT** a production capacity claim, SLA guarantee, or enterprise scale benchmark.

---

## 1. Test Environment

| Property | Value |
| :--- | :--- |
| **Operating System** | Windows 11 (AMD64) |
| **Python Runtime** | CPython 3.14.0 |
| **Database Target** | Hosted Supabase PostgreSQL (AWS `ap-southeast-1` transaction pooler) via `TEST_DATABASE_URL` |
| **Connection Safety** | Explicitly isolated test database; canonical host check verified distinct from `DATABASE_URL` |
| **Worker Concurrency** | 6 concurrent `WorkerService` worker threads |
| **Tenants Evaluated** | 2 isolated tenants (Tenant A: `aeedf718-413b-4474-af35-d7f07c130dc4`, Tenant B: `e2d05a79-055d-47ef-80dd-16d6d8864da1`) |

---

## 2. Ingestion Results (`POST /v1/tickets/bulk`)

Each tenant ingested 500 synthetic tickets via a single raw CSV upload over HTTP.

| Metric | Result |
| :--- | :--- |
| **Total Tickets Ingested** | 1,000 (500 for Tenant A + 500 for Tenant B) |
| **Batch Size per Request** | 500 rows |
| **HTTP Status Code** | `202 Accepted` |
| **Elapsed Ingestion Time** | 154.00 seconds |
| **Ingestion Throughput** | 6.5 tickets/second |
| **New Tickets Created** | 1,000 (500 Tenant A, 500 Tenant B) |
| **Replayed Tickets** | 0 |
| **Initial Jobs Created** | 1,000 (all status: `queued`, attempts: 0) |

---

## 3. Worker Processing Results (`WorkerService`)

Workers polled `jobs` concurrently using `SELECT ... FOR UPDATE SKIP LOCKED` scoped strictly to the two test tenants.

> **Important Note on Throughput:**
> The processing throughput reported below (9.0 jobs/second) was achieved using **six concurrent worker threads running in parallel**, not a single worker thread.

| Metric | Result |
| :--- | :--- |
| **Total Jobs Processed** | 1,000 |
| **Worker Concurrency** | 6 concurrent worker threads |
| **Completed Jobs** | 1000 |
| **Remaining Queued Jobs** | 0 |
| **Lingering Running Jobs** | 0 |
| **Failed Jobs** | 0 |
| **Elapsed Processing Time**| 111.53 seconds |
| **Processing Throughput** | 9.0 jobs/second (aggregate across 6 concurrent worker threads; ~1.5 jobs/sec per worker thread) |
| **Final Ticket States** | 1000 `open` (0 remaining `new`) |
| **Audit Log Events Created**| 1000 `ticket.processed` events |

---

## 4. Correctness Invariants Verified

1. **Transactional Atomicity**:
   - Tickets and matching `ticket_ingest` jobs committed together atomically in the bulk upload transaction.
2. **Composite Foreign Key Integrity**:
   - Foreign key `(tenant_id, ticket_id) REFERENCES tickets(tenant_id, id)` verified across all 1,000 rows.
3. **Queue Concurrency & Lease Generation**:
   - Competing worker threads claimed jobs using `FOR UPDATE SKIP LOCKED` without lock contention, deadlocks, or generation preemption failures.
4. **Ticket Transition Rules**:
   - Exactly 1,000 tickets transitioned from `new` to `open` upon job completion.
   - Raw subject and body preserved without mutation.
5. **Audit Trail Completeness**:
   - Exactly 1,000 `ticket.processed` audit events recorded with matching tenant and ticket identifiers.
6. **Dual-Tenant HTTP Isolation with Pagination**:
   - `GET /v1/tickets?limit=200&offset=N` paginated through all 500 tickets for Tenant A and Tenant B.
   - Verified 100% of tickets returned to Tenant A belonged strictly to Tenant A; 0 cross-tenant leaks.
   - Verified 100% of tickets returned to Tenant B belonged strictly to Tenant B; 0 cross-tenant leaks.
7. **Idempotent Replay Resilience**:
   - Re-uploading the identical 500-ticket CSVs to both tenants returned HTTP `202 Accepted` with `created: 0, replayed: 500`.
   - Verified zero duplicate tickets and zero duplicate jobs created in the database.

---

## 5. Limitations

1. **Network Latency Impact**:
   - Local test client communicating over public internet with AWS `ap-southeast-1` Supabase pooler adds ~50–90ms latency per batch network roundtrip compared to a co-located production VPC.
2. **Worker Concurrency Limit**:
   - Local concurrency was bounded to 6 threads to stay within the free-tier PostgreSQL connection pool limit. The 9.0 jobs/sec throughput is an aggregate multi-worker rate, not single-worker performance.
3. **Scope Boundary**:
   - Evaluated core transactional queueing, worker execution, conditional state transition, and bulk CSV parsing. No LLM reply drafting, embeddings, or dashboard rendering was included in Phase 2 scope.
