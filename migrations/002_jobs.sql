-- 002_jobs.sql - Phase 2 jobs queue schema and ticket constraints

-- 1. Add composite unique constraint to tickets to back composite foreign key
ALTER TABLE tickets 
    ADD CONSTRAINT uq_tickets_tenant_id_id UNIQUE (tenant_id, id);

-- 2. Create jobs table with composite tenant-ticket foreign key
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    ticket_id UUID NOT NULL,
    type TEXT NOT NULL DEFAULT 'ticket_ingest' CHECK (type IN ('ticket_ingest')),
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    attempts INT NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    max_attempts INT NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
    run_after TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    locked_at TIMESTAMPTZ NULL,
    locked_by TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ NULL,
    failed_at TIMESTAMPTZ NULL,
    error TEXT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::JSONB,

    -- Enforce strict tenant-ticket consistency at database level
    CONSTRAINT fk_jobs_tenant_ticket FOREIGN KEY (tenant_id, ticket_id)
        REFERENCES tickets (tenant_id, id) ON DELETE CASCADE,

    -- Enforce strictly one job per tenant/ticket/type across all states for Phase 2
    CONSTRAINT uq_jobs_tenant_ticket_type UNIQUE (tenant_id, ticket_id, type)
);

-- SPEC Section 5 mandatory multi-tenant index
CREATE INDEX idx_jobs_tenant_id_id ON jobs (tenant_id, id);

-- High-performance queue claim index (oldest eligible queued jobs first)
CREATE INDEX idx_jobs_claim ON jobs (run_after, created_at) WHERE status = 'queued';

-- Stale/orphaned lease recovery index
CREATE INDEX idx_jobs_stale_recovery ON jobs (status, locked_at) WHERE status = 'running';

