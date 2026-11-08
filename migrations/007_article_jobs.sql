-- 007_article_jobs.sql - Phase 4: Dedicated background queue for knowledge base article embedding
--
-- Supports:
-- Dedicated asynchronous queue for computing vector embeddings for knowledge-base article chunks.
-- Keeps the working ticket queue and pipeline completely unchanged and independent.
--
-- Multi-Tenancy & Referential Integrity:
-- - Composite foreign key (tenant_id, article_id) REFERENCES kb_articles(tenant_id, id) ON DELETE CASCADE.
-- - One active/pending embedding job per tenant and article via UNIQUE (tenant_id, article_id, type).
-- - Renewable lease-based locking with worker generation counter (attempts) and fenced atomic completion.

CREATE TABLE article_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    article_id UUID NOT NULL,
    type TEXT NOT NULL DEFAULT 'article_embedding' CHECK (type IN ('article_embedding')),
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

    -- Enforce strict tenant-article consistency at database level
    CONSTRAINT fk_article_jobs_tenant_article FOREIGN KEY (tenant_id, article_id)
        REFERENCES kb_articles (tenant_id, id) ON DELETE CASCADE,

    -- Enforce strictly one embedding job per tenant and article
    CONSTRAINT uq_article_jobs_tenant_article_type UNIQUE (tenant_id, article_id, type)
);

-- SPEC Section 5 mandatory multi-tenant index
CREATE INDEX idx_article_jobs_tenant_id_id ON article_jobs (tenant_id, id);

-- High-performance queue claim index (oldest eligible queued jobs first)
CREATE INDEX idx_article_jobs_claim ON article_jobs (run_after, created_at) WHERE status = 'queued';

-- Stale/orphaned lease recovery index
CREATE INDEX idx_article_jobs_stale_recovery ON article_jobs (status, locked_at) WHERE status = 'running';
