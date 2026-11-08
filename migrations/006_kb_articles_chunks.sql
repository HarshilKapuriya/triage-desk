-- 006_kb_articles_chunks.sql - Phase 4: Tenant-scoped knowledge base articles and chunks
--
-- Supports:
-- 1. kb_articles: Tenant-owned reference documentation and help articles.
-- 2. kb_chunks: Deterministic passage chunks with 384-dimensional vector embeddings
--    for exact cosine similarity retrieval against ticket embeddings.
--
-- Referential Integrity & Multi-Tenancy:
-- - All tables include tenant_id UUID NOT NULL referencing tenants(id) ON DELETE CASCADE.
-- - kb_chunks enforces composite foreign key (tenant_id, article_id) referencing
--   kb_articles(tenant_id, id) ON DELETE CASCADE, preventing cross-tenant chunk assignment.
-- - SPEC Section 5 index on (tenant_id, id) is created for all tables.

-- 1. Create kb_articles table
CREATE TABLE kb_articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'processing' CHECK (status IN ('processing', 'ready', 'failed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Composite unique constraint to back composite foreign key
    CONSTRAINT uq_kb_articles_tenant_id_id UNIQUE (tenant_id, id)
);

-- SPEC Section 5 mandatory multi-tenant index
CREATE INDEX idx_kb_articles_tenant_id_id ON kb_articles (tenant_id, id);

-- Index for tenant-scoped article listing ordered newest first
CREATE INDEX idx_kb_articles_tenant_created ON kb_articles (tenant_id, created_at DESC);


-- 2. Create kb_chunks table
CREATE TABLE kb_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    article_id UUID NOT NULL,
    chunk_index INT NOT NULL CHECK (chunk_index >= 0),
    content TEXT NOT NULL,
    char_count INT NOT NULL CHECK (char_count > 0),
    token_count INT NOT NULL CHECK (token_count > 0),
    embedding vector(384) NULL,
    model_name TEXT NULL,
    model_version TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Enforce strict tenant-article consistency at database level
    CONSTRAINT fk_kb_chunks_tenant_article FOREIGN KEY (tenant_id, article_id)
        REFERENCES kb_articles (tenant_id, id) ON DELETE CASCADE,

    -- Enforce strictly one chunk per index per article
    CONSTRAINT uq_kb_chunks_tenant_article_chunk UNIQUE (tenant_id, article_id, chunk_index)
);

-- SPEC Section 5 mandatory multi-tenant index
CREATE INDEX idx_kb_chunks_tenant_id_id ON kb_chunks (tenant_id, id);

-- Index for looking up all chunks of an article
CREATE INDEX idx_kb_chunks_tenant_article ON kb_chunks (tenant_id, article_id);

-- Partial index for tenant-scoped exact cosine retrieval filtering by model identity
CREATE INDEX idx_kb_chunks_retrieval ON kb_chunks (tenant_id, model_name, model_version)
    WHERE embedding IS NOT NULL;
