-- 004_embeddings_duplicates.sql - Phase 4 Ticket Embeddings and Duplicate Links Schema
-- Implements multi-tenant tables: ticket_embeddings and duplicate_links
-- Strict tenant isolation: every table has tenant_id referencing tenants(id) and index on (tenant_id, id).
-- Foreign keys enforce composite tenant consistency (tenant_id, ticket_id) and (tenant_id, source/target_ticket_id).

-- 1. Enable pgvector extension (available in Supabase PostgreSQL)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. ticket_embeddings: Store vector embeddings of ticket content for similarity search and deduplication
-- vector(384) corresponds to the 384-dimensional dense vector embeddings produced by sentence-transformers all-MiniLM-L6-v2.
-- Approximate vector indexing (HNSW / IVFFlat) is deliberately deferred until retrieval scale and evaluation justify it.
CREATE TABLE IF NOT EXISTS ticket_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    ticket_id UUID NOT NULL,
    embedding vector(384) NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Enforce strict tenant-ticket referential integrity at the database level
    CONSTRAINT fk_ticket_embeddings_tenant_ticket FOREIGN KEY (tenant_id, ticket_id)
        REFERENCES tickets (tenant_id, id) ON DELETE CASCADE,

    -- Guarantee idempotency and prevent duplicate embeddings for the same ticket and model version
    CONSTRAINT uq_ticket_embeddings_ticket_model UNIQUE (tenant_id, ticket_id, model_name, model_version),
    -- Back composite foreign key references from tenant-scoped child tables if needed
    CONSTRAINT uq_ticket_embeddings_tenant_id_id UNIQUE (tenant_id, id)
);

-- SPEC Section 5 mandatory multi-tenant index
CREATE INDEX IF NOT EXISTS idx_ticket_embeddings_tenant_id_id ON ticket_embeddings (tenant_id, id);
-- Lookup index by ticket
CREATE INDEX IF NOT EXISTS idx_ticket_embeddings_ticket_id ON ticket_embeddings (tenant_id, ticket_id);


-- 3. duplicate_links: Store pairwise similarity links between tickets within the same tenant
-- Relationship semantics: Undirected duplicate relation enforced canonically (source_ticket_id < target_ticket_id).
-- Metric: Cosine similarity between normalized sentence embeddings, bounded in [-1.0, 1.0].
CREATE TABLE IF NOT EXISTS duplicate_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    source_ticket_id UUID NOT NULL,
    target_ticket_id UUID NOT NULL,
    similarity_score DOUBLE PRECISION NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'suggested' CHECK (status IN ('suggested', 'confirmed', 'rejected')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ NULL,

    -- Enforce strict tenant ownership: both tickets must belong to the exact same tenant
    CONSTRAINT fk_duplicate_links_tenant_source FOREIGN KEY (tenant_id, source_ticket_id)
        REFERENCES tickets (tenant_id, id) ON DELETE CASCADE,
    CONSTRAINT fk_duplicate_links_tenant_target FOREIGN KEY (tenant_id, target_ticket_id)
        REFERENCES tickets (tenant_id, id) ON DELETE CASCADE,

    -- Disallow linking a ticket to itself
    CONSTRAINT chk_duplicate_links_no_self_link CHECK (source_ticket_id <> target_ticket_id),

    -- Undirected canonical ordering: source_ticket_id < target_ticket_id
    -- Rejects inverted/reversed pairs at the database constraint level
    CONSTRAINT chk_duplicate_links_canonical_order CHECK (source_ticket_id < target_ticket_id),

    -- Cosine similarity metric score contract: mathematical range [-1.0, 1.0]
    CONSTRAINT chk_duplicate_links_similarity_score CHECK (similarity_score >= -1.0 AND similarity_score <= 1.0),

    -- Uniqueness per undirected ticket pair within a tenant
    CONSTRAINT uq_duplicate_links_tenant_pair UNIQUE (tenant_id, source_ticket_id, target_ticket_id),
    CONSTRAINT uq_duplicate_links_tenant_id_id UNIQUE (tenant_id, id)
);

-- SPEC Section 5 mandatory multi-tenant index
CREATE INDEX IF NOT EXISTS idx_duplicate_links_tenant_id_id ON duplicate_links (tenant_id, id);
-- Query indices for finding duplicate links attached to a ticket
CREATE INDEX IF NOT EXISTS idx_duplicate_links_source_ticket ON duplicate_links (tenant_id, source_ticket_id);
CREATE INDEX IF NOT EXISTS idx_duplicate_links_target_ticket ON duplicate_links (tenant_id, target_ticket_id);
-- Filter index by status (e.g., finding unreviewed 'suggested' pairs)
CREATE INDEX IF NOT EXISTS idx_duplicate_links_status ON duplicate_links (tenant_id, status);
