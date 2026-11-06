-- 003_ml_predictions.sql - Phase 3 ML Model Versions and Predictions Schema
-- Implements multi-tenant tables: model_versions and ticket_predictions
-- Strict tenant isolation: every table has tenant_id referencing tenants(id) and index on (tenant_id, id).
-- Foreign keys enforce composite tenant consistency (tenant_id, ticket_id) and (tenant_id, model_version_id).

-- 1. model_versions: Track active and historic ML model versions per tenant
CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    task TEXT NOT NULL CHECK (task IN ('category', 'priority')),
    artifact_path TEXT NOT NULL,
    sha256_hash TEXT NOT NULL,
    classes JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Enforce unique model version per tenant, task, and name
    CONSTRAINT uq_model_versions_tenant_task_name_version UNIQUE (tenant_id, task, name, version),
    -- Back composite foreign key references from tenant-scoped child tables
    CONSTRAINT uq_model_versions_tenant_id_id UNIQUE (tenant_id, id)
);

-- SPEC Section 5 mandatory multi-tenant index
CREATE INDEX IF NOT EXISTS idx_model_versions_tenant_id_id ON model_versions (tenant_id, id);
-- Index to quickly resolve active model for a task within a tenant
CREATE INDEX IF NOT EXISTS idx_model_versions_tenant_task_active ON model_versions (tenant_id, task, is_active) WHERE is_active = TRUE;


-- 2. ticket_predictions: Store inference outputs separately from human-curated ticket values
CREATE TABLE IF NOT EXISTS ticket_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    ticket_id UUID NOT NULL,
    model_version_id UUID NOT NULL,
    task TEXT NOT NULL CHECK (task IN ('category', 'priority')),
    predicted_label TEXT NOT NULL,
    confidence DOUBLE PRECISION NULL,          -- Explicitly NULL for uncalibrated models (e.g. LinearSVC)
    probabilities JSONB NULL,                 -- Calibrated class probabilities if available
    decision_scores JSONB NULL,               -- Uncalibrated decision margins / distances
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Enforce strict tenant-ticket and tenant-model consistency at database level
    CONSTRAINT fk_ticket_predictions_tenant_ticket FOREIGN KEY (tenant_id, ticket_id)
        REFERENCES tickets (tenant_id, id) ON DELETE CASCADE,
    CONSTRAINT fk_ticket_predictions_tenant_model FOREIGN KEY (tenant_id, model_version_id)
        REFERENCES model_versions (tenant_id, id) ON DELETE CASCADE,

    -- Guarantee idempotency under retries: at most one prediction per tenant, ticket, and task
    CONSTRAINT uq_ticket_predictions_tenant_ticket_task UNIQUE (tenant_id, ticket_id, task),
    CONSTRAINT uq_ticket_predictions_tenant_id_id UNIQUE (tenant_id, id)
);

-- SPEC Section 5 mandatory multi-tenant index
CREATE INDEX IF NOT EXISTS idx_ticket_predictions_tenant_id_id ON ticket_predictions (tenant_id, id);
-- Fast ticket detail lookup for predictions
CREATE INDEX IF NOT EXISTS idx_ticket_predictions_ticket_id ON ticket_predictions (tenant_id, ticket_id);

