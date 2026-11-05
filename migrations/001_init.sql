# 001_init.sql - Phase 1 initial schema
-- Implements multi-tenant tables: tenants, users, api_keys, tickets, audit_log
-- Every table (except tenants) has tenant_id referencing tenants(id) and an index on (tenant_id, id).

-- Enable pgcrypto / gen_random_uuid support (built-in in PostgreSQL 13+)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 1. tenants
CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    plan TEXT NOT NULL DEFAULT 'free',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. users
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'agent')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_tenant_id_id ON users (tenant_id, id);

-- 3. api_keys
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    key_prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ NULL,
    revoked_at TIMESTAMPTZ NULL
);

CREATE INDEX IF NOT EXISTS idx_api_keys_tenant_id_id ON api_keys (tenant_id, id);

-- 4. tickets
CREATE TABLE IF NOT EXISTS tickets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    external_id TEXT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    customer_email TEXT NULL,
    status TEXT NOT NULL DEFAULT 'new' CHECK (status IN ('new', 'open', 'pending', 'resolved', 'closed')),
    category TEXT NULL,
    priority TEXT NULL CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    first_response_at TIMESTAMPTZ NULL,
    resolved_at TIMESTAMPTZ NULL,
    assignee_id UUID NULL REFERENCES users(id) ON DELETE SET NULL
);

-- Required indices for tickets
CREATE INDEX IF NOT EXISTS idx_tickets_tenant_id_id ON tickets (tenant_id, id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_tickets_tenant_id_external_id ON tickets (tenant_id, external_id) WHERE external_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tickets_tenant_status_created_at ON tickets (tenant_id, status, created_at DESC);

-- 5. audit_log
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_log_tenant_id_id ON audit_log (tenant_id, id);

