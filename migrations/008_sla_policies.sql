-- 008_sla_policies.sql - Phase 5: Service Level Agreement (SLA) policies
--
-- Supports:
-- Tenant-scoped SLA target policies keyed by priority ('low', 'normal', 'high', 'urgent', 'default').
-- Independent first-response and resolution target durations in integer minutes.
--
-- Multi-Tenancy & Integrity:
-- - tenant_id NOT NULL REFERENCES tenants(id) ON DELETE CASCADE.
-- - One policy per priority key per tenant via UNIQUE (tenant_id, priority).
-- - Target durations bounded to positive integer minutes (1 minute to 525,600 minutes = 365 days).
-- - Integer minutes units are explicitly documented and enforced.

CREATE TABLE sla_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    priority TEXT NOT NULL CHECK (priority IN ('low', 'normal', 'high', 'urgent', 'default')),
    first_response_minutes INT NOT NULL CHECK (first_response_minutes > 0 AND first_response_minutes <= 525600),
    resolution_minutes INT NOT NULL CHECK (resolution_minutes > 0 AND resolution_minutes <= 525600),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_sla_policies_tenant_priority UNIQUE (tenant_id, priority)
);

-- SPEC Section 5 mandatory multi-tenant index
CREATE INDEX idx_sla_policies_tenant_id_id ON sla_policies (tenant_id, id);
