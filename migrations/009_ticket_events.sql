-- 009_ticket_events.sql - Phase 5: Operational event history for ticket lifecycles and SLA telemetry
--
-- Supports:
-- - Immutable, append-only operational event logging for ticket lifecycle transitions.
-- - Capture of business occurrence time (occurred_at) alongside database recording time (recorded_at).
-- - Point-in-time SLA policy intake snapshots and transition telemetry without customer PII.
--
-- Concurrency & Idempotency:
-- - Partial UNIQUE indexes enforce one-time event guarantees per ticket for single-occurrence transitions.
-- - Composite foreign key on (tenant_id, ticket_id) strictly prevents cross-tenant event association.
-- - NOW() evaluates to transaction-start time in PostgreSQL, not commit time. Deterministic tie-breaker
--   (occurred_at ASC, recorded_at ASC, id ASC) provides stable replay ordering.

CREATE TABLE ticket_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    ticket_id UUID NOT NULL,
    event_type VARCHAR(64) NOT NULL CHECK (
        event_type IN (
            'ticket_created',
            'ticket_processed',
            'first_response_recorded',
            'ticket_resolved',
            'ticket_closed'
        )
    ),
    occurred_at TIMESTAMPTZ NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    actor_type VARCHAR(32) NOT NULL CHECK (actor_type IN ('system', 'agent', 'worker', 'customer')),
    actor_id VARCHAR(128) NULL,
    schema_version INT NOT NULL DEFAULT 1 CHECK (schema_version >= 1),
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Enforce strict composite multi-tenant integrity matching 002/003/004 patterns
    -- Backed by uq_tickets_tenant_id_id on tickets(tenant_id, id) from 002_jobs.sql
    CONSTRAINT fk_ticket_events_tickets FOREIGN KEY (tenant_id, ticket_id)
        REFERENCES tickets (tenant_id, id) ON DELETE CASCADE,

    -- Multi-tenant unique index backing SPEC Section 5 tenant-scoped lookup
    -- Avoids separate redundant CREATE INDEX on (tenant_id, id)
    CONSTRAINT uq_ticket_events_tenant_id_id UNIQUE (tenant_id, id)
);

-- Purpose: Chronological retrieval of a ticket's event history strictly scoped by tenant
-- Deterministic tie-breakers: occurred_at ASC, recorded_at ASC, id ASC
CREATE INDEX idx_ticket_events_lookup
    ON ticket_events (tenant_id, ticket_id, occurred_at ASC, recorded_at ASC, id ASC);

-- Database-level one-time occurrence constraints per ticket:
-- 1. Exactly one intake/creation event per ticket
CREATE UNIQUE INDEX uq_ticket_events_ticket_created
    ON ticket_events (tenant_id, ticket_id) WHERE event_type = 'ticket_created';

-- 2. Exactly one worker processing transition from 'new' to 'open' per ticket
CREATE UNIQUE INDEX uq_ticket_events_ticket_processed
    ON ticket_events (tenant_id, ticket_id) WHERE event_type = 'ticket_processed';

-- 3. Exactly one first-response recorded event per ticket
CREATE UNIQUE INDEX uq_ticket_events_first_response
    ON ticket_events (tenant_id, ticket_id) WHERE event_type = 'first_response_recorded';

-- 4. Exactly one resolution event per ticket
CREATE UNIQUE INDEX uq_ticket_events_resolved
    ON ticket_events (tenant_id, ticket_id) WHERE event_type = 'ticket_resolved';

-- 5. Exactly one closing event per ticket
CREATE UNIQUE INDEX uq_ticket_events_closed
    ON ticket_events (tenant_id, ticket_id) WHERE event_type = 'ticket_closed';

