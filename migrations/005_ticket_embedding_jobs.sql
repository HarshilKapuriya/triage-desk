-- 005_ticket_embedding_jobs.sql - Phase 4 Slice 3: Allow ticket_embedding jobs in queue
-- Expands allowed job types from ('ticket_ingest') to ('ticket_ingest', 'ticket_embedding').
-- The composite unique constraint (tenant_id, ticket_id, type) already handles uniqueness per type.

ALTER TABLE jobs DROP CONSTRAINT IF EXISTS jobs_type_check;
ALTER TABLE jobs ADD CONSTRAINT jobs_type_check CHECK (type IN ('ticket_ingest', 'ticket_embedding'));
