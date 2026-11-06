"""ML repository layer for multi-tenant model versions and ticket predictions.

Per SPEC.md Section 5, 6:
- All database access requires tenant_id and uses it in every query.
- No global-tenant bypass: every tenant has their own active model_versions record.
- Composite foreign keys guarantee database-level multi-tenant isolation.
- Prediction writes are idempotent on (tenant_id, ticket_id, task).
"""

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any
import uuid
import psycopg
from psycopg.rows import dict_row


@dataclass(frozen=True)
class ModelVersionRecord:
    """Read-only view of a model version record."""
    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    version: str
    task: str
    artifact_path: str
    sha256_hash: str
    classes: list[str]
    metadata: dict[str, Any]
    is_active: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class PredictionRecord:
    """Read-only view of a ticket prediction record."""
    id: uuid.UUID
    tenant_id: uuid.UUID
    ticket_id: uuid.UUID
    model_version_id: uuid.UUID
    task: str
    predicted_label: str
    confidence: float | None
    probabilities: dict[str, float] | None
    decision_scores: dict[str, float] | None
    metadata: dict[str, Any]
    created_at: datetime


def _row_to_model_version(row: dict[str, Any]) -> ModelVersionRecord:
    classes_val = row["classes"]
    if isinstance(classes_val, str):
        classes_val = json.loads(classes_val)
    metadata_val = row["metadata"]
    if isinstance(metadata_val, str):
        metadata_val = json.loads(metadata_val)

    return ModelVersionRecord(
        id=uuid.UUID(str(row["id"])),
        tenant_id=uuid.UUID(str(row["tenant_id"])),
        name=row["name"],
        version=row["version"],
        task=row["task"],
        artifact_path=row["artifact_path"],
        sha256_hash=row["sha256_hash"],
        classes=classes_val,
        metadata=metadata_val,
        is_active=bool(row["is_active"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_prediction(row: dict[str, Any]) -> PredictionRecord:
    probs = row["probabilities"]
    if isinstance(probs, str):
        probs = json.loads(probs)
    scores = row["decision_scores"]
    if isinstance(scores, str):
        scores = json.loads(scores)
    meta = row["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)

    return PredictionRecord(
        id=uuid.UUID(str(row["id"])),
        tenant_id=uuid.UUID(str(row["tenant_id"])),
        ticket_id=uuid.UUID(str(row["ticket_id"])),
        model_version_id=uuid.UUID(str(row["model_version_id"])),
        task=row["task"],
        predicted_label=row["predicted_label"],
        confidence=float(row["confidence"]) if row["confidence"] is not None else None,
        probabilities=probs,
        decision_scores=scores,
        metadata=meta,
        created_at=row["created_at"],
    )


def ensure_tenant_model_version(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    name: str,
    version: str,
    task: str,
    artifact_path: str,
    sha256_hash: str,
    classes: list[str],
    metadata: dict[str, Any] | None = None,
) -> ModelVersionRecord:
    """Ensure a model version record is registered and active for this tenant.

    Guarantees strict tenant isolation: every tenant has their own registered model row.
    Idempotent: updates existing record on conflict.
    """
    meta_json = json.dumps(metadata or {})
    classes_json = json.dumps(classes)

    query = """
        INSERT INTO model_versions (
            tenant_id, name, version, task, artifact_path,
            sha256_hash, classes, metadata, is_active
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s::JSONB, %s::JSONB, TRUE)
        ON CONFLICT (tenant_id, task, name, version)
        DO UPDATE SET
            artifact_path = EXCLUDED.artifact_path,
            sha256_hash = EXCLUDED.sha256_hash,
            classes = EXCLUDED.classes,
            metadata = EXCLUDED.metadata,
            is_active = TRUE,
            updated_at = NOW()
        RETURNING id, tenant_id, name, version, task, artifact_path,
                  sha256_hash, classes, metadata, is_active, created_at, updated_at;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            query,
            (
                tenant_id,
                name,
                version,
                task,
                artifact_path,
                sha256_hash,
                classes_json,
                meta_json,
            ),
        )
        row = cur.fetchone()
        assert row is not None
        return _row_to_model_version(row)


def get_active_model_version(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    task: str = "category",
) -> ModelVersionRecord | None:
    """Retrieve the currently active model version for tenant_id and task."""
    query = """
        SELECT id, tenant_id, name, version, task, artifact_path,
               sha256_hash, classes, metadata, is_active, created_at, updated_at
        FROM model_versions
        WHERE tenant_id = %s AND task = %s AND is_active = TRUE
        ORDER BY created_at DESC
        LIMIT 1;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (tenant_id, task))
        row = cur.fetchone()
        if not row:
            return None
        return _row_to_model_version(row)


def save_ticket_prediction(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    ticket_id: uuid.UUID,
    model_version_id: uuid.UUID,
    task: str,
    predicted_label: str,
    confidence: float | None = None,
    probabilities: dict[str, float] | None = None,
    decision_scores: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> PredictionRecord:
    """Persist prediction for a ticket idempotently.

    Retries do not create duplicate predictions (ON CONFLICT DO UPDATE).
    Enforces composite tenant foreign keys on (tenant_id, ticket_id) and (tenant_id, model_version_id).
    """
    probs_json = json.dumps(probabilities) if probabilities is not None else None
    scores_json = json.dumps(decision_scores) if decision_scores is not None else None
    meta_json = json.dumps(metadata or {})

    query = """
        INSERT INTO ticket_predictions (
            tenant_id, ticket_id, model_version_id, task,
            predicted_label, confidence, probabilities, decision_scores, metadata
        )
        VALUES (
            %s, %s, %s, %s,
            %s, %s, %s::JSONB, %s::JSONB, %s::JSONB
        )
        ON CONFLICT (tenant_id, ticket_id, task)
        DO UPDATE SET
            model_version_id = EXCLUDED.model_version_id,
            predicted_label = EXCLUDED.predicted_label,
            confidence = EXCLUDED.confidence,
            probabilities = EXCLUDED.probabilities,
            decision_scores = EXCLUDED.decision_scores,
            metadata = EXCLUDED.metadata,
            created_at = NOW()
        RETURNING id, tenant_id, ticket_id, model_version_id, task,
                  predicted_label, confidence, probabilities, decision_scores,
                  metadata, created_at;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            query,
            (
                tenant_id,
                ticket_id,
                model_version_id,
                task,
                predicted_label,
                confidence,
                probs_json,
                scores_json,
                meta_json,
            ),
        )
        row = cur.fetchone()
        assert row is not None
        return _row_to_prediction(row)


def get_predictions_for_ticket(
    conn: psycopg.Connection[Any],
    tenant_id: uuid.UUID,
    ticket_id: uuid.UUID,
) -> list[PredictionRecord]:
    """Retrieve all predictions for a ticket, strictly scoped to tenant_id."""
    query = """
        SELECT id, tenant_id, ticket_id, model_version_id, task,
               predicted_label, confidence, probabilities, decision_scores,
               metadata, created_at
        FROM ticket_predictions
        WHERE tenant_id = %s AND ticket_id = %s
        ORDER BY created_at ASC;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (tenant_id, ticket_id))
        rows = cur.fetchall()
        return [_row_to_prediction(r) for r in rows]

