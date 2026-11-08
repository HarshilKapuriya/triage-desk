"""Offline Unit Tests for SLA-Risk Feasibility Pipeline.

Tests strictly cover:
1. Label calculation and exact deadline boundary.
2. Feature allowlist and outcome-column exclusion.
3. Missing/invalid input handling.
4. Split disjointness and group separation.
5. Preprocessing fitted only on training data.
6. Serialization and reload consistency using the produced artifact.
7. Finite output scores within the expected [0.0, 1.0] range.
"""

import csv
import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
import joblib

from scripts.train_sla_risk import (
    FEATURE_ALLOWLIST,
    EXCLUDED_OUTCOME_COLUMNS,
    EXCLUDED_TRIAGE_AND_META_COLUMNS,
    SYNTHETIC_TARGET_MINUTES,
    compute_sla_breach_label,
    compute_sha256,
    build_feature_matrix,
    create_text_metadata_pipeline,
    create_metadata_pipeline,
    find_optimal_threshold,
    run_experiment,
    get_git_provenance,
)


def test_sla_breach_label_calculation_and_boundary():
    """Verify strict inequality and exact boundary behavior for SLA breach calculation.

    Definition: resolution_hours * 60.0 > 1440.0.
    Boundary 24.0 hours (1440.0 minutes) must be non-breach (0).
    """
    # Exact boundary: 24.0 hours == 1440 minutes -> 0 (non-breach)
    assert compute_sla_breach_label(24.0) == 0

    # Strictly below boundary
    assert compute_sla_breach_label(0.0) == 0
    assert compute_sla_breach_label(1.5) == 0
    assert compute_sla_breach_label(23.999) == 0

    # Strictly above boundary -> 1 (breach)
    assert compute_sla_breach_label(24.001) == 1
    assert compute_sla_breach_label(25.0) == 1
    assert compute_sla_breach_label(100.0) == 1

    # Custom target boundary check
    assert compute_sla_breach_label(2.0, target_minutes=120.0) == 0
    assert compute_sla_breach_label(2.01, target_minutes=120.0) == 1

    # Invalid input checks
    with pytest.raises(ValueError):
        compute_sla_breach_label(-1.0)
    with pytest.raises(ValueError):
        compute_sla_breach_label(float("nan"))


def test_feature_allowlist_excludes_outcome_and_leakage_columns():
    """Verify feature allowlist permits only T0 fields and excludes outcome/triage columns."""
    # Allowlist check
    assert set(FEATURE_ALLOWLIST) == {"message", "product", "channel"}

    # Excluded outcomes check
    for col in (
        "resolution_hours",
        "first_response_hours",
        "status",
        "csat_score",
    ):
        assert col in EXCLUDED_OUTCOME_COLUMNS
        assert col not in FEATURE_ALLOWLIST

    # Excluded triage and metadata check
    for col in ("ticket_id", "created_at", "category", "priority", "sentiment"):
        assert col in EXCLUDED_TRIAGE_AND_META_COLUMNS
        assert col not in FEATURE_ALLOWLIST

    # Verify build_feature_matrix strips disallowed columns
    raw_row = {
        "ticket_id": "fake-uuid-1234",
        "created_at": "2026-11-08 12:00:00",
        "message": "Payment failed on checkout",
        "product": "PayWallet",
        "channel": "Email",
        "category": "Billing",
        "priority": "Urgent",
        "status": "Resolved",
        "resolution_hours": "36.5",
        "csat_score": "1",
    }
    matrix = build_feature_matrix([raw_row])
    assert matrix.shape == (1, 3)
    assert matrix[0, 0] == "Payment failed on checkout"
    assert matrix[0, 1] == "PayWallet"
    assert matrix[0, 2] == "Email"


def test_missing_and_invalid_input_handling():
    """Verify pipeline handles empty text, whitespace, and unknown categorical values."""
    # Fit a tiny pipeline on controlled training data
    train_rows = [
        {"message": "Login crash error", "product": "PayWallet", "channel": "Email"},
        {"message": "Great interface update", "product": "CloudDrive", "channel": "Chat"},
        {"message": "Where is my refund", "product": "PayWallet", "channel": "Phone"},
        {"message": "Account locked out", "product": "CloudDrive", "channel": "Email"},
    ]
    y_train = np.array([1, 0, 1, 0])
    X_train = build_feature_matrix(train_rows)

    pipe = create_text_metadata_pipeline(random_state=42)
    pipe.fit(X_train, y_train)

    # Edge cases: empty string, whitespace only, completely unknown product/channel, None values
    edge_rows = [
        {"message": "", "product": "UnseenProduct", "channel": "UnseenChannel"},
        {"message": "   \n\t  ", "product": "PayWallet", "channel": "CarrierPigeon"},
        {"message": "Payment refund", "product": None, "channel": None},
    ]
    X_edge = build_feature_matrix(edge_rows)

    probs = pipe.predict_proba(X_edge)
    assert probs.shape == (3, 2)
    for p in probs[:, 1]:
        assert np.isfinite(p)
        assert 0.0 <= p <= 1.0


def test_split_disjointness_and_group_separation():
    """Verify split manifest indices are pairwise disjoint and have 0 message overlap."""
    run_dir = Path("experiments/runs/run_006_sla_risk_feasibility")
    manifest_path = run_dir / "split_manifest.json"
    metrics_path = run_dir / "metrics_sla_risk.json"

    assert manifest_path.exists(), "split_manifest.json missing"
    assert metrics_path.exists(), "metrics_sla_risk.json missing"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    splits = manifest["splits"]
    train_idx = set(splits["train_row_indices"])
    val_idx = set(splits["val_row_indices"])
    test_idx = set(splits["test_row_indices"])

    # Disjointness of row indices
    assert len(train_idx & val_idx) == 0, "Train and Val indices overlap"
    assert len(train_idx & test_idx) == 0, "Train and Test indices overlap"
    assert len(val_idx & test_idx) == 0, "Val and Test indices overlap"
    assert len(train_idx) + len(val_idx) + len(test_idx) == manifest["total_rows"]

    # Verify group separation checks recorded in metrics_sla_risk.json
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    overlap_checks = metrics["splits"]["group_overlap_checks"]
    assert overlap_checks["train_val_message_overlap"] == 0
    assert overlap_checks["train_test_message_overlap"] == 0
    assert overlap_checks["val_test_message_overlap"] == 0


def test_preprocessing_fitted_only_on_train():
    """Verify vectorizer and encoder vocabulary are frozen on train and not expanded by test."""
    train_rows = [
        {"message": "alpha beta", "product": "ProdA", "channel": "Email"},
        {"message": "gamma delta", "product": "ProdB", "channel": "Chat"},
    ]
    y_train = np.array([0, 1])
    X_train = build_feature_matrix(train_rows)

    pipe = create_text_metadata_pipeline(random_state=42)
    pipe.fit(X_train, y_train)

    preprocessor = pipe.named_steps["preprocessor"]
    text_transformer = preprocessor.named_transformers_["text"]
    cat_transformer = preprocessor.named_transformers_["cat"]

    initial_vocab_size = len(text_transformer.vocabulary_)
    initial_categories = [list(c) for c in cat_transformer.categories_]

    # Evaluate on rows with completely new vocabulary and new categories
    test_rows = [
        {"message": "omega epsilon zeta", "product": "UnseenProdX", "channel": "Social"},
    ]
    X_test = build_feature_matrix(test_rows)
    _ = pipe.predict_proba(X_test)

    # Vocabulary and categories must not have mutated
    assert len(text_transformer.vocabulary_) == initial_vocab_size
    assert [list(c) for c in cat_transformer.categories_] == initial_categories


def test_model_serialization_and_reload_consistency():
    """Verify saved model artifact can be reloaded and yields identical predictions."""
    run_dir = Path("experiments/runs/run_006_sla_risk_feasibility")
    model_path = run_dir / "model.joblib"
    sha256_path = run_dir / "model.joblib.sha256"

    assert model_path.exists(), "model.joblib missing"
    assert sha256_path.exists(), "model.joblib.sha256 missing"

    # Verify integrity hash
    expected_hash = compute_sha256(model_path)
    with open(sha256_path, "r", encoding="utf-8") as f:
        stored_hash = f.read().strip().split()[0]
    assert expected_hash == stored_hash, "Model SHA-256 mismatch with hash file"

    # Load serialized pipeline
    loaded_model = joblib.load(model_path)

    # Predict on test samples
    sample_rows = [
        {"message": "Urgent crash on PayWallet app", "product": "PayWallet", "channel": "Email"},
        {"message": "Thank you for the wonderful update", "product": "CloudDrive", "channel": "Chat"},
        {"message": "Device battery drained quickly", "product": "FitBand X", "channel": "Social Media"},
    ]
    X_sample = build_feature_matrix(sample_rows)
    probs = loaded_model.predict_proba(X_sample)

    assert probs.shape == (3, 2)
    # Reload again to verify deterministic idempotence
    reloaded_again = joblib.load(model_path)
    probs_again = reloaded_again.predict_proba(X_sample)
    np.testing.assert_allclose(probs, probs_again, atol=1e-12)


def test_model_output_scores_finite_and_bounded():
    """Verify all model risk scores are finite floats within [0.0, 1.0]."""
    run_dir = Path("experiments/runs/run_006_sla_risk_feasibility")
    model_path = run_dir / "model.joblib"
    loaded_model = joblib.load(model_path)

    test_samples = [
        {"message": "Billing statement incorrect charge", "product": "PayWallet", "channel": "Email"},
        {"message": "How do I export my data?", "product": "TaskFlow", "channel": "Chat"},
        {"message": "System down urgent attention needed", "product": "CloudDrive", "channel": "Web Form"},
        {"message": "", "product": "", "channel": ""},
    ]
    X_test = build_feature_matrix(test_samples)
    probs = loaded_model.predict_proba(X_test)

    for i in range(len(test_samples)):
        p0, p1 = probs[i, 0], probs[i, 1]
        assert np.isfinite(p0) and np.isfinite(p1)
        assert 0.0 <= p0 <= 1.0
        assert 0.0 <= p1 <= 1.0
        assert pytest.approx(p0 + p1, abs=1e-7) == 1.0


def test_threshold_selection_monotonicity():
    """Verify threshold tuner handles edge distributions and returns valid bounds."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.6, 0.8, 0.3, 0.7])
    th, f1 = find_optimal_threshold(y_true, y_prob)
    assert 0.05 <= th <= 0.95
    assert 0.0 <= f1 <= 1.0


def test_existing_run_directory_rejected_before_fitting(tmp_path, monkeypatch):
    """Verify existing run directory is rejected immediately with FileExistsError before model fitting."""
    existing_dir = tmp_path / "existing_run"
    existing_dir.mkdir()
    sentinel_file = existing_dir / "sentinel.txt"
    sentinel_content = b"ORIGINAL_SENTINEL_CONTENT_DO_NOT_ALTER"
    sentinel_file.write_bytes(sentinel_content)

    # Monkeypatch create_text_metadata_pipeline to ensure model creation is never reached
    def forbidden_call(*args, **kwargs):
        raise AssertionError("Model pipeline creation should not be reached when directory exists!")

    monkeypatch.setattr("scripts.train_sla_risk.create_text_metadata_pipeline", forbidden_call)

    with pytest.raises(FileExistsError) as exc_info:
        run_experiment(output_dir=str(existing_dir))

    assert "already exists" in str(exc_info.value).lower()
    # Verify existing sentinel file remains byte-for-byte unchanged
    assert sentinel_file.read_bytes() == sentinel_content
    # Verify no other files were created
    assert set(p.name for p in existing_dir.iterdir()) == {"sentinel.txt"}


def test_fresh_run_directory_can_be_reserved_and_writes_provenance(tmp_path):
    """Verify fresh run directory can be created with small synthetic fixture and provenance is captured."""
    mini_csv = tmp_path / "mini.csv"
    with open(mini_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ticket_id", "created_at", "product", "channel", "message",
            "category", "sentiment", "priority", "first_response_hours",
            "resolution_hours", "status", "csat_score"
        ])
        for i in range(24):
            # Alternate breached (>24h) and non-breached (<24h) rows
            res_hours = 36.0 if (i % 2 == 0) else 12.0
            w.writerow([
                f"uuid-{i}", "2026-11-08 10:00:00", "PayWallet", "Email",
                f"Unique message test content {i}", "Technical Issue", "Negative", "High",
                "1.0", str(res_hours), "Resolved", "4"
            ])

    fresh_run_dir = tmp_path / "fresh_run_001"
    res = run_experiment(csv_path=str(mini_csv), output_dir=str(fresh_run_dir), random_seed=42)
    assert res is not None
    assert fresh_run_dir.exists()
    assert (fresh_run_dir / "model.joblib").exists()
    assert (fresh_run_dir / "model_metadata.json").exists()
    assert not (fresh_run_dir / "INCOMPLETE.json").exists()

    with open(fresh_run_dir / "model_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "training_script_sha256" in meta
    assert len(meta["training_script_sha256"]) == 64
    assert "git_commit" in meta
    assert "git_dirty" in meta


def test_incomplete_marker_preserved_on_failure_and_prevents_reuse(tmp_path):
    """Verify aborted/failed runs leave INCOMPLETE.json and are rejected on subsequent attempts."""
    failing_dir = tmp_path / "failing_run"

    # Pass non-existent CSV path
    with pytest.raises(FileNotFoundError):
        run_experiment(csv_path=str(tmp_path / "non_existent.csv"), output_dir=str(failing_dir))

    assert failing_dir.exists()
    incomplete_file = failing_dir / "INCOMPLETE.json"
    assert incomplete_file.exists()

    with open(incomplete_file, "r", encoding="utf-8") as f:
        inc_data = json.load(f)
    assert inc_data["status"] == "failed"
    assert "non_existent.csv" in inc_data["error"]

    # Attempting to rerun against the failed directory must fail with FileExistsError
    with pytest.raises(FileExistsError) as exc_info:
        run_experiment(output_dir=str(failing_dir))
    assert "already exists" in str(exc_info.value).lower()


def test_git_provenance_handles_missing_git(monkeypatch):
    """Verify get_git_provenance handles missing git gracefully without raising or fabricating values."""
    def fake_subprocess_run(*args, **kwargs):
        raise FileNotFoundError("git binary not found")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    prov = get_git_provenance()
    assert prov == {"git_commit": None, "git_dirty": None}
