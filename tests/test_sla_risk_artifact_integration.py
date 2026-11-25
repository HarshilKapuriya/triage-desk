"""Saved-Artifact Integration Tests for SLA-Risk Feasibility Pipeline (run_006).

Contract:
- Integration tests requiring the actual trained model artifact provisioned on disk at
  experiments/runs/run_006_sla_risk_feasibility/model.joblib.
- Each test independently validates artifact presence and verifies that the file's SHA-256
  checksum matches the repository's committed expectation in model.joblib.sha256 BEFORE
  calling joblib.load.
- Does NOT claim the checksum alone establishes external provenance, model quality, or
  calibration. It verifies exact agreement with the repository's committed digest.
- In environments where model.joblib is not provisioned (e.g. fresh clones subject to
  the *.joblib gitignore rule), these tests fail closed asserting artifact presence.
"""

from pathlib import Path
import joblib
import numpy as np
import pytest

from scripts.train_sla_risk import (
    build_feature_matrix,
    compute_sha256,
)

# Resolve artifact paths relative to repository root rather than current working directory
REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = REPO_ROOT / "experiments" / "runs" / "run_006_sla_risk_feasibility"
MODEL_PATH = RUN_DIR / "model.joblib"
SHA256_PATH = RUN_DIR / "model.joblib.sha256"


def _verify_artifact_integrity() -> None:
    """Verify model artifact exists and matches committed SHA-256 digest before deserialization."""
    assert MODEL_PATH.exists(), (
        f"Trained model artifact missing at {MODEL_PATH}. "
        "Ensure the artifact is provisioned according to operational evaluation guidelines."
    )
    assert SHA256_PATH.exists(), f"Committed model SHA-256 digest missing at {SHA256_PATH}."

    expected_hash = compute_sha256(MODEL_PATH)
    with open(SHA256_PATH, "r", encoding="utf-8") as f:
        stored_hash = f.read().strip().split()[0]
    assert expected_hash == stored_hash, (
        f"Model SHA-256 mismatch for {MODEL_PATH}. "
        f"Computed: {expected_hash}, Expected from {SHA256_PATH.name}: {stored_hash}"
    )


def test_run_006_artifact_serialization_and_reload_consistency() -> None:
    """Verify saved run_006 model artifact matches committed SHA-256 and reloads identically.

    Contract: Integration check requiring the trained run_006 artifact at
    experiments/runs/run_006_sla_risk_feasibility/model.joblib.
    Verifies cryptographic integrity against the committed digest BEFORE loading,
    then verifies deterministic inference across consecutive joblib reloads.
    """
    _verify_artifact_integrity()

    # Load serialized pipeline
    loaded_model = joblib.load(MODEL_PATH)

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
    reloaded_again = joblib.load(MODEL_PATH)
    probs_again = reloaded_again.predict_proba(X_sample)
    np.testing.assert_allclose(probs, probs_again, atol=1e-12)


def test_run_006_artifact_output_scores_finite_and_bounded() -> None:
    """Verify all run_006 trained model risk scores are finite floats within [0.0, 1.0].

    Contract: Integration check requiring the trained run_006 artifact at
    experiments/runs/run_006_sla_risk_feasibility/model.joblib.
    Verifies cryptographic integrity against the committed digest BEFORE loading,
    then asserts that the loaded model outputs valid bounded probabilities across test samples.
    """
    _verify_artifact_integrity()

    loaded_model = joblib.load(MODEL_PATH)

    test_samples = [
        {"message": "Billing statement incorrect charge", "product": "PayWallet", "channel": "Email"},
        {"message": "How do I export my data?", "product": "TaskFlow", "channel": "Chat"},
        {"message": "System down urgent attention needed", "product": "CloudDrive", "channel": "Web Form"},
        {"message": "", "product": "", "channel": ""},
    ]
    X_test = build_feature_matrix(test_samples)
    probs = loaded_model.predict_proba(X_test)

    assert probs.shape == (len(test_samples), 2)
    for i in range(len(test_samples)):
        p0, p1 = probs[i, 0], probs[i, 1]
        assert np.isfinite(p0) and np.isfinite(p1)
        assert 0.0 <= p0 <= 1.0
        assert 0.0 <= p1 <= 1.0
        assert pytest.approx(p0 + p1, abs=1e-7) == 1.0

