"""Trusted ML Model Loader and Inference Module for Phase 3 Financial Complaint Demo.

Architectural and Operational Guarantees:
1. Loads trusted local artifact once per worker process, caching it in memory.
2. Validates artifact existence, SHA-256 integrity, pipeline structure, and expected classes.
3. Does NOT download or deserialize arbitrary or user-supplied model files.
4. LinearSVC decision scores are raw hyperplane margins, NOT calibrated probabilities.
   Confidence is left explicitly NULL / None until probability calibration is implemented.
5. Out-of-Domain Notice: This model is trained exclusively on CFPB consumer complaint narratives.
   It has no out-of-domain detection mechanism; generic or non-financial text will receive
   one of the 11 financial-product labels.
6. Input Text: Ticket body only. Trained on narrative text; subject is not prepended.
"""

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
from typing import Any
import joblib
import numpy as np
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Trusted baseline metadata from run_004 (frozen LinearSVC TF-IDF baseline)
TRUSTED_MODEL_NAME = "cfpb_financial_product_classifier"
TRUSTED_MODEL_VERSION = "run_004"
TRUSTED_TASK = "category"
EXPECTED_SHA256 = "7e833e8992a3150bdaf2489b9b9117d700f7875e9dde045e4f638bd1e656b360"
DEFAULT_ARTIFACT_REL_PATH = "experiments/runs/run_004_cfpb_imbalance_tuning/models/selected_cw_None_C_1.0.joblib"

EXPECTED_CLASSES: list[str] = [
    "Checking or savings account",
    "Credit card",
    "Credit reporting or other personal consumer reports",
    "Debt collection",
    "Debt or credit management",
    "Money transfer, virtual currency, or money service",
    "Mortgage",
    "Payday loan, title loan, personal loan, or advance loan",
    "Prepaid card",
    "Student loan",
    "Vehicle loan or lease",
]


class ModelArtifactError(Exception):
    """Base exception for model artifact errors."""


class ModelArtifactMissingError(ModelArtifactError):
    """Raised when the expected trusted model artifact is missing from disk."""


class ModelArtifactCorruptedError(ModelArtifactError):
    """Raised when artifact SHA-256 does not match the expected cryptographic digest."""


class ModelValidationError(ModelArtifactError):
    """Raised when the deserialized artifact fails structural or class validation."""


@dataclass(frozen=True)
class TicketInferenceResult:
    """Prediction outcome for a ticket."""
    predicted_label: str
    confidence: float | None  # Strictly None for uncalibrated LinearSVC
    decision_scores: dict[str, float]  # Raw margins
    model_name: str
    model_version: str
    sha256_hash: str


class ModelManager:
    """Worker-level singleton model manager."""

    def __init__(self) -> None:
        self._cached_pipeline: Pipeline | None = None
        self._cached_path: Path | None = None
        self._cached_hash: str | None = None

    def resolve_artifact_path(self, override_path: str | Path | None = None) -> Path:
        """Resolve the path to the trusted model artifact."""
        if override_path:
            return Path(override_path).resolve()

        env_path = os.getenv("MODEL_ARTIFACT_PATH")
        if env_path:
            return Path(env_path).resolve()

        # Try relative to cwd or project root
        base_dir = Path(__file__).resolve().parent.parent.parent
        candidate = base_dir / DEFAULT_ARTIFACT_REL_PATH
        if candidate.exists():
            return candidate.resolve()

        return Path(DEFAULT_ARTIFACT_REL_PATH).resolve()

    def load_model(
        self,
        artifact_path: str | Path | None = None,
        expected_sha256: str | None = None,
    ) -> Pipeline:
        """Load and validate the trusted model artifact.

        Validates:
        1. File exists and is a regular file.
        2. Cryptographic SHA-256 digest matches expected_sha256.
        3. Object is a scikit-learn Pipeline with 'tfidf' and 'clf' steps.
        4. Target classes match EXPECTED_CLASSES exactly.
        """
        path = self.resolve_artifact_path(artifact_path)
        expected_hash = expected_sha256 or EXPECTED_SHA256

        if not path.is_file():
            raise ModelArtifactMissingError(
                f"Trusted model artifact missing at {path}. "
                f"Ensure the artifact is provisioned according to ML_EXPERIMENTS.md."
            )

        # 1. Cryptographic integrity check
        hasher = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            actual_hash = hasher.hexdigest()
        except Exception as exc:
            raise ModelArtifactError(f"Failed to read model artifact at {path}: {exc}") from exc

        if actual_hash != expected_hash:
            raise ModelArtifactCorruptedError(
                f"Model artifact at {path} is corrupted or modified. "
                f"Expected SHA-256: {expected_hash}, actual: {actual_hash}."
            )

        # 2. Deserialization
        try:
            pipeline = joblib.load(path)
        except Exception as exc:
            raise ModelValidationError(f"Failed to deserialize model pipeline from {path}: {exc}") from exc

        # 3. Structural validation
        if not isinstance(pipeline, Pipeline):
            raise ModelValidationError(f"Expected scikit-learn Pipeline, got {type(pipeline).__name__}")

        step_names = [step[0] for step in pipeline.steps]
        if "tfidf" not in step_names or "clf" not in step_names:
            raise ModelValidationError(f"Pipeline missing required steps 'tfidf' or 'clf'. Steps: {step_names}")

        clf = pipeline.named_steps["clf"]
        if not hasattr(clf, "classes_"):
            raise ModelValidationError("Classifier step has no 'classes_' attribute.")

        loaded_classes = [str(c) for c in clf.classes_]
        if loaded_classes != EXPECTED_CLASSES:
            raise ModelValidationError(
                f"Model classes mismatch.\nExpected: {EXPECTED_CLASSES}\nGot: {loaded_classes}"
            )

        self._cached_pipeline = pipeline
        self._cached_path = path
        self._cached_hash = actual_hash
        logger.info(
            f"Successfully loaded trusted model '{TRUSTED_MODEL_NAME}' version '{TRUSTED_MODEL_VERSION}' "
            f"(SHA-256: {actual_hash[:8]}...) from {path}"
        )
        return pipeline

    def get_model(self) -> Pipeline:
        """Get the cached model pipeline, loading it if not yet initialized."""
        if self._cached_pipeline is not None:
            return self._cached_pipeline
        return self.load_model()

    def reset_cache(self) -> None:
        """Reset the cached model (useful for testing edge cases like missing/corrupt files)."""
        self._cached_pipeline = None
        self._cached_path = None
        self._cached_hash = None

    def predict_ticket(self, body: str) -> TicketInferenceResult:
        """Run classification on ticket body text.

        Input is strictly the ticket body (CFPB complaint narrative text).
        """
        pipeline = self.get_model()
        clf = pipeline.named_steps["clf"]

        # LinearSVC decision_function returns hyperplane margin distances
        # Shape: (1, n_classes)
        raw_scores: np.ndarray = pipeline.decision_function([body])[0]
        classes: list[str] = [str(c) for c in clf.classes_]

        best_idx = int(np.argmax(raw_scores))
        predicted_label = classes[best_idx]

        # Margin scores dictionary
        decision_scores = {
            cls_name: round(float(score), 4)
            for cls_name, score in zip(classes, raw_scores)
        }

        # LinearSVC decision scores are NOT calibrated probabilities.
        # Confidence is strictly None / null per specification.
        return TicketInferenceResult(
            predicted_label=predicted_label,
            confidence=None,
            decision_scores=decision_scores,
            model_name=TRUSTED_MODEL_NAME,
            model_version=TRUSTED_MODEL_VERSION,
            sha256_hash=self._cached_hash or EXPECTED_SHA256,
        )


# Global process singleton
_GLOBAL_MODEL_MANAGER = ModelManager()


def get_model_manager() -> ModelManager:
    """Return the worker process singleton ModelManager."""
    return _GLOBAL_MODEL_MANAGER

