"""Bounded Offline SLA-Risk Feasibility Training and Evaluation Pipeline.

Phase 5 Slice: Synthetic SLA-Risk Feasibility at Ticket Creation (T0).
Target: Binary resolution SLA breach (resolution_hours * 60 > 1440, i.e. > 24 hours).

Constraints & Design:
- CPU-only, using existing scikit-learn / numpy / joblib dependencies.
- Feature allowlist demonstrably available at T0: message, product, channel.
- Excludes all outcome columns (resolution_hours, first_response_hours, status, csat_score).
- Excludes identifiers (ticket_id) and creation timestamps (created_at).
- Excludes downstream triage labels (category, priority, sentiment) to prevent
  synthetic generator leakage and assume raw customer input at T0.
- Leakage-resistant GroupShuffleSplit on unique message to eliminate exact-duplicate leakage.
- Baselines: DummyClassifier(strategy="prior") and Metadata-only LogisticRegression.
- Threshold tuning performed strictly on validation set.
- All outputs documented as model risk scores, not calibrated operational probabilities.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Feature allowlist demonstrably available at ticket creation (T0)
FEATURE_ALLOWLIST = ("message", "product", "channel")

# Explicit outcome columns strictly excluded from model features
EXCLUDED_OUTCOME_COLUMNS = (
    "resolution_hours",
    "first_response_hours",
    "status",
    "csat_score",
)

# Triage-derived and identifier columns excluded from T0 feature allowlist
EXCLUDED_TRIAGE_AND_META_COLUMNS = (
    "ticket_id",
    "created_at",
    "category",
    "priority",
    "sentiment",
)

SYNTHETIC_TARGET_MINUTES = 1440.0  # 24 elapsed calendar hours


def compute_sha256(filepath: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_provenance() -> Dict[str, Any]:
    """Collect local Git HEAD revision and dirty status without network access."""
    commit_hash = None
    is_dirty = None
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        commit_hash = res.stdout.strip() or None
    except Exception:
        commit_hash = None

    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        is_dirty = bool(res.stdout.strip())
    except Exception:
        is_dirty = None

    return {
        "git_commit": commit_hash,
        "git_dirty": is_dirty,
    }


def compute_sla_breach_label(
    resolution_hours: float, target_minutes: float = SYNTHETIC_TARGET_MINUTES
) -> int:
    """Compute binary resolution SLA breach label.

    Strict inequality: resolution_hours * 60.0 > target_minutes.
    Boundary condition: resolution_hours == target_minutes / 60.0 is non-breach (0).
    """
    if np.isnan(resolution_hours) or resolution_hours < 0:
        raise ValueError(f"Invalid resolution_hours: {resolution_hours}")
    return 1 if (resolution_hours * 60.0 > target_minutes) else 0


def build_feature_matrix(
    rows: List[Dict[str, Any]],
    allowlist: Tuple[str, ...] = FEATURE_ALLOWLIST,
) -> np.ndarray:
    """Extract allowed features into a 2D numpy array of shape (N, len(allowlist)).

    Guarantees order: column 0 is 'message', column 1 is 'product', column 2 is 'channel'.
    """
    data = []
    for r in rows:
        row_features = [r.get(col, "") or "" for col in allowlist]
        data.append(row_features)
    return np.array(data, dtype=object)


def create_preprocessor() -> ColumnTransformer:
    """Create ColumnTransformer for Text + Metadata features.

    Column 0: message -> TfidfVectorizer (unigrams + bigrams, max 5000 features)
    Columns 1, 2: product, channel -> OneHotEncoder (handle_unknown='ignore')
    """
    return ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=5000,
                    stop_words="english",
                ),
                0,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                [1, 2],
            ),
        ]
    )


def create_text_metadata_pipeline(random_state: int = 42) -> Pipeline:
    """Create full Text + Metadata Logistic Regression pipeline."""
    return Pipeline(
        [
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    random_state=random_state,
                    max_iter=1000,
                    C=1.0,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def create_metadata_pipeline(random_state: int = 42) -> Pipeline:
    """Create Metadata-only Logistic Regression pipeline."""
    meta_preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                [1, 2],
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", meta_preprocessor),
            (
                "classifier",
                LogisticRegression(
                    random_state=random_state,
                    max_iter=1000,
                    C=1.0,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics.

    Metrics:
    - Average Precision (average_precision_score, explicitly named)
    - ROC-AUC
    - Brier score
    - Log loss
    - Precision, Recall, F1 at specified decision threshold
    - Confusion Matrix [[TN, FP], [FN, TP]]
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # Average precision
    ap = float(average_precision_score(y_true, y_prob))

    # ROC-AUC
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = None

    # Brier score
    brier = float(brier_score_loss(y_true, y_prob))

    # Log loss (clip probabilities to prevent numerical infinities)
    y_prob_clipped = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
    try:
        ll = float(log_loss(y_true, y_prob_clipped, labels=[0, 1]))
    except ValueError:
        ll = float("nan")

    # Binary decisions at threshold
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "average_precision": ap,
        "roc_auc": auc,
        "brier_score": brier,
        "log_loss": ll,
        "decision_threshold": float(threshold),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": {
            "matrix": cm,
            "format": "[[TN, FP], [FN, TP]]",
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1],
        },
    }


def find_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, num_steps: int = 91
) -> Tuple[float, float]:
    """Find the threshold in [0.05, 0.95] that maximizes F1 score on validation data."""
    best_th = 0.5
    best_f1 = -1.0
    for th in np.linspace(0.05, 0.95, num_steps):
        preds = (y_prob >= th).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_th = float(th)
    return best_th, best_f1


def run_experiment(
    csv_path: str = "data/customer_support_tickets.csv",
    output_dir: str = "experiments/runs/run_006_sla_risk_feasibility",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete offline SLA-risk feasibility training and evaluation."""
    start_time = time.time()
    out_path = Path(output_dir)

    # 0. Strict Overwrite Prevention (Fail immediately before dataset loading or training)
    if out_path.exists():
        raise FileExistsError(
            f"Experiment run directory already exists: '{out_path.resolve()}'. "
            f"Overwriting previous runs is strictly prohibited to preserve artifact integrity. "
            f"Please specify a new, unique output directory (e.g., experiments/runs/run_007_...)."
        )

    # Directory creation that prevents concurrent existence race
    try:
        out_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as e:
        raise FileExistsError(
            f"Experiment run directory already exists: '{out_path.resolve()}'. "
            f"Overwriting previous runs is strictly prohibited to preserve artifact integrity. "
            f"Please specify a new, unique output directory."
        ) from e

    # Mark directory as incomplete to prevent silent reuse if training is aborted or fails
    incomplete_marker = out_path / "INCOMPLETE.json"
    with open(incomplete_marker, "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "incomplete",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "output_dir": str(out_path.resolve()),
                "error": None,
            },
            f,
            indent=2,
        )

    try:
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        # 1. Dataset Verification & Ingestion
        dataset_sha256 = compute_sha256(csv_file)
        rows: List[Dict[str, Any]] = []
        with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            for r in reader:
                rows.append(r)

        total_rows = len(rows)
        if total_rows == 0:
            raise ValueError("Dataset contains zero rows.")

        # Calculate labels
        labels = [
            compute_sla_breach_label(float(r["resolution_hours"])) for r in rows
        ]
        labels_arr = np.array(labels, dtype=int)
        pos_count = int(np.sum(labels_arr))
        neg_count = total_rows - pos_count
        overall_prevalence = float(pos_count / total_rows)

        messages = [r["message"] for r in rows]

        # 2. Leakage-Resistant Group Split by Unique Message
        gss_outer = GroupShuffleSplit(
            n_splits=1, train_size=0.70, random_state=random_seed
        )
        train_idx, temp_idx = next(gss_outer.split(rows, labels_arr, groups=messages))

        temp_rows = [rows[i] for i in temp_idx]
        temp_labels = labels_arr[temp_idx]
        temp_messages = [messages[i] for i in temp_idx]

        gss_inner = GroupShuffleSplit(
            n_splits=1, train_size=0.50, random_state=random_seed
        )
        val_sub_idx, test_sub_idx = next(
            gss_inner.split(temp_rows, temp_labels, groups=temp_messages)
        )

        val_idx = temp_idx[val_sub_idx]
        test_idx = temp_idx[test_sub_idx]

        # Verify disjointness
        s_train, s_val, s_test = set(train_idx), set(val_idx), set(test_idx)
        assert len(s_train & s_val) == 0, "Train and Val row indices overlap!"
        assert len(s_train & s_test) == 0, "Train and Test row indices overlap!"
        assert len(s_val & s_test) == 0, "Val and Test row indices overlap!"

        # Verify group separation (message overlap)
        train_msgs = set(messages[i] for i in train_idx)
        val_msgs = set(messages[i] for i in val_idx)
        test_msgs = set(messages[i] for i in test_idx)
        assert (
            len(train_msgs & val_msgs) == 0
        ), "Message leak between Train and Val!"
        assert (
            len(train_msgs & test_msgs) == 0
        ), "Message leak between Train and Test!"
        assert len(val_msgs & test_msgs) == 0, "Message leak between Val and Test!"

        # Feature extraction
        X_train = build_feature_matrix([rows[i] for i in train_idx])
        X_val = build_feature_matrix([rows[i] for i in val_idx])
        X_test = build_feature_matrix([rows[i] for i in test_idx])

        y_train = labels_arr[train_idx]
        y_val = labels_arr[val_idx]
        y_test = labels_arr[test_idx]

        # Split distributions
        split_info = {
            "train": {
                "row_count": int(len(train_idx)),
                "positive_count": int(np.sum(y_train)),
                "negative_count": int(len(train_idx) - np.sum(y_train)),
                "prevalence": float(np.mean(y_train)),
                "unique_messages": len(train_msgs),
            },
            "validation": {
                "row_count": int(len(val_idx)),
                "positive_count": int(np.sum(y_val)),
                "negative_count": int(len(val_idx) - np.sum(y_val)),
                "prevalence": float(np.mean(y_val)),
                "unique_messages": len(val_msgs),
            },
            "test": {
                "row_count": int(len(test_idx)),
                "positive_count": int(np.sum(y_test)),
                "negative_count": int(len(test_idx) - np.sum(y_test)),
                "prevalence": float(np.mean(y_test)),
                "unique_messages": len(test_msgs),
            },
            "group_overlap_checks": {
                "train_val_message_overlap": len(train_msgs & val_msgs),
                "train_test_message_overlap": len(train_msgs & test_msgs),
                "val_test_message_overlap": len(val_msgs & test_msgs),
            },
        }

        # 3. Model Training: Baselines & Primary Pipeline
        # Baseline 1: DummyClassifier(strategy="prior")
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(X_train, y_train)
        p_dummy_val = dummy.predict_proba(X_val)[:, 1]
        p_dummy_test = dummy.predict_proba(X_test)[:, 1]

        # Baseline 2: Metadata-only Logistic Regression
        meta_pipe = create_metadata_pipeline(random_state=random_seed)
        meta_pipe.fit(X_train, y_train)
        p_meta_val = meta_pipe.predict_proba(X_val)[:, 1]
        p_meta_test = meta_pipe.predict_proba(X_test)[:, 1]

        # Primary Model: Text + Metadata Logistic Regression
        text_meta_pipe = create_text_metadata_pipeline(random_state=random_seed)
        text_meta_pipe.fit(X_train, y_train)
        p_text_val = text_meta_pipe.predict_proba(X_val)[:, 1]
        p_text_test = text_meta_pipe.predict_proba(X_test)[:, 1]

        # 4. Threshold Tuning on Validation Data
        val_th_tuned, val_best_f1 = find_optimal_threshold(y_val, p_text_val)

        # 5. Comprehensive Evaluations
        results = {
            "metadata": {
                "run_id": out_path.name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "target_definition": "resolution_hours * 60.0 > 1440.0 (strict inequality, 24 calendar hours)",
                "feature_allowlist": list(FEATURE_ALLOWLIST),
                "excluded_outcome_columns": list(EXCLUDED_OUTCOME_COLUMNS),
                "excluded_triage_and_meta_columns": list(
                    EXCLUDED_TRIAGE_AND_META_COLUMNS
                ),
                "random_seed": random_seed,
                "dataset_sha256": dataset_sha256,
                "total_rows": total_rows,
                "overall_prevalence": overall_prevalence,
                "validation_tuned_threshold": val_th_tuned,
            },
            "splits": split_info,
            "evaluations": {
                "dummy_prior": {
                    "validation": evaluate_predictions(
                        y_val, p_dummy_val, threshold=0.5
                    ),
                    "test": evaluate_predictions(
                        y_test, p_dummy_test, threshold=0.5
                    ),
                },
                "metadata_only_logistic_regression": {
                    "validation": evaluate_predictions(
                        y_val, p_meta_val, threshold=0.5
                    ),
                    "test": evaluate_predictions(
                        y_test, p_meta_test, threshold=0.5
                    ),
                },
                "text_metadata_logistic_regression_default_threshold_0_50": {
                    "validation": evaluate_predictions(
                        y_val, p_text_val, threshold=0.5
                    ),
                    "test": evaluate_predictions(
                        y_test, p_text_test, threshold=0.5
                    ),
                },
                "text_metadata_logistic_regression_val_tuned_threshold": {
                    "validation": evaluate_predictions(
                        y_val, p_text_val, threshold=val_th_tuned
                    ),
                    "test": evaluate_predictions(
                        y_test, p_text_test, threshold=val_th_tuned
                    ),
                },
            },
        }

        # 6. Save Model Artifact and Hashes
        model_artifact_path = out_path / "model.joblib"
        joblib.dump(text_meta_pipe, model_artifact_path)
        model_sha256 = compute_sha256(model_artifact_path)

        sha256_file = out_path / "model.joblib.sha256"
        with open(sha256_file, "w", encoding="utf-8") as f:
            f.write(f"{model_sha256}  model.joblib\n")

        results["metadata"]["model_sha256"] = model_sha256

        # 7. Save Metrics JSON
        metrics_path = out_path / "metrics_sla_risk.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # 8. Save Split Manifest (Non-sensitive row references)
        manifest = {
            "dataset_sha256": dataset_sha256,
            "total_rows": total_rows,
            "random_seed": random_seed,
            "splits": {
                "train_row_indices": train_idx.tolist(),
                "val_row_indices": val_idx.tolist(),
                "test_row_indices": test_idx.tolist(),
                "train_ticket_ids": [rows[i]["ticket_id"] for i in train_idx],
                "val_ticket_ids": [rows[i]["ticket_id"] for i in val_idx],
                "test_ticket_ids": [rows[i]["ticket_id"] for i in test_idx],
            },
        }
        manifest_path = out_path / "split_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    # 9. Save Model Metadata with Git Provenance & Script SHA-256
        git_info = get_git_provenance()
        try:
            script_path = Path(__file__).resolve()
            script_sha256 = compute_sha256(script_path) if script_path.exists() else None
        except Exception:
            script_sha256 = None

        metadata_out = {
            "run_id": out_path.name,
            "model_type": "Pipeline(ColumnTransformer([TfidfVectorizer, OneHotEncoder]), LogisticRegression)",
            "dataset_sha256": dataset_sha256,
            "model_sha256": model_sha256,
            "training_script_sha256": script_sha256,
            "git_commit": git_info.get("git_commit"),
            "git_dirty": git_info.get("git_dirty"),
            "python_version": sys.version,
            "dependencies": {
                "scikit-learn": "1.9.1",
                "joblib": "1.6.0",
                "numpy": "2.5.4",
            },
            "target": "resolution_hours * 60.0 > 1440.0",
            "features": list(FEATURE_ALLOWLIST),
            "val_tuned_threshold": val_th_tuned,
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
        with open(out_path / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata_out, f, indent=2)

        # 10. Clean up incomplete run marker upon successful completion
        if incomplete_marker.exists():
            incomplete_marker.unlink()

        print(f"Feasibility experiment complete in {time.time() - start_time:.2f}s")
        print(f"Artifacts written to: {out_path.resolve()}")
        return results

    except Exception as exc:
        # Record failure details in INCOMPLETE.json without deleting directory or partial files
        try:
            if out_path.exists() and incomplete_marker.exists():
                with open(incomplete_marker, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "status": "failed",
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                            "output_dir": str(out_path.resolve()),
                            "error": str(exc),
                        },
                        f,
                        indent=2,
                    )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run offline SLA-risk feasibility experiment."
    )
    parser.add_argument(
        "--csv-path",
        default="data/customer_support_tickets.csv",
        help="Path to customer support tickets CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/runs/run_006_sla_risk_feasibility",
        help="Output directory for experiment artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for group shuffle split",
    )
    args = parser.parse_args()

    run_experiment(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )
