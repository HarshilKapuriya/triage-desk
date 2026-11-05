"""Validation-driven LinearSVC Class Imbalance Tuning on CFPB Dataset.

Preserves:
- Date splits: Train (2024-01-01..2024-01-22), Val (2024-01-23..2024-01-26), Test (2024-01-27..2024-01-31)
- Input: 'Consumer complaint narrative' exclusively
- TF-IDF: fit strictly on X_train

Grid:
- class_weight in [None, 'balanced']
- C in [0.1, 1.0, 10.0]

Selection Criterion:
- Strict selection by Validation Macro-F1.
- Test set is evaluated ONLY after best configuration is selected from validation metrics.
- Subsequent test metrics are explicitly labeled as follow-up on an already inspected test set.
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib


def load_dataset(csv_path: str = "data/cfpb_complaints_jan2024.csv") -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows.sort(key=lambda r: (r["Date received"].strip(), int(r["Complaint ID"])))
    return rows


def evaluate_split(
    pipeline: Any,
    X: List[str],
    y: List[str],
    label_names: List[str],
    split_name: str,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    y_pred = pipeline.predict(X)
    infer_time = time.perf_counter() - t0

    acc = accuracy_score(y, y_pred)
    macro_p = precision_score(y, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y, y_pred, labels=label_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y, y_pred, labels=label_names).tolist()

    cm_sum = sum(sum(r) for r in cm)
    report_support_sum = sum(report[lbl]["support"] for lbl in label_names)

    assert cm_sum == len(X), f"CM sum {cm_sum} != split size {len(X)}"
    assert report_support_sum == len(X), f"Report support {report_support_sum} != split size {len(X)}"

    per_class_recall = {lbl: round(float(report[lbl]["recall"]), 4) for lbl in label_names}
    per_class_f1 = {lbl: round(float(report[lbl]["f1-score"]), 4) for lbl in label_names}

    return {
        "split_name": split_name,
        "sample_count": len(X),
        "confusion_matrix_sum": cm_sum,
        "inference_time_seconds": round(infer_time, 4),
        "latency_ms_per_sample": round(infer_time / len(X) * 1000.0, 4) if X else 0.0,
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def run_tuning(
    csv_path: str = "data/cfpb_complaints_jan2024.csv",
    run_id: str = "run_004_cfpb_imbalance_tuning",
) -> Dict[str, Any]:
    rows = load_dataset(csv_path)

    # Strictly day-aligned splits
    train_rows = [r for r in rows if r["Date received"] <= "2024-01-22"]
    val_rows = [r for r in rows if "2024-01-23" <= r["Date received"] <= "2024-01-26"]
    test_rows_chrono = [r for r in rows if r["Date received"] >= "2024-01-27"]

    # Purged test sets
    train_narratives = set(r["Consumer complaint narrative"].strip() for r in train_rows)
    val_narratives = set(r["Consumer complaint narrative"].strip() for r in val_rows)
    train_val_narratives = train_narratives.union(val_narratives)

    test_rows_r1 = [r for r in test_rows_chrono if r["Consumer complaint narrative"].strip() not in train_narratives]
    test_rows_r2 = [r for r in test_rows_chrono if r["Consumer complaint narrative"].strip() not in train_val_narratives]

    X_train = [r["Consumer complaint narrative"].strip() for r in train_rows]
    y_train = [r["Product"].strip() for r in train_rows]

    X_val = [r["Consumer complaint narrative"].strip() for r in val_rows]
    y_val = [r["Product"].strip() for r in val_rows]

    X_test_chrono = [r["Consumer complaint narrative"].strip() for r in test_rows_chrono]
    y_test_chrono = [r["Product"].strip() for r in test_rows_chrono]

    X_test_r1 = [r["Consumer complaint narrative"].strip() for r in test_rows_r1]
    y_test_r1 = [r["Product"].strip() for r in test_rows_r1]

    X_test_r2 = [r["Consumer complaint narrative"].strip() for r in test_rows_r2]
    y_test_r2 = [r["Product"].strip() for r in test_rows_r2]

    labels = sorted(list(set(y_train)))

    tfidf_params = {
        "ngram_range": (1, 2),
        "max_features": 25000,
        "sublinear_tf": True,
        "strip_accents": "unicode",
    }

    # Grid search across 6 configurations
    grid = [
        {"class_weight": None, "C": 0.1},
        {"class_weight": None, "C": 1.0},
        {"class_weight": None, "C": 10.0},
        {"class_weight": "balanced", "C": 0.1},
        {"class_weight": "balanced", "C": 1.0},
        {"class_weight": "balanced", "C": 10.0},
    ]

    print("=== STEP 1: VALIDATION EVALUATION (Test set kept CLOSED) ===")
    val_results = []
    fitted_models = {}

    for idx, cfg in enumerate(grid):
        cfg_name = f"cw_{cfg['class_weight']}_C_{cfg['C']}"
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LinearSVC(class_weight=cfg["class_weight"], C=cfg["C"], max_iter=2000, random_state=42))
        ])

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        res_val = evaluate_split(pipe, X_val, y_val, labels, "validation")
        res_val["config"] = cfg
        res_val["config_name"] = cfg_name
        res_val["fit_time_seconds"] = round(fit_time, 4)

        val_results.append(res_val)
        fitted_models[cfg_name] = pipe

        print(f"[{cfg_name:24}] Val Acc: {res_val['accuracy']:.4f} | Val Macro-F1: {res_val['macro_f1']:.4f} | Val Weighted-F1: {res_val['weighted_f1']:.4f} | Fit: {fit_time:.2f}s")

    # Select configuration strictly by Validation Macro-F1
    best_config_result = max(val_results, key=lambda r: r["macro_f1"])
    best_cfg_name = best_config_result["config_name"]
    best_cfg = best_config_result["config"]
    best_model = fitted_models[best_cfg_name]

    print(f"\n=== SELECTION: Best configuration chosen strictly by Validation Macro-F1: {best_cfg_name} ===")
    print(f"Validation Macro-F1: {best_config_result['macro_f1']:.4f} | Validation Accuracy: {best_config_result['accuracy']:.4f}")

    print("\n=== STEP 2: FOLLOW-UP TEST EVALUATION (On already inspected test set) ===")
    # Evaluate baseline unweighted C=1.0 for comparison
    baseline_model = fitted_models["cw_None_C_1.0"]
    test_baseline_chrono = evaluate_split(baseline_model, X_test_chrono, y_test_chrono, labels, "test_chronological")
    test_baseline_r1 = evaluate_split(baseline_model, X_test_r1, y_test_r1, labels, "test_rule1_purged")

    # Evaluate selected configuration
    test_selected_chrono = evaluate_split(best_model, X_test_chrono, y_test_chrono, labels, "test_chronological")
    test_selected_r1 = evaluate_split(best_model, X_test_r1, y_test_r1, labels, "test_rule1_purged")
    test_selected_r2 = evaluate_split(best_model, X_test_r2, y_test_r2, labels, "test_rule2_purged")

    print(f"Selected [{best_cfg_name}] Chrono Acc: {test_selected_chrono['accuracy']:.4f} | Macro-F1: {test_selected_chrono['macro_f1']:.4f} | Weighted-F1: {test_selected_chrono['weighted_f1']:.4f}")
    print(f"Selected [{best_cfg_name}] Rule 1 Acc: {test_selected_r1['accuracy']:.4f} | Macro-F1: {test_selected_r1['macro_f1']:.4f}")
    print(f"Baseline [cw_None_C_1.0] Chrono Acc: {test_baseline_chrono['accuracy']:.4f} | Macro-F1: {test_baseline_chrono['macro_f1']:.4f}")

    # Build versioned artifact record
    output_data = {
        "metadata": {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "selection_criterion": "Strictly Validation Macro-F1 across 6 configurations (class_weight in [None, 'balanced'], C in [0.1, 1.0, 10.0]). Test set kept closed until configuration selection was fixed.",
            "test_evaluation_notice": "Follow-up evaluation on an already inspected test set, not a fresh untouched final holdout.",
            "selected_configuration": best_cfg,
            "selected_configuration_name": best_cfg_name,
            "split_sizes": {
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "test_chronological_rows": len(test_rows_chrono),
                "test_rule1_purged_rows": len(test_rows_r1),
                "test_rule2_purged_rows": len(test_rows_r2),
            },
            "split_date_boundaries": {
                "train": "2024-01-01 to 2024-01-22 (22 days)",
                "val": "2024-01-23 to 2024-01-26 (4 days)",
                "test": "2024-01-27 to 2024-01-31 (5 days)",
            },
            "classes": labels,
        },
        "validation_grid_results": val_results,
        "selected_configuration_validation": best_config_result,
        "baseline_unweighted_validation": [r for r in val_results if r["config_name"] == "cw_None_C_1.0"][0],
        "follow_up_test_results": {
            "selected_config_chronological": test_selected_chrono,
            "selected_config_rule1_purged": test_selected_r1,
            "selected_config_rule2_purged": test_selected_r2,
            "baseline_unweighted_chronological": test_baseline_chrono,
            "baseline_unweighted_rule1_purged": test_baseline_r1,
        },
    }

    run_dir = Path("experiments/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, models_dir / f"selected_{best_cfg_name}.joblib")

    output_path = run_dir / "metrics_phase3.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nTuning experiment complete. Artifacts saved to {run_dir}/")
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="run_004_cfpb_imbalance_tuning")
    args = parser.parse_args()
    run_tuning(run_id=args.run_id)

