"""CFPB Product Category Classification Pipeline & Audit.

Dataset: data/cfpb_complaints_jan2024.csv
Source Export: CCDB_Export_5_September_2023_through_March_2024.csv
Source SHA-256: 7a8645f6f602762fcf8cdbf01005aaad9a5834debc9cab549ff94b6b128b1ab0
Extracted CSV SHA-256: 9586914d56e3205e92cae4251d6f200852918f9f40f1236ef821bb7a52b07b5f

Target: 'Product' (11 financial product categories).
Input: 'Consumer complaint narrative' exclusively.

Chronological Splits (strict day-boundary separation):
- Train: 2024-01-01 .. 2024-01-22 (22 days) = 13,606 rows (68.03%)
- Val:   2024-01-23 .. 2024-01-26 (4 days)  =  3,326 rows (16.63%)
- Test:  2024-01-27 .. 2024-01-31 (5 days)  =  3,068 rows (15.34%)

Evaluations:
- Full Chronological Test Set (N=3068)
- Rule 1 Purged Test Set (narratives not in Train)
- Rule 2 Purged Test Set (narratives not in Train nor Val)
"""

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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


def audit_dataset(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_rows = len(rows)
    narratives = [r["Consumer complaint narrative"].strip() for r in rows]
    unique_narratives = len(set(narratives))
    duplicate_narratives = total_rows - unique_narratives

    # Date breakdown
    dates = [r["Date received"].strip() for r in rows]
    date_counts = Counter(dates)

    # Product breakdown
    products = [r["Product"].strip() for r in rows]
    product_counts = Counter(products)

    # Length statistics
    word_counts = [len(n.split()) for n in narratives]
    char_counts = [len(n) for n in narratives]

    # Redaction analysis (XXXX patterns)
    redaction_counts = [len(re.findall(r"X{2,}", n)) for n in narratives]
    complaints_with_redaction = sum(1 for c in redaction_counts if c > 0)

    # Cross-split leakage check
    train_narratives = set(r["Consumer complaint narrative"].strip() for r in rows if r["Date received"] <= "2024-01-22")
    val_narratives = set(r["Consumer complaint narrative"].strip() for r in rows if "2024-01-23" <= r["Date received"] <= "2024-01-26")
    test_rows = [r for r in rows if r["Date received"] >= "2024-01-27"]
    test_narratives = set(r["Consumer complaint narrative"].strip() for r in test_rows)

    test_in_train = test_narratives.intersection(train_narratives)
    test_in_val = test_narratives.intersection(val_narratives)
    test_in_train_or_val = test_narratives.intersection(train_narratives.union(val_narratives))

    test_rows_rule1 = [r for r in test_rows if r["Consumer complaint narrative"].strip() not in train_narratives]
    test_rows_rule2 = [r for r in test_rows if r["Consumer complaint narrative"].strip() not in train_narratives.union(val_narratives)]

    return {
        "total_records": total_rows,
        "date_coverage": {
            "start_date": min(dates),
            "end_date": max(dates),
            "total_days": len(date_counts),
            "daily_counts": dict(sorted(date_counts.items())),
        },
        "product_distribution": dict(product_counts.most_common()),
        "narrative_metrics": {
            "unique_narratives": unique_narratives,
            "duplicate_narrative_occurrences": duplicate_narratives,
            "word_count_min": min(word_counts),
            "word_count_median": int(np.median(word_counts)),
            "word_count_mean": round(float(np.mean(word_counts)), 2),
            "word_count_max": max(word_counts),
            "char_count_median": int(np.median(char_counts)),
            "char_count_mean": round(float(np.mean(char_counts)), 2),
            "complaints_with_XXXX_redaction": complaints_with_redaction,
            "complaints_with_XXXX_redaction_pct": round(complaints_with_redaction / total_rows * 100, 2),
            "median_redactions_per_complaint": int(np.median(redaction_counts)),
            "mean_redactions_per_complaint": round(float(np.mean(redaction_counts)), 2),
        },
        "leakage_and_splits": {
            "train_days": "2024-01-01 to 2024-01-22",
            "train_rows": sum(1 for r in rows if r["Date received"] <= "2024-01-22"),
            "val_days": "2024-01-23 to 2024-01-26",
            "val_rows": sum(1 for r in rows if "2024-01-23" <= r["Date received"] <= "2024-01-26"),
            "test_days": "2024-01-27 to 2024-01-31",
            "test_rows": len(test_rows),
            "test_unique_narratives": len(test_narratives),
            "test_unique_narratives_in_train": len(test_in_train),
            "test_unique_narratives_in_train_pct": round(len(test_in_train) / len(test_narratives) * 100, 2),
            "test_unique_narratives_in_train_or_val": len(test_in_train_or_val),
            "test_unique_narratives_in_train_or_val_pct": round(len(test_in_train_or_val) / len(test_narratives) * 100, 2),
            "test_rule1_purged_rows": len(test_rows_rule1),
            "test_rule2_purged_rows": len(test_rows_rule2),
        },
    }


def evaluate_model(
    pipeline: Any,
    X_train: List[str],
    y_train: List[str],
    X_test: List[str],
    y_test: List[str],
    label_names: List[str],
    model_name: str,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    inference_time = time.perf_counter() - t1

    acc = accuracy_score(y_test, y_pred)
    macro_p = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    report_dict = classification_report(
        y_test, y_pred, labels=label_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=label_names).tolist()

    cm_sum = sum(sum(row) for row in cm)
    report_support_sum = sum(report_dict[lbl]["support"] for lbl in label_names)

    assert cm_sum == len(X_test), f"Confusion matrix sum {cm_sum} != test size {len(X_test)}"
    assert report_support_sum == len(X_test), f"Report support sum {report_support_sum} != test size {len(X_test)}"

    latency_per_sample_ms = (inference_time / len(X_test) * 1000.0) if X_test else 0.0

    return {
        "model": model_name,
        "target": "Product",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "confusion_matrix_sum": cm_sum,
        "report_support_sum": report_support_sum,
        "fit_time_seconds": round(fit_time, 4),
        "inference_time_seconds": round(inference_time, 4),
        "latency_ms_per_sample": round(latency_per_sample_ms, 4),
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "labels": label_names,
        "classification_report": report_dict,
        "confusion_matrix": cm,
    }


def run_cfpb_pipeline(
    csv_path: str = "data/cfpb_complaints_jan2024.csv",
    run_id: str = "run_003_cfpb_jan2024",
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    rows = []
    with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    rows.sort(key=lambda r: (r["Date received"].strip(), int(r["Complaint ID"])))

    # 1. Audit
    audit_data = audit_dataset(rows)

    # 2. Split rows
    train_rows = [r for r in rows if r["Date received"] <= "2024-01-22"]
    val_rows = [r for r in rows if "2024-01-23" <= r["Date received"] <= "2024-01-26"]
    test_rows_chrono = [r for r in rows if r["Date received"] >= "2024-01-27"]

    train_narratives_set = set(r["Consumer complaint narrative"].strip() for r in train_rows)
    val_narratives_set = set(r["Consumer complaint narrative"].strip() for r in val_rows)
    train_val_narratives_set = train_narratives_set.union(val_narratives_set)

    test_rows_rule1 = [r for r in test_rows_chrono if r["Consumer complaint narrative"].strip() not in train_narratives_set]
    test_rows_rule2 = [r for r in test_rows_chrono if r["Consumer complaint narrative"].strip() not in train_val_narratives_set]

    X_train = [r["Consumer complaint narrative"].strip() for r in train_rows]
    y_train = [r["Product"].strip() for r in train_rows]

    X_val = [r["Consumer complaint narrative"].strip() for r in val_rows]
    y_val = [r["Product"].strip() for r in val_rows]

    X_test_chrono = [r["Consumer complaint narrative"].strip() for r in test_rows_chrono]
    y_test_chrono = [r["Product"].strip() for r in test_rows_chrono]

    X_test_r1 = [r["Consumer complaint narrative"].strip() for r in test_rows_rule1]
    y_test_r1 = [r["Product"].strip() for r in test_rows_rule1]

    X_test_r2 = [r["Consumer complaint narrative"].strip() for r in test_rows_rule2]
    y_test_r2 = [r["Product"].strip() for r in test_rows_rule2]

    labels = sorted(list(set(y_train)))

    tfidf_params = {
        "ngram_range": (1, 2),
        "max_features": 25000,
        "sublinear_tf": True,
        "strip_accents": "unicode",
    }

    models_config = [
        ("majority_baseline", lambda: DummyClassifier(strategy="most_frequent")),
        ("tfidf_logistic_regression", lambda: Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        ])),
        ("tfidf_linear_svc", lambda: Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=42))
        ])),
    ]

    all_metrics = {
        "metadata": {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "source_archive": "CCDB_Export_5_September_2023_through_March_2024.csv",
            "source_hash_sha256": "7a8645f6f602762fcf8cdbf01005aaad9a5834debc9cab549ff94b6b128b1ab0",
            "extracted_csv": "data/cfpb_complaints_jan2024.csv",
            "extracted_csv_sha256": "9586914d56e3205e92cae4251d6f200852918f9f40f1236ef821bb7a52b07b5f",
            "selection_rule": "Deterministic systematic daily sampling from 43,920 eligible January 2024 records to yield exactly 20,000 records covering all 31 days proportionally without splitting any receipt date.",
            "split_sizes": {
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "test_chronological_rows": len(test_rows_chrono),
                "test_rule1_purged_rows": len(test_rows_rule1),
                "test_rule2_purged_rows": len(test_rows_rule2),
            },
            "split_date_boundaries": {
                "train": "2024-01-01 to 2024-01-22 (22 days)",
                "val": "2024-01-23 to 2024-01-26 (4 days)",
                "test": "2024-01-27 to 2024-01-31 (5 days)",
            },
            "tfidf_params": tfidf_params,
            "classes": labels,
        },
        "audit": audit_data,
        "results_chronological_test": {},
        "results_rule1_purged_test": {},
        "results_rule2_purged_test": {},
    }

    run_dir = Path("experiments/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    if save_artifacts:
        models_dir.mkdir(parents=True, exist_ok=True)

    print("\n================== CFPB PRODUCT CLASSIFICATION ==================")
    for name, model_fn in models_config:
        # Chronological test
        m_chrono = model_fn()
        res_chrono = evaluate_model(m_chrono, X_train, y_train, X_test_chrono, y_test_chrono, labels, name)
        all_metrics["results_chronological_test"][name] = res_chrono

        # Rule 1 purged test
        res_r1 = evaluate_model(m_chrono, X_train, y_train, X_test_r1, y_test_r1, labels, name)
        all_metrics["results_rule1_purged_test"][name] = res_r1

        # Rule 2 purged test
        res_r2 = evaluate_model(m_chrono, X_train, y_train, X_test_r2, y_test_r2, labels, name)
        all_metrics["results_rule2_purged_test"][name] = res_r2

        print(f"[{name:26}] Chrono Acc: {res_chrono['accuracy']:.4f} | Macro-F1: {res_chrono['macro_f1']:.4f} | Rule1 F1: {res_r1['macro_f1']:.4f} | Fit Time: {res_chrono['fit_time_seconds']:.2f}s | Latency: {res_chrono['latency_ms_per_sample']:.3f}ms")

        if save_artifacts and name != "majority_baseline":
            joblib.dump(m_chrono, models_dir / f"cfpb_product_{name}.joblib")

    with open(run_dir / "audit_results.json", "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    with open(run_dir / "metrics_phase3.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nCFPB Experiment complete. Versioned artifacts saved to {run_dir}/")
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="run_003_cfpb_jan2024")
    args = parser.parse_args()
    run_cfpb_pipeline(run_id=args.run_id)

