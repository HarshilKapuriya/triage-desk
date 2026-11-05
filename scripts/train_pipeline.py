"""Reproducible Phase 3 Training and Evaluation Pipeline.

Models evaluated for both Category and Priority:
1. Majority Class Baseline (DummyClassifier)
2. TF-IDF + LogisticRegression
3. TF-IDF + LinearSVC

Evaluates on three explicitly defined test sets:
- Chronological Test Set (N=3000): Raw 15% test split sorted by created_at.
- Test Rule 1 Purged Set (N=332): Test rows whose message never appeared in Train.
- Test Rule 2 Purged Set (N=215): Test rows whose message never appeared in Train OR Validation.

Asserts and verifies:
- Confusion matrix sum == evaluation split size.
- Classification report total support == evaluation split size.
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

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


PRIORITY_MAP = {
    "Low": "low",
    "Medium": "normal",
    "High": "high",
    "Urgent": "urgent",
}


def load_dataset(csv_path: str = "data/customer_support_tickets.csv") -> List[Dict[str, Any]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at {csv_path}")

    rows = []
    with open(path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    rows.sort(key=lambda r: datetime.strptime(r["created_at"], "%Y-%m-%d %H:%M:%S"))
    return rows


def prepare_splits(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_rows = rows[:train_end]
    val_rows = rows[train_end:val_end]
    test_rows = rows[val_end:]

    # Rule 1: Exclude messages in Train ONLY
    train_messages = set(r["message"].strip() for r in train_rows)
    test_rule1_rows = [r for r in test_rows if r["message"].strip() not in train_messages]

    # Rule 2: Exclude messages in Train OR Validation
    train_val_messages = train_messages.union(set(r["message"].strip() for r in val_rows))
    test_rule2_rows = [r for r in test_rows if r["message"].strip() not in train_val_messages]

    return {
        "train": train_rows,
        "val": val_rows,
        "test": test_rows,
        "test_dedup": test_rule1_rows,  # backwards-compatible key for tests
        "test_rule1": test_rule1_rows,
        "test_rule2": test_rule2_rows,
    }


def evaluate_model(
    pipeline: Any,
    X_train: List[str],
    y_train: List[str],
    X_test: List[str],
    y_test: List[str],
    label_names: List[str],
    target_name: str,
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

    # Verification assertions
    assert cm_sum == len(X_test), f"Confusion matrix sum {cm_sum} != test split size {len(X_test)}"
    assert report_support_sum == len(X_test), f"Report support sum {report_support_sum} != test split size {len(X_test)}"

    urgent_recall = None
    if target_name == "priority" and "urgent" in label_names:
        urgent_recall = report_dict.get("urgent", {}).get("recall", 0.0)

    latency_per_sample_ms = (inference_time / len(X_test) * 1000.0) if X_test else 0.0

    return {
        "model": model_name,
        "target": target_name,
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
        "urgent_recall": round(float(urgent_recall), 4) if urgent_recall is not None else None,
        "labels": label_names,
        "classification_report": report_dict,
        "confusion_matrix": cm,
    }


def run_experiment(
    csv_path: str = "data/customer_support_tickets.csv",
    run_id: str = "run_002_reconciled",
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    rows = load_dataset(csv_path)
    splits = prepare_splits(rows)

    X_train = [r["message"].strip() for r in splits["train"]]
    X_val = [r["message"].strip() for r in splits["val"]]
    X_test_chrono = [r["message"].strip() for r in splits["test"]]
    X_test_rule1 = [r["message"].strip() for r in splits["test_rule1"]]
    X_test_rule2 = [r["message"].strip() for r in splits["test_rule2"]]

    y_cat_train = [r["category"].strip() for r in splits["train"]]
    y_cat_test_chrono = [r["category"].strip() for r in splits["test"]]
    y_cat_test_rule1 = [r["category"].strip() for r in splits["test_rule1"]]
    y_cat_test_rule2 = [r["category"].strip() for r in splits["test_rule2"]]

    y_prio_train = [PRIORITY_MAP[r["priority"].strip()] for r in splits["train"]]
    y_prio_test_chrono = [PRIORITY_MAP[r["priority"].strip()] for r in splits["test"]]
    y_prio_test_rule1 = [PRIORITY_MAP[r["priority"].strip()] for r in splits["test_rule1"]]
    y_prio_test_rule2 = [PRIORITY_MAP[r["priority"].strip()] for r in splits["test_rule2"]]

    cat_labels = sorted(list(set(y_cat_train)))
    prio_labels = ["low", "normal", "high", "urgent"]

    tfidf_params = {
        "ngram_range": (1, 2),
        "max_features": 10000,
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

    all_results = {
        "metadata": {
            "run_id": run_id,
            "date": datetime.now().isoformat(),
            "random_seed": 42,
            "dataset_total_rows": len(rows),
            "split_sizes": {
                "train": len(splits["train"]),
                "val": len(splits["val"]),
                "test_chronological": len(splits["test"]),
                "test_rule1_purged_train_only": len(splits["test_rule1"]),
                "test_rule2_purged_train_and_val": len(splits["test_rule2"]),
            },
            "tfidf_params": tfidf_params,
            "priority_mapping": PRIORITY_MAP,
            "installed_versions": {
                "scikit-learn": "1.9.1",
                "numpy": "2.5.4",
                "scipy": "1.18.1",
                "joblib": "1.6.0",
                "python": "3.14.0",
            }
        },
        "category_results": {},
        "priority_results": {},
        "category_results_rule1_purged": {},
        "priority_results_rule1_purged": {},
        "category_results_rule2_purged": {},
        "priority_results_rule2_purged": {},
        # Backwards compatible alias for tests
        "category_results_dedup_test": {},
        "priority_results_dedup_test": {},
    }

    run_dir = Path("experiments/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    if save_artifacts:
        models_dir.mkdir(parents=True, exist_ok=True)

    for name, model_fn in models_config:
        # Category
        m_cat = model_fn()
        res_cat_chrono = evaluate_model(
            m_cat, X_train, y_cat_train, X_test_chrono, y_cat_test_chrono, cat_labels, "category", name
        )
        res_cat_r1 = evaluate_model(
            m_cat, X_train, y_cat_train, X_test_rule1, y_cat_test_rule1, cat_labels, "category", name
        )
        res_cat_r2 = evaluate_model(
            m_cat, X_train, y_cat_train, X_test_rule2, y_cat_test_rule2, cat_labels, "category", name
        )

        all_results["category_results"][name] = res_cat_chrono
        all_results["category_results_rule1_purged"][name] = res_cat_r1
        all_results["category_results_rule2_purged"][name] = res_cat_r2
        all_results["category_results_dedup_test"][name] = res_cat_r1

        # Priority
        m_prio = model_fn()
        res_prio_chrono = evaluate_model(
            m_prio, X_train, y_prio_train, X_test_chrono, y_prio_test_chrono, prio_labels, "priority", name
        )
        res_prio_r1 = evaluate_model(
            m_prio, X_train, y_prio_train, X_test_rule1, y_prio_test_rule1, prio_labels, "priority", name
        )
        res_prio_r2 = evaluate_model(
            m_prio, X_train, y_prio_train, X_test_rule2, y_prio_test_rule2, prio_labels, "priority", name
        )

        all_results["priority_results"][name] = res_prio_chrono
        all_results["priority_results_rule1_purged"][name] = res_prio_r1
        all_results["priority_results_rule2_purged"][name] = res_prio_r2
        all_results["priority_results_dedup_test"][name] = res_prio_r1

        if save_artifacts and name != "majority_baseline":
            joblib.dump(m_cat, models_dir / f"category_{name}.joblib")
            joblib.dump(m_prio, models_dir / f"priority_{name}.joblib")

    output_path = run_dir / "metrics_phase3.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="run_002_reconciled", help="Run directory ID")
    args = parser.parse_args()
    results = run_experiment(run_id=args.run_id)
    print(f"Pipeline executed successfully. Artifacts saved to experiments/runs/{args.run_id}/")
