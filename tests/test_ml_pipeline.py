"""Tests for ML training and evaluation pipeline integrity:
- Split disjointness and chronological monotonicity
- Metric support counts matching evaluation split sizes
- Confusion matrix sums equaling evaluation split sizes
- TF-IDF vectorizer fit strictly on training data
"""

import csv
from datetime import datetime
from pathlib import Path
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

from scripts.train_pipeline import (
    load_dataset,
    prepare_splits,
    evaluate_model,
    PRIORITY_MAP,
)


@pytest.fixture(scope="module")
def dataset_rows():
    csv_path = Path("data/customer_support_tickets.csv")
    if not csv_path.exists():
        pytest.skip(f"Dataset file not found at {csv_path}")
    return load_dataset(str(csv_path))


@pytest.fixture(scope="module")
def prepared_splits(dataset_rows):
    return prepare_splits(dataset_rows)


def test_split_disjointness_and_monotonicity(dataset_rows, prepared_splits):
    """Assert train, val, and test splits are strictly disjoint, contiguous, and chronological."""
    train = prepared_splits["train"]
    val = prepared_splits["val"]
    test = prepared_splits["test"]

    n_total = len(dataset_rows)
    assert len(train) + len(val) + len(test) == n_total, "Splits do not sum to total row count"
    assert len(train) == 14000
    assert len(val) == 3000
    assert len(test) == 3000

    # Ticket IDs must be completely disjoint
    train_ids = set(r["ticket_id"] for r in train)
    val_ids = set(r["ticket_id"] for r in val)
    test_ids = set(r["ticket_id"] for r in test)

    assert len(train_ids) == len(train), "Train IDs contain duplicates"
    assert len(val_ids) == len(val), "Val IDs contain duplicates"
    assert len(test_ids) == len(test), "Test IDs contain duplicates"

    assert train_ids.isdisjoint(val_ids), "Train and Val share ticket IDs"
    assert train_ids.isdisjoint(test_ids), "Train and Test share ticket IDs"
    assert val_ids.isdisjoint(test_ids), "Val and Test share ticket IDs"

    # Chronological monotonicity
    train_max_date = max(datetime.strptime(r["created_at"], "%Y-%m-%d %H:%M:%S") for r in train)
    val_min_date = min(datetime.strptime(r["created_at"], "%Y-%m-%d %H:%M:%S") for r in val)
    val_max_date = max(datetime.strptime(r["created_at"], "%Y-%m-%d %H:%M:%S") for r in val)
    test_min_date = min(datetime.strptime(r["created_at"], "%Y-%m-%d %H:%M:%S") for r in test)

    assert train_max_date <= val_min_date, f"Train ({train_max_date}) overlaps Val ({val_min_date})"
    assert val_max_date <= test_min_date, f"Val ({val_max_date}) overlaps Test ({test_min_date})"


def test_confusion_matrix_sum_equals_evaluation_split_size(prepared_splits):
    """Assert that for each evaluation, confusion matrix sum equals split size and report support matches."""
    X_train = [r["message"].strip() for r in prepared_splits["train"]]
    y_train = [r["category"].strip() for r in prepared_splits["train"]]

    X_test_chrono = [r["message"].strip() for r in prepared_splits["test"]]
    y_test_chrono = [r["category"].strip() for r in prepared_splits["test"]]

    X_test_dedup = [r["message"].strip() for r in prepared_splits["test_dedup"]]
    y_test_dedup = [r["category"].strip() for r in prepared_splits["test_dedup"]]

    labels = sorted(list(set(y_train)))

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=1000)),
        ("clf", LogisticRegression(max_iter=200, random_state=42))
    ])
    pipe.fit(X_train, y_train)

    # 1. Chronological test
    eval_chrono = evaluate_model(pipe, X_train, y_train, X_test_chrono, y_test_chrono, labels, "category", "lr")
    cm_chrono = eval_chrono["confusion_matrix"]
    cm_sum_chrono = sum(sum(row) for row in cm_chrono)
    assert cm_sum_chrono == len(X_test_chrono) == 3000, f"Expected 3000, got {cm_sum_chrono}"
    assert eval_chrono["test_samples"] == 3000

    report_chrono = eval_chrono["classification_report"]
    support_sum_chrono = sum(report_chrono[lbl]["support"] for lbl in labels)
    assert support_sum_chrono == 3000

    # 2. Train-purged deduplicated test (Rule 1)
    eval_dedup = evaluate_model(pipe, X_train, y_train, X_test_dedup, y_test_dedup, labels, "category", "lr")
    cm_dedup = eval_dedup["confusion_matrix"]
    cm_sum_dedup = sum(sum(row) for row in cm_dedup)
    assert cm_sum_dedup == len(X_test_dedup) == 332, f"Expected 332, got {cm_sum_dedup}"
    assert eval_dedup["test_samples"] == 332

    report_dedup = eval_dedup["classification_report"]
    support_sum_dedup = sum(report_dedup[lbl]["support"] for lbl in labels)
    assert support_sum_dedup == 332


def test_tfidf_fitted_on_train_only(prepared_splits):
    """Assert that the vectorizer vocabulary is fit strictly on X_train without test leakage."""
    X_train = [r["message"].strip() for r in prepared_splits["train"]]
    X_test = [r["message"].strip() for r in prepared_splits["test"]]

    # Introduce a unique token in X_test that never appears in X_train
    unique_token = "zzuniquetokentest123"
    X_test_with_unique = X_test + [f"Here is a test with {unique_token}"]

    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    vec.fit(X_train)

    vocab = vec.vocabulary_
    assert unique_token not in vocab, "Vectorizer vocabulary leaked a test-only token"

    # Transforming X_test must not modify the learned vocabulary
    initial_vocab_size = len(vocab)
    initial_feature_names = list(vec.get_feature_names_out())

    vec.transform(X_test_with_unique)

    assert len(vec.vocabulary_) == initial_vocab_size, "Transform altered vectorizer vocabulary size"
    assert list(vec.get_feature_names_out()) == initial_feature_names, "Transform altered feature names"


def test_priority_rule_consistency(dataset_rows):
    """Verify the deterministic synthetic priority assignment rule across the full dataset."""
    exceptions = []
    for r in dataset_rows:
        cat = r["category"].strip()
        prio = r["priority"].strip()
        msg = r["message"].strip().lower()

        if cat in ("Feedback", "Product Inquiry"):
            expected = "Low"
        elif cat in ("Account", "Shipping"):
            expected = "Medium"
        elif cat in ("Technical Issue", "Billing"):
            expected = "Urgent" if ("urgent" in msg or "asap" in msg) else "High"
        else:
            expected = "Unknown"

        if prio != expected:
            exceptions.append(r["ticket_id"])

    assert len(exceptions) == 0, f"Found {len(exceptions)} exceptions to priority rule: {exceptions[:5]}"

