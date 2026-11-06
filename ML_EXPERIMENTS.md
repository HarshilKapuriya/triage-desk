# Machine Learning Experiments & Classification Baselines

This document records the Phase 3 exploration, dataset audits, baseline comparisons, and the frozen financial-product classification baseline (`run_004`).

---

## 1. Domain Boundary & Scope Notice

* **Prediction Domain**: **Financial-product complaint classification**, trained on real Consumer Financial Protection Bureau (CFPB) complaint narratives.
* **Non-Goals / Scope Restrictions**:
  * This model classifies financial product lines (`Credit reporting`, `Debt collection`, `Credit card`, `Mortgage`, etc.). It does **not** classify internal helpdesk categories (`Billing`, `Technical Issue`, `Account`, `Shipping`).
  * **Priority classification remains unresolved**: CFPB complaints do not possess operational priority labels (e.g. `urgent`, `high`, `normal`, `low`). No priority labels were inferred, generated, or assigned. Priority prediction remains explicitly deferred.
  * **Integration Scope**: The frozen LinearSVC model from `run_004` is integrated as a financial complaint triage demo via `model_versions`, `ticket_predictions`, the background worker (`app/worker/handler.py`), and `GET /v1/tickets/{id}`. No dashboard UI is included.

---

## 2. Synthetic Dataset Smoke Test & Template Leakage Analysis

* **Dataset Candidate**: `biswajitpasi/customer-support-tickets` (20,000 rows).
* **Initial Observation**:
  * Both TF-IDF + Logistic Regression and TF-IDF + LinearSVC achieved **1.0000 Accuracy** and **1.0000 Macro-F1** on a 70/15/15 chronological split.
* **Root-Cause Audit (`run_001` & `run_002`)**:
  * **Severe Artificial Redundancy**: Out of 20,000 rows, only 6,104 unique messages existed (13,896 duplicate occurrences).
  * **Template Monoculture**: The dataset was synthetically generated from **exactly 192 core sentence templates** (4 semantic intents per category across 8 product names), prefixed by 5 greetings and 6 sign-offs.
  * **Test-Set Contamination**:
    * **88.6%** (2,114 / 2,385) of unique test messages appeared verbatim in the training split.
    * **100.0%** (660 / 660) of abstract message templates in the test split appeared in the training split.
    * A template-purged chronological split yielded **0 test samples** because all 192 templates were exhausted before the training cutoff date.
  * **Contradictory Priority Assignment**: Priority was assigned by rigid generation rules rather than textual evidence: `Feedback` and `Product Inquiry` were 100% labeled `Low` (even when text began with `"Urgent:"` or ended with `"Need this fixed ASAP."`), while `Technical Issue` and `Billing` were labeled `Urgent` strictly if containing `"Urgent"` or `"ASAP"`, and `High` otherwise.
* **Conclusion**: The dataset served strictly as an infrastructure smoke test verifying pipeline mechanics, vectorization isolation, and metrics calculation. Its scores reflect template memorization rather than real-world generalizability.

---

## 3. CFPB Financial Complaint Dataset (`run_003` & `run_004`)

### A. Provenance, License & Selection Protocol
* **Source Archive**: Official CFPB FOIA Electronic Reading Room — *CCDB Export September 2023 through March 2024* (`CCDB_Export_5_September_2023_through_March_2024.csv`, 618,266,146 bytes).
  * Source SHA-256: `7a8645f6f602762fcf8cdbf01005aaad9a5834debc9cab549ff94b6b128b1ab0`.
* **Legal Reuse**: U.S. Government work (17 U.S.C. § 105) dedicated to the public domain under Creative Commons CC0 1.0 Universal.
* **Extraction Protocol**:
  * Date Window: `2024-01-01` to `2024-01-31`.
  * `Date received` reflects the date the complaint was officially received by the CFPB, not an internal helpdesk ticket creation time.
  * Target Population: 43,920 complaints received in January 2024 with non-empty `Consumer complaint narrative`.
  * Deterministic Selection Rule: Grouped by date, sorted by integer `Complaint ID` ascending, and sampled systematically using proportional daily step allocations ($k_d = \text{round}(N_d \times \frac{20000}{43920})$).
  * Extracted Dataset: Exactly 20,000 records saved to `data/cfpb_complaints_jan2024.csv`.
  * Extracted SHA-256: `9586914d56e3205e92cae4251d6f200852918f9f40f1236ef821bb7a52b07b5f`.

### B. Narrative Quality & Preprocessing
* Input Feature: `Consumer complaint narrative` exclusively. All 15 metadata columns (e.g. `Company`, `State`, `Issue`, `Submitted via`) are excluded.
* Length: Min = 2 words, Median = 117 words, Mean = 184.8 words, Max = 5,882 words.
* PII Redactions: 72.2% of complaints contain `XXXX` scrubbed tokens (median 5 redactions per narrative).
* Narrative Uniqueness: 15,731 unique narratives; 4,269 exact duplicates (primarily standardized credit dispute letters).

### C. Chronological Day-Boundary Splits
To satisfy strict temporal isolation without splitting any single receipt date across boundaries:
* **Train Split** (`2024-01-01` through `2024-01-22`, 22 complete days): **13,606 rows (68.03%)**, all 11 classes present.
* **Validation Split** (`2024-01-23` through `2024-01-26`, 4 complete days): **3,326 rows (16.63%)**, all 11 classes present.
* **Test Split** (`2024-01-27` through `2024-01-31`, 5 complete days): **3,068 rows (15.34%)**, all 11 classes present.
* **Test Duplicate Leakage Controls**:
  * Out of 2,528 unique test narratives, only **111 (4.39%)** appeared in Train.
  * **Rule 1 Purged Test** ($N = 2,624$, 85.5% of test): Excludes narratives seen in Train.
  * **Rule 2 Purged Test** ($N = 2,558$, 83.4% of test): Excludes narratives seen in Train or Validation.

---

## 4. Model Selection & Validation Grid Tuning (`run_004`)

To address severe class imbalance (where `Credit reporting` comprises 69.4% of total volume), a 6-configuration grid search was evaluated **strictly on the Validation Split** with the Test Split kept closed.

* **Vectorization**: `TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True, strip_accents='unicode')`, fit **only** on `X_train`.

### Validation Grid Results ($N = 3,326$)
| Configuration | Validation Accuracy | Validation Macro-F1 | Validation Weighted-F1 | Training Time |
| :--- | :---: | :---: | :---: | :---: |
| `class_weight=None, C=0.1` | 0.8214 | 0.5180 | 0.7989 | 5.70s |
| **`class_weight=None, C=1.0` (Selected)** | **0.8289** | **0.5990** | **0.8194** | 7.23s |
| `class_weight=None, C=10.0` | 0.8247 | 0.5930 | 0.8187 | 14.12s |
| `class_weight='balanced', C=0.1` | 0.8244 | 0.5977 | 0.8226 | 7.19s |
| `class_weight='balanced', C=1.0` | 0.8262 | 0.5894 | 0.8255 | 9.40s |
| `class_weight='balanced', C=10.0` | 0.8205 | 0.5915 | 0.8175 | 16.18s |

* **Selection Criterion**: Selected strictly by highest **Validation Macro-F1** (`class_weight=None, C=1.0`, Macro-F1 = **0.5990**).

### Class Imbalance Tradeoff on Validation Set
Comparing the unweighted winner against `class_weight='balanced', C=0.1` demonstrates the accuracy/minority-recall tradeoff in absolute terms:
* Balancing weights decreases overall accuracy slightly from 82.89% to 82.44% (-0.45 percentage points), driven by majority `Credit reporting` recall falling from 94.2% to 90.5% (-3.7 percentage points).
* In exchange, recall on under-represented classes improves significantly:
  * `Payday loan...`: 38.1% $\to$ **61.9%** (+23.8 percentage points)
  * `Vehicle loan or lease`: 41.5% $\to$ **58.5%** (+17.0 percentage points)
  * `Mortgage`: 74.0% $\to$ **83.1%** (+9.1 percentage points)
  * `Credit card`: 64.5% $\to$ **72.7%** (+8.2 percentage points)
  * `Student loan`: 74.6% $\to$ **81.4%** (+6.8 percentage points)
  * `Debt collection`: 44.9% $\to$ **49.6%** (+4.7 percentage points)

---

## 5. Frozen Baseline: Test Performance (`run_004`)

> **Holdout Integrity Notice**: The test evaluation below is a **follow-up evaluation on an already inspected test set**, not a fresh untouched holdout.

### Summary Metrics Across Splits
| Model & Partition | Evaluated Set | Accuracy | Macro-F1 | Weighted-F1 |
| :--- | :--- | :---: | :---: | :---: |
| **Majority Baseline** | Chronological ($N=3068$) | 0.6887 | 0.0742 | 0.5618 |
| **Selected LinearSVC (`C=1.0`)** | **Chronological ($N=3068$)** | **0.8611** | **0.6206** | **0.8542** |
| **Selected LinearSVC (`C=1.0`)** | **Rule 1 Purged ($N=2624$)** | **0.8422** | **0.6208** | **0.8342** |
| **Selected LinearSVC (`C=1.0`)** | **Rule 2 Purged ($N=2558$)** | **0.8385** | **0.6200** | **0.8303** |

*Inference Latency*: 0.254 ms per sample (single thread, AMD64).

### Per-Class Test Performance (LinearSVC, $N = 3,068$)
| Product Class | Precision | Recall | F1-Score | Test Support | Train Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Credit reporting or personal consumer reports** | 0.910 | 0.957 | **0.933** | 2,113 | 9,528 |
| **Student loan** | 0.915 | 0.818 | **0.864** | 66 | 297 |
| **Mortgage** | 0.864 | 0.854 | **0.859** | 82 | 297 |
| **Checking or savings account** | 0.718 | 0.823 | **0.767** | 164 | 653 |
| **Credit card** | 0.741 | 0.673 | **0.705** | 217 | 880 |
| **Prepaid card** | 0.933 | 0.560 | **0.700** | 25 | 126 |
| **Money transfer, virtual currency, or money service** | 0.800 | 0.537 | **0.643** | 67 | 246 |
| **Debt collection** | 0.662 | 0.534 | **0.591** | 264 | 1,226 |
| **Vehicle loan or lease** | 0.529 | 0.450 | **0.486** | 40 | 175 |
| **Payday, title, or personal loan** | 0.357 | 0.227 | **0.278** | 22 | 149 |
| **Debt or credit management** | 0.000 | 0.000 | **0.000** | 8 | 29 |

---

## 6. Known Limitations & Unmeasured Leakage

1. **Near-Duplicate / Boilerplate Leakage**: Exact verbatim duplicates were measured and purged (affecting 4.39% of unique test narratives). However, **near-duplicate boilerplate** (such as credit repair templates with altered dates or minor sentence reorderings) remains unmeasured and may inflate test metrics.
2. **Extreme Class Imbalance & Long-Tail Failure**: Linear models fail to predict the rarest class (`Debt or credit management`, F1 = 0.000) due to severe lack of training data (only 29 train instances).
3. **No Domain Transfer to Helpdesk Categories**: These weights and features classify financial disputes, not customer helpdesk tickets.
4. **No Priority Model**: Priority classification remains an open architectural task.

---

## 7. Versioned Artifact Inventory

* **Extraction**: [`scripts/extract_cfpb_jan2024.py`](file:///C:/Users/91931/OneDrive/Desktop/Major%20Project/Ticket%20Intelligence%20Platform/triage-desk/scripts/extract_cfpb_jan2024.py)
* **Pipelines**:
  * [`scripts/train_pipeline.py`](file:///C:/Users/91931/OneDrive/Desktop/Major%20Project/Ticket%20Intelligence%20Platform/triage-desk/scripts/train_pipeline.py) (Synthetic baseline)
  * [`scripts/train_cfpb_pipeline.py`](file:///C:/Users/91931/OneDrive/Desktop/Major%20Project/Ticket%20Intelligence%20Platform/triage-desk/scripts/train_cfpb_pipeline.py) (CFPB baseline)
  * [`scripts/tune_cfpb_imbalance.py`](file:///C:/Users/91931/OneDrive/Desktop/Major%20Project/Ticket%20Intelligence%20Platform/triage-desk/scripts/tune_cfpb_imbalance.py) (CFPB validation grid search)
* **Pipeline Integrity Tests**: [`tests/test_ml_pipeline.py`](file:///C:/Users/91931/OneDrive/Desktop/Major%20Project/Ticket%20Intelligence%20Platform/triage-desk/tests/test_ml_pipeline.py)
* **Metrics (Sanitized JSON, No Narratives)**:
  * `experiments/runs/run_001_initial_audit/`
  * `experiments/runs/run_002_reconciled/`
  * `experiments/runs/run_003_cfpb_jan2024/`
  * `experiments/runs/run_004_cfpb_imbalance_tuning/`
* **Regression Suite Status**: **105 passed, 0 failed, 0 skipped, 1 warning** across 15 test modules in 225.55s.

---

## 8. Worker Pipeline Integration & Demo

* **Integrated Components**:
  * **Model Loader**: [`app/ml/loader.py`](file:///C:/Users/91931/OneDrive/Desktop/Major%20Project/Ticket%20Intelligence%20Platform/triage-desk/app/ml/loader.py) preloads and caches the trusted local artifact once per worker process.
  * **Expected SHA-256 Digest**: `7e833e8992a3150bdaf2489b9b9117d700f7875e9dde045e4f638bd1e656b360`.
  * **Required Dependencies**: Python 3.11+, `scikit-learn==1.9.1`, `numpy==2.5.4`, `scipy==1.18.1`, `joblib==1.6.0` (as declared in `requirements.txt`).
  * **Provisioning Notice**: Retraining via `scripts/tune_cfpb_imbalance.py` reproduces hyperparameters and splits, but platform/compiler build differences do not guarantee a bit-for-bit identical binary file hash. Environments should provision the exact verified artifact matching the SHA-256 digest above.
  * **Worker Inference**: Evaluates `ticket.body` strictly (narrative text). Missing or corrupted artifacts raise explicit operational errors (`ModelArtifactMissingError`, `ModelArtifactCorruptedError`) and requeue/fail the job without fabricating predictions.
  * **Schema & Multi-Tenancy**: `migrations/003_ml_predictions.sql` defines `model_versions` and `ticket_predictions` with composite foreign keys enforcing multi-tenant isolation. Each tenant has an explicit active row in `model_versions` (`tenant_id NOT NULL`) without global-tenant bypasses.
  * **Scores vs. Probabilities**: LinearSVC margin distances are uncalibrated; `confidence` is left strictly `NULL`, and raw margin scores are stored in `decision_scores JSONB`.
  * **Priority Status**: Priority prediction remains explicitly deferred.
  * **Integration Tests**: [`tests/test_ml_integration_db.py`](file:///C:/Users/91931/OneDrive/Desktop/Major%20Project/Ticket%20Intelligence%20Platform/triage-desk/tests/test_ml_integration_db.py) (8 tests covering valid labels, detail endpoint exposure, tenant isolation, retry idempotency, stale claim rollback, and missing/corrupt artifact handling).
  * **CI Status**: GitHub Actions CI remains marked blocked due to the outstanding repository billing lock; gated via local test execution against `TEST_DATABASE_URL`.

