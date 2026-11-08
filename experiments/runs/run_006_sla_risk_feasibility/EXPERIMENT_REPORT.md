# Experiment Report: Resolution SLA Breach Risk Feasibility (T0)

- **Experiment Identifier**: `run_006_sla_risk_feasibility`
- **Execution Date**: 2026-11-08
- **Hardware & Environment**: CPU-only, Python 3.14.0, scikit-learn 1.9.1, numpy 2.5.4, joblib 1.6.0
- **Execution Wall Clock**: ~1.04 seconds

---

## 1. Executive Summary
This offline feasibility experiment evaluates whether machine learning models can estimate the risk of support ticket resolution SLA breaches using only information available at the initial moment of ticket creation ($T_0$).

To establish an un-compromised evaluation pipeline without modifying the production database schema or interfering with live services, this experiment was conducted strictly offline using `data/customer_support_tickets.csv` with existing CPU-friendly scikit-learn dependencies.

### Key Findings
1. **Metadata-only model does not improve on prior baseline**: In this experiment, a metadata-only logistic regression using ticket intake metadata (`product` and `channel`) achieved an Average Precision of **0.1517** and an ROC-AUC of **0.4858** on the test set—failing to improve upon the chance prior baseline (`DummyClassifier`, AP = **0.1560**, ROC-AUC = **0.5000**). While this demonstrates that linear encoding of intake metadata offered no predictive separation on this synthetic split, it does not establish that metadata inherently carries zero signal in general or under alternative representations.
2. **Text features capture generator-induced risk signals**: Combining intake text (`message` via TF-IDF) with metadata improved test Average Precision to **0.2383** (+0.0823 over baseline) and ROC-AUC to **0.6334** (+0.1334 over baseline). In this synthetic benchmark, complaint text contains keywords associated with categories (e.g. feedback vs urgent billing issues) that parameterize the synthetic resolution time distributions.
3. **Threshold calibration on validation data**: At the standard default threshold ($0.50$), the model predicted 0 true positive breaches on the test set due to class imbalance (~15.6% prevalence). Tuning the decision threshold on validation data ($th = 0.19$) yielded **41.77% recall** with **25.53% precision** on the test set. As disclosed below, exploratory test metrics were inspected during prototyping before the validation sweep was finalized; hence reported test metrics represent exploratory synthetic evidence rather than an untouched-holdout validation.

---

## 2. Dataset Verification and Target Definition

### Dataset Provenance & Integrity
- **Source File**: `data/customer_support_tickets.csv`
- **Dataset SHA-256**: `c93d5d57e69bee8ac8cb8447af3469fc150c5414761f7197fd1775c6de2a5f2e`
- **Total Records**: 20,000 rows
- **Missing Values**: Exactly 0 across all 12 columns
- **Unique Messages**: 6,104 unique strings (13,896 duplicate message rows)

### Label Definition
$$\text{breach} = \mathbb{I}(\text{resolution\_hours} \times 60.0 > 1440.0)$$

- **Target Threshold**: 24 elapsed calendar hours ($1440.0\text{ minutes}$).
- **Boundary Semantics**: Strict inequality. A ticket resolved at exactly 24.0 hours (22 occurrences in dataset) is classified as **non-breach (0)**.
- **Overall Class Balance**:
  - Non-breach ($0$): 16,933 records (84.665%)
  - Breach ($1$): 3,067 records (15.335%)

---

## 3. Feature Selection & Leakage Prevention

| Column Name | Status | Justification / Leakage Rationale |
| :--- | :--- | :--- |
| `message` | **Included (Feature)** | Raw customer problem description; demonstrably available at $T_0$. |
| `product` | **Included (Feature)** | Selected product line; available at ticket intake. |
| `channel` | **Included (Feature)** | Ingestion channel (Email, Chat, Phone, Web Form, Social Media); available at $T_0$. |
| `ticket_id` | **Excluded** | Artificial surrogate key; non-predictive. |
| `created_at` | **Excluded** | Intake timestamp; not monotonic; excluded to prevent temporal memorization. |
| `category` | **Excluded** | Downstream triage label; in production, derived after ingestion. In synthetic benchmark, directly dictates synthetic priority. |
| `priority` | **Excluded** | Deterministic generator rule based on category and message substrings; using it leaks the synthetic generator. |
| `sentiment` | **Excluded** | Downstream triage annotation. |
| `first_response_hours` | **Excluded (Outcome)** | Post-creation event; target leakage. |
| `resolution_hours` | **Excluded (Target)** | Ground-truth outcome used solely to compute target label. |
| `status` | **Excluded (Outcome)** | Final lifecycle progression (`Closed`, `Resolved`). |
| `csat_score` | **Excluded (Outcome)** | Post-resolution survey. |

---

## 4. Split Design & Group Separation

To prevent data leakage from repeated message templates, a **GroupShuffleSplit on unique message** was executed:
- **Outer Split**: 70% Train (14,079 rows), 30% Temporary (5,921 rows), `random_state=42`.
- **Inner Split**: 50% Validation (2,960 rows), 50% Test (2,961 rows), `random_state=42`.

### Group Separation Audit
- **Train vs Validation Message Overlap**: Exactly 0.
- **Train vs Test Message Overlap**: Exactly 0.
- **Validation vs Test Message Overlap**: Exactly 0.
- **Row Index Disjointness**: $S_{\text{train}} \cap S_{\text{val}} = \emptyset$, $S_{\text{train}} \cap S_{\text{test}} = \emptyset$, $S_{\text{val}} \cap S_{\text{test}} = \emptyset$.
- **Test Set Quarantine Disclosure**: In the final training script (`train_sla_risk.py`), preprocessing is fitted exclusively on training data and the decision threshold ($th = 0.19$) is selected purely by optimizing $F_1$ on validation data prior to evaluating the test set. However, during earlier interactive prototyping (transcript steps 8283 and 8285), exploratory test metrics were computed and inspected prior to designing the validation threshold sweep. Consequently, the reported test performance represents exploratory synthetic evidence rather than a strictly untouched-holdout validation.

---

## 5. Model Architecture & Baselines

All preprocessing transformers were fitted exclusively on training data:
1. **Dummy Baseline**: `DummyClassifier(strategy="prior")` predicting training class prevalence ($0.1510$).
2. **Metadata-only Baseline**:
   - `OneHotEncoder(handle_unknown="ignore")` on `['product', 'channel']`
   - `LogisticRegression(random_state=42, max_iter=1000, C=1.0)`
3. **Text + Metadata Primary Model**:
   - `TfidfVectorizer(ngram_range=(1,2), max_features=5000, stop_words="english")` on `message`
   - `OneHotEncoder(handle_unknown="ignore")` on `['product', 'channel']`
   - `LogisticRegression(random_state=42, max_iter=1000, C=1.0)`

### Threshold Selection
- Validation search over $[0.05, 0.95]$ in 91 steps.
- Peak validation $F_1$ achieved at **$th = 0.19$** (Val Precision: 0.3026, Val Recall: 0.5616, Val $F_1$: 0.3933).

---

## 6. Comprehensive Evaluation Results

### Test Set Metrics ($N = 2,961$, Prevalence = $15.60\%$)

| Metric | Dummy (Prior) | Metadata-Only LogReg | Text + Metadata (Default $th=0.50$) | Text + Metadata (Tuned $th=0.19$) |
| :--- | :---: | :---: | :---: | :---: |
| **Average Precision** | 0.1560 | 0.1517 | **0.2383** | **0.2383** |
| **ROC-AUC** | 0.5000 | 0.4858 | **0.6334** | **0.6334** |
| **Brier Score** | 0.1317 | 0.1320 | **0.1282** | **0.1282** |
| **Log Loss** | 0.4331 | 0.4342 | **0.4218** | **0.4218** |
| **Decision Threshold** | 0.50 | 0.50 | 0.50 | 0.19 |
| **Precision** | 0.0000 | 0.0000 | 0.0000 | **0.2553** |
| **Recall** | 0.0000 | 0.0000 | 0.0000 | **0.4177** |
| **$F_1$ Score** | 0.0000 | 0.0000 | 0.0000 | **0.3169** |

### Confusion Matrix on Test Set ($th = 0.19$)
$$\begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix} = \begin{pmatrix} 1,936 & 563 \\ 269 & 193 \end{pmatrix}$$

- **True Negatives**: 1,936 non-breach tickets correctly cleared.
- **False Positives**: 563 non-breach tickets flagged as high risk.
- **False Negatives**: 269 actual breaches missed.
- **True Positives**: 193 actual breaches caught at ticket creation.

---

## 7. Artifacts & Integrity Hashes

All experiment artifacts are isolated in `experiments/runs/run_006_sla_risk_feasibility/`:
- `model.joblib`: Serialized Text + Metadata scikit-learn pipeline.
  - **SHA-256**: `4d20957606f09b99039ce9e547efe4badf45bfad025baf617d0f384d654ca93a`
- `model.joblib.sha256`: Hash verification file.
- `metrics_sla_risk.json`: Machine-readable metrics across all splits and baselines.
- `split_manifest.json`: Non-sensitive row and ticket ID partition indices.
- `model_metadata.json`: Hyperparameters, execution duration, and dependency versions.
- `PROVENANCE_ADDENDUM.md`: Dated review and provenance audit detailing commit state, working-tree status, and development workflow.
- `LIMITATIONS.md`: Prominent disclosure of synthetic dataset constraints.
- `OPERATIONAL_DATA_COLLECTION_PLAN.md`: Proposed data collection roadmap for production SLA modeling.
