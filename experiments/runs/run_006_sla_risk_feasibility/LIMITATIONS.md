# LIMITATIONS: SYNTHETIC FEASIBILITY ONLY

> [!WARNING]
> **SYNTHETIC BENCHMARK ONLY — DO NOT DEPLOY TO PRODUCTION**
> The model trained in this experiment (`run_006_sla_risk_feasibility`) is strictly an offline feasibility prototype evaluated on synthetic benchmark data (`data/customer_support_tickets.csv`). It does **not** establish predictive validity for real financial-support tickets or operational customer service workflows.

---

## 1. Dataset Provenance and Generator Properties
- **Synthetic Origin**: `data/customer_support_tickets.csv` consists of 20,000 synthetic rows generated from a template grammar.
- **Template Repetition**: As audited in Phase 3 (`run_002_reconciled`), the 20,000 rows were generated from only **24 abstract core intents**, **192 core sentence templates**, and **680 abstract templates**. Out of 20,000 rows, there are only 6,104 unique message strings, representing 13,896 exact duplicate messages.
- **Deterministic Priority Generation**: The dataset's `priority` field was generated using a 100% deterministic rule with zero exceptions:
  - `Feedback` & `Product Inquiry` $\to$ `Low`
  - `Account` & `Shipping` $\to$ `Medium`
  - `Technical Issue` & `Billing` $\to$ `Urgent` if `urgent` or `asap` in message, else `High`
- **Synthetic Duration Distribution**: The outcome fields `resolution_hours` (range 1.3h to 173.3h) and `first_response_hours` (0.12h to 104.5h) were drawn from synthetic distributions parameterized directly by this artificial `priority` assignment.

---

## 2. Leakage Controls and Residual Risks in This Experiment
- **Leakage Elimination**:
  - All direct outcome columns (`resolution_hours`, `first_response_hours`, `status`, `csat_score`) were strictly excluded from model inputs.
  - Database identifiers (`ticket_id`) and timestamps (`created_at`) were excluded from features.
  - Downstream triage outputs (`category`, `priority`, `sentiment`) were excluded to prevent leaking the synthetic generator's deterministic priority rules.
  - Exact-message duplicates were controlled using `GroupShuffleSplit(groups=message, random_state=42)` across 70% train (14,079 rows), 15% validation (2,960 rows), and 15% test (2,961 rows), ensuring **zero exact message overlap** between splits.
- **Residual Template Leakage**:
  - Because only 680 abstract templates exist across the entire 20,000 rows, messages in the test set inevitably share underlying syntactic frames (e.g. greetings, product slots, closing phrasing) with messages in the training set. Grouping by exact string eliminates literal memorization, but cannot fully decouple shared template grammar.

---

## 3. Scope of Claims and Performance Interpretation
- **Target Assumption**: The target threshold of 24 elapsed calendar hours ($1440\text{ minutes}$) is an experimental assumption chosen for offline feasibility, **not** a standard support SLA or an application configuration.
- **Model Output Terminology**: Outputs from this model represent uncalibrated **"Model Risk Scores"** in $[0.0, 1.0]$. They must **not** be interpreted as calibrated operational probabilities of breach without prospective calibration evidence (reliability diagrams, Brier score calibration).
- **Predictive Associations**:
  - The text + metadata logistic regression achieves an **Average Precision of 0.2383** (against a test prevalence of 0.1560) and an **ROC-AUC of 0.6334**.
  - A metadata-only baseline (`product` and `channel`) yields Average Precision of 0.1517 and ROC-AUC of 0.4858 (chance level), showing that linear combinations of product and channel did not improve upon the prior baseline on this synthetic split (though this does not prove metadata inherently carries zero signal in other formulations or operational contexts).
  - The model's modest edge over baseline stems entirely from TF-IDF picking up keywords associated with the generator's template categories (e.g. feedback/inquiries having longer resolution times).
- **Exploratory Workflow Disclosure**:
  - Exploratory test-set metrics were inspected during interactive development before finalizing the validation threshold sweep in `train_sla_risk.py`. Reported test metrics provide exploratory synthetic benchmark evidence, not a blind untouched-holdout evaluation.
- **Operational Non-Transferability**: Real support operations involve agent shifts, queue backlogs, ticket reassignments, customer responsiveness, third-party dependencies, and business hours. None of these operational mechanisms exist in this static synthetic file.

---

## 4. Operational Boundaries
- **No Production Integration**: This model must not be registered in `model_versions` or loaded by production background workers.
- **No Dynamic Rescoring Claim**: This slice evaluates risk strictly at ticket creation ($T_0$). Dynamic rescoring over the ticket lifecycle requires snapshot-based event logging, which does not exist in this static dataset.
- **Deterministic vs ML Facts**: Overdue states and target completions must always remain deterministic operational facts computed from timestamps and tenant policies, never inferred as an ML risk score of 1.0.
