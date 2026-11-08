# Provenance Addendum: Experiment run_006_sla_risk_feasibility

- **Addendum Date**: 2026-11-08
- **Purpose**: Clarify code provenance, working-tree state, and development workflow for `run_006_sla_risk_feasibility` without retroactively mutating the original `model_metadata.json` artifact.

---

## 1. Repository & Commit State at Original Execution
- **Evidenced Git HEAD**: `dc9b35f06e03193444d32cc46e4c61fc46731d7b` (Phase 5 Slice 3: tenant operational dashboard).
- **Working-Tree Status**: **Dirty (Uncommitted)**.
  - The training code was created in `scripts/train_sla_risk.py` during execution (transcript step 8297) and executed at step 8299 before any commit was created.
  - `DECISIONS.md` contained an uncommitted record addition.
  - The working tree had untracked files: `scripts/train_sla_risk.py`, `tests/test_sla_risk_unit.py`, and `experiments/runs/run_006_sla_risk_feasibility/`.
- **Evidence Source**:
  - Local Git log and branch tracking (`git status --porcelain=v2 --branch`).
  - Recorded tool call execution trajectory in `transcript_full.jsonl` (steps 8283 through 8299).

---

## 2. Integrity Hashes of Original Artifacts
- **Dataset (`data/customer_support_tickets.csv`)**:
  - SHA-256: `c93d5d57e69bee8ac8cb8447af3469fc150c5414761f7197fd1775c6de2a5f2e`
- **Model Artifact (`experiments/runs/run_006_sla_risk_feasibility/model.joblib`)**:
  - SHA-256: `4d20957606f09b99039ce9e547efe4badf45bfad025baf617d0f384d654ca93a`
- **Metrics JSON (`experiments/runs/run_006_sla_risk_feasibility/metrics_sla_risk.json`)**:
  - SHA-256: `3ee7796561e06398da1439664702426081182d70e85444025032ea6d161da2bb`
- **Split Manifest (`experiments/runs/run_006_sla_risk_feasibility/split_manifest.json`)**:
  - SHA-256: `9a6eff555153829d04e0e4768995ed89ff093bbba99759071a184eb7bec4fb02`
- **Original Model Metadata (`experiments/runs/run_006_sla_risk_feasibility/model_metadata.json`)**:
  - SHA-256: `abec8263154dc2fcabeb368bc4381148ed3247943859548941fae6b524df9eb0`

---

## 3. Distinction Between Original Training Facts and Review Discoveries

### Facts Captured at Execution Time (Original `model_metadata.json`)
1. Model structure: `Pipeline(ColumnTransformer([TfidfVectorizer, OneHotEncoder]), LogisticRegression)`
2. Target definition: `resolution_hours * 60.0 > 1440.0`
3. Environment: Python 3.14.0, scikit-learn 1.9.1, joblib 1.6.0, numpy 2.5.4
4. Execution time: 1.04 seconds
5. Validation-selected threshold: 0.19

### Facts Established During Subsequent Acceptance Review
1. **Interactive Workflow History**: Exploratory test set metrics were computed and inspected in step 8283 before the validation threshold sweep in step 8285 and script authoring in step 8297. While the final script strictly calculates the optimal threshold on validation data without test feedback, the interactive development workflow had already exposed test performance.
2. **Current Training Script Source**: The file `scripts/train_sla_risk.py` has been updated with artifact overwrite guards and git provenance logging for future runs. Consequently, the current repository file is an upgraded version of the script and is **not** the exact verbatim source that initially generated `model.joblib` at step 8299.
