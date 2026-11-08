# Retrieval Benchmark Dataset: Hand-Crafted Question Set

This directory contains the synthetic knowledge-base corpus and labeled evaluation query set used to measure offline retrieval quality for Phase 4 of the Ticket Intelligence Platform.

---

## 1. Domain Boundary & Disclaimer

- **Synthetic Nature**: All articles and tickets in this dataset are **entirely fictional, synthetic demonstration fixtures**. They do **not** reflect real banking policies, operational procedures, or legal guidance of any financial institution.
- **Evaluation Status**: Relevance annotations in this dataset are **developer-authored judgments**, created to assess ranking precision and recall under known topical mapping. They do **not** represent an independent, double-blind, human-annotated gold standard.
- **Scope Restriction**: This benchmark evaluates **retrieval quality** (embedding similarity ranking), not database SQL execution, tenant isolation, API latency, or production infrastructure scalability.

---

## 2. Frozen Dataset Files & Hashes

The dataset files are frozen prior to scoring to prevent post-hoc tuning against evaluation results:

| File | Content | Records | SHA-256 Digest |
| :--- | :--- | :---: | :--- |
| `articles.json` | Fictional banking reference policy documents | 10 articles (12 deterministic chunks) | `18ae28b9f7a7ce7cb86eb9485ececf69e8fcc1c89491086ea107b3dfbe9cf59e` |
| `queries.json` | Support tickets with ground-truth relevance references | 30 queries (`q01`–`q30`) | `314ebfdd88c667a2c1be50df109fe663127b56b8c14f1c854157f071eb524e8a` |

---

## 3. Corpus Composition

The corpus spans 10 distinct banking support domains designed to evaluate semantic discrimination, confusable policy topics, and multi-relevant inquiries:

1. **`kb-art-001`**: ATM Deposit Hold Cutoff and Overdraft Fee Waiver Policy (2 chunks)
2. **`kb-art-002`**: Electronic Funds Transfer and Unauthorized Debit Card Disputes (Reg E, 1 chunk)
3. **`kb-art-003`**: Domestic Wire Irrevocability & International Remittance Cancellation (1 chunk)
4. **`kb-art-004`**: Credit Card Billing Error Resolution & Chargeback Evidence (FCBA, 1 chunk)
5. **`kb-art-005`**: Mortgage Escrow Analysis, Shortage Repayment, and Surplus Refunds (RESPA, 1 chunk)
6. **`kb-art-006`**: Auto Loan Grace Periods, Delinquency Late Fees, and Default Notice (1 chunk)
7. **`kb-art-007`**: Identity Theft Account Lockdown, Fraud Affidavit, and FCRA Credit Block (1 chunk)
8. **`kb-art-008`**: Personal Check Stop Payment Order Validity and Expiration Rules (1 chunk)
9. **`kb-art-009`**: Dormant Deposit Account Inactivity Fees and State Escheatment (1 chunk)
10. **`kb-art-010`**: Certificate of Deposit (CD) Early Withdrawal Penalties and Maturity Grace Period (2 chunks)

---

## 4. Query Taxonomy

- **Paraphrased Inquiries**: Tickets using conversational language distinct from formal policy terminology (e.g., `"ATM ate my check"`, `"car repossession notice"`).
- **Confusable Scenarios**: Questions exercising boundary conditions between similar topics:
  - Unauthorized debit card fraud (Reg E) vs Credit card billing disputes (FCBA) vs Identity theft freeze.
  - Wire cancellation recall vs Personal check stop payment order.
  - ATM overdraft fee waiver vs Auto loan late fee vs Dormant account maintenance fee.
- **Multi-Relevant Tickets**: Complex complaints touching multiple distinct passage chunks (e.g., deposit hold delay causing bounced check fees requiring both hold rules and fee waiver authorization).
