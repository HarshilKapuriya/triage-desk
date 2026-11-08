# Phase 4 Offline Knowledge-Base Retrieval Quality Baseline

> **DISCLAIMER: SYNTHETIC DATASET NOTICE**
> All articles, queries, policies, dollar amounts, and procedures in this evaluation are **synthetic fixtures** created solely for automated offline regression and quality benchmarking. They do not represent real bank policies, legal standards, regulatory advice, or live customer inquiries. Ground-truth relevance judgments are **developer-authored annotations**, not an external, certified, or human-annotated golden standard.

---

## 1. Executive Summary

This report establishes the baseline offline retrieval quality for the Knowledge-Base feature implemented in Phase 4. Using the project's provisioned local embedding model (`sentence-transformers/all-MiniLM-L6-v2`), the benchmark evaluates exact cosine similarity ranking across a frozen synthetic banking corpus and labeled support queries.

### Macro Metrics Summary

| Metric | Baseline Score | Exploratory Target | Target Status |
| :--- | :---: | :---: | :---: |
| **Recall@1** | **0.9167** (91.7%) | — | Informational |
| **Recall@3** | **1.0000** (100.0%) | — | Informational |
| **Recall@5** | **1.0000** (100.0%) | $\ge 0.85$ | **MET** |
| **Hit Rate@1** | **0.9333** (93.3%) | — | Informational |
| **Hit Rate@3** | **1.0000** (100.0%) | — | Informational |
| **Hit Rate@5** | **1.0000** (100.0%) | — | Informational |
| **Mean Reciprocal Rank (MRR)** | **0.9611** | $\ge 0.65$ | **MET** |

*Note: Exploratory targets ($\ge 0.85$ Recall@5, $\ge 0.65$ MRR) are suggested non-SPEC engineering guardrails to detect regression.*

---

## 2. Dataset & Model Provenance

### Model Identification
- **Model Name**: `sentence-transformers/all-MiniLM-L6-v2`
- **Revision SHA**: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- **Architecture**: ONNX Runtime (384-dimensional dense vectors, mean pooling, L2 normalization)
- **Token Limit**: 256 subword tokens (including `[CLS]` and `[SEP]`)

### Dataset Provenance & Pre-Scoring Chronology Disclosure
1. **Initial Draft & Pre-Scoring Validation Halt**:
   The initial draft of `articles.json` (`cc8908d152dc...`) was authorially intended to yield 12 chunks, but because paragraphs were short, `chunk_article_text` grouped them into 10 single-chunk articles. When `scripts/evaluate_retrieval.py` was first invoked, the pre-run validator immediately raised `ValueError: Query q01 references non-existent chunk (article_id='kb-art-001', chunk_index=1)` before computing any embeddings or similarity scores.
2. **Fixture Repair Prior to Scoring**:
   The bodies of `kb-art-001` and `kb-art-010` were expanded with complete synthetic policy text so that each cleanly splits into two deterministic chunks under the 160-token target parameter.
3. **Dataset Freezing**:
   The repaired `articles.json` (SHA-256: `18ae28b9f7a7ce7cb86eb9485ececf69e8fcc1c89491086ea107b3dfbe9cf59e`) and `queries.json` (SHA-256: `314ebfdd88c667a2c1be50df109fe663127b56b8c14f1c854157f071eb524e8a`) were frozen and hashed prior to initiating any model inference.
4. **Scored Benchmark Execution**:
   The benchmark was then executed for the first time, generating run artifact `run_005_retrieval_benchmark`. **Zero query rankings or aggregate scores were observed or computed prior to freezing the final dataset**, and no fixtures or relevance labels were tuned after seeing results.

---

## 3. Metric Definitions & Mathematical Formulation

All metrics are macro-averaged across the set of $N = 30$ evaluated queries $Q$:

1. **Recall@k**:
   $$\text{Recall@k}(q) = \frac{|\text{Retrieved}_k(q) \cap \text{Relevant}(q)|}{|\text{Relevant}(q)|}, \quad \text{Macro Recall@k} = \frac{1}{|Q|} \sum_{q \in Q} \text{Recall@k}(q)$$

2. **Hit Rate@k**:
   $$\text{HitRate@k}(q) = \mathbb{I}[|\text{Retrieved}_k(q) \cap \text{Relevant}(q)| \ge 1], \quad \text{Macro HitRate@k} = \frac{1}{|Q|} \sum_{q \in Q} \text{HitRate@k}(q)$$

3. **Mean Reciprocal Rank (MRR)**:
   $$\text{RR}(q) = \begin{cases} \frac{1}{\text{rank}_1(q)} & \text{if first relevant result appears at rank } \le k \\ 0 & \text{otherwise} \end{cases}, \quad \text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \text{RR}(q)$$

4. **Deterministic Ranking & Tie-Breaking**:
   Matches production SQL (`ORDER BY kc.embedding <=> %s::vector ASC, ka.id ASC, kc.chunk_index ASC`):
   Sorted by descending cosine similarity (`score DESC`), with ties broken by `article_id ASC`, then `chunk_index ASC`.

---

## 4. Execution Timing Breakdown

Timing recorded during run `run_005_retrieval_benchmark` on local CPU:

| Phase | Duration | Details |
| :--- | :---: | :--- |
| **Model Initialization** | 0.468s | ONNX session setup & tokenizer load |
| **Validation & Chunking** | 0.029s | Parsing & deterministic chunk generation (12 chunks) |
| **Corpus Embedding** | 0.433s | Generating 12 chunk vectors (384-dim) |
| **Query Embedding** | 1.047s | Generating 30 ticket vectors (subject + body formatted) |
| **Cosine Ranking & Scoring** | 0.008s | 360 vector comparisons and macro aggregation |
| **Total Pipeline** | **~1.985s** | Offline end-to-end execution |

> [!NOTE]
> Timings reflect single-threaded local CPU execution on synthetic fixtures. They do **not** represent production database latency, connection pooling, or network round-trips.

---

## 5. Detailed Query Analysis & Sub-Ranked Results

### Complete Misses (0 queries)
There were **0 complete misses** at $k=5$ (Hit Rate@5 is 100.0%). Every query retrieved at least one ground-truth passage chunk in the top 5 results.

### First-Rank Hits (28 / 30 queries)
28 out of 30 queries retrieved a relevant passage at **Rank #1** (Hit Rate@1 = 93.33%).

### Sub-Ranked Inquiries (2 queries)
Two queries retrieved their first relevant passage at rank $> 1$:

#### 1. Query `q05`: "Provisional credit delay on debit dispute"
- **Query Body**: *"I disputed an unauthorized charge on my debit card over two weeks ago. Your policy says provisional credit is issued after 10 days if the investigation isn't done. Why haven't the funds been credited to my account yet?"*
- **Ground-Truth Target**: `kb-art-002` chunk 0 (Electronic Funds Transfer & Unauthorized Debit Card Dispute Policy)
- **Observed Ranking**:
  - **Rank 1**: `kb-art-004` chunk 0 (Credit Card Billing Error Inquiries and Merchant Chargebacks) — Cosine Similarity: `0.5053`
  - **Rank 2**: `kb-art-007` chunk 0 (Identity Theft Account Lockdown & Fraud Affidavit) — Cosine Similarity: `0.4941`
  - **Rank 3** (Target): `kb-art-002` chunk 0 — Cosine Similarity: `0.4876`
- **Qualitative Interpretation**: This inquiry exhibits semantic proximity to general card dispute terminology (Credit Card disputes vs Debit Card disputes). The top 3 results are all card-fraud/dispute policies within 0.018 cosine distance of each other. *(Note: This explanation represents a plausible qualitative interpretation of similarity scores, not a proven causal certainty.)*

#### 2. Query `q06`: "Dispute reported after monthly statement arrived"
- **Query Body**: *"I just noticed an incorrect charge on my checking statement that was mailed 3 weeks ago. Am I still eligible to dispute this transaction or is there a 2-day limit?"*
- **Ground-Truth Target**: `kb-art-002` chunk 0 (Regulation E 60-day periodic statement liability window)
- **Observed Ranking**:
  - **Rank 1**: `kb-art-004` chunk 0 (Credit Card Billing Error notice within 60 days of statement) — Cosine Similarity: `0.6382`
  - **Rank 2** (Target): `kb-art-002` chunk 0 — Cosine Similarity: `0.6231`
  - **Rank 3**: `kb-art-007` chunk 0 — Cosine Similarity: `0.4800`
- **Qualitative Interpretation**: The ticket inquires about a disputed charge appearing on a periodic statement with a 60-day vs 2-day limit. Both `kb-art-004` (FCBA credit card 60-day billing statement rule) and `kb-art-002` (Reg E debit card 60-day statement rule) share near-identical phrasing regarding statement mailing dates and dispute deadlines. *(Note: This explanation represents a plausible qualitative interpretation of similarity scores, not a proven causal certainty.)*

---

## 6. Scope Limitations & Non-Goals

1. **Embedding Quality Only**: This offline benchmark evaluates mathematical cosine similarity and embedding representation quality on local CPU. It does **not** evaluate live PostgreSQL query execution, pgvector HNSW index construction, connection pooling, SQL correctness, or Row-Level Security (RLS) tenant isolation.
2. **Synthetic Corpus & Developer Judgments**: The evaluated corpus comprises 10 small fictional banking policy documents and 30 developer-authored tickets. These scores do not establish production retrieval accuracy, customer satisfaction, scalability to thousands of articles, or compliance with real-world banking regulations. Relevance labels reflect developer annotations, not certified external or human-annotated gold standards.
3. **Exploratory Targets Are Non-Binding**: Targets ($\ge 0.85$ Recall@5, $\ge 0.65$ MRR) are suggested engineering guardrails to detect regression; they are **not** SPEC-mandated contractual acceptance thresholds.
4. **No LLM or Generation Quality Evaluated**: This evaluation measures retrieval quality only; it does not evaluate generative answer drafting, summarization, or hallucination rates.
