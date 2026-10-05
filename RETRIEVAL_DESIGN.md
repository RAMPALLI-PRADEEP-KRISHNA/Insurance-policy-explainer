# Insurance Policy Explainer — Retrieval Design

> **Version:** 2.0 (Post-Adversarial Review)
> **Date:** 2026-10-05

---

## 1. Design Philosophy

Simple vector similarity search (top-k cosine/L2) is **not** assumed to be sufficient
for grounded answers in a legal-domain application. However, a cross-encoder reranker
is **not** included in the MVP pipeline — at insurance document scale (< 500 chunks),
the complexity and latency cost is not justified. Instead, the pipeline uses:

1. A multilingual embedding model covering both English and Tamil.
2. FAISS exact search (no approximation).
3. Keyword signal boosting for numerically critical terms.
4. An evidence threshold with four explicit states.
5. Complete document isolation at every stage.

The cross-encoder is an **optional post-MVP enhancement**, to be added only if
Phase 7 evaluation shows Retrieval Recall@5 < 0.75.

---

## 2. Retrieval Pipeline (Stages)

```
Query
  │
  ▼  Stage 1: Query Embedding
Embed query using sentence-transformers (same model as document chunks)
  │
  ▼  Stage 2: Candidate Retrieval (FAISS)
Search doc-specific FAISS IndexFlatIP for top-10 candidates
(L2-normalized embeddings → inner product = cosine similarity)
Return: List[(Chunk, raw_score)]
  │
  ▼  Stage 3: Keyword Signal Boost
For each candidate chunk, add a small boost if chunk contains:
  - Numeric values matching query context (e.g., "20%" "₹5,000" "30 days")
  - Risk keywords from query (exclusion, waiting, co-pay, sub-limit)
  - chunk_type == TABLE (tables often contain numeric answers)
Boost magnitude: 0.05 or 0.10 (configurable)
  │
  ▼  Stage 4: Evidence Threshold Filter
Sort candidates by (raw_score + keyword_boost) descending
Select top-5 results above LOW_THRESHOLD as candidate evidence set
  │
  ▼  Stage 5: Evidence Validation (4-State)
Evaluate the candidate evidence set:
  SUFFICIENT   → top_1_score ≥ HIGH_THRESHOLD AND len ≥ MIN_RESULTS
  MARGINAL     → top_1_score ≥ LOW_THRESHOLD (below HIGH_THRESHOLD)
  CONFLICTING  → top results contain contradictory coverage signals
  INSUFFICIENT → top_1_score < LOW_THRESHOLD OR no results pass
  │
  ▼  Stage 6: Route by Evidence Status
INSUFFICIENT → AbstentionAnswer (no LLM call)
CONFLICTING  → LLMGenerator(evidence, conflict_aware_prompt)
SUFFICIENT/MARGINAL → LLMGenerator(evidence, standard_grounding_prompt)
```

---

## 3. Embedding Model

| Property | Value |
|---|---|
| Model | `paraphrase-multilingual-mpnet-base-v2` (sentence-transformers) |
| Dimension | 768 |
| Languages | 50+ including English and Tamil |
| Hosting | Local; no API call required |
| Loading | Single instance per process (cached in `embedder.py`) |

**Rationale for multilingual model:**
Tamil queries must produce embeddings in the same vector space as English document
chunks. A multilingual model handles this without requiring separate Tamil indexes or
query translation.

**Alternative considered:** `all-MiniLM-L6-v2` (English-only, faster) — rejected
because it would fail on Tamil queries.

---

## 4. FAISS Index Configuration

| Property | Value |
|---|---|
| Index type | `IndexFlatIP` (exact inner product search) |
| Normalization | Embeddings L2-normalized before indexing; inner product equals cosine similarity |
| Serialization | `faiss.write_index` to `data/indexes/<doc_id>/index.faiss` |
| Chunk metadata | Stored separately in `data/indexes/<doc_id>/chunks.json` |
| Loading | Lazy-load on first query for a doc_id; LRU cache with configurable capacity |

**Why `IndexFlatIP` over `IndexFlatL2`:**
Sentence-transformer embeddings are designed to be compared by cosine similarity, not
Euclidean distance. L2-normalizing the vectors before indexing and using inner product
search is numerically equivalent to cosine similarity and is the standard convention
for sentence-transformer models.

**Why Flat (exact) search:**
Insurance documents are typically 10–200 pages. Even with generous chunking,
a single document will rarely exceed 1,000 chunks. Flat exact search is fast enough
at this scale and eliminates approximation error that could affect citation precision.

**Per-document index structure:**
```
data/indexes/
└── <doc_id>/
    ├── index.faiss       # FAISS IndexFlatIP file
    └── chunks.json       # JSON array of Chunk dicts (embedding field excluded)
```

---

### 5. Keyword Signal Boost

Applied after FAISS retrieval; replaces the cross-encoder reranker in the MVP.

**Boost signals (additive; applied before threshold filter):**

| Signal | Boost |
|---|---|
| Query contains a numeric pattern (₹, %, "days", "months", "years") AND chunk contains matching numeric context | +0.10 |
| Query contains risk keywords (exclusion, waiting, co-pay, sub-limit, deductible) AND chunk.section matches | +0.10 |
| `chunk_type == TABLE` AND query is likely numeric (contains %, ₹, amounts) | +0.05 |
| No signals match | 0.00 |

**Rationale:** Keyword boosting is transparent, requires no model download, adds negligible
latency, and directly targets the most error-prone retrieval failure mode in insurance
documents: numeric/percentage values that are semantically similar across many chunks
but where the exact figure matters.

**Post-MVP enhancement:** If Phase 7 evaluation shows Recall@5 < 0.75, add cross-encoder
reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) as an optional stage between FAISS
and threshold filtering.

---

## 6. Evidence Threshold and Four-State Validator

### 6.1 Threshold Definition

| EvidenceStatus | Condition |
|---|---|
| `SUFFICIENT` | `top_1_score >= HIGH_THRESHOLD` AND `len(results) >= MIN_RESULTS` |
| `MARGINAL` | `top_1_score >= LOW_THRESHOLD` (but below HIGH_THRESHOLD) |
| `CONFLICTING` | `top_1_score >= LOW_THRESHOLD` AND contradictory signals detected in top results |
| `INSUFFICIENT` | `top_1_score < LOW_THRESHOLD` OR no results pass threshold |

**Conflict detection heuristic:** A result set is flagged as `CONFLICTING` when:
- At least 2 results pass threshold, AND
- One result contains strong coverage terms ("covered", "included", "payable") AND
  another contains strong exclusion terms ("excluded", "not covered", "not applicable",
  "except", "provided that") **for the same apparent subject**.

**Default thresholds (configurable in `config.py`):**

| Constant | Default Value |
|---|---|
| `EVIDENCE_HIGH_THRESHOLD` | 0.65 |
| `EVIDENCE_LOW_THRESHOLD` | 0.40 |
| `MIN_EVIDENCE_RESULTS` | 2 |
| `RETRIEVAL_CANDIDATE_K` | 10 |
| `RETRIEVAL_FINAL_K` | 5 |

### 6.2 Threshold Tuning

Thresholds are tuned during evaluation (Phase 8). Initial values are conservative
(prefer abstention over hallucination). The evaluation dataset includes "should-abstain"
questions to measure false abstention rate alongside false answer rate.

---

## 8. Insufficient Evidence Behavior

When `Evidence.status == INSUFFICIENT`:

1. The LLM is **not** called.
2. An `AbstentionAnswer` is returned with `AnswerType.ABSTENTION`.
3. The `text` field contains a message such as:
   > "I could not find sufficient information about this in the uploaded policy document.
   >  Please consult the policy document directly or contact your insurer for clarification."
4. The response includes the nearest retrieved chunks (even if below threshold) as
   diagnostic context — these are marked clearly as "Low-confidence matches" and are
   **not** presented as citations.
5. `citations` is empty.

---

## 9. Retrieval for Risk Scanner

The risk scanner uses targeted queries per category:

| Category | Query Templates |
|---|---|
| EXCLUSION | "what is excluded from coverage", "not covered", "exclusions" |
| WAITING_PERIOD | "waiting period", "initial waiting period", "days before coverage begins" |
| CO_PAYMENT | "co-payment", "copay", "co-pay percentage", "patient share" |
| SUB_LIMIT | "sub-limit", "maximum payable", "capped at", "limit per claim" |
| RESTRICTION | "condition", "restriction", "not applicable", "only if", "subject to" |

Each category runs up to 3 template queries and merges deduplicated results by
`chunk_id`. The top results per category are used as `RiskFinding.source_chunks`.

---

## 10. Retrieval for Comparison

The comparison engine uses attribute-specific queries:

```python
COMPARISON_SCHEMA = [
    {"attribute": "sum_insured",           "label": "Sum Insured",              "query": "sum insured coverage amount"},
    {"attribute": "waiting_period",        "label": "Waiting Period",           "query": "waiting period initial days"},
    {"attribute": "exclusions",            "label": "Key Exclusions",           "query": "what is excluded from this policy"},
    {"attribute": "co_payment",            "label": "Co-Payment",               "query": "co-payment percentage patient share"},
    {"attribute": "room_rent_limit",       "label": "Room Rent Limit",          "query": "room rent limit per day"},
    {"attribute": "pre_existing",          "label": "Pre-Existing Disease",     "query": "pre-existing disease coverage waiting period"},
    {"attribute": "network_hospitals",     "label": "Network Hospitals",        "query": "network hospital cashless"},
    {"attribute": "no_claim_bonus",        "label": "No Claim Bonus",           "query": "no claim bonus NCB discount"},
    {"attribute": "maternity_benefit",     "label": "Maternity Benefit",        "query": "maternity benefit coverage"},
    {"attribute": "premium_payment",       "label": "Premium Payment Terms",    "query": "premium payment annual monthly"},
    {"attribute": "renewal_conditions",    "label": "Renewal Conditions",       "query": "renewal terms conditions"},
    {"attribute": "portability",           "label": "Portability",              "query": "policy portability transfer"},
]
```

Each attribute query is run independently against each document's FAISS index.
Results from the two documents are never concatenated before the LLM call.

---

## 11. Known Limitations and Mitigations

| Limitation | Mitigation |
|---|---|
| Dense retrieval may miss exact keyword matches | Planned: add BM25 sparse retrieval as a complementary signal in Phase 6 |
| Cross-encoder slow on long chunks | Chunk character count capped at `MAX_CHUNK_CHARS` (default: 800) |
| Tamil query embedding quality depends on model | Evaluated in Phase 7; multilingual model selected specifically for Tamil coverage |
| Documents without section headings lose section metadata | `section` field defaults to `None`; retrieval still works; section boost simply not applied |
| Very short documents may have too few chunks | Minimum chunk count validated; warning shown to user if < 5 chunks |
