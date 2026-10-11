# Insurance Policy Explainer — Evaluation Design

> **Version:** 2.0 (Post-Adversarial Review)
> **Date:** 2026-10-05

---

## 1. Evaluation Philosophy

A single "accuracy" number is insufficient and misleading for a grounded RAG system.
This evaluation plan defines separate dimensions, each with its own metric and dataset
subset. Claims about system performance must be tied to specific dimension scores.

**Core principle:** The evaluation must be able to falsify the system's quality claims.
If the system cannot be shown to *fail* under certain conditions, the evaluation is
not rigorous.

---

## 2. Evaluation Dimensions

| Dimension | What it measures | Metric(s) |
|---|---|---|
| **D1: Answer Correctness** | Is the generated answer factually correct given the policy? | Manual binary + partial-credit score |
| **D2: Citation Correctness** | Do the cited page numbers match the actual source pages? | Page-level precision, recall, F1 |
| **D3: Evidence/Retrieval Quality** | Does the retrieval pipeline surface the relevant chunks? | Chunk-level recall@5, MRR |
| **D4: Abstention Correctness** | Does the system correctly abstain on unanswerable questions? | Precision/Recall for abstention class |
| **D5: Comparison Correctness** | Does the comparison engine correctly identify attribute differences? | Per-attribute accuracy |
| **D6: Grounding (Hallucination Check)** | Does the answer contain claims not present in the evidence? | Manual annotation: "no extra claims" |
| **D7: Language Quality (Tamil)** | Is the Tamil translation fluent and accurate? | Manual fluency + accuracy rating |
| **D8: Numeric Accuracy** | Are numeric values (%, ₹, days) extracted and reported correctly? | Exact match for %, ±5% for ₹ amounts |
| **D9: Tamil A/B Comparison** | Does direct Tamil generation outperform translate-from-English? | D8 scores compared across both approaches |

---

## 3. Evaluation Dataset Structure

### 3.1 Dataset File: `evaluation/test_data/eval_dataset.json`

```json
{
  "version": "1.0",
  "created_at": "2026-10-05",
  "questions": [
    {
      "question_id": "Q001",
      "category": "coverage",
      "subcategory": "sum_insured",
      "question_text": "What is the maximum sum insured under this policy?",
      "doc_filename": "sample_health_policy_A.pdf",
      "expected_answer": "The maximum sum insured is ₹10 lakhs.",
      "expected_page_numbers": [3, 4],
      "expected_has_answer": true,
      "expected_abstain": false,
      "relevant_chunk_keywords": ["sum insured", "10 lakhs", "maximum coverage"],
      "evaluation_notes": "Page 3 has the table; page 4 has explanatory text.",
      "requires_comparison": false,
      "language": "en"
    },
    {
      "question_id": "Q002",
      "category": "abstention",
      "subcategory": "out_of_scope",
      "question_text": "What is the stock price of the insurance company?",
      "doc_filename": "sample_health_policy_A.pdf",
      "expected_answer": null,
      "expected_page_numbers": [],
      "expected_has_answer": false,
      "expected_abstain": true,
      "relevant_chunk_keywords": [],
      "evaluation_notes": "Completely out of scope; system must abstain.",
      "requires_comparison": false,
      "language": "en"
    }
  ]
}
```

### 3.2 Dataset Fields

| Field | Type | Description |
|---|---|---|
| `question_id` | str | Unique identifier, e.g., "Q001" |
| `category` | str | One of the question categories (Section 4) |
| `subcategory` | str | Fine-grained topic |
| `question_text` | str | The actual query as the user would type it |
| `doc_filename` | str | Filename of the PDF to use (must be in test set) |
| `expected_answer` | str \| null | Reference answer (null for abstention cases) |
| `expected_page_numbers` | List[int] | Pages where the answer can be found (1-indexed) |
| `expected_has_answer` | bool | True if the document contains the answer |
| `expected_abstain` | bool | True if the system should abstain |
| `relevant_chunk_keywords` | List[str] | Keywords expected in retrieved chunks |
| `evaluation_notes` | str | Annotator notes explaining the expected answer |
| `requires_comparison` | bool | True for comparison dimension questions |
| `language` | str | "en" or "ta" |

---

## 4. Question Categories

> **Important:** The question count targets below are minimums. The actual dataset
> is built from real policy documents after they are available. Do not fabricate
> questions to reach a count.

| Category | Min Count | Description |
|---|---|---|
| `coverage` | 5 | Questions about what is covered |
| `exclusions` | 5 | Questions about what is excluded |
| `waiting_period` | 4 | Questions about waiting periods |
| `co_payment` | 4 | Questions about co-payment and cost sharing |
| `sub_limits` | 4 | Questions about per-item limits |
| `numeric` | 8 | **NEW** Questions requiring an exact numeric answer (%, ₹, days) |
| `table_based` | 4 | **NEW** Questions whose answer lives in a table |
| `multi_hop` | 4 | **NEW** Questions requiring information from 2+ clauses |
| `claims_procedure` | 3 | **NEW** Questions about the claims process |
| `ambiguous` | 3 | **NEW** Questions where policy wording is genuinely ambiguous |
| `premiums` | 3 | Questions about premium structure |
| `renewal` | 3 | Questions about renewal terms |
| `comparison` | 5 | Cross-policy attribute comparisons |
| `abstention_oos` | 4 | Out-of-scope questions (must abstain) |
| `abstention_missing` | 4 | Questions about topics genuinely absent from the doc |
| `tamil` | 5 | Tamil-language queries (gated on Tamil reviewer availability) |
| **Minimum Total** | **68** | Minimum viable evaluation dataset |

The evaluation is not run until at least **20 questions** spanning at least 5 categories
are manually verified. This is the minimum to validate the pipeline before full evaluation.

---

## 5. Expected Answer Format

For annotators, the expected answer format is:

```
{
  "short_answer": "One-sentence factual answer.",
  "key_facts": ["fact 1", "fact 2"],     // Individual verifiable claims
  "source_pages": [3, 4],                // Pages where these facts appear
  "verbatim_quote": "Exact quote from policy if available"
}
```

Evaluation scoring uses `key_facts` to compute partial credit — the answer need not be
verbatim but must include each key fact.

---

## 6. Evaluation Metrics — Detailed Definitions

### 6.1 D1: Answer Correctness

**Measurement:** Manual binary score per question; each question is scored 0, 0.5, or 1.

| Score | Meaning |
|---|---|
| 1.0 | All key_facts present and correct |
| 0.5 | Some key_facts correct; partial answer |
| 0.0 | Answer incorrect, hallucinated, or missing key facts |

**Aggregate metric:** Mean answer score across all non-abstention, non-error answers.

**Automated approximation (D1-Auto):** For graded evaluation runs, use ROUGE-L overlap
between `answer.text` and `expected_answer` as a proxy. Treat D1-Auto as indicative only;
manual scoring is authoritative.

### 6.2 D2: Citation Correctness

**Measurement:** For each answer, compare `answer.citations[*].page_number` to
`expected_page_numbers`.

```
Citation Precision = |cited_pages ∩ expected_pages| / |cited_pages|
Citation Recall    = |cited_pages ∩ expected_pages| / |expected_pages|
Citation F1        = 2 * P * R / (P + R)
```

**Critical:** A citation pointing to a page that does not contain the answer is
a **citation error**, regardless of answer correctness.

**Aggregate metric:** Mean Citation F1 across all GROUNDED answers.

### 6.3 D3: Evidence/Retrieval Quality

**Measurement:** Check whether the relevant chunks (identified by `relevant_chunk_keywords`)
appear in the `answer.source_chunks` list.

```
Chunk Recall@5  = # questions where relevant chunk in top-5 / # total questions
MRR             = mean(1 / rank_of_first_relevant_chunk)
```

Computed automatically since `source_chunks` are stored in `Answer`.

### 6.4 D4: Abstention Correctness

**Measurement:** For each question with `expected_abstain == True`, check whether
`answer.answer_type == ABSTENTION`.

```
Abstention Precision = TP_abstain / (TP_abstain + FP_abstain)
Abstention Recall    = TP_abstain / (TP_abstain + FN_abstain)
```

Where:
- TP_abstain: question should abstain AND system abstained
- FP_abstain: question should NOT abstain but system abstained (false refusal)
- FN_abstain: question should abstain but system gave an answer (potentially hallucinated)

**Critical failure mode:** FN_abstain (failing to abstain when evidence is absent)
is considered a more serious failure than FP_abstain.

### 6.5 D5: Comparison Correctness

**Measurement:** For each comparison question, manually annotate whether the
`ComparisonResult` correctly identifies the attribute value for each policy.

```
Per-attribute accuracy = # correct cells / # total cells
```

A cell is correct if:
- `ComparisonCell.status == NOT_FOUND` when the attribute is genuinely absent, OR
- `ComparisonCell.value` matches the expected value (fuzzy match for numeric values ±10%)

### 6.6 D6: Grounding Check

**Measurement:** Manual annotation of a random 20% sample of GROUNDED answers.

Annotator checks: does the answer contain any claim **not** directly supported by
the `source_chunks` text?

```
Hallucination Rate = # answers with at least one unsupported claim / # reviewed answers
```

**Target:** Hallucination Rate < 10% in the reviewed sample.

### 6.7 D7: Tamil Language Quality

**Measurement:** Manual rating by a Tamil-proficient annotator.

| Score | Meaning |
|---|---|
| 2 | Fluent and factually accurate translation |
| 1 | Minor fluency issues but factually correct |
| 0 | Mistranslation or factual change introduced by translation |

**Aggregate metric:** Mean Tamil Quality Score across all Tamil answers.

---

## 7. Evaluation Execution

### 7.1 CLI Command

```bash
python -m evaluation.evaluator \
    --dataset evaluation/test_data/eval_dataset.json \
    --output evaluation/results/run_<timestamp>.json \
    --dimensions D2,D3,D4  # automated dimensions
```

### 7.2 Automated vs. Manual Dimensions

| Dimension | Automated? |
|---|---|
| D1 Answer Correctness | Partially (ROUGE proxy); requires manual for authoritative score |
| D2 Citation Correctness | **Fully automated** |
| D3 Retrieval Quality | **Fully automated** |
| D4 Abstention Correctness | **Fully automated** |
| D5 Comparison | Partially automated (NOT_FOUND detection); requires manual for value accuracy |
| D6 Grounding | **Manual only** |
| D7 Tamil Quality | **Manual only** |

### 7.3 Evaluation Report

The evaluation module produces:

1. `run_<timestamp>.json` — per-question `EvaluationResult` objects.
2. `summary_<timestamp>.csv` — aggregate metrics per dimension.
3. `failure_analysis_<timestamp>.md` — auto-generated list of worst-performing questions.

---

## 8. Failure Analysis

For each evaluation run, the evaluator identifies:

| Failure Type | Detection |
|---|---|
| Citation mismatch | F1 < 0.5 on D2 |
| Hallucinated answer | Manual D6 annotation |
| False abstention | FP in D4 |
| Failed abstention | FN in D4 (highest priority failure) |
| Retrieval miss | Chunk Recall@5 == 0 |
| Comparison cell error | Per-attribute D5 score == 0 |
| Tamil accuracy drop | D7 score == 0 |

Failures are logged per `question_id` with the retrieved chunks, generated answer,
and a diagnosis note to guide future threshold tuning or prompt revision.

---

## 9. Pre-Evaluation Checklist

Before running evaluation:

- [ ] At least 2 real insurance policy PDFs are available in test set.
- [ ] All questions in `eval_dataset.json` have been verified by a human reviewer.
- [ ] `expected_page_numbers` have been confirmed by manually checking the PDF.
- [ ] Abstention questions have been confirmed: the attribute is genuinely absent from
      the document (not just hard to find).
- [ ] Tamil questions have been written by or reviewed by a Tamil speaker.
- [ ] The system is running in its deployed configuration (not test mocks).

---

## 10. Metrics Summary Table

| Metric | Dimension | Type | Target (initial) |
|---|---|---|---|
| Mean Answer Score | D1 | Manual | ≥ 0.7 |
| ROUGE-L Proxy | D1-Auto | Automated | ≥ 0.4 |
| Citation F1 | D2 | Automated | ≥ 0.75 |
| Chunk Recall@5 | D3 | Automated | ≥ 0.80 |
| MRR | D3 | Automated | ≥ 0.60 |
| Abstention Precision | D4 | Automated | ≥ 0.90 |
| Abstention Recall | D4 | Automated | ≥ 0.85 |
| Comparison Accuracy | D5 | Partial | ≥ 0.70 |
| Hallucination Rate | D6 | Manual | ≤ 0.10 |
| Tamil Quality Score | D7 | Manual | ≥ 1.5/2.0 |

> **Important:** These targets are *initial aspirational values* set before any
> evaluation run. They must be reviewed and updated after the first evaluation run.
> Do not report these as "achieved" metrics before running evaluation.
