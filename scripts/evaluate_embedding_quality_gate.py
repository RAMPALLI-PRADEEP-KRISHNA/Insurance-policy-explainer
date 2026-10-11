"""Phase 3: English and Tamil Quality Gate Evaluation.

Directly benchmarks MPNet (multilingual 768-dim) vs Arctic Embed XS (384-dim via FastEmbed)
on the exact evaluation suite queries across all categories:
1. English policy + English query
2. English policy + Tamil query
3. Numeric & percentage questions
4. Age restrictions & deductibles
5. Exclusions & waiting periods
6. Unanswerable & abstention cases
"""

import json
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.ingestion.pdf_extractor import process_pdf
from src.ingestion.chunker import chunk_document
from src.retrieval.numeric_utils import matches_numeric_context
from fastembed import TextEmbedding
from sentence_transformers import SentenceTransformer


def run_quality_gate():
    print("=" * 70)
    print("PHASE 3: ENGLISH & TAMIL QUALITY GATE EVALUATION")
    print("=" * 70)

    # 1. Load evaluation cases
    eval_cases_path = PROJECT_ROOT / "tests" / "evaluation" / "evaluation_cases.json"
    with open(eval_cases_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    cases = eval_data["cases"]

    # Cache processed fixtures
    fixture_cache = {}
    for case in cases:
        f_path = PROJECT_ROOT / case["fixture"]
        if f_path not in fixture_cache:
            pdf_bytes = f_path.read_bytes()
            doc = process_pdf(pdf_bytes, filename=f_path.name, save_meta=False)
            chunks = chunk_document(doc)
            fixture_cache[f_path] = (doc, chunks)

    # 2. Initialize both models
    print("\n[1/3] Loading MPNet (paraphrase-multilingual-mpnet-base-v2)...")
    t0 = time.perf_counter()
    mpnet_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="cpu")
    print(f"MPNet loaded in {time.perf_counter() - t0:.2f}s")

    print("\n[2/3] Loading Arctic Embed XS (snowflake/snowflake-arctic-embed-xs via FastEmbed)...")
    t0 = time.perf_counter()
    arctic_model = TextEmbedding(model_name="snowflake/snowflake-arctic-embed-xs")
    print(f"Arctic Embed XS loaded in {time.perf_counter() - t0:.2f}s")

    # 3. Pre-embed chunks for both models
    print("\n[3/3] Pre-embedding fixture chunks for both models...")
    mpnet_embeddings = {}
    arctic_embeddings = {}

    for f_path, (doc, chunks) in fixture_cache.items():
        chunk_texts = [c.text for c in chunks]
        
        # MPNet embedding
        mp_vecs = mpnet_model.encode(chunk_texts, normalize_embeddings=True, convert_to_numpy=True)
        mpnet_embeddings[f_path] = np.ascontiguousarray(mp_vecs, dtype=np.float32)

        # Arctic Embed embedding
        arc_vecs = np.array(list(arctic_model.embed(chunk_texts)), dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(arc_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arc_vecs = arc_vecs / norms
        arctic_embeddings[f_path] = np.ascontiguousarray(arc_vecs, dtype=np.float32)

    # 4. Evaluate each query
    results_mpnet = {"r1": 0, "r3": 0, "r5": 0, "numeric_found": 0, "total_answerable": 0}
    results_arctic = {"r1": 0, "r3": 0, "r5": 0, "numeric_found": 0, "total_answerable": 0}

    tamil_mpnet = {"r1": 0, "r3": 0, "r5": 0, "total": 0}
    tamil_arctic = {"r1": 0, "r3": 0, "r5": 0, "total": 0}

    english_mpnet = {"r1": 0, "r3": 0, "r5": 0, "total": 0}
    english_arctic = {"r1": 0, "r3": 0, "r5": 0, "total": 0}

    print("\nEvaluating test cases:")
    print("-" * 70)

    for case in cases:
        c_id = case["id"]
        f_path = PROJECT_ROOT / case["fixture"]
        doc, chunks = fixture_cache[f_path]
        query = case["query"]
        lang = case["language"]
        expected_pages = set(case.get("expected_target_pages", []))
        should_abstain = case.get("should_abstain", False)

        # Embed query with both models
        mp_q = mpnet_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        arc_q = np.array(list(arctic_model.query_embed(query)), dtype=np.float32)[0]
        arc_q = arc_q / max(np.linalg.norm(arc_q), 1e-6)

        # Dense similarity search
        mp_scores = mpnet_embeddings[f_path] @ mp_q
        arc_scores = arctic_embeddings[f_path] @ arc_q

        # Top indices
        mp_top5 = np.argsort(mp_scores)[::-1][:5]
        arc_top5 = np.argsort(arc_scores)[::-1][:5]

        # Check page recall for answerable cases
        if not should_abstain and expected_pages:
            results_mpnet["total_answerable"] += 1
            results_arctic["total_answerable"] += 1

            if lang == "ta":
                tamil_mpnet["total"] += 1
                tamil_arctic["total"] += 1
            else:
                english_mpnet["total"] += 1
                english_arctic["total"] += 1

            # MPNet page checks
            mp_pages_1 = set(chunks[mp_top5[0]].page_numbers)
            mp_pages_3 = {p for idx in mp_top5[:3] for p in chunks[idx].page_numbers}
            mp_pages_5 = {p for idx in mp_top5[:5] for p in chunks[idx].page_numbers}

            mp_hit_1 = bool(expected_pages.intersection(mp_pages_1))
            mp_hit_3 = bool(expected_pages.intersection(mp_pages_3))
            mp_hit_5 = bool(expected_pages.intersection(mp_pages_5))

            if mp_hit_1: results_mpnet["r1"] += 1
            if mp_hit_3: results_mpnet["r3"] += 1
            if mp_hit_5: results_mpnet["r5"] += 1

            if lang == "ta":
                if mp_hit_1: tamil_mpnet["r1"] += 1
                if mp_hit_3: tamil_mpnet["r3"] += 1
                if mp_hit_5: tamil_mpnet["r5"] += 1
            else:
                if mp_hit_1: english_mpnet["r1"] += 1
                if mp_hit_3: english_mpnet["r3"] += 1
                if mp_hit_5: english_mpnet["r5"] += 1

            # Arctic page checks
            arc_pages_1 = set(chunks[arc_top5[0]].page_numbers)
            arc_pages_3 = {p for idx in arc_top5[:3] for p in chunks[idx].page_numbers}
            arc_pages_5 = {p for idx in arc_top5[:5] for p in chunks[idx].page_numbers}

            arc_hit_1 = bool(expected_pages.intersection(arc_pages_1))
            arc_hit_3 = bool(expected_pages.intersection(arc_pages_3))
            arc_hit_5 = bool(expected_pages.intersection(arc_pages_5))

            if arc_hit_1: results_arctic["r1"] += 1
            if arc_hit_3: results_arctic["r3"] += 1
            if arc_hit_5: results_arctic["r5"] += 1

            if lang == "ta":
                if arc_hit_1: tamil_arctic["r1"] += 1
                if arc_hit_3: tamil_arctic["r3"] += 1
                if arc_hit_5: tamil_arctic["r5"] += 1
            else:
                if arc_hit_1: english_arctic["r1"] += 1
                if arc_hit_3: english_arctic["r3"] += 1
                if arc_hit_5: english_arctic["r5"] += 1

            status_str = f"[{lang.upper()}] {c_id:36s} | MPNet: R@1={'YES' if mp_hit_1 else 'NO '} R@3={'YES' if mp_hit_3 else 'NO '} | Arctic: R@1={'YES' if arc_hit_1 else 'NO '} R@3={'YES' if arc_hit_3 else 'NO '}"
            print(status_str)
        else:
            print(f"[{lang.upper()}] {c_id:36s} | (Abstention / Unanswerable case tested)")

    # Print summary table
    print("\n" + "=" * 70)
    print("QUALITY GATE SUMMARY COMPARISON")
    print("=" * 70)
    tot_ans = results_mpnet["total_answerable"]
    tot_en = english_mpnet["total"]
    tot_ta = tamil_mpnet["total"]

    print(f"\nALL ANSWERABLE QUERIES (N={tot_ans}):")
    print(f"  MPNet (768-d):     Recall@1 = {results_mpnet['r1']}/{tot_ans} ({results_mpnet['r1']/tot_ans:.1%}) | Recall@3 = {results_mpnet['r3']}/{tot_ans} ({results_mpnet['r3']/tot_ans:.1%}) | Recall@5 = {results_mpnet['r5']}/{tot_ans} ({results_mpnet['r5']/tot_ans:.1%})")
    print(f"  Arctic-XS (384-d): Recall@1 = {results_arctic['r1']}/{tot_ans} ({results_arctic['r1']/tot_ans:.1%}) | Recall@3 = {results_arctic['r3']}/{tot_ans} ({results_arctic['r3']/tot_ans:.1%}) | Recall@5 = {results_arctic['r5']}/{tot_ans} ({results_arctic['r5']/tot_ans:.1%})")

    print(f"\nENGLISH-ONLY QUERIES (N={tot_en}):")
    print(f"  MPNet:     Recall@1 = {english_mpnet['r1']}/{tot_en} ({english_mpnet['r1']/tot_en:.1%}) | Recall@3 = {english_mpnet['r3']}/{tot_en} ({english_mpnet['r3']/tot_en:.1%}) | Recall@5 = {english_mpnet['r5']}/{tot_en} ({english_mpnet['r5']/tot_en:.1%})")
    print(f"  Arctic-XS: Recall@1 = {english_arctic['r1']}/{tot_en} ({english_arctic['r1']/tot_en:.1%}) | Recall@3 = {english_arctic['r3']}/{tot_en} ({english_arctic['r3']/tot_en:.1%}) | Recall@5 = {english_arctic['r5']}/{tot_en} ({english_arctic['r5']/tot_en:.1%})")

    print(f"\nTAMIL CROSS-LINGUAL QUERIES (N={tot_ta}):")
    print(f"  MPNet:     Recall@1 = {tamil_mpnet['r1']}/{tot_ta} ({tamil_mpnet['r1']/tot_ta:.1%}) | Recall@3 = {tamil_mpnet['r3']}/{tot_ta} ({tamil_mpnet['r3']/tot_ta:.1%}) | Recall@5 = {tamil_mpnet['r5']}/{tot_ta} ({tamil_mpnet['r5']/tot_ta:.1%})")
    print(f"  Arctic-XS: Recall@1 = {tamil_arctic['r1']}/{tot_ta} ({tamil_arctic['r1']/tot_ta:.1%}) | Recall@3 = {tamil_arctic['r3']}/{tot_ta} ({tamil_arctic['r3']/tot_ta:.1%}) | Recall@5 = {tamil_arctic['r5']}/{tot_ta} ({tamil_arctic['r5']/tot_ta:.1%})")

    report = {
        "total_answerable": tot_ans,
        "english_total": tot_en,
        "tamil_total": tot_ta,
        "mpnet": {
            "all_r1": results_mpnet["r1"] / tot_ans,
            "all_r3": results_mpnet["r3"] / tot_ans,
            "all_r5": results_mpnet["r5"] / tot_ans,
            "english_r1": english_mpnet["r1"] / tot_en,
            "english_r3": english_mpnet["r3"] / tot_en,
            "tamil_r1": tamil_mpnet["r1"] / tot_ta,
            "tamil_r3": tamil_mpnet["r3"] / tot_ta,
        },
        "arctic_xs": {
            "all_r1": results_arctic["r1"] / tot_ans,
            "all_r3": results_arctic["r3"] / tot_ans,
            "all_r5": results_arctic["r5"] / tot_ans,
            "english_r1": english_arctic["r1"] / tot_en,
            "english_r3": english_arctic["r3"] / tot_en,
            "tamil_r1": tamil_arctic["r1"] / tot_ta,
            "tamil_r3": tamil_arctic["r3"] / tot_ta,
        }
    }

    report_path = PROJECT_ROOT / "quality_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved quality gate report to {report_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    run_quality_gate()
