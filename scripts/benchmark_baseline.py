"""Baseline performance benchmark script for Insurance Policy Explainer.

Measures all Phase 1 components with time.perf_counter().
Runs on real policy PDF: Policy_Wordings_Surrosafe_Health_Insurance.pdf
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_baseline_benchmark():
    results = {}
    print("=" * 60)
    print("RUNNING BASELINE PERFORMANCE BENCHMARK (PHASE 1)")
    print("=" * 60)

    # A. Application startup / critical imports
    t0 = time.perf_counter()
    import backend.config
    import backend.models.data_models
    import backend.utils.timing
    t_app_startup = time.perf_counter() - t0
    results["A_app_startup_seconds"] = t_app_startup
    print(f"A. Application base module import: {t_app_startup * 1000:.2f} ms")

    # B. Sentence-Transformer import time
    t0 = time.perf_counter()
    import sentence_transformers
    t_st_import = time.perf_counter() - t0
    results["B_sentence_transformers_import_seconds"] = t_st_import
    print(f"B. Sentence-Transformers import: {t_st_import * 1000:.2f} ms")

    # C. Cold MPNet initialization time
    from backend.ingestion.embedder import Embedder
    from backend.config import config
    cold_embedder = Embedder(model_name=config.EMBEDDING_MODEL_NAME, device="cpu")
    t0 = time.perf_counter()
    cold_embedder.load()
    t_cold_embed_init = time.perf_counter() - t0
    results["C_cold_mpnet_init_seconds"] = t_cold_embed_init
    print(f"C. Cold MPNet model initialization: {t_cold_embed_init:.4f} s")

    # D. Warm MPNet inference (single query, multiple runs for median & p95)
    test_query = "What is the waiting period for pre-existing diseases?"
    warm_query_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = cold_embedder.embed_query(test_query)
        warm_query_times.append(time.perf_counter() - t0)
    warm_query_times.sort()
    med_warm_query = warm_query_times[len(warm_query_times) // 2]
    p95_warm_query = warm_query_times[int(len(warm_query_times) * 0.95)]
    results["D_warm_mpnet_query_median_seconds"] = med_warm_query
    results["D_warm_mpnet_query_p95_seconds"] = p95_warm_query
    print(f"D. Warm MPNet query inference (10 runs): median={med_warm_query * 1000:.2f} ms, p95={p95_warm_query * 1000:.2f} ms")

    # PDF file to test
    pdf_path = PROJECT_ROOT / "Policy_Wordings_Surrosafe_Health_Insurance.pdf"
    if not pdf_path.exists():
        # Fallback to test fixture if root PDF not found
        pdf_path = PROJECT_ROOT / "tests" / "fixtures" / "sample_policy_fixture.pdf"
    print(f"\nUsing test PDF: {pdf_path.name}")

    # E. PDF read bytes time
    t0 = time.perf_counter()
    pdf_bytes = pdf_path.read_bytes()
    t_pdf_read = time.perf_counter() - t0
    results["E_pdf_read_seconds"] = t_pdf_read
    print(f"E. PDF read bytes ({len(pdf_bytes)} bytes): {t_pdf_read * 1000:.2f} ms")

    # F. PDF extraction time
    from backend.ingestion.pdf_extractor import process_pdf
    t0 = time.perf_counter()
    doc = process_pdf(pdf_bytes, filename=pdf_path.name, save_meta=False)
    t_pdf_extract = time.perf_counter() - t0
    results["F_pdf_extract_seconds"] = t_pdf_extract
    results["F_pdf_page_count"] = doc.page_count
    print(f"F. PDF extraction ({doc.page_count} pages): {t_pdf_extract:.4f} s ({t_pdf_extract / max(doc.page_count, 1) * 1000:.2f} ms/page)")

    # G. OCR time
    ocr_pages = sum(1 for p in doc.pages if p.extraction_method.value == "OCR")
    results["G_ocr_pages_count"] = ocr_pages
    results["G_ocr_time_seconds"] = 0.0 if ocr_pages == 0 else "APPLIED"
    print(f"G. OCR pages: {ocr_pages} (native text available on all pages)")

    # H. Chunking time
    from backend.ingestion.chunker import chunk_document
    chunk_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        chunks = chunk_document(doc)
        chunk_times.append(time.perf_counter() - t0)
    chunk_times.sort()
    med_chunk_time = chunk_times[len(chunk_times) // 2]
    results["H_chunking_median_seconds"] = med_chunk_time
    results["H_chunk_count"] = len(chunks)
    print(f"H. Chunking ({len(chunks)} chunks): median={med_chunk_time * 1000:.2f} ms")

    # I. Document embedding time
    t0 = time.perf_counter()
    vectors = cold_embedder.embed_chunks(chunks)
    t_doc_embed = time.perf_counter() - t0
    results["I_doc_embedding_seconds"] = t_doc_embed
    results["I_chunks_per_second"] = len(chunks) / max(t_doc_embed, 1e-6)
    print(f"I. Document embedding ({len(chunks)} chunks): {t_doc_embed:.4f} s ({len(chunks) / max(t_doc_embed, 1e-6):.1f} chunks/sec)")

    # J. FAISS/NumPy index construction time & K. Index persistence time
    from database.index_store import IndexStore
    test_store_dir = PROJECT_ROOT / "data" / "benchmark_test_index"
    store = IndexStore(indexes_dir=test_store_dir, embedder_instance=cold_embedder)

    t0 = time.perf_counter()
    meta = store.build_index(doc, chunks)
    t_index_build_persist = time.perf_counter() - t0
    results["J_K_index_build_and_persist_seconds"] = t_index_build_persist
    print(f"J & K. Index build + persistence: {t_index_build_persist * 1000:.2f} ms")

    # L. Same-document reuse time (cache hit / load existing index)
    reuse_times = []
    for _ in range(10):
        # Clear in-memory cache to measure disk load
        store._cache.clear()
        t0 = time.perf_counter()
        _loaded_idx, _loaded_chunks, _loaded_meta = store.load_index(doc.doc_id)
        reuse_times.append(time.perf_counter() - t0)
    reuse_times.sort()
    med_reuse = reuse_times[len(reuse_times) // 2]
    p95_reuse = reuse_times[int(len(reuse_times) * 0.95)]
    results["L_same_doc_reuse_median_seconds"] = med_reuse
    results["L_same_doc_reuse_p95_seconds"] = p95_reuse
    print(f"L. Same-document reuse from disk (10 runs): median={med_reuse * 1000:.2f} ms, p95={p95_reuse * 1000:.2f} ms")

    # M. Query embedding time
    # (Measured in D: ~med_warm_query)
    results["M_query_embedding_seconds"] = med_warm_query

    # N. Vector retrieval time (raw index search)
    q_vec = cold_embedder.embed_query(test_query)
    retrieval_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _candidates = store.search(doc.doc_id, q_vec, top_k=10)
        retrieval_times.append(time.perf_counter() - t0)
    retrieval_times.sort()
    med_vec_search = retrieval_times[len(retrieval_times) // 2]
    results["N_vector_retrieval_median_seconds"] = med_vec_search
    print(f"N. Vector similarity search (20 runs): median={med_vec_search * 1000:.3f} ms")

    # O. Hybrid scoring time
    from backend.retrieval.retriever import Retriever
    retriever_instance = Retriever(index_store=store, embedder_instance=cold_embedder)
    
    t0 = time.perf_counter()
    retrieval_results = retriever_instance.retrieve(doc_id=doc.doc_id, query=test_query, top_k=5)
    t_full_retrieval = time.perf_counter() - t0
    t_hybrid_scoring = max(0.0, t_full_retrieval - med_warm_query - med_vec_search)
    results["O_hybrid_scoring_seconds"] = t_hybrid_scoring
    print(f"O. Hybrid scoring & reranking: {t_hybrid_scoring * 1000:.3f} ms")

    # P. Evidence validation time
    from backend.retrieval.evidence_validator import EvidenceValidator
    validator = EvidenceValidator()
    val_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        evidence = validator.validate(query=test_query, results=retrieval_results, doc_id=doc.doc_id)
        val_times.append(time.perf_counter() - t0)
    val_times.sort()
    med_val_time = val_times[len(val_times) // 2]
    results["P_evidence_validation_median_seconds"] = med_val_time
    print(f"P. Evidence validation (20 runs): median={med_val_time * 1000:.3f} ms (Status: {evidence.status.value})")

    # Q. Citation assembly time
    from backend.generation.citation_assembler import assemble_citations
    cit_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        citations = assemble_citations(evidence)
        cit_times.append(time.perf_counter() - t0)
    cit_times.sort()
    med_cit_time = cit_times[len(cit_times) // 2]
    results["Q_citation_assembly_median_seconds"] = med_cit_time
    print(f"Q. Citation assembly (20 runs): median={med_cit_time * 1000:.3f} ms ({len(citations)} citations)")

    # R. LLM request time
    results["R_llm_request_seconds"] = "NOT MEASURED (MOCKED TO PRESERVE API QUOTAS)"
    print("R. LLM request time: NOT MEASURED (mocked to prevent external API consumption)")

    # S. Total local query latency (Query Embed + Vector Search + Hybrid Boost + Evidence Validation + Citation Assembly)
    t_local_query = med_warm_query + med_vec_search + t_hybrid_scoring + med_val_time + med_cit_time
    results["S_total_local_query_latency_seconds"] = t_local_query
    print(f"S. Total local query latency (no LLM): {t_local_query * 1000:.2f} ms")

    # T. Total question-to-answer latency
    results["T_total_qa_latency_seconds"] = f"Local: {t_local_query * 1000:.2f} ms + LLM Network Latency"
    print(f"T. Total Q&A latency: {t_local_query * 1000:.2f} ms + LLM Network Latency")

    # Clean up benchmark store
    import shutil
    if test_store_dir.exists():
        shutil.rmtree(test_store_dir, ignore_errors=True)

    # Save benchmark report to JSON
    report_path = PROJECT_ROOT / "benchmark_baseline_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nBaseline report saved to: {report_path.name}")
    print("=" * 60)
    return results


if __name__ == "__main__":
    run_baseline_benchmark()
