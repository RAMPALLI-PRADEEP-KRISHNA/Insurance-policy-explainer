import os
import csv
import sys
import tempfile
from pathlib import Path
import pymupdf

os.environ['LLM_PROVIDER'] = 'fake'
os.environ['HF_HUB_OFFLINE'] = '1'
# Initialize torch early to prevent multithreading issues with sentence-transformers
import torch
os.environ['OMP_NUM_THREADS'] = '1'

from src.config import config
config.LLM_PROVIDER = 'fake'

from src.ingestion.pdf_extractor import process_pdf
from src.ingestion.chunker import chunk_document
from src.ingestion.embedder import embedder
from src.ingestion.index_store import index_store
from src.retrieval.evidence_validator import EvidenceValidator

def main():
    csv_path = Path("eval/real_policy_gold.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        sys.exit(1)
        
    pdf_path = Path("Policy_Wordings_Surrosafe_Health_Insurance.pdf")
    doc_pdf = pymupdf.open(pdf_path)
    real_page_count = doc_pdf.page_count
    doc_pdf.close()
    
    print(f"Real page count: {real_page_count}")
    
    # Read rows and validate
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            ans = row['answerable'].strip().lower() == 'true'
            
            exp_pages = []
            if row['expected_pages'].strip():
                exp_pages = [int(p) for p in row['expected_pages'].split(';') if p.strip()]
                for p in exp_pages:
                    if p < 1 or p > real_page_count:
                        print(f"Validation Error: Row {i+1} has expected page {p} outside valid range (1-{real_page_count})")
                        sys.exit(1)
            
            rows.append({
                'question': row['question'],
                'language': row['language'],
                'expected_pages': exp_pages,
                'answerable': ans
            })
            
    print(f"Loaded {len(rows)} validated rows.")
    
    # Isolate FAISS index
    temp_dir = tempfile.TemporaryDirectory()
    index_store.base_dir = Path(temp_dir.name)
    
    # Process PDF
    doc = process_pdf(pdf_path, save_meta=False)
    
    validator = EvidenceValidator()
    
    def evaluate_config(max_len, name):
        embedder.load()
        embedder._model.max_seq_length = max_len
            
        chunks = chunk_document(doc)
        
        truncated_count = 0
        tokenizer = getattr(embedder._model, 'tokenizer', None)
        if tokenizer:
            for c in chunks:
                tokens = tokenizer(c.text, truncation=False)['input_ids']
                if len(tokens) > max_len:
                    truncated_count += 1
        
        index_store.build_index(doc, chunks, force_rebuild=True)
        
        en_r1, en_r3, en_r5, en_tot = 0, 0, 0, 0
        ta_r1, ta_r3, ta_r5, ta_tot = 0, 0, 0, 0
        
        print(f"\nEvaluating {name} (max_len={max_len}, chunks={len(chunks)}, truncated={truncated_count})")
        
        for idx, row in enumerate(rows):
            q_vec = embedder.embed_query(row['question'])
            results = index_store.search(doc.doc_id, q_vec, top_k=5)
            
            # The signature is validate(query, results, doc_id)
            # In index_store.search results are Tuple[Chunk, float] but validator expects List[Chunk]
            chunks_for_validator = [r[0] for r in results]
            evidence = validator.validate(row['question'], chunks_for_validator, doc_id=doc.doc_id)
            
            top_score = results[0][1] if results else 0.0
            retrieved_pages = [int(r[0].page_number) for r in results]
            
            if not row['answerable']:
                print(f"  [Q{idx+1} UNANSWERABLE] '{row['question']}'")
                print(f"    -> Top Score: {top_score:.4f}, Evidence State: {evidence.status.value}")
                continue
                
            # It's answerable
            exp = set(row['expected_pages'])
            r1 = set(retrieved_pages[:1])
            r3 = set(retrieved_pages[:3])
            r5 = set(retrieved_pages[:5])
            
            hit1 = len(exp.intersection(r1)) > 0
            hit3 = len(exp.intersection(r3)) > 0
            hit5 = len(exp.intersection(r5)) > 0
            
            if row['language'] == 'en':
                en_tot += 1
                if hit1: en_r1 += 1
                if hit3: en_r3 += 1
                if hit5: en_r5 += 1
            elif row['language'] == 'ta':
                ta_tot += 1
                if hit1: ta_r1 += 1
                if hit3: ta_r3 += 1
                if hit5: ta_r5 += 1
                
            print(f"  [Q{idx+1} {row['language']}] Expected: {list(exp)} | Retrieved Top-5: {retrieved_pages}")
            
        en_metrics = f"{en_r1/max(1,en_tot):.2f}/{en_r3/max(1,en_tot):.2f}/{en_r5/max(1,en_tot):.2f}"
        ta_metrics = f"{ta_r1/max(1,ta_tot):.2f}/{ta_r3/max(1,ta_tot):.2f}/{ta_r5/max(1,ta_tot):.2f}"
        
        return (name, max_len, len(chunks), truncated_count, en_metrics, ta_metrics)
        
    res_a = evaluate_config(128, "MPNet")
    res_b = evaluate_config(256, "MPNet")
    
    print("\n--- FINAL TABLE ---")
    print("| Config | Chunks | Truncated | EN Recall@1/3/5 | TA Recall@1/3/5 |")
    print("|---|---|---|---|---|")
    for r in [res_a, res_b]:
        print(f"| {r[0]} (max={r[1]}) | {r[2]} | {r[3]} | {r[4]} | {r[5]} |")
        
if __name__ == '__main__':
    main()
