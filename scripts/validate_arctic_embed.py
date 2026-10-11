"""Validate FastEmbed with snowflake/snowflake-arctic-embed-xs on Windows / Python 3.12."""

import time
import numpy as np
from fastembed import TextEmbedding

def validate():
    print("=" * 60)
    print("VALIDATING SNOWFLAKE/SNOWFLAKE-ARCTIC-EMBED-XS ON FASTEMBED")
    print("=" * 60)

    model_name = "snowflake/snowflake-arctic-embed-xs"
    
    # 1. Measure cold initialization time
    t0 = time.perf_counter()
    model = TextEmbedding(model_name=model_name)
    cold_init_time = time.perf_counter() - t0
    print(f"1. Cold initialization time: {cold_init_time:.4f} s")

    # 2. Document embedding (multiple sentences, batching)
    sample_texts = [
        "Pre-existing diseases have a waiting period of 36 months from policy inception.",
        "Room rent is capped at 1% of the sum insured per day.",
        "Maternity expenses are covered after a waiting period of 24 months.",
        "Cataract surgery has a specific limit of Rs. 40,000 per eye.",
        "ஏற்கனவே உள்ள நோய்களுக்கான காத்திருப்பு காலம் 36 மாதங்கள் ஆகும்." # Tamil text
    ]

    t0 = time.perf_counter()
    doc_embeddings = list(model.embed(sample_texts))
    warm_doc_time = time.perf_counter() - t0
    doc_arr = np.array(doc_embeddings, dtype=np.float32)

    print(f"2. Embedded {len(sample_texts)} texts in {warm_doc_time * 1000:.2f} ms ({len(sample_texts) / warm_doc_time:.1f} texts/sec)")
    print(f"   Shape: {doc_arr.shape}")
    print(f"   Dimension: {doc_arr.shape[1]}")
    assert doc_arr.shape[1] == 384, f"Expected 384, got {doc_arr.shape[1]}"
    assert np.all(np.isfinite(doc_arr)), "Non-finite values detected in doc embeddings!"

    # 3. Check L2 normalization
    norms = np.linalg.norm(doc_arr, axis=1)
    print(f"   Norms (min={norms.min():.4f}, max={norms.max():.4f}): {np.allclose(norms, 1.0, atol=1e-3)}")

    # 4. Query embedding (passage query)
    query_text = "What is the waiting period for pre-existing disease?"
    t0 = time.perf_counter()
    query_embeddings = list(model.passage_embed([query_text]))
    query_time = time.perf_counter() - t0
    query_arr = np.array(query_embeddings, dtype=np.float32)
    print(f"3. Query embedding in {query_time * 1000:.2f} ms, shape={query_arr.shape}")
    assert query_arr.shape[1] == 384
    assert np.all(np.isfinite(query_arr))

    # 5. Measure Warm query inference over 20 runs
    warm_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = list(model.query_embed(query_text))
        warm_times.append(time.perf_counter() - t0)
    warm_times.sort()
    med_q = warm_times[len(warm_times) // 2]
    p95_q = warm_times[int(len(warm_times) * 0.95)]
    print(f"4. Warm query inference (20 runs): median={med_q * 1000:.2f} ms, p95={p95_q * 1000:.2f} ms")

    # 6. Test empty text / edge cases
    empty_result = list(model.embed([""]))
    print(f"5. Empty string handling: returns {len(empty_result)} embedding with shape {np.array(empty_result).shape}")

    print("=" * 60)
    print("FAST-EMBED ARCTIC-EMBED-XS VALIDATION PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    validate()
