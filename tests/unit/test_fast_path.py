import pytest
from pathlib import Path
import json
from src.app import process_uploaded_policy
from src.ingestion.index_store import IndexStore
from src.models.data_models import Document, Chunk

class FakeEmbedder:
    def __init__(self, dimension=768, model_name="fake-model"):
        self.dimension = dimension
        self.model_name = model_name
        self.embed_calls = 0
        self.batch_size = 32

    def embed_chunks(self, chunks):
        self.embed_calls += 1
        import numpy as np
        return np.random.rand(len(chunks), self.dimension).astype('float32')

@pytest.fixture
def temp_index_store(tmp_path):
    store = IndexStore(indexes_dir=tmp_path)
    store.embedder = FakeEmbedder()
    return store

def test_fast_path_same_bytes_twice(temp_index_store, monkeypatch):
    process_pdf_calls = 0
    def mock_process_pdf(temp_path, save_meta=True):
        nonlocal process_pdf_calls
        process_pdf_calls += 1
        from src.models.data_models import Document, DocumentStatus, DocumentQuality
        return Document(doc_id="dummy", filename="dummy", file_hash="dummy", page_count=1, status=DocumentStatus.READY, quality=DocumentQuality.GOOD)
        
    monkeypatch.setattr("src.app.process_pdf", mock_process_pdf)

    file_bytes = b"%PDF-1.4\nPDF content 1"
    
    # 1st run
    res1 = process_uploaded_policy(file_bytes, "test.pdf", indexes_store=temp_index_store)
    assert not res1.cache_hit
    assert process_pdf_calls == 1
    assert temp_index_store.embedder.embed_calls >= 1
    
    # Reset counters
    process_pdf_calls = 0
    temp_index_store.embedder.embed_calls = 0

    # 2nd run
    res2 = process_uploaded_policy(file_bytes, "test.pdf", indexes_store=temp_index_store)
    assert res2.cache_hit
    assert process_pdf_calls == 0
    assert temp_index_store.embedder.embed_calls == 0

def test_fast_path_simulated_refresh(temp_index_store, tmp_path, monkeypatch):
    process_pdf_calls = 0
    def mock_process_pdf(*args, **kwargs):
        nonlocal process_pdf_calls
        process_pdf_calls += 1
        from src.models.data_models import Document, DocumentStatus, DocumentQuality
        return Document(doc_id="dummy", filename="dummy", file_hash="dummy", page_count=1, status=DocumentStatus.READY, quality=DocumentQuality.GOOD)
    monkeypatch.setattr("src.app.process_pdf", mock_process_pdf)

    file_bytes = b"%PDF-1.4\nPDF content refresh"
    process_uploaded_policy(file_bytes, "test.pdf", indexes_store=temp_index_store)
    
    # Simulated refresh: fresh objects
    new_store = IndexStore(indexes_dir=tmp_path)
    new_store.embedder = FakeEmbedder()
    
    process_pdf_calls = 0
    res2 = process_uploaded_policy(file_bytes, "test.pdf", indexes_store=new_store)
    assert res2.cache_hit
    assert process_pdf_calls == 0
    assert new_store.embedder.embed_calls == 0

def test_fast_path_different_bytes(temp_index_store, monkeypatch):
    def mock_process_pdf(temp_path, save_meta=True):
        from src.models.data_models import Document, DocumentStatus, DocumentQuality
        return Document(doc_id="dummy", filename="dummy", file_hash="dummy", page_count=1, status=DocumentStatus.READY, quality=DocumentQuality.GOOD)
    monkeypatch.setattr("src.app.process_pdf", mock_process_pdf)
    
    res1 = process_uploaded_policy(b"%PDF-1.4\nFile A", "A.pdf", indexes_store=temp_index_store)
    assert not res1.cache_hit
    embeds1 = temp_index_store.embedder.embed_calls
    
    temp_index_store.embedder.embed_calls = 0
    res2 = process_uploaded_policy(b"%PDF-1.4\nFile B", "B.pdf", indexes_store=temp_index_store)
    assert not res2.cache_hit
    assert temp_index_store.embedder.embed_calls > 0

def test_fast_path_model_change_invalidates(temp_index_store, monkeypatch):
    def mock_process_pdf(temp_path, save_meta=True):
        from src.models.data_models import Document, DocumentStatus, DocumentQuality
        return Document(doc_id="dummy", filename="dummy", file_hash="dummy", page_count=1, status=DocumentStatus.READY, quality=DocumentQuality.GOOD)
    monkeypatch.setattr("src.app.process_pdf", mock_process_pdf)

    file_bytes = b"%PDF-1.4\nPDF content model change"
    res1 = process_uploaded_policy(file_bytes, "test.pdf", indexes_store=temp_index_store)
    assert not res1.cache_hit
    
    # Change model name
    temp_index_store.embedder.model_name = "new-model"
    
    res2 = process_uploaded_policy(file_bytes, "test.pdf", indexes_store=temp_index_store)
    assert not res2.cache_hit

def test_fast_path_corrupt_index_rebuilds(temp_index_store, monkeypatch):
    def mock_process_pdf(temp_path, save_meta=True):
        from src.models.data_models import Document, DocumentStatus, DocumentQuality
        return Document(doc_id="dummy", filename="dummy", file_hash="dummy", page_count=1, status=DocumentStatus.READY, quality=DocumentQuality.GOOD)
    monkeypatch.setattr("src.app.process_pdf", mock_process_pdf)

    file_bytes = b"%PDF-1.4\nPDF content corrupt"
    res1 = process_uploaded_policy(file_bytes, "test.pdf", indexes_store=temp_index_store)
    assert not res1.cache_hit
    
    # Corrupt the index (e.g., delete chunks.json)
    import hashlib
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    idx_dir = temp_index_store.indexes_dir / file_hash
    (idx_dir / "chunks.json").unlink()
    
    # Should rebuild
    res2 = process_uploaded_policy(file_bytes, "test.pdf", indexes_store=temp_index_store)
    assert not res2.cache_hit
    assert res2[0].file_hash == file_hash




