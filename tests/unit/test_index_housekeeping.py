import pytest
import os
import json
from pathlib import Path
from database.index_store import IndexStore

def test_cleanup_invalid_folders(tmp_path):
    store = IndexStore(indexes_dir=tmp_path)
    
    # Valid
    valid_hash = "a" * 64
    (tmp_path / valid_hash).mkdir()
    
    # Invalid
    (tmp_path / "1234").mkdir()
    (tmp_path / "dummy-uuid-folder").mkdir()
    
    before, after = store.cleanup_invalid_folders()
    assert before == 3
    assert after == 1
    assert (tmp_path / valid_hash).exists()
    assert not (tmp_path / "1234").exists()

def test_enforce_index_limit(tmp_path):
    store = IndexStore(indexes_dir=tmp_path)
    
    import time
    # Create 25 valid folders with varying mtimes
    for i in range(25):
        h = f"{i:064d}"
        folder = tmp_path / h
        folder.mkdir()
        meta = folder / "metadata.json"
        meta.write_text("{}")
        
        # Set mtime (older to newer)
        os.utime(meta, (time.time() + i, time.time() + i))
        
    assert len(list(tmp_path.iterdir())) == 25
    
    store.enforce_index_limit(20)
    
    remaining = list(tmp_path.iterdir())
    assert len(remaining) == 20
    
    # The oldest 5 should be deleted (0 to 4)
    for i in range(5):
        assert not (tmp_path / f"{i:064d}").exists()
        
    # 5 to 24 should remain
    for i in range(5, 25):
        assert (tmp_path / f"{i:064d}").exists()
