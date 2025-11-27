import pytest
import os
import pickle
import numpy as np
from unittest.mock import MagicMock, patch
from rag_logic import RAGManager

@pytest.fixture(autouse=True)
def mock_sentence_transformer():
    with patch('rag_logic.SentenceTransformer') as mock_st:
        # Mock the encode method to return fixed-dimension random vectors
        mock_instance = mock_st.return_value
        def mock_encode(sentences, **kwargs):
            import numpy as np
            count = len(sentences) if isinstance(sentences, list) else 1
            return np.random.rand(count, 384).astype('float32')
        mock_instance.encode.side_effect = mock_encode
        yield mock_instance

def test_rag_manager_initialization(tmp_path):
    """Test that RAGManager initializes with default values."""
    manager = RAGManager(index_folder=str(tmp_path / "empty"))
    assert manager.chunk_size == 750
    assert manager.chunk_overlap == 75

def test_file_discovery_robust(tmp_path):
    """Test that RAGManager finds the expected files and ignores hidden ones."""
    # Setup temp directory
    d = tmp_path / "test_me"
    d.mkdir()
    (d / "test.pdf").write_text("content")
    (d / "test.txt").write_text("content")
    (d / ".hidden.txt").write_text("hidden")
    (d / "~$temp.docx").write_text("temp")
    
    manager = RAGManager(index_folder=str(tmp_path / "empty"))
    files = manager.list_source_files(str(d))
    
    basenames = [os.path.basename(f) for f in files]
    assert "test.pdf" in basenames
    assert "test.txt" in basenames
    assert ".hidden.txt" not in basenames
    assert "~$temp.docx" not in basenames
    assert len(files) == 2

def test_chunking_logic(tmp_path):
    """Test that text is correctly chunked."""
    manager = RAGManager(index_folder=str(tmp_path / "empty"), chunk_size=100, chunk_overlap=10)
    text = "A" * 250  # 250 characters
    chunks = manager.chunk_text(text)
    
    # With chunk_size=100 and overlap=10:
    # Chunk 1: 0-100
    # Chunk 2: 90-190
    # Chunk 3: 180-250
    assert len(chunks) == 3
    for chunk in chunks:
        assert len(chunk) <= 100

def test_generate_embeddings(tmp_path):
    """Test that embeddings are generated with correct dimensions."""
    manager = RAGManager(index_folder=str(tmp_path / "empty"))
    chunks = ["This is a test chunk.", "Another test chunk."]
    embeddings = manager.generate_embeddings(chunks)
    
    # dimensions for all-MiniLM-L6-v2 is 384
    assert len(embeddings) == 2
    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == 'float32'

def test_faiss_indexing(tmp_path):
    """Test that FAISS index correctly stores embeddings."""
    manager = RAGManager(index_folder=str(tmp_path / "empty"))
    chunks = ["Chunk 1", "Chunk 2"]
    embeddings = manager.generate_embeddings(chunks)
    
    manager.add_to_index(embeddings)
    assert manager.index.ntotal == 2

def test_faiss_indexing_dimension_mismatch(tmp_path):
    """Test that FAISS index raises error on dimension mismatch."""
    manager = RAGManager(index_folder=str(tmp_path / "empty"))
    wrong_dim_embeddings = np.random.rand(1, 128).astype('float32')
    
    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        manager.add_to_index(wrong_dim_embeddings)

def test_ingest_documents_orchestration(tmp_path):
    """Test that ingest_documents orchestrates embedding and indexing."""
    d = tmp_path / "test_ingest"
    d.mkdir()
    (d / "doc.txt").write_text("This is valid text for embedding.")
    
    manager = RAGManager(index_folder=str(tmp_path / "index"))
    chunks = manager.sync_index(str(d))
    
    assert len(chunks) > 0
    assert manager.index.ntotal == len(chunks)

def test_save_load_index(tmp_path):
    """Test that FAISS index and metadata are saved and reloaded correctly."""
    index_dir = tmp_path / "index"
    
    # Instance 1: sync index and save
    manager1 = RAGManager(index_folder=str(index_dir))
    
    # Setup test file
    d = tmp_path / "data"
    d.mkdir()
    (d / "test.txt").write_text("Persistence test chunk.")
    
    manager1.sync_index(str(d))
    
    # Verify files exist
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "metadata.pkl").exists()
    assert len(manager1.all_chunks) == manager1.index.ntotal == 1
    
    # Instance 2: Load index and metadata
    manager2 = RAGManager(index_folder=str(index_dir))
    # Note: load_index is called automatically in __init__
    
    assert len(manager2.all_chunks) == manager2.index.ntotal == 1
    assert "Persistence test chunk." in manager2.all_chunks[0]

def test_sequential_ingestion_rebuild(tmp_path):
    """Test that multiple ingestions rebuilding the index instead of appending."""
    index_dir = tmp_path / "index"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    manager = RAGManager(index_folder=str(index_dir))
    
    # First ingestion (1 file)
    (data_dir / "doc1.txt").write_text("Chunk one.")
    manager.sync_index(str(data_dir))
    assert len(manager.all_chunks) == manager.index.ntotal == 1
    
    
    # Second ingestion (2 files: doc1 and doc2)
    (data_dir / "doc2.txt").write_text("Chunk two.")
    manager.sync_index(str(data_dir))
    
    # Total should be 2 (rebuilt with doc1.txt and doc2.txt)
    # NOT 3 (which would happen if it appended the second run result to the first)
    assert len(manager.all_chunks) == manager.index.ntotal == 2
    assert manager.all_chunks[0] == "Chunk one."
    assert "Chunk two." in manager.all_chunks[1]


def test_get_query_embedding(tmp_path):
    """Test that query embeddings are generated with correct shape and type."""
    manager = RAGManager(index_folder=str(tmp_path / "empty"))
    query = "What is RAG?"
    embedding = manager.get_query_embedding(query)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.dtype == 'float32'
    assert embedding.shape == (1, 384)
    

def test_get_query_embedding_empty_input(tmp_path):
    """Test that empty query raises ValueError."""
    manager = RAGManager(index_folder=str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="Query cannot be empty"):
        manager.get_query_embedding("")

def test_search_logic(tmp_path):
    """Test that search returns the expected text chunks ranking by relevance."""
    index_dir = tmp_path / "index"
    manager = RAGManager(index_folder=str(index_dir))
    
    # Manually add some controlled data
    # Vector 1: [1, 0, 0, ...]
    # Vector 2: [0, 1, 0, ...]
    # Vector 3: [0, 0, 1, ...]
    vectors = np.zeros((3, 384), dtype='float32')
    vectors[0, 0] = 1.0
    vectors[1, 1] = 1.0
    vectors[2, 2] = 1.0
    
    manager.all_chunks = ["Chunk A", "Chunk B", "Chunk C"]
    manager.index.add(vectors)
    
    # Search for something close to Vector 2
    query_vector = np.zeros((1, 384), dtype='float32')
    query_vector[0, 1] = 0.9
    
    results = manager.search(query_vector, k=2)
    
    assert len(results) == 2
    assert "Chunk B" in results[0] # Should be first
    assert "Chunk A" in results[1] or "Chunk C" in results[1]

def test_search_empty_index(tmp_path):
    """Test that search handles empty index gracefully."""
    manager = RAGManager(index_folder=str(tmp_path / "empty"))
    query_vector = np.zeros((1, 384), dtype='float32')
    
    results = manager.search(query_vector, k=3)
    assert results == []

def test_search_k_greater_than_total(tmp_path):
    """Test search when k is larger than the number of available chunks."""
    manager = RAGManager(index_folder=str(tmp_path / "empty"))
    vectors = np.random.rand(2, 384).astype('float32')
    manager.all_chunks = ["Doc 1", "Doc 2"]
    manager.index.add(vectors)
    
    query_vector = np.random.rand(1, 384).astype('float32')
    results = manager.search(query_vector, k=5)
    
    assert len(results) == 2 # Should return all available, not 5

def test_search_invalid_input(tmp_path):
    """Test that search raises appropriate errors for invalid inputs."""
    manager = RAGManager(index_folder=str(tmp_path / "empty"))
    
    # Test non-numpy input
    with pytest.raises(TypeError, match="must be a numpy.ndarray"):
        manager.search([1, 2, 3])
        
    # Test wrong shape
    with pytest.raises(ValueError, match="must have shape"):
        manager.search(np.zeros((1, 128)))
        
    # Test dtype conversion (float64 -> float32)
    query_vector_64 = np.zeros((1, 384), dtype='float64')
    # Should not raise error and return empty list (or results)
    results = manager.search(query_vector_64, k=1)
    assert results == [] # empty index case

def test_format_context(tmp_path):
    """Test that context is formatted correctly per AC 1 and 2."""
    manager = RAGManager(index_folder=str(tmp_path / "empty"))
    chunks = ["Chunk 1", "Chunk 2"]
    formatted = manager.format_context(chunks)
    
    assert "## Contextual Evidence (from professional documents):" in formatted
    assert "Chunk 1" in formatted
    assert "Chunk 2" in formatted
    assert "prioritize the evidence" in formatted

def test_format_context_empty(tmp_path):
    """Test that empty context returns empty string per AC 3."""
    manager = RAGManager(index_folder=str(tmp_path / "empty"))
    assert manager.format_context([]) == ""
