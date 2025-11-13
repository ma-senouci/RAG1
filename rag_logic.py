import os
import sys
import logging
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Configure logging
PROJECT_LOGGER_NAME = "RAG1"
logger = logging.getLogger(PROJECT_LOGGER_NAME)
logger.setLevel(logging.INFO)

# Handler console
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Prevent inheritance issues
logger.propagate = False

# Suppress noise from third-party libraries
for name in logging.root.manager.loggerDict:
    if not name.startswith(PROJECT_LOGGER_NAME):
        logging.getLogger(name).setLevel(logging.WARNING)


class RAGManager:
    """
    Manages document ingestion, text extraction, chunking, and FAISS indexing for RAG.
    """
    def __init__(self, index_folder="index", chunk_size=750, chunk_overlap=75):
        self.index_folder = index_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        self._model = None  # Lazy loaded
        self.index = faiss.IndexFlatL2(384)
        self.all_chunks = []
        
        logger.info(f"RAGManager initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    @property
    def model(self):
        """Lazy loader for SentenceTransformer to save memory/startup time."""
        if self._model is None:
            if os.environ.get("RAG_MOCK_MODEL") == "true":
                logger.info("RAG_MOCK_MODEL=true detected. Using dummy mock for SentenceTransformer.")
                from unittest.mock import MagicMock
                mock = MagicMock()
                # Simulate the encode method returning random vectors
                def mock_encode(sentences, **kwargs):
                    import numpy as np
                    count = len(sentences) if isinstance(sentences, list) else 1
                    return np.random.rand(count, 384).astype('float32')
                mock.encode.side_effect = mock_encode
                self._model = mock
            else:
                logger.info("Loading SentenceTransformer model...")
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._model

    def list_source_files(self, folder_path="me"):
        """
        Lists all valid professional documents in the target folder.
        Explicitly ignores hidden files and system metadata.
        """
        valid_extensions = (".pdf", ".txt", ".md")
        files = []
        if not os.path.exists(folder_path):
            logger.warning(f"Folder not found: {folder_path}")
            return files
            
        for file in os.listdir(folder_path):
            # Ignore hidden files (starting with .) or system files (starting with ~)
            if file.startswith(".") or file.startswith("~$"):
                continue
                
            if file.lower().endswith(valid_extensions):
                files.append(os.path.join(folder_path, file))
        
        logger.info(f"Discovered {len(files)} files in '{folder_path}'")
        return sorted(files)

    def extract_text(self, file_path):
        """
        Extracts raw text from various file formats with robust error handling.
        """
        text = ""
        try:
            if file_path.lower().endswith(".pdf"):
                reader = PdfReader(file_path)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            elif file_path.lower().endswith((".txt", ".md")):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {str(e)}")
            
        return text

    def chunk_text(self, text):
        """
        Splits text into semantically preserved chunks.
        """
        return self.splitter.split_text(text)

    def generate_embeddings(self, chunks, batch_size=32):
        """
        Generates vector embeddings for a list of text chunks using batching to manage memory.
        """
        if not chunks:
            return np.array([], dtype='float32')
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks (batch_size={batch_size})")
        embeddings = self.model.encode(chunks, batch_size=batch_size, show_progress_bar=False)
        return embeddings

    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Converts a user query string into a vector embedding.
        Ensures consistency with ingestion by using the same model and 384 dimensions.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
            
        logger.info(f"Generating embedding for query: '{query[:50]}...'")
        embedding = self.model.encode([query])
        
        # Ensure embeddings is a float32 numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        if embedding.dtype != 'float32':
            embedding = embedding.astype('float32')
        
        # Validate dimensions (Expected: 384 for all-MiniLM-L6-v2)
        if embedding.shape != (1, 384):
             raise ValueError(f"Query embedding dimension mismatch. Expected (1, 384), got {embedding.shape}")
             
        return embedding

    def add_to_index(self, embeddings):
        """
        Adds vector embeddings to the FAISS index with dimension verification.
        """
        if len(embeddings) == 0:
            return
            
        # Ensure embeddings is a float32 numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        if embeddings.dtype != 'float32':
            embeddings = embeddings.astype('float32')
        
        # Validate dimensions (Expected: 384 for all-MiniLM-L6-v2)
        if embeddings.shape[1] != 384:
            raise ValueError(f"Embedding dimension mismatch. Expected 384, got {embeddings.shape[1]}")
        
        logger.info(f"Adding {len(embeddings)} vectors to FAISS index")
        self.index.add(embeddings)

    def search(self, query_vector: np.ndarray, k: int = 3):
        """
        Retrieves the top-k most relevant text chunks from the FAISS index.
        """
        if not isinstance(query_vector, np.ndarray):
            logger.error("Search called with non-numpy array input.")
            raise TypeError(f"query_vector must be a numpy.ndarray, got {type(query_vector)}")
            
        if query_vector.shape != (1, 384):
            logger.error(f"Search called with invalid vector shape: {query_vector.shape}")
            raise ValueError(f"query_vector must have shape (1, 384), got {query_vector.shape}")

        if query_vector.dtype != 'float32':
            logger.info("Converting query_vector to float32 for FAISS compatibility.")
            query_vector = query_vector.astype('float32')

        if self.index.ntotal == 0:
            logger.warning("Search called on an empty index.")
            return []

        # Ensure k doesn't exceed total indexed chunks
        k = min(k, self.index.ntotal)
        
        logger.info(f"Performing similarity search for k={k}")
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.all_chunks):
                results.append(self.all_chunks[idx])
            else:
                logger.warning(f"FAISS returned invalid index: {idx}")
                
        return results

    def format_context(self, context_chunks: list[str]) -> str:
        """
        Formats retrieved context chunks for system prompt injection.
        """
        if not context_chunks:
            return ""
            
        header = "## Contextual Evidence (from professional documents):\n\n"
        joined_chunks = "\n\n".join(context_chunks)
        instructions = (
            "\n\nUse the following evidence to provide factual, persona-aligned answers. "
            "If the evidence contradicts your general knowledge, prioritize the evidence."
        )
        
        return f"{header}{joined_chunks}{instructions}"
