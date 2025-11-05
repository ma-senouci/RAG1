import os
import sys
import logging

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
        self._model = None  # Lazy loaded
        self.all_chunks = []
        
        logger.info(f"RAGManager initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
