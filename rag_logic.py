import os
import sys
import logging
from pypdf import PdfReader

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
