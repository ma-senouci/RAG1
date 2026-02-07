# RAG Interactive Persona from scratch

A Retrieval-Augmented Generation (RAG) system designed to create an interactive AI persona based on professional documents (CVs, summaries, blog posts). The system uses FAISS for efficient vector similarity search and a Gradio interface for seamless user interaction.

## 🚀 Features

- **Document Ingestion**: Supports PDF, TXT, and Markdown files.
- **Smart Chunking**: Uses `RecursiveCharacterTextSplitter` to maintain semantic context.
- **Vector Search**: Leverages `SentenceTransformer` and `FAISS` for high-performance retrieval.
- **Interactive Chat**: A polished UI built with Gradio, featuring tool/function calling for lead capture and feedback.
- **Automated Lead Management**: Integrated with Pushover for real-time notifications when users provide contact details or ask unanswered questions.

## 🛠️ Architecture

- **Core**: `rag_logic.py` handles the heavy lifting—extraction, embedding, and indexing.
- **UI**: `app.py` manages the LLM orchestration (DeepSeek) and the user interface.
- **Tools**: Includes custom tools to record user interest and unknown queries.

## 📦 Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ma-senouci/RAG1.git
   cd RAG1
   ```

2. **Environment Setup**:
   Create a `.env` file from the provided template:
   ```env
   DEEPSEEK_API_KEY=your_key_here
   PUSHOVER_TOKEN=your_token_here
   PUSHOVER_USER=your_user_here
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Data**:
   Place your professional documents (PDF, TXT, MD) in the `me/` directory.

## 🏃 Usage

### 1. Synchronize the Index
Before running the app, ensure your documents are indexed:
```bash
python rag_logic.py --sync
```

### 2. Launch the Application
Start the interactive persona:
```bash
python app.py
```

## 🧪 Testing & CI/CD

This project follows professional standards with automated testing and CI/CD:

- **Automated Tests**: Unit and integration tests are powered by `pytest`.
- **CI/CD Pipeline**: A GitHub Actions workflow (`python-tests.yml`) automatically runs the full test suite on every push and pull request to ensure code quality.

To run tests locally:
```bash
pytest
```

---
![Python Tests](https://github.com/ma-senouci/RAG1/actions/workflows/python-tests.yml/badge.svg)
*Created with passion for building intelligent personal platforms.*
