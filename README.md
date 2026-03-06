# RAG From Scratch — Framework-less RAG Chatbot

[![Live Demo](https://img.shields.io/badge/🤗_Live_Demo-Try_It-green?style=for-the-badge)](https://huggingface.co/spaces/m-senouci/rag-from-scratch)
[![CI](https://github.com/ma-senouci/rag-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/ma-senouci/rag-from-scratch/actions/workflows/ci.yml)

**A Retrieval-Augmented Generation (RAG) system built from scratch**
with a custom FAISS-based retrieval pipeline, streaming responses, and tool calling implemented directly on top of the OpenAI-compatible API — without any RAG frameworks or high-level abstractions.

> Ask questions grounded in your document collection — every answer is retrieved through a **custom FAISS-based retrieval pipeline** and backed by verifiable source evidence.

While this demo uses `mysummary.txt` to simulate a personal creator profile, the system can ingest any collection of PDF, TXT, and Markdown files for versatile, document-grounded Q&A.

## ✨ Key Features

- **Semantic Search** — Queries are matched against document embeddings using FAISS L2 similarity (equivalent to cosine similarity since embeddings are pre-normalized), not keyword matching
- **Evidence-Grounded Responses** — The LLM cites specific portfolio content; no hallucinated claims
- **Multi-Format Ingestion** — Supports PDF, TXT, and Markdown documents out of the box
- **Tool Calling** — Collects user contact information and flags unanswered questions via Pushover notifications
- **Streaming Responses** — Real-time token-by-token output with streamed tool-call reassembly for a responsive chat experience
- **Persistent Index** — FAISS index + pickle metadata stored on disk; no re-indexing on restart
- **Local Embeddings** — Uses `all-MiniLM-L6-v2` for zero-cost, offline vector generation with lazy loading for fast startup
- **Multi-Provider LLM** — 8 interchangeable backends (DeepSeek, OpenAI, Gemini, Grok, Groq, OpenRouter, Mistral, Ollama) via a single `LLM_BACKEND` env var — all OpenAI-compatible

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      app.py  (Orchestrator)                  │
│                                                              │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Gradio  │→ │ Persona  │→ │Streaming │→ │ Tool Calling │  │
│  │ ChatUI  │  │ Prompt   │  │ LLM Chat │  │              │  │
│  └─────────┘  └──────────┘  └────┬─────┘  └──────┬───────┘  │
└───────┬──────────────────────────┼────────────────┼──────────┘
        │                          │                │
  get_query_embedding()       get_llm()         push()
   + search()                     │                │
        │                          │                │
        ▼                          ▼                ▼
┌──────────────────┐   ┌────────────────────┐  ┌────────────┐
│  rag_logic.py    │   │  llm_factory.py    │  │ Pushover   │
│  (RAGManager)    │   │                    │  │ API        │
│                  │   │  8 providers via   │  └────────────┘
│  ┌────────────┐  │   │  OpenAI-compat API │──→ LLM API
│  │   FAISS    │  │   └────────────────────┘
│  │   (L2)     │  │
│  └─────▲──────┘  │
│  ┌─────┴──────┐  │
│  │ Sentence   │  │
│  │ Transformer│  │
│  └────────────┘  │
└──────────────────┘
```

**Modular Architecture:**
| File | Responsibility |
|------|---------------|
| `app.py` | UI, persona, streaming LLM chat with tool-call loop, and orchestration |
| `llm_factory.py` | LLM backend selection and client initialization |
| `rag_logic.py` | Text extraction, chunking, embedding, FAISS indexing, and semantic retrieval |

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Orchestration | Custom Python Pipeline | Framework-less control over indexing, retrieval, and prompt augmentation |
| Vector Store | FAISS (L2) | Raw, high-performance similarity search — no abstraction layer |
| Embeddings | `all-MiniLM-L6-v2` (Sentence-Transformers) | Local, free, fast (~80MB) |
| Text Splitting | `RecursiveCharacterTextSplitter` | Semantic-aware chunking with overlap |
| Metadata Persistence | `Pickle` | Persists text chunks to disk and maps FAISS vector indices back to source text |
| LLM Provider | OpenAI-Compatible API | 8 interchangeable backends — cloud and local — all via a unified OpenAI-compatible interface |
| Interface | Gradio | Chat UI with message history — renders the hand-rolled streaming output |
| Notifications | Pushover API | Real-time alerts for unanswered questions and user lead capture |
| Document Loading | PyPDF + native I/O | PDF, TXT, Markdown support — no framework dependency |

## 🧠 LLM Provider Catalog

The system supports **8 interchangeable LLM backends**, all accessed through a unified OpenAI-compatible interface. Switch providers with a single environment variable — no code changes required.

### Default Model Selection

For **local** inference, the system uses **Llama 3.1 8B Instruct (q4_0)** — a quantized, instruction-tuned model that runs on consumer hardware with ~4.7 GB of memory.

For **cloud** inference, all defaults prioritize **low cost, fast responses, and strong reasoning** for RAG workloads. **OpenRouter** provides access to models from Anthropic, Meta, Google, Mistral, and others through a single API key, serving as an API aggregator. **Groq** deploys the full-scale **Llama 3.3 70B Versatile** — too large for local inference, yet delivered with near-instant latency, enabled by Groq's specialized LPU hardware.

### Provider Matrix

| Provider | Default Model | Type |
|----------|--------------|------|
| **DeepSeek** | `deepseek-chat` | Cloud API |
| **OpenAI** | `gpt-4.1-mini` | Cloud API |
| **Gemini** | `gemini-3.1-flash-lite-preview` | Cloud API |
| **Grok** | `grok-4.1-fast` | Cloud API |
| **Groq** | `llama-3.3-70b-versatile` | Inference API |
| **OpenRouter** | `anthropic/claude-haiku-4.5` | API Aggregator |
| **Mistral** | `mistral-small-latest` | Cloud API |
| **Ollama** | `llama3.1:8b-instruct-q4_0` | Local |

**Type legend:** *Cloud API* — provider hosts their own model · *Inference API* — runs open-source models on specialized hardware · *API Aggregator* — single key, multiple providers · *Local* — runs on your own machine

### Design Philosophy

- **Zero code changes** to switch providers — just update `LLM_BACKEND` in `.env`
- **Curated defaults** — each model is pre-selected for cost-efficient RAG workloads
- **Fully overridable** — set any model via provider-specific env vars (e.g., `OPENAI_MODEL=gpt-4.1`)

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or 3.12
- A provider API key (e.g., [DeepSeek](https://platform.deepseek.com/) or [OpenAI](https://platform.openai.com/))

### Installation

```bash
git clone https://github.com/ma-senouci/rag-from-scratch.git
cd rag-from-scratch

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Create a `.env` file from the template:

```env
# Select your backend: deepseek (default) | openai | gemini | grok | groq | openrouter | mistral | ollama
LLM_BACKEND=deepseek
```

Each provider requires its own API key and has a pre-configured default model. See `.env.example` for the full template with all 8 providers.

> [!NOTE]
> Only the credentials for your **active** `LLM_BACKEND` are required. You don't need API keys for providers you aren't using.

### Usage

**Step 1 — Index your documents**

Place your PDF, TXT, or MD files in the `me/` folder, then run:

```bash
python rag_logic.py --sync
```

**Step 2 — Chat**

```bash
python app.py
```

This launches the Gradio chat interface at `http://localhost:7860`.

## 🔄 Streaming

This project uses a **hand-rolled streaming pipeline** — no framework abstractions or black-box handlers:

1. The LLM response is streamed token-by-token via the OpenAI-compatible API
2. Content tokens are yielded immediately to Gradio for real-time display
3. Tool-call fragments are buffered and reassembled across stream chunks
4. After tool execution, the conversation loops back for a follow-up LLM turn
5. A `max_turns=10` guard prevents infinite tool-call loops

This gives users instant visual feedback while the full response is still being generated.

## 🐳 Docker Support

For users who prefer containerized environments, this project is fully Docker-ready. This ensures a consistent, isolated setup regardless of host OS.

### 1. Build the Image
```bash
docker build -t rag-from-scratch .
```

### 2. Run the Container
Pass your API key at runtime using the `-e` flag.

```bash
# Default backend (DeepSeek)
docker run -p 7860:7860 -e DEEPSEEK_API_KEY=your-api-key-here rag-from-scratch

# Switch provider and override default model
docker run -p 7860:7860 -e LLM_BACKEND=openai -e OPENAI_MODEL=gpt-4.1 -e OPENAI_API_KEY=your-key rag-from-scratch
```

> [!TIP]
> This image is a "Complete Package" — it comes pre-bundled with the `me/` documents and `index/` FAISS index so it works immediately. To use your own documents, mount your local folder and re-sync:
> ```bash
> docker run -p 7860:7860 -e DEEPSEEK_API_KEY=xxx -v /path/to/docs:/app/me rag-from-scratch sh -c "python rag_logic.py --sync && python app.py"
> ```

## 📁 Project Structure

```
rag-from-scratch/
├── app.py                # Chat UI, streaming RAG + generation, tool calling
├── rag_logic.py          # Knowledge base: text extraction, chunking, embedding, FAISS indexing, retrieval
├── llm_factory.py        # LLM backend selection and client initialization
├── me/                   # Source documents (PDF, TXT, MD)
├── index/                # Persisted FAISS index + metadata (manually generated via sync)
├── tests/
│   ├── test_rag_logic.py
│   ├── test_llm_factory.py
│   ├── test_me.py
│   ├── test_orchestration.py
│   └── test_cli.py
├── Dockerfile            # Container build definition
├── .dockerignore
├── .github/workflows/ci.yml  # CI pipeline
├── verify_ingestion.py   # Index synchronization verification script
├── requirements.txt      # Pinned dependencies
├── .env.example          # Environment variable template
├── LICENSE
└── README.md
```

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

Tests cover:
- LLM Provider Backend selection and model fallback logic
- Document discovery and multi-format loading
- Text chunking with configurable parameters
- FAISS indexing and persistence verification
- Semantic search with top-k retrieval
- Context formatting and prompt injection
- Streaming orchestration with tool-call reassembly
- Multi-turn tool-call loop and max_turns guard
- Error handling and graceful degradation

## 📝 How It Works

1. **Sync** — Documents in `me/` are manually loaded, split into 750-char chunks with 75 overlap (configurable in `rag_logic.py` via `RAGManager(chunk_size, chunk_overlap)`), embedded with `all-MiniLM-L6-v2`, and stored as a FAISS index with Pickle metadata on disk for persistence.
2. **Query** — When a user asks a question, the query is embedded and the top-3 most similar chunks are retrieved via L2 similarity search (configurable in `rag_logic.py` via `RAGManager.search(k=3)`).
3. **Augment** — Retrieved chunks are formatted with descriptive headers via `format_context` and injected into the system prompt alongside the defined AI persona.
4. **Generate** — The LLM produces a streamed, grounded response. If it can't answer, it logs the unknown question via tool calling. If the user provides contact info, it captures it automatically.

## 🌐 Deployment

> **🚀 [Try the live demo on HuggingFace Spaces →](https://huggingface.co/spaces/m-senouci/rag-from-scratch)**

This application is deployed on [HuggingFace Spaces](https://huggingface.co/spaces) using the Gradio SDK.

### Deploy Your Own:

1. **Create a New Space:** On HuggingFace, create a new Space and select **Gradio** as the SDK.
2. **Upload Files:** Upload the following files to the Space repository:
   - `app.py`
   - `rag_logic.py`
   - `llm_factory.py`
   - `requirements.txt`
   - `me/` (source documents)
   - `index/` (pre-built FAISS index, to avoid re-indexing on startup)
3. **Configure Secrets:** In your Space's **Settings** tab, add the following as "Variables" or "Secrets":
   - `LLM_BACKEND` (defaults to `deepseek`)
   - API key for your chosen provider (e.g., `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`)
   - `PUSHOVER_TOKEN` / `PUSHOVER_USER` (optional)

The Space will automatically build and launch the interface, providing a public URL for your RAG chatbot.

## License

MIT
