# 🔬 Nexus AI Research Hub

An open-source **Retrieval-Augmented Generation (RAG)** research assistant
with a two-stage hybrid retrieval pipeline, cross-encoder reranking,
and streaming responses.

Upload PDFs or paste YouTube URLs → the system indexes them into a vector
store and answers your questions using **Llama 3.3 70B** (via Groq),
grounded entirely in your data.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Hybrid Search** | BM25 (keyword) + Vector (semantic) ensemble retrieval |
| **Cross-Encoder Reranking** | Second-stage re-scoring with `ms-marco-MiniLM` for precision |
| **Streaming Responses** | Token-by-token answer display via Groq's ~800 tok/s inference |
| **Conversation Memory** | Ask follow-up questions naturally |
| **Source Citations** | Every answer shows the exact chunks it was derived from |
| **Hallucination Guard** | LLM is instructed to refuse when context is insufficient |
| **Configurable Pipeline** | Tune chunk size, search strategy, k, reranking from the sidebar |
| **Modular Architecture** | `core/` business logic is framework-agnostic and reusable |

---

## 🏗️ Architecture

```
nexus-ai-research-hub/
├── app.py                        # Thin entry point — wires UI to Core
├── config.py                     # Centralized configuration (dataclass)
├── core/                         # Business logic (framework-agnostic)
│   ├── document_processor.py     # PDF/YouTube loading, chunking, dedup
│   ├── embeddings.py             # BGE embedding model (cached)
│   ├── vector_store.py           # ChromaDB lifecycle management
│   ├── retriever.py              # Hybrid BM25+Vector → Reranker
│   ├── rag_chain.py              # Prompt → LLM → Stream orchestration
│   ├── llm.py                    # Groq LLM client (cached, streaming)
│   └── prompts.py                # All prompt templates
├── ui/                           # Streamlit presentation layer
│   ├── styles.py                 # CSS design system
│   ├── sidebar.py                # Control panel components
│   └── chat.py                   # Chat interface + source display
└── utils/                        # Cross-cutting concerns
    └── logger.py                 # Structured logging
```

### Retrieval Pipeline

```
Query → BM25 (keyword, weight 0.3) ──┐
                                      ├── Ensemble (k=8) → Cross-Encoder Rerank → Top 4 → LLM → Answer
Query → Vector Search (MMR, weight 0.7) ┘
```

1. **Ingestion** — PDFs via `PyPDFLoader`; YouTube via `YoutubeLoader`
2. **Chunking** — 1500-char chunks, 200-char overlap, content-hash deduplication
3. **Embedding** — `BAAI/bge-base-en-v1.5` (768-dim, top-5 on MTEB)
4. **Storage** — ChromaDB with cosine similarity (HNSW index)
5. **Retrieval** — Hybrid BM25 + Vector ensemble, weighted merge
6. **Reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores top candidates
7. **Generation** — Llama 3.3 70B via Groq, streaming tokens

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/nexus-ai-research-hub.git
cd nexus-ai-research-hub

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Create your .env from the template
cp .env.example .env

# Add your Groq API key
nano .env
```

### Run

```bash
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

---

## ⚙️ Configuration

All settings can be overridden in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | HuggingFace embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq-hosted LLM |
| `CHUNK_SIZE` | `1500` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVER_K` | `8` | Candidates from hybrid search |
| `RERANKER_TOP_N` | `4` | Chunks after reranking |
| `BM25_WEIGHT` | `0.3` | BM25 weight in ensemble |
| `VECTOR_WEIGHT` | `0.7` | Vector weight in ensemble |
| `CHROMA_PERSIST_DIR` | `./nexus_db` | Vector store path |
| `MAX_CHAT_HISTORY` | `10` | Conversation turns in memory |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| LLM | Llama 3.3 70B (via Groq) |
| Embeddings | `BAAI/bge-base-en-v1.5` (768-dim) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector Store | ChromaDB |
| Keyword Search | BM25 (rank_bm25) |
| Orchestration | LangChain |
| PDF Parsing | PyPDF |
| YouTube | youtube-transcript-api |

---

## 📄 License

MIT