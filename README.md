# 🔬 AI Research Assistant

> Production-ready RAG (Retrieval-Augmented Generation) system for intelligent document Q&A with citations, confidence scoring, and evaluation metrics.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.0-orange)](https://www.trychroma.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-blue)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                       │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │  Chat UI │  │  Evaluation  │  │   Document Library    │ │
│  │  +History│  │   Metrics    │  │   Chunk Preview       │ │
│  └────┬─────┘  └──────┬───────┘  └───────────┬───────────┘ │
└───────┼───────────────┼──────────────────────┼─────────────┘
        │               │    REST API           │
┌───────▼───────────────▼──────────────────────▼─────────────┐
│                    FastAPI Backend                           │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   API Routes                         │   │
│  │  /documents  /chat  /sessions  /evaluation  /health  │   │
│  └──────┬──────────┬──────────┬──────────┬────────────┘   │
│         │          │          │          │                  │
│  ┌──────▼──┐ ┌─────▼───┐ ┌───▼────┐ ┌───▼──────────────┐ │
│  │Document │ │Retriever│ │Session │ │ Evaluation        │ │
│  │Processor│ │Service  │ │Manager │ │ Metrics Engine    │ │
│  └──────┬──┘ └─────┬───┘ └───┬────┘ └──────────────────┘ │
│         │          │          │                            │
│  ┌──────▼──────────▼──┐  ┌───▼────┐                       │
│  │  Embedding Service  │  │SQLite  │                       │
│  │  (Sentence Trans.)  │  │Sessions│                       │
│  └──────┬──────────────┘  └────────┘                       │
│         │                                                   │
└─────────┼───────────────────────────────────────────────────┘
          │
┌─────────▼───────────────┐    ┌─────────────────────────────┐
│       ChromaDB           │    │       Ollama LLM             │
│  Vector Store            │    │   mistral / llama2           │
│  Cosine Similarity       │    │   Local Inference            │
│  Persistent Storage      │    │   OpenAI-compatible API      │
└─────────────────────────┘    └─────────────────────────────┘
```

---

## 📁 Folder Structure

```
ai-research-assistant/
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── requirements.txt
│   ├── .env.example
│   ├── api/
│   │   └── routes/
│   │       ├── documents.py       # Upload, list, delete, preview chunks
│   │       ├── chat.py            # RAG Q&A with citations
│   │       ├── sessions.py        # Conversation history management
│   │       ├── evaluation.py      # Retrieval quality metrics
│   │       └── health.py          # Health checks
│   ├── core/
│   │   ├── config.py              # Pydantic Settings + env vars
│   │   ├── database.py            # ChromaDB init and client
│   │   └── logging_config.py      # Structured logging
│   ├── models/
│   │   └── schemas.py             # All Pydantic request/response models
│   ├── services/
│   │   ├── document_processor.py  # PDF load → chunk → embed → store
│   │   ├── embedder.py            # Sentence Transformers embedding service
│   │   ├── retriever.py           # Semantic search + citation building
│   │   ├── llm_service.py         # LangChain + Ollama LLM wrapper
│   │   ├── session_manager.py     # SQLite-backed conversation history
│   │   └── confidence_scorer.py   # Composite confidence scoring
│   └── evaluation/
│       └── metrics.py             # MRR, P@K, R@K, NDCG implementations
├── frontend/
│   ├── app.py                     # Streamlit multi-page app
│   └── requirements.txt
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── tests/
│   ├── unit/
│   │   ├── test_metrics.py
│   │   └── test_confidence.py
│   └── integration/
├── scripts/
│   └── setup.sh
└── README.md
```

---

## 🔌 API Endpoints

### Documents
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/documents/upload` | Upload & process PDF |
| `GET` | `/api/v1/documents/` | List all indexed documents |
| `DELETE` | `/api/v1/documents/{doc_id}` | Delete document + chunks |
| `GET` | `/api/v1/documents/{doc_id}/chunks` | Preview document chunks |

### Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/chat/` | Ask question, get cited answer |

### Sessions
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/sessions/` | Create new session |
| `GET` | `/api/v1/sessions/` | List all sessions |
| `GET` | `/api/v1/sessions/{id}` | Get session details |
| `GET` | `/api/v1/sessions/{id}/history` | Get conversation history |
| `DELETE` | `/api/v1/sessions/{id}` | Delete session |

### Evaluation
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/evaluation/` | Run retrieval quality evaluation |
| `GET` | `/api/v1/evaluation/benchmark` | Get metric definitions |

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `GET` | `/docs` | Swagger UI |

---

## 🗄 Database Schema

### ChromaDB Collections

**`research_documents`** — Vector store
```
id:        {doc_id}_chunk_{index}   (string, primary key)
document:  chunk text content        (string)
embedding: float vector [384-dim]    (list[float])
metadata: {
  doc_id:       string,              # SHA-256 derived document ID
  filename:     string,              # Original PDF filename
  chunk_index:  int,                 # Position within document
  page_number:  int,                 # Source page
  session_id:   string,              # Associated session
  tags:         string (CSV),        # User-assigned tags
  char_count:   int                  # Characters in chunk
}
```

### SQLite Tables

**`sessions`**
```sql
session_id  TEXT  PRIMARY KEY
name        TEXT  NOT NULL
created_at  TEXT  NOT NULL   -- ISO 8601
last_active TEXT  NOT NULL   -- ISO 8601
doc_ids     TEXT  DEFAULT '[]'  -- JSON array
```

**`messages`**
```sql
id          INTEGER  PRIMARY KEY AUTOINCREMENT
session_id  TEXT     REFERENCES sessions(session_id)
role        TEXT     CHECK(role IN ('user','assistant','system'))
content     TEXT     NOT NULL
timestamp   TEXT     NOT NULL   -- ISO 8601
```

---

## 📊 Evaluation Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **MRR** | Mean Reciprocal Rank — rewards first relevant result | [0, 1] |
| **Precision@K** | Fraction of top-K results that are relevant | [0, 1] |
| **Recall@K** | Fraction of relevant docs retrieved in top-K | [0, 1] |
| **NDCG** | Normalized Discounted Cumulative Gain | [0, 1] |
| **Confidence Score** | Composite: retrieval quality + answer completeness | [0, 1] |

**Relevance threshold:** `score ≥ 0.5` → considered relevant  
**Confidence labels:** High (≥0.75) · Medium (≥0.50) · Low (≥0.25) · Very Low (<0.25)

---

## 🚀 Quick Start

### Option A — Local (Recommended for development)

```bash
# 1. Clone and enter project
git clone <repo-url>
cd ai-research-assistant

# 2. Install Ollama and pull model
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull mistral

# 3. Run setup script
chmod +x scripts/setup.sh && ./scripts/setup.sh

# 4. Start backend
cd backend
source .venv/bin/activate
python main.py
# API available at http://localhost:8000

# 5. Start frontend (new terminal)
cd frontend
source .venv/bin/activate
streamlit run app.py
# UI available at http://localhost:8501
```

### Option B — Docker Compose

```bash
# Build and start all services
cd docker
docker compose up --build

# Pull LLM model into Ollama container
docker exec ra_ollama ollama pull mistral

# Services:
# Frontend  → http://localhost:8501
# Backend   → http://localhost:8000
# API Docs  → http://localhost:8000/docs
```

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and customize:

```env
# LLM Model (change to llama3, phi3, etc.)
OLLAMA_MODEL=mistral

# Chunk settings (tune for your documents)
CHUNK_SIZE=512
CHUNK_OVERLAP=64

# Retrieval quality
TOP_K_RETRIEVAL=5
MIN_RELEVANCE_SCORE=0.3

# Embedding model
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

## 🧪 Running Tests

```bash
cd backend
source .venv/bin/activate
pip install pytest
pytest ../tests/unit/ -v
```

---

## 🗺 Implementation Roadmap

### Phase 1 — Foundation ✅
- [x] FastAPI backend with all endpoints
- [x] PDF ingestion pipeline (load → chunk → embed → store)
- [x] ChromaDB vector store integration
- [x] Sentence Transformers embeddings
- [x] Ollama LLM integration via LangChain
- [x] RAG pipeline with context injection
- [x] Source citations with page numbers

### Phase 2 — Quality & UX ✅
- [x] Confidence scoring (composite metric)
- [x] Retrieval evaluation (MRR, P@K, R@K, NDCG)
- [x] Conversation history (SQLite)
- [x] Multi-PDF support with doc filtering
- [x] Streamlit frontend with dark theme
- [x] Docker Compose deployment

### Phase 3 — Production Enhancements (Roadmap)
- [ ] Streaming responses (SSE)
- [ ] Async ChromaDB client
- [ ] Redis caching for embeddings
- [ ] HyDE (Hypothetical Document Embeddings)
- [ ] Re-ranking with cross-encoders
- [ ] User authentication (JWT)
- [ ] PostgreSQL for metadata at scale
- [ ] Kubernetes deployment manifests
- [ ] Prometheus metrics + Grafana dashboard
- [ ] Multi-modal support (images in PDFs)

---

## 🤝 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit 1.35 | Interactive UI with dark theme |
| **Backend** | FastAPI 0.111 | Async REST API |
| **Vector DB** | ChromaDB 0.5 | Semantic vector storage |
| **Embeddings** | Sentence Transformers | `all-MiniLM-L6-v2` (384-dim) |
| **LLM** | Ollama + Mistral 7B | Local inference, no API cost |
| **Orchestration** | LangChain 0.2 | Document loading & LLM chains |
| **Sessions** | SQLite | Lightweight conversation storage |
| **Container** | Docker + Compose | Reproducible deployment |

---

## 📄 License

MIT — free to use, modify, and distribute.

---

*Built as a portfolio-grade RAG system demonstrating production patterns: clean architecture, evaluation-driven development, and observable AI.*
#
