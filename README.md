🔬 AI Research Assistant (RAG System)

A production-grade Retrieval-Augmented Generation (RAG) system for intelligent document Q&A with citations, confidence scoring, and evaluation metrics.

🚀 Overview

The AI Research Assistant is a full-stack RAG system that allows users to upload documents and ask intelligent questions with context-aware answers backed by citations.

It is designed with production-level architecture, evaluation-driven development, and modular backend services.

✨ Key Features
💬 Intelligent Q&A (RAG)
Ask questions over PDFs using natural language
Semantic retrieval using embeddings
LLM-generated answers (Ollama: Mistral / Llama)
Every answer includes source citations (page-level grounding)
📄 Document Processing
PDF upload and parsing
Chunking with overlap for better retrieval
Embedding generation using Sentence Transformers
Persistent vector storage with ChromaDB
📊 Evaluation System
Retrieval quality metrics:
Mean Reciprocal Rank (MRR)
Precision@K
Recall@K
NDCG
Benchmark API for testing retrieval performance
🧠 Confidence Scoring

Composite scoring system based on:

Retrieval similarity strength
Context completeness
Answer quality signals

Confidence Levels:

🟢 High (≥ 0.75)
🟡 Medium (≥ 0.50)
🟠 Low (≥ 0.25)
🔴 Very Low (< 0.25)
💾 Session Management
Multi-session chat history
Persistent conversation storage (SQLite)
Document-session linking
🏗 Architecture
User (Streamlit UI)
        ↓
FastAPI Backend
        ↓
Document Retriever (ChromaDB)
        ↓
Embedding Model (MiniLM)
        ↓
Top-K Relevant Chunks
        ↓
LLM (Ollama: Mistral / Llama)
        ↓
Cited Answer + Confidence Score
        ↓
Streamlit UI Response
📁 Project Structure
ai-research-assistant/
├── backend/
│   ├── api/               # API routes (chat, docs, sessions, eval)
│   ├── core/              # Config, DB, logging
│   ├── services/          # RAG pipeline logic
│   ├── models/            # Pydantic schemas
│   ├── evaluation/       # IR metrics engine
│   └── main.py           # FastAPI entry point
│
├── frontend/
│   └── app.py            # Streamlit UI
│
├── docker/
├── scripts/
├── tests/
└── README.md
🔌 API Endpoints
📄 Documents
Method	Endpoint	Description
POST	/api/v1/documents/upload	Upload & process PDF
GET	/api/v1/documents/	List documents
DELETE	/api/v1/documents/{doc_id}	Delete document
GET	/api/v1/documents/{doc_id}/chunks	Preview chunks
💬 Chat
Method	Endpoint	Description
POST	/api/v1/chat/	Ask question using RAG
📊 Evaluation
Method	Endpoint	Description
POST	/api/v1/evaluation/	Run retrieval evaluation
GET	/api/v1/evaluation/benchmark	View metrics info
❤️ Health
Method	Endpoint
GET	/health
GET	/docs
🗄 Data Storage
📌 ChromaDB Collection: research_documents

Stores:

Document chunks
Embeddings (384-dim vectors)
Metadata:
document ID
filename
page number
chunk index
session ID
🗃 SQLite Schema
Sessions
session_id TEXT PRIMARY KEY,
name TEXT,
created_at TEXT,
last_active TEXT,
doc_ids TEXT
Messages
id INTEGER PRIMARY KEY AUTOINCREMENT,
session_id TEXT,
role TEXT,
content TEXT,
timestamp TEXT
📊 Evaluation Metrics
Metric	Description
MRR	Ranking quality of first relevant result
Precision@K	Accuracy of top-K results
Recall@K	Coverage of relevant chunks
NDCG	Ranking quality with position weighting
⚙️ Configuration

Create a .env file:

OLLAMA_MODEL=mistral

CHUNK_SIZE=512
CHUNK_OVERLAP=64

TOP_K_RETRIEVAL=5
MIN_RELEVANCE_SCORE=0.3

EMBEDDING_MODEL=all-MiniLM-L6-v2
🚀 Quick Start
1️⃣ Clone Repository
git clone <repo-url>
cd ai-research-assistant
2️⃣ Install Ollama + Model
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull mistral
3️⃣ Backend Setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
4️⃣ Frontend Setup
cd frontend
pip install -r requirements.txt
streamlit run app.py
🐳 Docker Deployment
cd docker
docker compose up --build
🧪 Testing
pytest tests/unit/
pytest tests/integration/
🧭 Roadmap
✅ Completed
RAG pipeline (PDF → retrieval → LLM)
Citation system
Confidence scoring
Evaluation metrics
Session management
Streamlit UI
Docker setup
🚧 Upcoming
Streaming responses (SSE)
Cross-encoder reranking
Redis caching
JWT authentication
PostgreSQL migration
Kubernetes deployment
Monitoring (Prometheus + Grafana)
Multi-modal document support
🧰 Tech Stack
Layer	Technology
Frontend	Streamlit
Backend	FastAPI
Vector DB	ChromaDB
Embeddings	Sentence Transformers
LLM	Ollama (Mistral / Llama)
Database	SQLite
DevOps	Docker
🎯 Why This Project Matters
Production-style RAG architecture (not a toy project)
Built-in evaluation system (rare in most RAG apps)
Fully local inference (privacy-first AI system)
Clean modular backend design (scalable & maintainable)
Real-world AI engineering patterns
📄 License

MIT License — free to use, modify, and extend.
