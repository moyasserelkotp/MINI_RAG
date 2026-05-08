# 🚀 MINI RAG: Advanced Retrieval-Augmented Generation System

> **Production-Ready AI System** that transforms static documents into intelligent, context-aware conversational AI experts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)](https://fastapi.tiangolo.com/)

---

## 📚 Documentation

> **New to this project?** Start here:
> - **Quick Overview** → This README (you are here)
> - **Comprehensive Guide** → [📖 README_COMPREHENSIVE.md](./README_COMPREHENSIVE.md) - Detailed architecture, schemas, APIs, and examples
> - **Configuration** → [⚙️ src/.env.example](./src/.env.example) - All configuration options with comments

---

## 🎯 What is MINI-TOURISM RAG?

**RAG (Retrieval-Augmented Generation)** combines document retrieval with LLM intelligence:

```
📄 Your Documents → 🔍 Search → 📍 Retrieve Relevant Chunks → 🤖 LLM + Context → 💬 Smart Response
```

**MINI-TOURISM RAG** provides:
- 📁 **Document Upload & Processing** - PDF, DOCX, TXT, CSV, HTML, MD
- 🔗 **Vector Search** - Find relevant information from your documents
- 🧠 **Smart Memory** - Window, Summary, Entity, and Semantic Cache memory
- 🤖 **Multiple LLM Options** - Cohere, OpenAI, Gemini, or local Llama
- 💾 **Scalable Storage** - MongoDB + Vector DB (Qdrant, FAISS, Pinecone, Chroma)
- 🌍 **Multilingual** - Arabic and English support out of the box

---

## ⭐ Key Features

### 🧠 Five Types of Memory

| Memory | Purpose | Example |
|--------|---------|---------|
| **Vector Memory** | Search your knowledge base semantically | User asks about "pyramids" → finds related content |
| **Window Memory** | Keep recent conversation context | Remembers last 5 messages |
| **Summary Memory** | Auto-compress long conversations | Condenses 50+ messages into summary |
| **Entity Memory** | Extract and remember user facts | "User prefers 5-star hotels" |
| **Semantic Cache** | Return cached responses to similar questions | Same answer for "Best time to visit?" and "When should I go?" |

### 🔍 Intelligent Search

- **Hybrid Search**: Combine vector (meaning) + keyword (exact) search
- **Smart Reranking**: Verify relevance with Cohere's reranker
- **Flexible Chunking**: 6 strategies (fixed, overlapping, recursive, semantic, etc.)
- **Context Awareness**: Automatically reformulate follow-up questions

### 🌍 Multilingual & Provider-Agnostic

```env
# Just change these to switch backends!
GENERATION_BACKEND="COHERE"        # or OPENAI, GEMINI, LLAMA, HUGGINGFACE
EMBEDDING_BACKEND="COHERE"         # or OPENAI, GEMINI, LLAMA, HUGGINGFACE
VECTOR_DB_BACKEND="QDRANT"         # or FAISS, PINECONE, CHROMA
```

---

## 🛠 Tech Stack

| Component | Technologies |
|-----------|---|
| **Backend** | FastAPI, Uvicorn, Async Python |
| **LLM APIs** | Cohere, OpenAI, Gemini, Hugging Face, Llama |
| **Vector DB** | Qdrant, FAISS, Pinecone, Chroma |
| **Database** | MongoDB (sessions, metadata) |
| **Parsing** | PyMuPDF, docx2txt, BeautifulSoup, NLTK |
| **Ranking** | BM25, Cohere Reranker |
| **Monitoring** | Prometheus, Grafana |
| **Infrastructure** | Docker, Docker Compose, Nginx |

---

## 🚀 Quick Start (5 minutes)

### Option 1: Docker Compose (Recommended)

```bash
# Clone & setup
git clone https://github.com/your-org/MINI-TOURISM_RAG.git
cd MINI-TOURISM_RAG

# Configure environment
cp .env.example .env
nano .env  # Add your API keys

# Start all services
docker-compose -f docker/docker-compose.yml up -d

# API is ready at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Option 2: Local Development

```bash
cd src

# Setup Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp ../.env.example ../.env
nano ../.env

# Run (ensure MongoDB is running)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Verify Installation

```bash
# Check API health
curl http://localhost:8000/api/v1/info

# View API documentation
open http://localhost:8000/docs
```

---

## 📝 Basic Usage

### 1️⃣ Upload Documents

```bash
curl -X POST "http://localhost:8000/api/v1/data/upload/my_project" \
  -F "file=@tourism_guide.pdf"
```

Response:
```json
{
  "signal": "FILE_UPLOAD_SUCCESS",
  "file_id": "abc123",
  "asset_id": "507f191e810c19729de860ea"
}
```

### 2️⃣ Process Files into Chunks

```bash
curl -X POST "http://localhost:8000/api/v1/data/process/my_project" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "abc123",
    "chunk_size": 512,
    "overlap_size": 50
  }'
```

### 3️⃣ Index into Vector Database

```bash
curl -X POST "http://localhost:8000/api/v1/nlp/index/push/my_project" \
  -H "Content-Type: application/json" \
  -d '{"do_reset": 1}'
```

### 4️⃣ Chat with Your Documents

```bash
curl -X POST "http://localhost:8000/api/v1/nlp/chat/my_project" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the best tourist attractions?",
    "session_id": "user_123",
    "top_k": 5
  }'
```

Response:
```json
{
  "signal": "CHAT_SUCCESS",
  "message": "Based on the documents, the best attractions are...",
  "sources": [
    {
      "text": "Excerpt from document...",
      "source": "tourism_guide.pdf"
    }
  ]
}
```

---

## 📊 API Endpoints Overview

### Projects
- `GET /api/v1/projects/` - List all projects
- `DELETE /api/v1/projects/{project_id}` - Delete project & all data

### Data Management
- `POST /api/v1/data/upload/{project_id}` - Upload file
- `POST /api/v1/data/process/{project_id}` - Process file into chunks
- `GET /api/v1/data/assets/{project_id}` - List project assets

### NLP & Chat
- `POST /api/v1/nlp/search/{project_id}` - Search documents
- `POST /api/v1/nlp/chat/{project_id}` - Chat with RAG
- `GET /api/v1/nlp/sessions/{session_id}` - Get chat history
- `POST /api/v1/nlp/index/push/{project_id}` - Index chunks into vector DB
- `GET /api/v1/nlp/index/info/{project_id}` - Get index information

👉 **Full API Documentation**: [README_COMPREHENSIVE.md#api-documentation](./README_COMPREHENSIVE.md#-api-documentation)

---

## ⚙️ Configuration

### Quick Setup

1. **Copy template**:
   ```bash
   cp src/.env.example src/.env
   ```

2. **Add your API keys**:
   ```env
   COHERE_API_KEY="your-key"           # if using Cohere
   OPENAI_API_KEY="sk-..."             # if using OpenAI
   MONGODB_URL="mongodb://..."         # MongoDB connection
   ```

3. **Choose backends**:
   ```env
   GENERATION_BACKEND="COHERE"         # Text generation
   EMBEDDING_BACKEND="COHERE"          # Embeddings
   VECTOR_DB_BACKEND="QDRANT"          # Vector storage
   ```

### Popular Configurations

**🔥 Recommended (OpenAI + Pinecone):**
```env
GENERATION_BACKEND="OPENAI"
GENERATION_MODEL_ID="gpt-4"
EMBEDDING_BACKEND="OPENAI"
VECTOR_DB_BACKEND="PINECONE"
USE_RERANK=True
```

**💰 Budget-Friendly (Cohere + Local Vector DB):**
```env
GENERATION_BACKEND="COHERE"
EMBEDDING_BACKEND="COHERE"
VECTOR_DB_BACKEND="QDRANT"
USE_RERANK=False
```

**🔒 Privacy-First (Local LLM):**
```env
GENERATION_BACKEND="LLAMA"
EMBEDDING_BACKEND="LLAMA"
LLAMA_API_URL="http://localhost:11434"
VECTOR_DB_BACKEND="QDRANT"
```

👉 **Complete Configuration Guide**: [README_COMPREHENSIVE.md#configuration-guide](./README_COMPREHENSIVE.md#%EF%B8%8F-configuration-guide)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Client Application                     │
└────────────────────────┬────────────────────────────────┘
                    HTTP │ REST
┌────────────────────────▼───────────────────────────────┐
│               FastAPI Server                           │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Data Routes   │  │ NLP Routes   │  │ Project Rtes │ │
│  └───────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
└──────────┼─────────────────┼─────────────────┼─────────┘
           │                 │                 │
    ┌──────▼────────┐ ┌──────▼───────┐ ┌───────▼──────┐
    │ File Upload   │ │ NLP Control  │ │ Project Ctrl │
    │ Processing    │ │ Embedding    │ │ Management   │
    │ Validation    │ │ Vector Search│ │              │
    └──────┬────────┘ └──────┬───────┘ └───────┬──────┘
           │                 │                 │
      ┌────▼───────┐  ┌──────▼─────────┐  ┌────▼──────┐
      │ File       │  │ LLM Providers  │  │ MongoDB   │
      │ Storage    │  │ ├─ Cohere      │  │ (Data)    │
      │            │  │ ├─ OpenAI      │  │           │
      │            │  │ ├─ Gemini      │  └───────────┘
      └────────────┘  │ └─ Llama       │
                      └───────┬────────┘
                              │
                      ┌───────▼──────────┐
                      │ Vector DB        │
                      │ ├─ Qdrant        │
                      │ ├─ FAISS         │
                      │ ├─ Pinecone      │
                      │ └─ Chroma        │
                      └──────────────────┘
```

👉 **Detailed Architecture**: [README_COMPREHENSIVE.md#architecture--system-design](./README_COMPREHENSIVE.md#%EF%B8%8F-architecture--system-design)

---

## 📊 Database Schema

### Collections

| Collection | Purpose | Key Fields |
|---|---|---|
| **projects** | Project metadata | `project_id`, `created_at` |
| **assets** | Uploaded files | `asset_name`, `asset_size`, `asset_project_id` |
| **chunks** | Text chunks from documents | `chunk_text`, `chunk_metadata`, `chunk_order` |
| **chat_sessions** | Conversation sessions | `session_id`, `summary`, `message_count` |
| **chat_messages** | Individual messages | `session_id`, `role`, `text` |

👉 **Detailed Schema**: [README_COMPREHENSIVE.md#database-schema](./README_COMPREHENSIVE.md#-database-schema)

---

## 📁 Project Structure

```
src/
├── main.py                      # FastAPI app initialization
├── controllers/                 # Business logic
│   ├── DataController.py        # File upload & processing
│   ├── NLPController.py         # Search, chat, embedding
│   └── ...
├── models/                      # Data models & schemas
│   ├── db_schemes/              # MongoDB schemas
│   └── enums/                   # Enumerations
├── routes/                      # API endpoints
│   ├── data.py                  # File operations
│   ├── nlp.py                   # Search & chat
│   └── projects.py              # Project management
├── stores/                      # Provider factories
│   ├── llm/providers/           # LLM implementations
│   └── vectordb/providers/      # VectorDB implementations
└── utils/                       # Monitoring, metrics
```

👉 **Full Project Structure**: [README_COMPREHENSIVE.md#-project-structure](./README_COMPREHENSIVE.md#-project-structure)

---

## 🔄 Data Flow

### Ingestion Flow
```
Upload → Validate → Store File → Extract Text → Apply Chunking → Store Chunks → Generate Embeddings → Index in Vector DB
```

### Query Flow
```
User Query → Generate Embedding → Vector Search → Optional Reranking → Get Memory Context → Build Prompt → LLM Generation → Response
```

👉 **Detailed Flows**: [README_COMPREHENSIVE.md#data-flow--pipelines](./README_COMPREHENSIVE.md#-data-flow--pipelines)

---

## 📚 Usage Examples

### Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# Upload document
with open("guide.pdf", "rb") as f:
    r = requests.post(f"{BASE_URL}/data/upload/tourism", files={"file": f})
    file_id = r.json()["file_id"]

# Process into chunks
requests.post(f"{BASE_URL}/data/process/tourism", json={
    "file_id": file_id,
    "chunk_size": 512,
    "overlap_size": 50
})

# Index into vector DB
requests.post(f"{BASE_URL}/nlp/index/push/tourism", json={"do_reset": 1})

# Chat
response = requests.post(f"{BASE_URL}/nlp/chat/tourism", json={
    "query": "Best attractions?",
    "session_id": "user123"
})
print(response.json()["message"])
```

👉 **More Examples**: [README_COMPREHENSIVE.md#-usage-examples](./README_COMPREHENSIVE.md#-usage-examples)

---

## 🐳 Deployment

### Docker Compose

```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f minirag

# Stop services
docker-compose -f docker/docker-compose.yml down
```

### Services Included
- **MongoDB** - Document storage
- **Qdrant** - Vector database
- **FastAPI** - Main application
- **Nginx** - Reverse proxy & load balancer
- **Prometheus** - Metrics collection

👉 **Production Deployment**: [README_COMPREHENSIVE.md#-deployment](./README_COMPREHENSIVE.md#-deployment)

---

## 📊 Monitoring

The application exposes Prometheus metrics at `/metrics`

**Key Metrics:**
- `chunking_latency_seconds` - Document chunking time
- `embedding_generation_latency_seconds` - Embedding generation time
- `vector_search_latency_seconds` - Search latency
- `llm_generation_latency_seconds` - LLM response time
- `chat_total_latency_seconds` - Total chat latency
- `api_requests_total` - Total API requests
- `api_errors_total` - Total errors

View dashboard: http://localhost:3000 (with Grafana)

👉 **Monitoring Details**: [README_COMPREHENSIVE.md#-monitoring--metrics](./README_COMPREHENSIVE.md#-monitoring--metrics)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](./LICENSE) for details.

---

## 🔗 Links

- 📖 **Comprehensive Documentation**: [README_COMPREHENSIVE.md](./README_COMPREHENSIVE.md)
- ⚙️ **Configuration Guide**: [src/.env.example](./src/.env.example)
- 📮 **Postman Collection**: [src/assets/mini-rag-app.postman_collection.json](./src/assets/mini-rag-app.postman_collection.json)
- 🐳 **Docker Setup**: [docker/README.md](./docker/README.md)
- 🚀 **FastAPI Docs**: https://fastapi.tiangolo.com/
- 📚 **Vector DB Docs**: https://qdrant.tech/documentation/

---

**Made with ❤️ for Tourism AI**