# 🚀 MINI- RAG: Advanced AI Retrieval System

**MINI- RAG** is a high-performance, feature-rich Retrieval-Augmented Generation (RAG) system built with **FastAPI**. It is designed to transform static documents into conversational AI experts capable of remembering user context, handling multiple languages, and delivering precise, cross-referenced answers.

---

## 🌟 Key Features

### 🧠 Intelligent Conversational Memory
*   **Window Memory**: Retains the last $N$ turns for immediate context.
*   **Summary Memory**: Automatically compresses long conversations into concise summaries to maintain long-term context without hitting token limits.
*   **Entity Memory**: Extracts and remembers specific facts about the user (e.g., "My name is Yasser").
*   **Context-Aware Query Condensation**: Automatically re-writes follow-up questions (e.g., "Where was he born?") into standalone queries ("Where was Mohamed Salah born?") for superior search results.

### 🔍 Advanced Retrieval & Search
*   **Hybrid Search**: Combines Dense Vector Search (Meaning) with BM25 Keyword Search (Exact Matches).
*   **Cohere Re-ranking**: Uses state-of-the-art re-rankers to verify the relevance of retrieved chunks before providing them to the LLM.
*   **Semantic Caching**: In-memory vector caching for lightning-fast responses to repeated or similar queries.
*   **Flexible Chunking**: Supports Fixed-Size, Overlapping, Semantic, and Sentence-based splitting (NLTK).

### 🌍 Multilingual & Multi-Provider
*   **Native Arabic Support**: Full prompt engineering and template support for both Arabic and English.
*   **Agnostic Backend**: Switch between **Cohere**, **OpenAI**, **Gemini**, or **Local LLMs** (via Ollama) by simply changing an environment variable.

---

## 🛠 Tech Stack

| Category          | Technology                                                                 |
|-------------------|----------------------------------------------------------------------------|
| **Framework**     | [FastAPI](https://fastapi.tiangolo.com/)                                   |
| **Vector DB**     | [Qdrant](https://qdrant.tech/) / [FAISS](https://github.com/facebookresearch/faiss) |
| **NoSQL DB**      | [MongoDB](https://www.mongodb.com/) (Session & Metadata Storage)           |
| **NLP**           | [NLTK](https://www.nltk.org/) (Text Processing)                            |
| **LLM APIs**      | Cohere, OpenAI, Gemini                                                     |
| **Infrastructure**| Docker, Docker Compose                                                     |

---

## 🚀 Getting Started

### 1. Prerequisites
*   Python 3.9+
*   Docker & Docker Compose

### 2. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd MINI-_RAG

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r src/requirements.txt
```

### 3. Environment Setup
Copy the example environment file and fill in your API keys:
```bash
cp src/.env.example src/.env
```

| Variable | Description |
|----------|-------------|
| `GENERATION_BACKEND` | `COHERE`, `OPENAI`, `GEMINI`, or `LLAMA` |
| `EMBEDDING_BACKEND` | `COHERE`, `OPENAI`, or `LLAMA` |
| `VECTOR_DB_BACKEND` | `QDRANT`, `CHROMA`, `PINECONE`, or `FAISS` |
| `USE_SEMANTIC_CACHE` | Enable/Disable vector caching (`True`/`False`) |
| `USE_RERANK` | Enable/Disable Cohere Re-ranker |

### 4. Run Services
```bash
# Start MongoDB and Qdrant
cd docker
docker-compose up -d

# Start the FastAPI Server
cd ../src
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

---

## 📖 API Usage Summary

### 📂 Document Management
*   `POST /api/v1/data/project/create` - Create a new  project container.
*   `POST /api/v1/data/upload/{project_id}` - Upload files (`.txt`, `.pdf`, `.docx`, etc.).
*   `POST /api/v1/data/process/{project_id}` - Chunk and index uploaded files into the Vector DB.

### 💬 RAG & Chat
*   **Search**: `POST /api/v1/nlp/index/search/{project_id}` - Direct hybrid search over chunks.
*   **Answer**: `POST /api/v1/nlp/index/answer/{project_id}` - Context-aware conversational RAG.
    *   *Payload:*
    ```json
    {
      "text": "When is the best time to visit Egypt?",
      "use_hybrid": true,
      "score_threshold": 0.5,
      "session_id": "unique_user_session"
    }
    ```

---

## 🛡 Security & Design
*   **Validation**: Strict Pydantic schemas for all request/response payloads.
*   **Error Handling**: Integrated `ResponseSignal` system for predictable cross-platform error states.
*   **Architecture**: Decoupled `Stores` layer for easy addition of new LLM or VectorDB providers.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.