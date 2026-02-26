# GovGig AI - Python LangGraph Backend

Python backend for GovGig AI using LangGraph for multi-agent orchestration and FastAPI for serving.

## 🎯 Overview

This is a **Phase 1** implementation that includes:

- ✅ **LangGraph Multi-Agent System**: Orchestrated workflows with state management
- ✅ **Data Retrieval Agent**: Vector search with hybrid (dense + sparse) embeddings
- ✅ **FastAPI Backend**: REST and WebSocket APIs
- ✅ **PostgreSQL + pgvector**: Existing vector database integration
- ✅ **Streaming Support**: Real-time response streaming
- ✅ **Chain-of-Thought**: Optional reasoning mode
- ⏳ **Additional Agents**: Document analysis, generation (coming in Phase 2+)

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│         FastAPI Application         │
│  (REST API + WebSocket Endpoints)   │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│      GovGigOrchestrator             │
│      (LangGraph StateGraph)         │
│                                     │
│  ┌─────────┐   ┌──────────────┐   │
│  │ Router  │──▶│ Data         │   │
│  │  Node   │   │ Retrieval    │   │
│  └─────────┘   │ Agent        │   │
│                └──────┬───────┘   │
│                       │            │
│                       ▼            │
│                ┌──────────────┐   │
│                │ Synthesizer  │   │
│                │   Node       │   │
│                └──────────────┘   │
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│       Tools Layer                   │
│  • Vector Search (hybrid)           │
│  • Embedding Generation             │
│  • Document Formatting              │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│    PostgreSQL + pgvector            │
│  • embeddings_dense (OpenAI)        │
│  • embeddings_sparse (BM25/FTS)     │
└─────────────────────────────────────┘
```

## 📋 Prerequisites

- **Python**: 3.11+ (3.11 recommended)
- **PostgreSQL**: 14+ with pgvector extension
- **OpenAI API Key**: For embeddings and LLM
- **Existing Data**: Ingested documents in PostgreSQL (from `ingest_python/`)

## 🚀 Quick Start

### 1. Create Virtual Environment

```bash
cd backend_python
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your actual values
```

Minimum required variables:
```env
OPENAI_API_KEY=your-key-here
PG_HOST=localhost
PG_DB=daedalus
PG_USER=postgres
PG_PASSWORD=your-password
JWT_SECRET_KEY=your-secret-key
```

### 4. Verify Database Connection

```bash
python -c "from src.db.connection import test_connection; print('✅ Connected' if test_connection() else '❌ Failed')"
```

### 5. Run the Application

```bash
# Development mode with auto-reload
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 6. Test the API

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Query endpoint
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the requirements for small business set-asides in FAR?"}'
```

### 7. Access Documentation

- **Swagger UI**: http://localhost:8000/api/v1/docs
- **ReDoc**: http://localhost:8000/api/v1/redoc

## 🐳 Docker Deployment

### Build and Run with Docker Compose

```bash
# Create .env file with required variables
cp .env.example .env

# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down
```

### Docker Only (without postgres)

```bash
# Build image
docker build -t govgig-backend .

# Run container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -e PG_HOST=host.docker.internal \
  -e PG_PASSWORD=your-password \
  govgig-backend
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_vector_search.py -v
```

## 📡 API Endpoints

### REST API

#### `GET /api/v1/health`
Health check endpoint

#### `POST /api/v1/query`
Query regulatory documents

**Request:**
```json
{
  "query": "What are the requirements for cost accounting standards?",
  "person_id": "user123",
  "history": [],
  "cot": false
}
```

**Response:**
```json
{
  "response": "According to FAR 30.201...",
  "documents": [...],
  "confidence": 0.89,
  "agent_path": ["DataRetrievalAgent: Starting..."],
  "regulation_types": ["FAR"],
  "errors": []
}
```

### WebSocket API

#### `WS /ws/chat`
Streaming chat endpoint

**Send:**
```json
{
  "query": "Find safety requirements for excavation",
  "cot": true
}
```

**Receive (streaming):**
```json
{"type": "step", "data": {...}}
{"type": "complete", "data": {"response": "..."}}
{"type": "done", "data": "[DONE]"}
```

## 🔧 Configuration

Key configuration options in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `gpt-4o` | OpenAI model for agents |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `TEMPERATURE` | `0.2` | LLM temperature |
| `DENSE_TOP_K` | `10` | Dense search results |
| `HYBRID_DENSE_WEIGHT` | `0.7` | Dense score weight |
| `MAX_ITERATIONS` | `10` | Max LangGraph iterations |
| `LOG_LEVEL` | `INFO` | Logging level |

## 📊 Monitoring

### LangSmith Integration (Optional)

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=govgig-ai
```

View traces at: https://smith.langchain.com/

### Logging

Logs are output to console with configurable level:

```bash
LOG_LEVEL=DEBUG python -m uvicorn src.api.main:app
```

## 🛠️ Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

### Project Structure

```
backend_python/
├── src/
│   ├── agents/          # LangGraph agents
│   │   ├── base.py
│   │   ├── data_retrieval.py
│   │   ├── orchestrator.py
│   │   └── prompts.py
│   ├── api/             # FastAPI application
│   │   └── main.py
│   ├── db/              # Database connections
│   │   ├── connection.py
│   │   └── queries.py
│   ├── state/           # LangGraph state
│   │   └── graph_state.py
│   ├── tools/           # Agent tools
│   │   ├── vector_search.py
│   │   └── llm_tools.py
│   └── config.py        # Configuration
├── tests/               # Test suite
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🔄 Migration from Node.js

This backend can run **alongside** the existing Node.js backend:

1. Node.js continues on port 3000
2. Python backend runs on port 8000
3. Frontend can connect to either endpoint
4. Gradual migration of features

## 🐛 Troubleshooting

### Database Connection Issues

```bash
# Test connection
python -c "from src.db.connection import test_connection; test_connection()"

# Check pgvector extension
psql -U postgres -d daedalus -c "SELECT * FROM pg_extension WHERE extname='vector';"
```

### Import Errors

```bash
# Ensure you're in the backend_python directory
cd backend_python

# Run with module path
python -m src.api.main
```

### Port Already in Use

```bash
# Change port in .env or command line
uvicorn src.api.main:app --port 8001
```

## 📝 Next Steps (Phase 2+)

- [ ] Add Document Analysis Agent
- [ ] Add Document Generation Agent
- [ ] Implement Help Agent
- [ ] Add user authentication
- [ ] Add feedback collection
- [ ] Add LangGraph visualization
- [ ] Add performance metrics
- [ ] Add rate limiting
- [ ] Add caching layer

## 📄 License

MIT License - See root LICENSE file

## 🤝 Contributing

Contributions welcome! Please ensure:
- Tests pass (`pytest`)
- Code is formatted (`black`)
- Types are correct (`mypy`)
- Documentation is updated
