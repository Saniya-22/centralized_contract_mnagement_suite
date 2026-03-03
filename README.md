# GovGig AI - Python LangGraph Backend

Python backend for GovGig AI using LangGraph for multi-agent orchestration and FastAPI for serving.

## ✅ Migration Status

This repository has been successfully migrated to the Python backend.

- Runtime API: `src/api/main.py` (FastAPI on port `8000`)
- Orchestration: `src/agents/orchestrator.py` (LangGraph)
- Retrieval: `src/agents/data_retrieval.py` + `src/tools/vector_search.py`
- Legacy Node.js runtime routes are not part of the active production path

## 🎯 Overview

This is a **Phase 1** implementation that includes:

- ✅ **LangGraph Multi-Agent System**: Orchestrated workflows with state management
- ✅ **Query Classifier (Router)**: Smart intent categorization (e.g. `clause_lookup`, `regulation_search`)
- ✅ **JWT Authentication**: Secured endpoints with token-based access
- ✅ **Conversation Persistence**: LangGraph state stored in PostgreSQL (Checkpointing)
- ✅ **Scoped API Response Caching**: Postgres-based caching for identical queries (scoped by user + thread, 24h TTL)
- ✅ **Data Retrieval Agent**: Vector search with hybrid (dense + sparse) embeddings
- ✅ **Reflection + Self-Healing Loop**: Retrieval critique and automatic recovery when confidence is low
- ✅ **Optional Sovereign-AI Guardrail Layer**: Post-synthesis safety verdict (`allow/warn/block`) with soft or hard enforcement modes
- ✅ **Direct Clause Lookup**: Optimized database fetching for exact clause references
- ✅ **FastAPI Backend**: REST and WebSocket APIs
- ✅ **PostgreSQL + pgvector**: Existing vector database integration
- ✅ **Streaming Support**: Real-time response streaming
- ✅ **Pilot Quality Signals (Non-Blocking)**: `quality_metrics` + `low_confidence` output for reviewer visibility
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

### 1. Create Virtual Environment (Python 3.11+ required)

```bash
# From repository root — use Python 3.11+ (e.g. python3.11 from Homebrew)
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Switch to project environment

If a `venv` already exists (created with Python 3.11+), activate it:

```bash
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate   # Windows
```

Then confirm: `python --version` should show **3.11** or higher. The project uses type hints that require Python 3.10+ (3.11 recommended).

### 3. Install Dependencies

```bash
pip install -r src/requirements.txt
```

### 4. Configure Environment

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

Optional guardrail integration:
```env
SOVEREIGN_GUARD_ENABLED=true
SOVEREIGN_GUARD_BASE_URL=http://localhost:8001
SOVEREIGN_GUARD_DETECT_PATH=/detect
SOVEREIGN_GUARD_TIMEOUT_SECONDS=3.0
SOVEREIGN_GUARD_FAIL_OPEN=true
SOVEREIGN_GUARD_BLOCK_MODE=soft
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

# Query endpoint (JWT required)
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer <your_jwt_token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the requirements for small business set-asides in FAR?"}'
```

### 7. Access Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

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

### Quality Gate (Deploy Check)

Run a real end-to-end quality gate against your configured DB/OpenAI services.
The command exits non-zero on regression, so it can be used directly in CI/CD.

```bash
python scripts/quality_gate.py \
  --queries-file scripts/quality_queries.json \
  --min-pass-rate 0.85 \
  --max-fallback-rate 0.20 \
  --min-citation-rate 0.85 \
  --max-avg-latency 12 \
  --report-out /tmp/quality_gate_report.json
```

### Pilot Testing Notes

- Reflection loop remains enabled for low-confidence retrieval recovery (`src/reflection/` + `src/agents/data_retrieval.py`).
- Answers are not hard-blocked by pilot guardrails; instead API returns:
  - `quality_metrics`: citation coverage, groundedness, evidence, composite quality.
  - `low_confidence`: soft reviewer signal when answer grounding looks weak.
- Recommended pilot workflow:
  - Treat `low_confidence=true` responses as reviewer-required.
  - Track pass-rate/fallback-rate/citation-rate from `scripts/quality_gate.py`.

## 📡 API Endpoints

### REST API

#### `GET /api/v1/health`
Health check endpoint

#### `POST /api/v1/query`
Query regulatory documents (**Requires JWT `Authorization: Bearer <token>` header**)

**Request:**
```json
{
  "query": "What are the requirements for cost accounting standards?",
  "thread_id": "session_abc_123",
  "person_id": "user123",
  "history": [],
  "cot": true
}
```

**Response:**
```json
{
  "response": "According to FAR 30.201...",
  "documents": [...],
  "confidence": 0.89,
  "quality_metrics": {
    "citation_coverage": 0.92,
    "groundedness_score": 0.81,
    "evidence_score": 0.64,
    "quality_score": 0.78,
    "low_confidence": false
  },
  "low_confidence": false,
  "agent_path": ["DataRetrievalAgent: Starting..."],
  "regulation_types": ["FAR"],
  "errors": []
}
```

`low_confidence=true` means the answer was still returned, but evidence/citation grounding looked weak and should be reviewed.

### WebSocket API

#### `WS /ws/chat`
Streaming chat endpoint (**Requires `token` field in the first message**)

**Send:**
```json
{
  "token": "your_jwt_token",
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

### Direct Clause Lookup API
Fast access to a specific clause without executing the LLM and RAG pipeline.

#### `GET /api/v1/clause/{clause_reference}`
**Requires JWT `Authorization: Bearer <token>` header**

**Example:**
```bash
curl -H "Authorization: Bearer <your_jwt_token>" http://localhost:8000/api/v1/clause/FAR%2052.219-8
```

**Response:**
```json
{
  "found": true,
  "clause_reference": "FAR 52.219-8",
  "clause": { ... },
  "context": "..."
}
```

## ⚡ Performance & Persistence

### API Caching
The application uses PostgreSQL to store a cache of API responses.
- **Mechanism:** Queries are hashed (SHA-256) and stored with final JSON response.
- **Scope:** Cache key includes query + CoT + user/thread scope to avoid cross-user leakage.
- **TTL:** Default expiration is **24 hours**.
- **Impact:** Repeat queries return in **< 50ms**, bypassing the entire LLM pipeline.

### Conversation Persistence
Using LangGraph's `PostgresSaver`, all conversation states are persisted.
- **Threads:** Use the `thread_id` field in requests to maintain state across different sessions.
- **Resilience:** Conversations can be resumed even after server restarts.

### Reflection + Self-Healing
When retrieval quality is low, the system runs a reflection loop:
1. Critique retrieval confidence and regulation alignment.
2. Expand/refine query.
3. Re-run search and merge supplemental documents.

This flow is implemented in:
- `src/reflection/`
- `src/agents/data_retrieval.py`

## 🔧 Configuration

Key configuration options in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `gpt-4o-mini` | Tool-selector model |
| `SYNTHESIZER_MODEL` | `gpt-4o-mini` | Final response synthesis model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `TEMPERATURE` | `0.2` | LLM temperature |
| `RETRIEVAL_TOP_K` | `6` | Primary retrieval depth |
| `RAG_TOKEN_LIMIT` | `3200` | Max context tokens before synthesis |
| `REFLECTION_THRESHOLD` | `0.35` | Reflection trigger threshold |
| `MAX_ITERATIONS` | `10` | Max LangGraph iterations |
| `LOG_LEVEL` | `INFO` | Logging level |

### Authentication Note
`/api/v1/query` and `/api/v1/clause/*` require a valid JWT.
This service validates JWTs; token issuance is expected from your auth flow/upstream system.

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
govgig-feature-python-ai-assistant/
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

Migration is complete for this repository.

- Use only the Python backend on port `8000`
- Do not use legacy Node.js routes or scripts for runtime operations
- Use `ingest_python/pipeline.py` for indexing and re-indexing

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
# Ensure you're in repository root
pwd

# Run with module path
python -m src.api.main
```

## ✅ Current Validation Snapshot (March 1, 2026)

- Python test suite: `35 passed`
- Reflection tests: `7/7 passed`
- E2E quality gate (post soft-quality update, 7 cases): `PASS`
  - `pass_rate=100.00%`
  - `fallback_rate=0.00%`
  - `citation_rate=100.00%`
  - `avg_latency=8.40s`
- Pilot quality diagnostics (same 7-case set):
  - `low_confidence` flagged in `2/7` responses (soft review signal, not blocked)

### Port Already in Use

```bash
# Change port in .env or command line
uvicorn src.api.main:app --port 8001
```

## 📝 Next Steps (Phase 2+)

- [ ] Add Document Analysis Agent
- [ ] Add Document Generation Agent
- [ ] Implement Help Agent
- [ ] Add feedback collection
- [ ] Add LangGraph visualization
- [ ] Add performance metrics
- [ ] Add rate limiting

## 📄 License

MIT License - See root LICENSE file

## 🤝 Contributing

Contributions welcome! Please ensure:
- Tests pass (`pytest`)
- Code is formatted (`black`)
- Types are correct (`mypy`)
- Documentation is updated
