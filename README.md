# GovGig AI - Python LangGraph Backend

Regulatory document RAG system for **FAR**, **DFARS**, and **EM 385-1-1** powered by LangGraph multi-agent orchestration, hybrid vector search (dense + sparse), and FastAPI.

## System Capabilities

| Feature | Status | Details |
|---------|--------|---------|
| **Multi-Agent Orchestration** | Supported | LangGraph `StateGraph` with Router -> DataRetrieval -> Synthesizer |
| **Deterministic Query Classifier** | Supported | Zero-LLM routing: `clause_lookup`, `regulation_search`, `out_of_scope` |
| **Hybrid Retrieval (Dense + Sparse)** | Supported | OpenAI `text-embedding-3-small` + BM25/FTS fused via RRF |
| **Reflection & Self-Healing** | Supported | Auto critique -> query expansion -> re-search on low-confidence results |
| **Optional LLM Reranker** | Supported | GPT-4o-mini relevance scoring (toggle via `RERANKER_ENABLED`) |
| **JWT Authentication** | Supported | All query endpoints secured with Bearer token |
| **In-Memory Rate Limiter** | Supported | 10 req/min per user (sliding window) |
| **Conversation Persistence** | Supported | LangGraph state in PostgreSQL via `PostgresSaver` |
| **API Response Caching** | Supported | SHA-256 hashed, scoped by user + thread, 24h TTL |
| **Sovereign-AI Guardrails** | Supported | Post-synthesis safety verdict (`allow`/`warn`/`block`, soft or hard mode) |
| **Direct Clause Lookup** | Supported | DB-direct fetch for exact clause references (no LLM) |
| **Quality Metrics** | Supported | `citation_coverage`, `groundedness_score`, `evidence_score`, `quality_score` |
| **Streamlit Dashboard** | Supported | Glassmorphism UI with real-time performance gauges |
| **REST + WebSocket APIs** | Supported | Non-streaming POST + streaming WS |

## Architecture

```text
┌──────────────────────────────────────────────────┐
│              FastAPI Application                  │
│   REST: POST /api/v1/query                       │
│   WS: /ws/chat    GET: /api/v1/clause/{ref}      │
│   Auth: JWT  │  Rate Limit: 10 req/min/user      │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│          GovGigOrchestrator (LangGraph)           │
│                                                   │
│  ┌──────────────┐    ┌────────────────────────┐  │
│  │   Router     │───>│   DataRetrievalAgent   │  │
│  │ (Deterministic│    │                        │  │
│  │  Classifier) │    │  VectorSearch (hybrid)  │  │
│  └──────────────┘    │  DirectClauseLookup     │  │
│        │             │  ReflectionRAG          │  │
│   out_of_scope       └───────────┬────────────┘  │
│   -> immediate                   │               │
│     refusal                      ▼               │
│                      ┌────────────────────────┐  │
│                      │    Synthesizer Node     │  │
│                      │  GPT-4o-mini (streaming)│  │
│                      │  + Quality Assessment   │  │
│                      │  + SovereignGuard       │  │
│                      └────────────────────────┘  │
└──────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│            PostgreSQL + pgvector                  │
│  * embeddings_dense (OpenAI 1536-dim)            │
│  * FTS (tsvector + ts_rank_cd)                   │
│  * RRF fusion (k=60)                             │
│  * LangGraph checkpointer (conversation state)   │
│  * Response cache (SHA-256, 24h TTL)             │
└──────────────────────────────────────────────────┘
```

## Project Structure

```text
govgig-feature-python-ai-assistant/
├── src/
│   ├── agents/
│   │   ├── base.py              # BaseAgent ABC (non-streaming LLM)
│   │   ├── data_retrieval.py    # Retrieval agent + reflection integration
│   │   ├── orchestrator.py      # LangGraph StateGraph orchestrator
│   │   └── prompts.py           # System prompts for all agents
│   ├── api/
│   │   ├── main.py              # FastAPI app (REST + WS + rate limiter)
│   │   └── auth.py              # JWT validation middleware
│   ├── db/
│   │   ├── connection.py        # PostgreSQL pool + CheckpointerManager
│   │   └── queries.py           # Vector search, RRF, clause lookup, caching
│   ├── reflection/
│   │   ├── critique.py          # RetrievalCritique (confidence + reg mismatch)
│   │   ├── expansion.py         # QueryExpansion (GPT-4o-mini, async)
│   │   └── manager.py           # ReflectionManager coordinator
│   ├── services/
│   │   ├── reranker.py          # GPT-4o-mini LLM reranker (optional)
│   │   └── sovereign_guard.py   # Post-synthesis safety guardrails
│   ├── state/
│   │   └── graph_state.py       # GovGigState (TypedDict + operator.add reducers)
│   ├── tools/
│   │   ├── vector_search.py     # VectorSearchTool (hybrid dense+sparse)
│   │   ├── query_classifier.py  # Deterministic intent classifier (zero LLM)
│   │   └── llm_tools.py         # Embedding, tokenization, formatting utils
│   ├── config.py                # Pydantic Settings (all env vars)
│   └── requirements.txt
├── ingest_python/
│   ├── pipeline.py              # PDF ingestion: extract -> chunk -> embed -> store
│   ├── config.py                # Ingestion configuration
│   └── parsing/
│       ├── classifier.py        # Line-level structural classification
│       └── rules.py             # Regex patterns for FAR/DFARS/EM385
├── dashboard/
│   └── app.py                   # Streamlit glassmorphism dashboard
├── tests/                       # 7 test modules (pytest)
├── scripts/                     # Quality gate, benchmarks, migration tools
├── specifications/              # Source PDF corpus (16 files)
├── unified_test.py              # End-to-end smoke test
├── gen_test_token.py            # JWT token generator for testing
├── run.sh                       # Dev environment setup + server start
├── run_dashboard.sh             # Streamlit dashboard launcher
├── Dockerfile                   # Production container
└── docker-compose.yml           # Full-stack deployment
```

## Prerequisites

- **Python** 3.11+
- **PostgreSQL** 14+ with `pgvector` extension
- **OpenAI API Key** - for embeddings (`text-embedding-3-small`) and synthesis (`gpt-4o-mini`)

## Quick Start

### 1. Setup and Run (recommended)

```bash
# One-command setup: creates venv, installs deps, starts server
bash run.sh
```

This script handles Python version check, virtual environment, dependency installation, `.env` validation, and starts Uvicorn on port 8000.

### 2. Manual Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r src/requirements.txt

cp .env.example .env
# Edit .env with your values (see Configuration section)

python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

### 3. Verify

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Generate a test JWT token
python gen_test_token.py

# Query with token
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the requirements for small business set-asides in FAR?"}'
```

### 4. Run Dashboard

```bash
bash run_dashboard.sh
# Opens Streamlit at http://localhost:8501
```

### 5. API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### `GET /api/v1/health`

Returns system status (database, orchestrator).

### `POST /api/v1/query` *(JWT required)*

Main query endpoint for regulatory document search and synthesis.

**Request:**
```json
{
  "query": "What are the safety requirements for excavation in EM 385?",
  "thread_id": "optional-for-multi-turn",
  "person_id": "user123",
  "history": [],
  "cot": true
}
```

> **Note:** If `thread_id` is omitted, each request automatically gets a unique UUID (fresh state). Provide an explicit `thread_id` only for multi-turn conversations where you want state continuity.

**Response:**
```json
{
  "response": "According to EM 385-1-1 Chapter 25...",
  "documents": [{ "content": "...", "source": "EM385", "section": "25-8", "score": 0.016 }],
  "confidence": 0.89,
  "quality_metrics": {
    "citation_coverage": 0.92,
    "groundedness_score": 0.81,
    "evidence_score": 0.64,
    "quality_score": 0.78,
    "low_confidence": false
  },
  "low_confidence": false,
  "agent_path": ["Router: intent=regulation_search ...", "DataRetrievalAgent: ...", "Synthesizer: ..."],
  "thought_process": ["[DataRetrievalAgent] Regulatory query detected..."],
  "regulation_types": ["EM385"],
  "errors": []
}
```

- `low_confidence: true` means the answer was returned but evidence grounding is weak - flag for human review.
- `errors: []` contains only current-run errors (stale errors from previous sessions are filtered out).

### `GET /api/v1/clause/{clause_reference}` *(JWT required)*

Direct clause lookup - no LLM, database-only.

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/clause/FAR%2052.219-8
```

### `GET /api/v1/analytics/summary` *(JWT required)*

Returns aggregate query analytics for the last N hours.

### `WS /ws/chat`

Streaming chat via WebSocket. First message must contain `token` for authentication.

```json
{ "token": "jwt_token", "query": "Find safety requirements for excavation", "cot": true }
```

Streams `{"type": "token", "data": "..."}` events, followed by `{"type": "complete", ...}` and `{"type": "done"}`.

## Query Processing Pipeline

```text
User Query
    │
    ▼
┌─ QueryClassifier (deterministic, zero-LLM) ──────────────┐
│  Outputs: intent, confidence, regulation_type, clause_ref │
│  Intents: clause_lookup | regulation_search | out_of_scope│
└───────────────────────────────────────────────────────────┘
    │
    ├─ out_of_scope -> Immediate refusal (no DB/LLM calls)
    │
    ├─ clause_lookup -> Direct DB fetch by clause reference
    │                   (e.g., "FAR 52.219-8")
    │
    └─ regulation_search -> Hybrid vector search:
                            1. Dense: cosine similarity (pgvector)
                            2. Sparse: Full-text search (tsvector)
                            3. RRF fusion (k=60)
                            4. Optional LLM reranker
                            5. Token-budget trimming
                                │
                                ▼
                           ┌─ ReflectionRAG ────────────────────┐
                           │  Critique confidence + reg alignment│
                           │  If low -> expand query -> re-search  │
                           │  Merge supplemental documents       │
                           └────────────────────────────────────┘
                                │
                                ▼
                           ┌─ Synthesizer (GPT-4o-mini) ────────┐
                           │  Format documents -> LLM prompt      │
                           │  Assess quality (citation, grounding)│
                           │  Apply low_confidence label if weak  │
                           │  SovereignGuard safety check         │
                           └─────────────────────────────────────┘
```

## Configuration

All settings loaded from `.env` via Pydantic `BaseSettings`.

### Required

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `PG_PASSWORD` | PostgreSQL password |
| `JWT_SECRET_KEY` | JWT signing secret |

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `gpt-4o-mini` | Tool-selector / fallback model |
| `SYNTHESIZER_MODEL` | `gpt-4o-mini` | Response synthesis model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model (1536 dims) |
| `TEMPERATURE` | `0.2` | LLM temperature |

### Retrieval and RAG

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVAL_TOP_K` | `6` | Primary retrieval depth |
| `DENSE_TOP_K` | `10` | Dense search candidates |
| `SPARSE_TOP_K` | `10` | Sparse/FTS search candidates |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `RAG_TOKEN_LIMIT` | `2400` | Max context tokens for synthesis |
| `MAX_DOC_CHARS_FOR_SYNTHESIS` | `1200` | Per-document content trim |
| `RERANKER_ENABLED` | `true` | Toggle LLM reranker (saves ~1s if disabled) |
| `RERANKER_MODEL` | `gpt-4o-mini` | Reranker model |

### Reflection and Self-Healing

| Variable | Default | Description |
|----------|---------|-------------|
| `REFLECTION_THRESHOLD` | `0.35` | Confidence threshold before self-healing |
| `REFLECTION_HEALING_MARGIN` | `0.05` | Skip retries for near-threshold scores |
| `SELF_HEALING_SEARCH_K` | `3` | Per expanded query search depth |
| `SELF_HEALING_MAX_QUERIES` | `1` | Max expanded queries to execute |
| `SELF_HEALING_MAX_DOCS` | `4` | Max additional docs from self-healing |

### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `PG_HOST` | `localhost` | PostgreSQL host |
| `PG_PORT` | `5432` | PostgreSQL port |
| `PG_DB` | `daedalus` | Database name |
| `PG_USER` | `postgres` | Database user |
| `PG_POOL_MIN` | `2` | Connection pool minimum |
| `PG_POOL_MAX` | `10` | Connection pool maximum |

### Sovereign Guard (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `SOVEREIGN_GUARD_ENABLED` | `false` | Enable guardrail layer |
| `SOVEREIGN_GUARD_BLOCK_MODE` | `soft` | `soft` (label) or `hard` (replace response) |
| `SOVEREIGN_GUARD_FAIL_OPEN` | `true` | Allow response if guard is unreachable |

## Testing

### Unit Tests

```bash
pytest                              # Run all 7 test modules
pytest --cov=src --cov-report=html  # With coverage report
pytest tests/test_query_classifier.py -v  # Specific module
```

### End-to-End Smoke Test

```bash
python unified_test.py
```

Runs 3 test queries (Small Business, FAR clause lookup, EM-385 safety) against the running server, validates responses, and reports latency.

### Quality Gate (CI/CD)

```bash
python scripts/quality_gate.py \
  --queries-file scripts/quality_queries.json \
  --min-pass-rate 0.85 \
  --max-fallback-rate 0.20 \
  --min-citation-rate 0.85 \
  --max-avg-latency 12 \
  --report-out /tmp/quality_gate_report.json
```

## Validation Snapshot

| Metric | Result |
|--------|--------|
| **Unified E2E** | 3/3 PASS |
| **Small Business Search** | PASS 7.81s |
| **FAR 52.219-8(a) Lookup** | PASS 4.22s |
| **EM-385 Safety** | PASS 4.74s |
| **Average Latency** | 5.59s |
| **Stale State Errors** | 0 (fixed via unique thread_ids + state slicing) |

## Data Ingestion

The ingestion pipeline lives in `ingest_python/`:

```bash
cd ingest_python
pip install -r requirements.txt
python pipeline.py
```

**Pipeline**: PDF -> Text extraction (PyMuPDF) -> Section-aware chunking (clause-boundary splitting) -> Embedding (OpenAI `text-embedding-3-small`) -> PostgreSQL/pgvector

**Corpus**: 16 PDFs (FAR, DFARS, EM 385-1-1) - 3,462 pages, ~1.27M words.

**Chunking**: Target 250-550 tokens per chunk, max cap 800 tokens, 200-token overlap, clause-boundary splitting at 700 tokens.

## Docker

```bash
cp .env.example .env
docker-compose up -d        # Full stack
docker-compose logs -f backend
```

Or standalone:

```bash
docker build -t govgig-backend .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -e PG_HOST=host.docker.internal \
  -e PG_PASSWORD=your-password \
  govgig-backend
```

## Monitoring

### LangSmith (Optional)

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=govgig-ai
```

### Logging

```bash
LOG_LEVEL=DEBUG python -m uvicorn src.api.main:app
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Database connection failed | `python -c "from src.db.connection import test_connection; test_connection()"` |
| pgvector missing | `psql -d daedalus -c "CREATE EXTENSION IF NOT EXISTS vector;"` |
| Port 8000 in use | `lsof -ti:8000 \| xargs kill -9` or use `--port 8001` |
| Import errors | Ensure you're running from repository root |
| Stale errors in response | Use unique `thread_id` per session or omit it (auto-UUID) |

## Pilot Testing Notes

- **Low-confidence handling**: Answers are never hard-blocked by default. `low_confidence: true` signals weak evidence grounding - treat as reviewer-required in pilot workflows.
- **Rate limiting**: 10 requests per minute per authenticated user. Returns HTTP 429 if exceeded.
- **State isolation**: Each REST request gets a fresh LangGraph state by default. The orchestrator slices accumulated lists to prevent stale data from prior sessions from leaking into responses.
- **Recommended workflow**: Track `quality_metrics` pass rates using `scripts/quality_gate.py`.

## License

MIT License - See root LICENSE file.

