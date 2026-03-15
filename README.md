# GovGig AI - Python LangGraph Backend

Regulatory RAG for **FAR**, **DFARS**, and **EM 385-1-1**: LangGraph multi-agent orchestration (router → retrieval → synthesis), hybrid vector search (dense + sparse), JWT-secured FastAPI, and optional Streamlit dashboard. See `docs/` for full capability list and design docs.

## Architecture

```text
┌──────────────────────────────────────────────────┐
│              FastAPI Application                  │
│   REST: POST /api/v1/query   WS: /ws/chat        │
│   Auth: JWT  │  Rate Limit: 10 req/min/user      │
└───────────────────────┬──────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────┐
│          GovGigOrchestrator (LangGraph)           │
│  Router → DataRetrieval (hybrid + reflection)    │
│  → LetterDrafter / Synthesizer + SovereignGuard   │
└───────────────────────┬──────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────┐
│            PostgreSQL + pgvector                  │
│  embeddings_dense, FTS, RRF, checkpointer, cache │
└──────────────────────────────────────────────────┘
```

## Project Structure

```text
govgig-feature-python-ai-assistant/
├── src/                        # Main application (FastAPI, LangGraph, RAG)
│   ├── agents/                 # Orchestrator, data_retrieval, prompts
│   ├── api/                    # FastAPI app, auth
│   ├── db/                     # Pool, vector search, queries, cache
│   ├── reflection/             # Critique, query expansion, self-healing
│   ├── services/               # Reranker, sovereign_guard
│   ├── state/                  # Graph state (TypedDict)
│   ├── tools/                  # Vector search, query_classifier, llm_tools
│   ├── scripts/                # DB setup: SQL + legacy Node (initDB, ingestRegulations, etc.)
│   ├── config.py               # App config (Pydantic Settings, env)
│   └── requirements.txt
├── ingest_python/              # PDF ingestion: chunk → embed → store
│   ├── config.py               # Ingestion config (env: chunk sizes, DATABASE_URL, etc.)
│   ├── pipeline.py
│   └── parsing/
├── dashboard/                  # Streamlit dashboard
├── scripts/                    # Python tooling: run_test_queries, dedup_embeddings,
│                               # quality_gate, benchmark_latency, gen_test_token, migrations
├── tests/                      # Pytest + tests/unified_test.py (E2E smoke)
├── docs/                       # Architecture and design docs
├── results/                    # Test run outputs (gitignored)
├── infra/                      # Terraform (AWS)
├── run.sh, run_dashboard.sh
├── Dockerfile, docker-compose.yml
└── .env.example
```

**Notes:**
- **Config:** App config = `src/config.py` (Pydantic). Ingestion config = `ingest_python/config.py` (env vars).
- **Scripts:** Root `scripts/` = Python tooling (tests, quality gate, latency benchmark). `src/scripts/` = DB setup (SQL + legacy Node).
- **Stack:** Primary API is FastAPI (Python). Node in `src/scripts/` is for DB/legacy setup only.

## Prerequisites

- **Python** 3.11+
- **PostgreSQL** 14+ with pgvector
- **OpenAI API key** (embeddings + synthesis)

## Quick Start

```bash
# One-command setup and run
bash run.sh
```

Manual:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r src/requirements.txt
cp .env.example .env   # Edit with OPENAI_API_KEY, PG_PASSWORD, JWT_SECRET_KEY
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Verify:

```bash
curl http://localhost:8000/api/v1/health
python scripts/gen_test_token.py   # Use token in Authorization header
python tests/unified_test.py       # E2E smoke (server must be running)
```

Dashboard: `bash run_dashboard.sh` (Streamlit on port 8501).

## API Summary

- **GET /api/v1/health** — Status and DB check.
- **POST /api/v1/query** *(JWT)* — Main query endpoint (request body: `query`, optional `thread_id`, `person_id`, `history`, `cot`).
- **GET /api/v1/clause/{ref}** *(JWT)* — Direct clause lookup.
- **WS /ws/chat** — Streaming chat (first message: `token`, `query`).
- Swagger: http://localhost:8000/docs

## Configuration

Required env: `OPENAI_API_KEY`, `PG_PASSWORD`, `JWT_SECRET_KEY`. All settings in `src/config.py` (Pydantic). Key options: `RETRIEVAL_TOP_K`, `RERANKER_ENABLED`, `CORS_ORIGINS`, `PG_*`. See `.env.example` and `src/config.py` for full list.

## Testing

```bash
pytest                              # All tests
pytest tests/test_query_classifier.py -v
pytest tests/test_api.py -v         # API tests
python tests/unified_test.py         # E2E smoke (server running)
```

Query list runs (output to `results/`):

```bash
python scripts/run_test_queries.py --api --list scripts/queries.txt -o results/results.csv --csv
```

Latency benchmark: `python scripts/benchmark_latency.py`.

## Data Ingestion

```bash
cd ingest_python
pip install -r requirements.txt
python pipeline.py
```

Uses `ingest_python/config.py` and env (e.g. `DATABASE_URL`, chunk sizes). See `docs/INGESTION_CHUNKING_STRATEGY.md`.

## Docker

```bash
cp .env.example .env
docker-compose up -d
```

## AWS (Terraform)

Infra in `infra/`. Deploy: `./scripts/deploy.sh`. After RDS is up, run `src/scripts/*.sql` and the ingest pipeline. See `docs/` and infra README for details.

## More

- Full capabilities and pipeline details: **docs/**
- Quality gate: `python scripts/quality_gate.py --queries-file scripts/quality_queries.json ...`
