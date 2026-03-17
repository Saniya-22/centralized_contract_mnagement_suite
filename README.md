# GovGig AI – Python Backend

A **regulatory RAG and assistant** for federal contracting: **FAR**, **DFARS**, and **EM 385-1-1**. Built with **Python**, **FastAPI**, **LangGraph**, and **PostgreSQL (pgvector)**. Supports clause lookup, regulation search, letter/REA drafting, procedural guidance, and JWT-secured REST/WebSocket APIs.

---

## 1. Overview

GovGig AI helps contractors and staff answer regulatory questions and generate compliance-oriented documents. The system:

- **Looks up** specific clauses by reference (e.g. FAR 52.236-2, DFARS 252.204-7012, EM 385).
- **Searches** regulations using hybrid vector + full-text search with optional reranking and self-healing (reflection/query expansion).
- **Drafts** letters, REAs, and RFIs (Letter Drafter) grounded in retrieved regulations.
- **Answers** procedural and general questions with step-by-step or structured guidance (Synthesizer).
- **Routes** document-type requests: only letter requests go to the Letter Drafter; checklist and form requests use the Synthesizer.

The backend is **Python-only** (FastAPI, LangGraph, OpenAI). Node.js in `src/scripts/` is legacy/DB setup only.

---

## 2. Features

| Feature | Description |
|--------|-------------|
| **Clause lookup** | Exact fetch by regulation + clause number (FAR, DFARS, EM385, 48 CFR, OSHA, etc.). |
| **Regulation search** | Hybrid dense (embeddings) + FTS + RRF; optional OpenAI reranker; reflection/query expansion when confidence is low. |
| **Letter / REA / RFI drafting** | Letter Drafter produces drafts with citations from retrieved chunks. |
| **Procedural guidance** | “How do I…”, “steps” → numbered **Recommended steps** from Synthesizer. |
| **Document routing** | Checklist and form requests → Synthesizer; letter requests → Letter Drafter. |
| **Auth** | JWT (signup/login), rate limit per user, optional admin API key. |
| **Persistence** | Chat history, API response cache, analytics, feedback in PostgreSQL. |
| **Streaming** | WebSocket `/ws/chat` for streaming responses. |

---

## 3. Architecture

```text
┌──────────────────────────────────────────────────┐
│              FastAPI Application                  │
│   REST: POST /api/v1/query   WS: /ws/chat        │
│   Auth: JWT  │  Rate Limit: 10 req/min/user      │
└───────────────────────┬──────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────┐
│          GovGigOrchestrator (LangGraph)           │
│  Router → DataRetrieval (hybrid + reflection)     │
│  → LetterDrafter / Synthesizer + SovereignGuard   │
└───────────────────────┬──────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────┐
│            PostgreSQL + pgvector                   │
│  embeddings_dense, FTS, RRF, cache, chat_history  │
└──────────────────────────────────────────────────┘
```

- **Router:** Classifier decides intent (clause_lookup, regulation_search, out_of_scope, document_request). Clause lookup is served from DB; regulation search goes to Data Retrieval.
- **Data Retrieval:** Hybrid vector + FTS, optional rerank, optional reflection (query expansion + re-search when confidence is low).
- **After retrieval:** Letter requests → Letter Drafter; else → Synthesizer. Optional Sovereign Guard on generated text.
- **Data:** Single PostgreSQL instance: embeddings (dense + sparse), FTS, cache table, chat history, users, analytics.

See **docs/GOVGIG_ARCHITECTURE.md** for diagrams and environment reference.

---

## 4. Project Structure

```text
govgig-feature-python-ai-assistant/
├── src/                        # Main application
│   ├── agents/                 # Orchestrator, data_retrieval, prompts
│   ├── api/                    # FastAPI app, routes, auth
│   ├── db/                     # Connection pool, queries (vector, FTS, cache, chat)
│   ├── reflection/             # Critique and query expansion (self-healing)
│   ├── services/               # Reranker, sovereign_guard
│   ├── state/                  # LangGraph state (TypedDict)
│   ├── tools/                  # vector_search, query_classifier, llm_tools
│   ├── scripts/                # DB setup (SQL + legacy Node)
│   ├── config.py               # Pydantic Settings (env)
│   └── requirements.txt
├── ingest_python/              # PDF → chunk → embed → PostgreSQL
│   ├── config.py               # Ingestion env (DATABASE_URL, chunk sizes)
│   ├── pipeline.py
│   └── parsing/
├── dashboard/                  # Streamlit UI (port 8501)
├── scripts/                    # run_test_queries, gen_test_token, benchmark_latency, migrations
├── tests/                      # Pytest + unified_test.py (E2E)
├── docs/                       # Architecture, ingestion, test references
├── results/                    # Test outputs (gitignored)
├── infra/                      # Terraform (AWS: VPC, ECS, RDS, ALB, ECR)
├── run.sh                      # One-command setup and run
├── run_dashboard.sh
├── Dockerfile, docker-compose.yml
└── .env.example
```

**Config:** App settings = `src/config.py` (Pydantic). Ingestion = `ingest_python/config.py` (env vars).

---

## 5. Prerequisites

- **Python** 3.11+
- **PostgreSQL** 14+ with **pgvector** extension
- **OpenAI API key** (embeddings and LLM)

---

## 6. Installation

```bash
git clone <repo-url>
cd govgig-feature-python-ai-assistant

python3.11 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r src/requirements.txt
cp .env.example .env
```

Edit `.env`: set `OPENAI_API_KEY`, `PG_PASSWORD`, `JWT_SECRET_KEY`. Set `PG_HOST`, `PG_PORT`, `PG_DB`, `PG_USER` if not using local defaults.

---

## 7. Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key. |
| `PG_PASSWORD` | Yes | PostgreSQL password. |
| `JWT_SECRET_KEY` | Yes | Secret for JWT signing. |
| `PG_HOST`, `PG_PORT`, `PG_DB`, `PG_USER` | No | Defaults: localhost, 5432, daedalus, daedalus_admin. |
| `PG_SSLMODE` | No | `disable` (local) or `require` (e.g. RDS). |
| `RETRIEVAL_TOP_K` | No | Primary retrieval size (default 12). |
| `RERANKER_ENABLED` | No | Use LLM reranker after hybrid search (default true). |
| `CORS_ORIGINS` | No | JSON list of allowed origins (default localhost:3000, 3001). |

Full list: **src/config.py** and **.env.example**.

---

## 8. Running

**One command (recommended):**

```bash
bash run.sh
```

Creates/activates venv, installs deps, checks `.env` and DB, then starts the server at **http://localhost:8000**.

**Manual:**

```bash
source venv/bin/activate
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Dashboard (Streamlit):**

```bash
bash run_dashboard.sh
```

Dashboard: **http://localhost:8501**. Backend default: http://localhost:8000 (override with `API_BASE_URL` if needed).

**Health check:**

```bash
curl http://localhost:8000/api/v1/health
```

---

## 9. API Reference

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/v1/health` | GET | No | Status and DB check. |
| `/api/v1/query` | POST | JWT | Main query. Body: `query`, optional `thread_id`, `person_id`, `history`, `cot`. |
| `/api/v1/clause/{ref}` | GET | JWT | Direct clause lookup by reference (e.g. FAR 52.236-2). |
| `/api/v1/auth/signup` | POST | No | Register (email, password, full_name). |
| `/api/v1/auth/login` | POST | No | Login; returns `access_token`. |
| `/api/v1/chat/threads` | GET | JWT | List current user's chat threads. |
| `/api/v1/chat/history` | GET | JWT | Chat history for a thread (query: `thread_id`). |
| `/api/v1/feedback` | POST | JWT | Submit feedback (query_id, response). |
| `/api/v1/analytics/summary` | GET | JWT | Aggregate analytics (query: `hours`). |
| `/ws/chat` | WebSocket | Token in first message | Streaming chat. |

Interactive docs: **http://localhost:8000/docs**.

---

## 10. Authentication

- **Signup:** `POST /api/v1/auth/signup` with `email`, `password`, `full_name`.
- **Login:** `POST /api/v1/auth/login` with `email`, `password` → use `access_token` as Bearer token.
- **Test token (no real user):**  
  `python scripts/gen_test_token.py`  
  Use the printed JWT as `Authorization: Bearer <token>`.

---

## 11. Testing

```bash
# Unit / integration
pytest
pytest tests/test_query_classifier.py -v
pytest tests/test_api.py -v

# E2E smoke (server must be running)
python tests/unified_test.py

# Batch queries via API (set API_TEST_EMAIL, API_TEST_PASSWORD or use --email / --password)
python scripts/run_test_queries.py --api --list scripts/queries.txt -o results/results.csv --csv
```

**Latency benchmark:** `python scripts/benchmark_latency.py`  
**Quality gate:** `python scripts/quality_gate.py --queries-file scripts/quality_queries.json ...` (see script help).

---

## 12. Data Ingestion

Regulations (PDFs) are chunked, embedded, and stored in PostgreSQL (`embeddings_dense`, `embeddings_sparse`, FTS).

```bash
cd ingest_python
pip install -r requirements.txt
# Set DATABASE_URL (and optionally chunk/embed settings) in env
python pipeline.py
```

See **docs/INGESTION_CHUNKING_STRATEGY.md** for chunking and table layout.

---

## 13. Docker

```bash
cp .env.example .env
# Edit .env as needed
docker-compose up -d
```

PostgreSQL (with pgvector) and the app service are defined in `docker-compose.yml`.

---

## 14. AWS (Terraform)

Infrastructure is in **infra/**: VPC, ECS Fargate, RDS PostgreSQL (pgvector), ALB, ECR, Secrets Manager, IAM, optional Route53.

1. Configure Terraform backend (e.g. S3) and variables (see `infra/` and `backend.hcl.example`).
2. Run `./scripts/deploy.sh` (or `terraform init/plan/apply`).
3. After RDS is up: run DB migrations/SQL and the ingest pipeline so the app has data.

See **docs/** and any README in **infra/** for details.

---

## 15. Troubleshooting

| Issue | Check |
|-------|--------|
| **DB connection failed** | `PG_*` in `.env`, pgvector extension (`CREATE EXTENSION vector;`), DB running. |
| **ImportError (langgraph.checkpoint.postgres)** | Install optional deps; conftest imports the full app. |
| **401 on /api/v1/query** | Valid JWT (login or `scripts/gen_test_token.py`). |
| **Empty or poor answers** | Run ingestion; ensure `embeddings_dense`/FTS populated; check `RETRIEVAL_TOP_K` and `RERANKER_ENABLED`. |

---

## 16. Further Reading

- **Architecture and env:** docs/GOVGIG_ARCHITECTURE.md  
- **Ingestion and chunking:** docs/INGESTION_CHUNKING_STRATEGY.md  
- **Test queries and expectations:** docs/TEST_QUERIES_REFERENCE.md  
- **Other design/verification docs:** docs/
