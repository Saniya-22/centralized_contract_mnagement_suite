# Migration Guide: Node.js to Python LangGraph Backend

> Status: Migration completed. This repository now runs in Python-only mode.
> Legacy Node.js runtime commands are retained only as historical context.

## Overview

This guide walks through migrating from the existing Node.js/Vercel AI SDK backend to the new Python/LangGraph backend.

## Phase 1: Setup (Python-Only)

### ✅ Completed

- [x] Python project structure created
- [x] LangGraph multi-agent system implemented
- [x] Data retrieval agent with vector search
- [x] FastAPI backend with REST and WebSocket APIs
- [x] Database connection with existing PostgreSQL/pgvector
- [x] JWT Authentication
- [x] Persistent Chat History (PostgreSQL)
- [x] API Response Caching (PostgreSQL)
- [x] Docker containerization
- [x] Testing framework
- [x] Documentation

### Setup Steps

#### 1. Install Python Backend

```bash
# From project root
./run.sh
```

Or manually:

```bash
python -m venv venv
source venv/bin/activate
pip install -r src/requirements.txt
cp .env.example .env
# Edit .env with your values
python -m uvicorn src.api.main:app --reload
```

#### 2. Verify Setup

```bash
./test_setup.sh
```

#### 3. Test Endpoints

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Query test
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are FAR small business requirements?"}'
```

### Runtime Mode

Run only the Python backend:

```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

## Phase 2: Additional Agents (NEXT)

### To Implement

1. **Document Analysis Agent**
   - File: `src/agents/document_analysis.py`
   - Purpose: Analyze regulations, check compliance, identify risks
   - Tools: Analysis templates, compliance checkers

2. **Document Generation Agent**
   - File: `src/agents/document_generation.py`
   - Purpose: Generate proposals, summaries, checklists
   - Tools: Document templates, formatting utilities

3. **Help Agent**
   - File: `src/agents/help.py`
   - Purpose: Answer system questions, guide users
   - Tools: System documentation, FAQs

### Implementation Template

```python
# src/agents/document_analysis.py
from src.agents.base import BaseAgent
from src.state.graph_state import GovGigState

class DocumentAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="DocumentAnalysisAgent")
        # Initialize tools
    
    def get_system_prompt(self, state: GovGigState) -> str:
        return """You are a document analysis specialist..."""
    
    def run(self, state: GovGigState) -> GovGigState:
        # Implementation
        return state
```

Update orchestrator:
```python
# src/agents/orchestrator.py
from src.agents.document_analysis import DocumentAnalysisAgent

class GovGigOrchestrator:
    def __init__(self):
        # ...
        self.document_analysis = DocumentAnalysisAgent()
        
    def _build_graph(self):
        # ...
        workflow.add_node("document_analysis", self.document_analysis.run)
        # Add edges
```

## Phase 3: Frontend Integration

### Update Frontend to Use Python Backend

#### Option A: Python API Base URL

```typescript
// config.ts
const API_BASE_URL = 'http://localhost:8000/api/v1';
```

#### Option B: Direct Python Call

```typescript
// services/api.ts
export const queryDocument = async (query: string) => {
  const response = await fetch('http://localhost:8000/api/v1/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query })
  });
  return await response.json();
};
```

### WebSocket Integration

```typescript
// services/websocket.ts
const ws = new WebSocket('ws://localhost:8000/ws/chat');

ws.onopen = () => {
  ws.send(JSON.stringify({
    query: 'What are the requirements for...?',
    cot: true
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'step') {
    // Handle streaming step
    console.log('Step:', data.data);
  } else if (data.type === 'complete') {
    // Handle final response
    console.log('Complete:', data.data);
  }
};
```

## Phase 4: Feature Parity

### Map Node.js Features to Python

| Node.js Feature | Python Equivalent | Status |
|----------------|-------------------|--------|
| Orchestrator (Vercel AI SDK) | LangGraph StateGraph | ✅ Done |
| Data Retrieval Agent | DataRetrievalAgent | ✅ Done |
| Vector Search | VectorSearchTool | ✅ Done |
| Document Analysis | DocumentAnalysisAgent | ✅ Done |
| Document Generation | DocumentGenerationAgent | ✅ Done |
| Help Agent | HelpAgent | ✅ Done |
| Authentication | JWT + FastAPI (HTTPBearer) | ✅ Done |
| Chat History | PostgreSQL (LangGraph Saver) | ✅ Done |
| Feedback System | API Endpoints | ✅ Done |

### Authentication Implementation

Authentication is implemented using FastAPI's `HTTPBearer` security scheme and the `python-jose` library for JWT validation.

```python
# src/api/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET_KEY, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        user_id = payload.get("sub") or payload.get("id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

# Use in main.py
@app.post("/api/v1/query")
async def query(request: QueryRequest, user: Dict[str, Any] = Depends(get_current_user)):
    # user contains the JWT payload
    return await process_query(request, user)
```

### Chat History & Persistence

Persistence is handled by `langgraph-checkpoint-postgres`, which stores the full agent state and message history in PostgreSQL.

1. **Configuration**: The `CheckpointerManager` handles the connection pool for the checkpointer.
2. **Usage**: Pass a `thread_id` in the configuration to maintain session state.

```python
# src/agents/orchestrator.py
config = {"configurable": {"thread_id": "user_session_123"}}
async for event in self.app.astream_events(initial_state, config=config, version="v1"):
    # State is automatically saved to 'checkpoints' and 'writes' tables
    pass
```

## Phase 5: Performance Optimization

### 1. Caching Layer (PostgreSQL)

Instead of Redis, we use a dedicated PostgreSQL table `api_response_cache` for persistent response caching. This simplifies the stack while maintaining high performance.

- **Storage**: JSONB column for flexible response data.
- **Eviction**: TTL-based expiration (`expires_at` column).
- **Lookup**: SHA-256 hashing of query text and settings for O(1) retrieval.

```python
# src/db/queries.py
def get_cached_response(query: str, cot: bool = True):
    # Generates a hash and looks up in api_response_cache table
    # Returns the response if not expired
    pass

def set_cached_response(query: str, response: dict, ttl_hours: int = 24):
    # Upserts the response into the cache table
    pass
```

### 2. Async Database Operations

To prevent blocking the FastAPI event loop, all synchronous database operations are executed in a thread pool using `run_in_threadpool`.

```python
# src/db/connection.py
from fastapi.concurrency import run_in_threadpool

async def execute_in_db(func, *args, **kwargs):
    """Execute a database function safely in the thread pool."""
    return await run_in_threadpool(func, *args, **kwargs)
```

### 3. Load Balancing

```yaml
# docker-compose.yml
services:
  backend1:
    build: .
    ports: ["8001:8000"]
  
  backend2:
    build: .
    ports: ["8002:8000"]
  
  nginx:
    image: nginx
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    ports: ["80:80"]
```

## Phase 6: Production Deployment

### AWS Deployment (Suggested)

```bash
# 1. Build Docker image
docker build -t govgig-backend .

# 2. Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <ecr-url>
docker tag govgig-backend:latest <ecr-url>/govgig-backend:latest
docker push <ecr-url>/govgig-backend:latest

# 3. Deploy to ECS/Fargate
aws ecs update-service --cluster govgig --service backend --force-new-deployment
```

### Environment Variables for Production

```env
# Production .env
DEBUG=False
LOG_LEVEL=WARNING
WORKERS=4

# Use managed database
PG_HOST=your-rds-endpoint.amazonaws.com
PG_PORT=5432

# Enable LangSmith for monitoring
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-key>

# Security
CORS_ORIGINS=["https://yourdomain.com"]
JWT_SECRET_KEY=<strong-random-key>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Database Pooling
PG_POOL_MIN=2
PG_POOL_MAX=20
```

## Phase 7: Retirement of Node.js Backend

### Checklist Before Retiring Node.js

- [ ] All agents implemented and tested
- [ ] Frontend fully migrated
- [ ] Authentication working
- [ ] Performance meets requirements
- [ ] Load testing completed
- [ ] Monitoring and logging set up
- [ ] Backup and rollback plan ready
- [ ] Team trained on Python backend

### Migration Completion Steps

1. **Parallel running period**: 2-4 weeks
2. **Traffic split testing**: Route 10% → 50% → 100% to Python
3. **Monitor metrics**: Response time, error rate, resource usage
4. **Freeze Node.js changes**: No new features in old backend
5. **Final cutover**: Update DNS/load balancer
6. **Keep Node.js running**: 1 week for emergency rollback
7. **Decommission**: Archive code, shut down services

## Troubleshooting Common Issues

### Database Connection Errors

```bash
# Check PostgreSQL is running
psql -U postgres -h localhost -d daedalus -c "SELECT 1"

# Verify pgvector extension
psql -U postgres -d daedalus -c "SELECT * FROM pg_extension WHERE extname='vector';"

# Test Python connection
python -c "from src.db.connection import test_connection; test_connection()"
```

### OpenAI API Errors

```bash
# Verify API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Test embedding generation
python -c "from src.tools.llm_tools import get_embedding; print(len(get_embedding('test')))"
```

### LangGraph State Issues

```python
# Debug state transitions
from src.agents.orchestrator import GovGigOrchestrator

orch = GovGigOrchestrator()
result = orch.run_sync("test query", {"cot": True})
print(result['thought_process'])  # See agent reasoning
print(result['agent_path'])  # See execution path
```

## Performance Comparison

### Benchmark Results (Expected)

| Metric | Node.js | Python | Improvement |
|--------|---------|--------|-------------|
| Query latency | 2.5s | 1.8s | 28% faster |
| Memory usage | 450MB | 380MB | 15% less |
| Concurrent users | 50 | 100 | 2x |
| Vector search | 180ms | 120ms | 33% faster |

### Run Benchmarks

```bash
# Install benchmark tool
pip install locust

# Run load test
locust -f tests/locustfile.py --host http://localhost:8000
```

## Support and Resources

- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **pgvector Docs**: https://github.com/pgvector/pgvector
- **LangSmith**: https://smith.langchain.com/

## Questions?

For support:
1. Check documentation in `README.md`
2. Review test files in `tests/`
3. Check logs: `docker-compose logs -f backend`
4. Test locally: `./run.sh`
