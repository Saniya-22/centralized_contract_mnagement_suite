# Migration Guide: Node.js to Python LangGraph Backend

## Overview

This guide walks through migrating from the existing Node.js/Vercel AI SDK backend to the new Python/LangGraph backend.

## Phase 1: Setup & Parallel Running (CURRENT)

### ✅ Completed

- [x] Python project structure created
- [x] LangGraph multi-agent system implemented
- [x] Data retrieval agent with vector search
- [x] FastAPI backend with REST and WebSocket APIs
- [x] Database connection with existing PostgreSQL/pgvector
- [x] Docker containerization
- [x] Testing framework
- [x] Documentation

### Setup Steps

#### 1. Install Python Backend

```bash
cd backend_python
./run.sh
```

Or manually:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
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

### Running Both Backends

**Node.js Backend:**
```bash
cd /path/to/govgig
yarn dev  # Runs on port 3000
```

**Python Backend:**
```bash
cd backend_python
python -m uvicorn src.api.main:app --reload --port 8000
```

Both can run simultaneously for testing and comparison.

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

#### Option A: Environment Variable Switch

```typescript
// config.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_USE_PYTHON_BACKEND 
  ? 'http://localhost:8000/api/v1'
  : 'http://localhost:3000/api';
```

#### Option B: Gradual Migration

```typescript
// services/api.ts
export const queryDocument = async (query: string) => {
  // Try Python backend first
  try {
    const response = await fetch('http://localhost:8000/api/v1/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    return await response.json();
  } catch (error) {
    // Fallback to Node.js backend
    console.warn('Python backend unavailable, using Node.js');
    return await legacyQuery(query);
  }
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
| Document Analysis | DocumentAnalysisAgent | ⏳ TODO |
| Document Generation | DocumentGenerationAgent | ⏳ TODO |
| Help Agent | HelpAgent | ⏳ TODO |
| Authentication | JWT + FastAPI | ⏳ TODO |
| Chat History | PostgreSQL + State | ⏳ TODO |
| Feedback System | API Endpoints | ⏳ TODO |

### Authentication Migration

```python
# src/api/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Implement JWT validation
    pass

# Use in endpoints
@app.post("/api/v1/query")
async def query(request: QueryRequest, user = Depends(get_current_user)):
    # Protected endpoint
    pass
```

## Phase 5: Performance Optimization

### 1. Caching Layer

```python
# src/utils/cache.py
from functools import lru_cache
import redis

# In-memory cache for embeddings
@lru_cache(maxsize=1000)
def cached_embedding(text: str) -> List[float]:
    return get_embedding(text)

# Redis for query results
redis_client = redis.Redis()

def cache_query_result(query: str, result: dict, ttl: int = 3600):
    key = f"query:{hash(query)}"
    redis_client.setex(key, ttl, json.dumps(result))
```

### 2. Async Database Queries

```python
# Upgrade to asyncpg
import asyncpg

async def get_async_pool():
    return await asyncpg.create_pool(
        host=settings.PG_HOST,
        database=settings.PG_DB,
        user=settings.PG_USER,
        password=settings.PG_PASSWORD
    )
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

For migration support:
1. Check documentation in `backend_python/README.md`
2. Review test files in `backend_python/tests/`
3. Check logs: `docker-compose logs -f backend`
4. Test locally: `./test_setup.sh`
