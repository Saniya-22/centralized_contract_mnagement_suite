"""FastAPI application with WebSocket and REST endpoints"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

from src.config import settings
from src.agents.orchestrator import GovGigOrchestrator
from src.db.connection import test_connection, close_db_pool, CheckpointerManager
from src.db.queries import VectorQueries
from src.api.auth import get_current_user

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="GovGig AI - Python LangGraph Backend for Regulatory Document RAG",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all exception handler for the API"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "response": "An unexpected error occurred. Please try again later.",
            "documents": [],
            "errors": ["Internal server error"],
            "agent_path": ["GlobalExceptionHandler: Caught unhandled error"]
        }
    )

# Initialize orchestrator
try:
    orchestrator = GovGigOrchestrator()
    logger.info("Orchestrator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize orchestrator: {e}")
    orchestrator = None


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., min_length=1, description="User query")
    person_id: Optional[str] = Field(None, description="User ID for personalization")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Chat history")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation persistence")
    cot: bool = Field(default=True, description="Enable Chain-of-Thought reasoning")


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    response: str
    documents: List[Dict[str, Any]]
    confidence: Optional[float] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    low_confidence: Optional[bool] = None
    agent_path: List[str]
    thought_process: Optional[List[str]] = None
    regulation_types: List[str]
    ui_action: Optional[Dict[str, Any]] = None
    errors: List[str]


class ClauseResponse(BaseModel):
    """Response model for clause lookup endpoint"""
    found: bool
    clause_reference: str
    clause: Optional[Dict[str, Any]] = None
    context: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    database: str
    orchestrator: str
    timestamp: str


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Initialize cache table and database connection
    db_ok = test_connection()
    if db_ok:
        VectorQueries.init_cache_table()
    
    if not db_ok:
        logger.warning("Database connection test failed at startup")
    else:
        # Initialize checkpointer and update orchestrator
        try:
            checkpointer = await CheckpointerManager.get_checkpointer()
            global orchestrator
            orchestrator = GovGigOrchestrator(checkpointer=checkpointer)
            logger.info("Orchestrator re-initialized with Postgres checkpointer")
        except Exception as e:
            logger.error(f"Failed to initialize checkpointer: {e}")
            logger.warning("Running orchestrator without persistence")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down application")
    await CheckpointerManager.close()
    close_db_pool()


# Health check endpoint
@app.get(
    f"{settings.API_PREFIX}/health",
    response_model=HealthResponse,
    tags=["Health"]
)
async def health_check():
    """Check application health status"""
    db_status = "healthy" if test_connection() else "unhealthy"
    orch_status = "healthy" if orchestrator is not None else "unhealthy"
    
    return HealthResponse(
        status="healthy" if db_status == "healthy" and orch_status == "healthy" else "degraded",
        version=settings.APP_VERSION,
        database=db_status,
        orchestrator=orch_status,
        timestamp=datetime.now().isoformat()
    )


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }


# Query endpoint (REST)
@app.post(
    f"{settings.API_PREFIX}/query",
    response_model=QueryResponse,
    tags=["Query"]
)
async def query_endpoint(
    request: QueryRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Process a query and return results (Authenticated)."""
    if orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator not available"
        )
    
    try:
        user_id = user.get("sub") or user.get("id") or user.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload: missing user identifier"
            )

        context = {
            "person_id": request.person_id or user_id,
            "thread_id": request.thread_id or request.person_id or user_id or "default_thread",
            "history": request.history,
            "cot": request.cot,
            "current_date": datetime.now().strftime("%A, %B %d, %Y")
        }
        cache_scope = f"user:{user_id}|thread:{context['thread_id']}"
        
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # 1. Check Cache
        cached_result = VectorQueries.get_cached_response(
            request.query,
            request.cot,
            cache_scope=cache_scope,
        )
        if cached_result:
            return QueryResponse(**cached_result)

        # 2. If no cache, run orchestrator
        # Run orchestrator asynchronously
        result = await orchestrator.run_async(request.query, context)
        
        # 3. Store in Cache
        VectorQueries.set_cached_response(
            request.query,
            result,
            request.cot,
            cache_scope=cache_scope,
        )
        
        return QueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Query processing failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Query processing failed"
        )


# WebSocket endpoint for streaming
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat"""
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    if orchestrator is None:
        await websocket.send_json({
            "type": "error",
            "data": {"message": "Orchestrator not available"}
        })
        await websocket.close()
        return
    
    try:
        authenticated = False
        user_id = None

        while True:
            # Receive query from client
            data = await websocket.receive_json()
            
            # 1. Handle Authentication (if not yet authenticated)
            if not authenticated:
                token = data.get("token")
                if not token:
                    await websocket.send_json({"type": "error", "data": {"message": "Authentication required. Please provide a 'token'."}})
                    await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                    return
                
                try:
                    from jose import jwt
                    payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
                    user_id = payload.get("sub") or payload.get("id")
                    if not user_id:
                        await websocket.send_json({"type": "error", "data": {"message": "Invalid token payload"}})
                        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                        return
                    authenticated = True
                    logger.info(f"WebSocket authenticated for user: {user_id}")
                except Exception as e:
                    await websocket.send_json({"type": "error", "data": {"message": "Invalid token"}})
                    await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                    return

            query = data.get("query")
            if not query:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Query is required"}
                })
                continue
            
            context = {
                "person_id": data.get("person_id") or user_id,
                "thread_id": data.get("thread_id") or data.get("person_id") or user_id or "default_thread",
                "history": data.get("history", []),
                "cot": data.get("cot", True),
                "current_date": datetime.now().strftime("%A, %B %d, %Y")
            }
            
            logger.info(f"WebSocket query: {query[:100]}...")
            
            # Stream responses
            async for event in orchestrator.run(query, context):
                await websocket.send_json(event)
            
            # Send done message
            await websocket.send_json({
                "type": "done",
                "data": "[DONE]"
            })
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "data": {"message": "Internal server error"}
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass




# Clause reference lookup endpoint
@app.get(
    f"{settings.API_PREFIX}/clause/{{clause_reference:path}}",
    response_model=ClauseResponse,
    tags=["Clause Lookup"]
)
async def clause_lookup(
    clause_reference: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Look up a specific regulation clause by its reference number (Authenticated).

    Supports FAR, DFARS, and EM385 clause references.

    Examples:
    - /api/v1/clause/FAR 52.236-2
    - /api/v1/clause/DFARS 252.204-7012
    - /api/v1/clause/EM 385 Section 05.A
    """
    from urllib.parse import unquote
    clause_reference = unquote(clause_reference)
    logger.info(f"Clause lookup request: '{clause_reference}'")

    try:
        result = await run_in_threadpool(VectorQueries.get_clause_by_reference, clause_reference)
        return ClauseResponse(
            found=result["found"],
            clause_reference=clause_reference,
            clause=result.get("clause"),
            context=result.get("context"),
        )
    except Exception as e:
        logger.error(f"Clause lookup failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Clause lookup failed"
        )


if __name__ == "__main__":

    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
