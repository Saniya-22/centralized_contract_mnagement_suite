"""FastAPI application with WebSocket and REST endpoints"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import jwt

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


# ── In-memory rate limiter (no external deps) ─────────────────────────────────

class InMemoryRateLimiter:
    """Simple sliding-window rate limiter keyed by user ID."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: Dict[str, list] = defaultdict(list)

    def check(self, user_id: str) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        now = time.time()
        window_start = now - self.window_seconds
        # Prune old entries
        self._hits[user_id] = [t for t in self._hits[user_id] if t > window_start]
        if len(self._hits[user_id]) >= self.max_requests:
            return False
        self._hits[user_id].append(now)
        return True


rate_limiter = InMemoryRateLimiter(
    max_requests=settings.RATE_LIMIT_MAX_REQUESTS,
    window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
)

_MAX_QUERY_LENGTH = 2000  # DoS guard


def _user_id_to_uuid(uid: Any) -> str:
    """Convert user_id from JWT to a UUID string for storage (user_feedback table)."""
    if uid is None:
        return str(uuid.uuid4())
    s = str(uid)
    try:
        uuid.UUID(s)
        return s
    except (ValueError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., min_length=1, max_length=_MAX_QUERY_LENGTH, description="User query")
    person_id: Optional[str] = Field(None, description="User ID for personalization")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Chat history")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation persistence")
    cot: bool = Field(default=True, description="Enable Chain-of-Thought reasoning")


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    response: str
    documents: List[Dict[str, Any]]
    confidence: Optional[float] = None
    mode: Optional[str] = None  # grounded | copilot | refusal | clarify
    quality_metrics: Optional[Dict[str, Any]] = None
    low_confidence: Optional[bool] = None
    reflection_triggered: Optional[bool] = None
    agent_path: List[str]
    thought_process: Optional[List[str]] = None
    regulation_types: List[str]
    ui_action: Optional[Dict[str, Any]] = None
    errors: List[str]
    user_id: str  # UUID string (authenticated user)
    query_id: str  # UUID string (this request)


class ClauseResponse(BaseModel):
    """Response model for clause lookup endpoint"""
    found: bool
    clause_reference: str
    clause: Optional[Dict[str, Any]] = None
    context: Optional[str] = None


class SignupRequest(BaseModel):
    """Request model for signup endpoint"""
    full_name: str = Field(..., min_length=1, max_length=255)
    email: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    confirm_password: str = Field(..., min_length=8, max_length=128)

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("password and confirm_password do not match")
        return v


class LoginRequest(BaseModel):
    """Request model for login endpoint"""
    email: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=1, max_length=128)


class FeedbackRequest(BaseModel):
    """Request model for feedback endpoint"""
    query_id: str = Field(..., description="UUID of the query this feedback refers to")
    response: str = Field(..., description="'good' (thumbs up) or 'bad' (thumbs down)")

    @field_validator("response")
    @classmethod
    def response_must_be_good_or_bad(cls, v: str) -> str:
        if v not in ("good", "bad"):
            raise ValueError("response must be 'good' or 'bad'")
        return v

    @field_validator("query_id")
    @classmethod
    def query_id_must_be_uuid(cls, v: str) -> str:
        try:
            uuid.UUID(v)
            return v
        except (ValueError, TypeError):
            raise ValueError("query_id must be a valid UUID")


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
    
    # Initialize cache, analytics, user_feedback, and auth tables
    db_ok = test_connection()
    if db_ok:
        VectorQueries.init_cache_table()
        VectorQueries.init_analytics_table()
        VectorQueries.init_user_feedback_table()
        VectorQueries.init_auth_tables()
        VectorQueries.init_chat_history_table()
        if VectorQueries.auth_tables_exist():
            logger.info("Required auth tables (users, auth_audit_log) confirmed in database")
        else:
            logger.warning("Auth tables (users, auth_audit_log) may be missing; check database and init_auth_tables")
    
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

        # ── Rate limit check ──────────────────────────────────────────────
        if not rate_limiter.check(user_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please wait a moment before trying again.",
            )

        query_id = str(uuid.uuid4())
        start_time = time.time()

        thread_id = request.thread_id or str(uuid.uuid4())
        history = request.history if request.history else await run_in_threadpool(
            VectorQueries.get_chat_history, thread_id, user_id
        )
        context = {
            "person_id": request.person_id or user_id,
            "thread_id": thread_id,
            "history": history,
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
            # Log cache hit to analytics
            latency_ms = int((time.time() - start_time) * 1000)
            try:
                _log_analytics_from_result(
                    cached_result, request.query, user_id,
                    context["thread_id"], latency_ms,
                    was_cached=True, source="rest",
                )
            except Exception:
                pass
            # Persist to chat_history on cache hit too
            try:
                response_text = cached_result.get("response") or ""
                await run_in_threadpool(
                    VectorQueries.insert_chat_message,
                    thread_id, user_id, "user", request.query,
                )
                await run_in_threadpool(
                    VectorQueries.insert_chat_message,
                    thread_id, user_id, "assistant", response_text,
                )
            except Exception as e:
                logger.warning(f"Chat history persist (cache hit) failed: {e}")
            return QueryResponse(**{**cached_result, "user_id": user_id, "query_id": query_id})

        # 2. If no cache, run orchestrator
        result = await orchestrator.run_async(request.query, context)
        
        # 3. Store in Cache — but skip caching OUT_OF_SCOPE / empty results
        #    to avoid polluting cache with refusal messages.
        has_documents = bool(result.get("documents"))
        if has_documents:
            VectorQueries.set_cached_response(
                request.query,
                result,
                request.cot,
                cache_scope=cache_scope,
            )

        # 4. Log to analytics
        latency_ms = int((time.time() - start_time) * 1000)
        try:
            _log_analytics_from_result(
                result, request.query, user_id,
                context["thread_id"], latency_ms,
                was_cached=False, source="rest",
            )
        except Exception:
            pass

        # 5. Persist to chat_history (thread_id = session_id)
        response_text = result.get("response") or ""
        try:
            await run_in_threadpool(
                VectorQueries.insert_chat_message,
                thread_id, user_id, "user", request.query,
            )
            await run_in_threadpool(
                VectorQueries.insert_chat_message,
                thread_id, user_id, "assistant", response_text,
            )
        except Exception as e:
            logger.warning(f"Chat history persist failed: {e}")

        return QueryResponse(**{**result, "user_id": user_id, "query_id": query_id})

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
            
            thread_id = data.get("thread_id") or data.get("person_id") or user_id or "default_thread"
            history = data.get("history") or await run_in_threadpool(
                VectorQueries.get_chat_history, thread_id, user_id
            )
            context = {
                "person_id": data.get("person_id") or user_id,
                "thread_id": thread_id,
                "history": history,
                "cot": data.get("cot", True),
                "current_date": datetime.now().strftime("%A, %B %d, %Y")
            }

            ws_query_id = str(uuid.uuid4())
            logger.info(f"WebSocket query: {query[:100]}...")
            ws_start_time = time.time()

            # Stream responses — capture the last complete event for analytics
            last_complete = None
            async for event in orchestrator.run(query, context):
                if isinstance(event, dict) and event.get("type") == "complete":
                    event["data"] = {**(event.get("data") or {}), "user_id": user_id, "query_id": ws_query_id}
                    last_complete = event["data"]
                await websocket.send_json(event)
            
            # Send done message
            await websocket.send_json({
                "type": "done",
                "data": "[DONE]"
            })

            # Log WebSocket query to analytics
            ws_latency = int((time.time() - ws_start_time) * 1000)
            try:
                if last_complete and isinstance(last_complete, dict):
                    _log_analytics_from_result(
                        last_complete, query, user_id,
                        context.get("thread_id"), ws_latency,
                        was_cached=False, source="websocket",
                    )
            except Exception:
                pass

            # Persist to chat_history (thread_id = session_id)
            if thread_id and user_id:
                try:
                    response_text = (last_complete or {}).get("response") or ""
                    await run_in_threadpool(
                        VectorQueries.insert_chat_message,
                        thread_id, user_id, "user", query,
                    )
                    await run_in_threadpool(
                        VectorQueries.insert_chat_message,
                        thread_id, user_id, "assistant", response_text,
                    )
                except Exception as e:
                    logger.warning(f"Chat history persist (ws) failed: {e}")
            
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


# ── Analytics helper ──────────────────────────────────────────────────────

def _extract_intent_from_result(result: Dict[str, Any]) -> Optional[str]:
    """Extract the intent from agent_path entries."""
    for step in result.get("agent_path", []):
        if "intent=" in step:
            # e.g. "Router: intent=regulation_search conf=0.80"
            try:
                return step.split("intent=")[1].split()[0]
            except (IndexError, AttributeError):
                pass
    return None


def _log_analytics_from_result(
    result: Dict[str, Any],
    query: str,
    user_id: str,
    thread_id: str,
    latency_ms: int,
    was_cached: bool = False,
    source: str = "rest",
):
    """Extract analytics fields from an orchestrator result and log them."""
    VectorQueries.log_query_analytics({
        "query_text": query,
        "user_id": user_id,
        "thread_id": thread_id,
        "intent": _extract_intent_from_result(result),
        "regulation_types": result.get("regulation_types", []),
        "confidence": result.get("confidence"),
        "mode": result.get("mode"),
        "quality_metrics": result.get("quality_metrics"),
        "low_confidence": result.get("low_confidence"),
        "doc_count": len(result.get("documents", [])),
        "reflection_triggered": bool(result.get("reflection_triggered")),
        "agent_path": result.get("agent_path", []),
        "errors": result.get("errors", []),
        "was_cached": was_cached,
        "latency_ms": latency_ms,
        "source": source,
    })


# ── Analytics summary endpoint ────────────────────────────────────────────

@app.get(
    f"{settings.API_PREFIX}/analytics/summary",
    tags=["Analytics"],
)
async def analytics_summary(
    hours: int = 24,
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Returns aggregate query analytics for the last N hours plus feedback counts (JWT required)."""
    summary = await run_in_threadpool(VectorQueries.get_analytics_summary, hours)
    return summary


# ── Auth: signup and login ────────────────────────────────────────────────────


@app.post(
    f"{settings.API_PREFIX}/auth/signup",
    status_code=status.HTTP_201_CREATED,
    tags=["Auth"],
)
async def signup(body: SignupRequest):
    """Register a new user. Returns full_name and status."""
    email = body.email.strip().lower()
    existing = await run_in_threadpool(VectorQueries.get_user_by_email, email)
    if existing:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"user_id": None, "full_name": body.full_name.strip(), "status": "Signup unsuccessful", "detail": "Email already registered"},
        )
    hashed = bcrypt.hashpw(body.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    try:
        user_row = await run_in_threadpool(
            VectorQueries.create_user,
            body.full_name.strip(),
            email,
            hashed,
        )
    except Exception as e:
        logger.exception("Signup create_user failed")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "user_id": None,
                "full_name": body.full_name.strip(),
                "status": "Signup unsuccessful",
                "detail": str(e),
            },
        )
    if not user_row:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"user_id": None, "full_name": body.full_name.strip(), "status": "Signup unsuccessful", "detail": "Registration failed"},
        )
    return {"user_id": str(user_row["user_id"]), "full_name": body.full_name.strip(), "status": "Signup successful"}


@app.post(
    f"{settings.API_PREFIX}/auth/login",
    tags=["Auth"],
)
async def login(body: LoginRequest):
    """Authenticate with email and password. Returns access_token and status. Lockout after max failed attempts for configured minutes."""
    email = body.email.strip().lower()
    user = await run_in_threadpool(VectorQueries.get_user_by_email, email)
    if not user:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"user_id": None, "access_token": "", "status": "Login unsuccessful", "detail": "Invalid email or password"},
        )
    lock_until = user.get("lock_until")
    if lock_until:
        now = datetime.now(timezone.utc)
        if lock_until.tzinfo is None:
            lock_until = lock_until.replace(tzinfo=timezone.utc)
        if lock_until > now:
            delta = lock_until - now
            remaining = int(delta.total_seconds())
            return JSONResponse(
                status_code=status.HTTP_423_LOCKED,
                content={
                    "user_id": str(user["user_id"]),
                    "access_token": "",
                    "status": "Login unsuccessful",
                    "detail": "Account locked. Try again later.",
                    "remaining_seconds": remaining,
                },
            )
    if not bcrypt.checkpw(body.password.encode("utf-8"), user["hashed_password"].encode("utf-8")):
        attempts = (user.get("failed_login_attempts") or 0) + 1
        lock_at_five = attempts >= settings.LOGIN_MAX_ATTEMPTS
        lock_until_dt = (datetime.now(timezone.utc) + timedelta(minutes=settings.LOGIN_LOCKOUT_MINUTES)) if lock_at_five else None
        await run_in_threadpool(VectorQueries.update_login_failure, user["id"], lock_until_dt)
        if lock_at_five:
            await run_in_threadpool(
                VectorQueries.log_auth_audit,
                user["id"],
                email,
                "lockout",
                {"reason": "max_attempts_reached", "lock_minutes": settings.LOGIN_LOCKOUT_MINUTES},
            )
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "user_id": str(user["user_id"]),
                "access_token": "",
                "status": "Login unsuccessful",
                "detail": "Invalid email or password",
                **({"locked": True, "remaining_seconds": settings.LOGIN_LOCKOUT_MINUTES * 60} if lock_at_five else {}),
            },
        )
    await run_in_threadpool(VectorQueries.update_login_success, user["id"])
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    user_id_uuid = str(user["user_id"])
    to_encode = {"sub": user_id_uuid, "email": email, "name": user.get("full_name"), "exp": expire}
    access_token = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return {
        "user_id": user_id_uuid,
        "access_token": access_token,
        "token_type": "bearer",
        "status": "Login successful",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "email": email,
        "full_name": user.get("full_name") or "",
    }


@app.post(
    f"{settings.API_PREFIX}/feedback",
    status_code=status.HTTP_201_CREATED,
    tags=["Feedback"],
)
async def feedback_endpoint(
    body: FeedbackRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Record user feedback (thumbs up/down) for a query (JWT required)."""
    user_id_raw = user.get("sub") or user.get("id") or user.get("user_id")
    if not user_id_raw:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload: missing user identifier",
        )
    user_id_uuid = _user_id_to_uuid(user_id_raw)
    ok = await run_in_threadpool(
        VectorQueries.insert_user_feedback,
        user_id_uuid,
        body.query_id,
        body.response,
    )
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save feedback",
        )
    return {"ok": True}


@app.get(
    f"{settings.API_PREFIX}/chat/threads",
    tags=["Chat"],
)
async def list_chat_threads_endpoint(
    user: Dict[str, Any] = Depends(get_current_user),
    limit: int = 50,
):
    """List the current user's chat threads (conversations). JWT required. Returns thread_id, updated_at, preview."""
    user_id = user.get("sub") or user.get("id") or user.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload: missing user identifier",
        )
    threads = await run_in_threadpool(
        VectorQueries.list_chat_threads,
        str(user_id),
        min(limit, 100),
    )
    return {"threads": threads}


@app.get(
    f"{settings.API_PREFIX}/chat/history",
    tags=["Chat"],
)
async def get_chat_history_endpoint(
    thread_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Return chat history for a thread (conversation). JWT required. Scoped to current user."""
    user_id = user.get("sub") or user.get("id") or user.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload: missing user identifier",
        )
    history = await run_in_threadpool(
        VectorQueries.get_chat_history,
        thread_id.strip(),
        str(user_id),
        50,
    )
    return {"thread_id": thread_id, "history": history}


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
