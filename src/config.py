"""Configuration management for GovGig AI Backend"""

import json
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    APP_NAME: str = "GovGig AI Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_PREFIX: str = "/api/v1"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4

    # OpenAI
    OPENAI_API_KEY: str
    MODEL_NAME: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    TEMPERATURE: float = 0.2
    MAX_TOKENS: Optional[int] = None

    # Database
    PG_HOST: str = "localhost"
    PG_PORT: int = 5432
    PG_DB: str = "daedalus"
    PG_USER: str = "daedalus_admin"
    PG_PASSWORD: str
    PG_SSLMODE: str = "disable"  # disable|require (default disable for local)
    PG_POOL_MIN: int = 2
    PG_POOL_MAX: int = 10

    # Database Tables
    PG_DENSE_TABLE: str = "embeddings_dense"
    PG_SPARSE_TABLE: str = "embeddings_sparse"
    REGULATIONS_NAMESPACE: str = "public-regulations"

    # Vector Search
    DENSE_TOP_K: int = 15
    SPARSE_TOP_K: int = 10
    HYBRID_DENSE_WEIGHT: float = 0.7
    HYBRID_SPARSE_WEIGHT: float = 0.3

    # Advanced RAG
    RRF_K: int = 60  # Reciprocal Rank Fusion constant
    RERANKER_MODEL: str = "gpt-4o-mini"  # Model used for LLM reranking
    RERANKER_ENABLED: bool = True  # Set False to skip LLM rerank, use RRF only
    RAG_TOKEN_LIMIT: int = 2400  # Max tokens assembled into context for faster speed
    RETRIEVAL_TOP_K: int = 20  # Primary retrieval size for regulation_search path
    REFLECTION_THRESHOLD: float = (
        0.50  # Retrieval critique pass threshold (higher = stricter = more healing triggers)
    )
    REFLECTION_HEALING_MARGIN: float = (
        0.05  # Skip retries for near-threshold borderline scores
    )
    SELF_HEALING_SEARCH_K: int = 3  # Per expanded query search depth
    SELF_HEALING_MAX_QUERIES: int = 2  # Max expanded queries to execute
    SELF_HEALING_MAX_DOCS: int = 4  # Max additional docs added from self-healing
    QUALITY_GATE_ENABLED: bool = (
        True  # Post-synthesis quality gate: re-retrieves if answer confidence is low
    )
    QUALITY_GATE_THRESHOLD: float = (
        0.35  # confidence_score below this triggers quality-gate healing (0..1)
    )
    MAX_DOC_CHARS_FOR_SYNTHESIS: int = (
        1200  # Per-document content trim before prompt assembly
    )
    # LLM Models (separate concerns)
    # MODEL_NAME:       used for the DataRetrieval tool-selector fallback
    # SYNTHESIZER_MODEL: used by _synthesize_response
    SYNTHESIZER_MODEL: str = "gpt-4o"  # Synthesizer model (quality)

    # LangGraph
    MAX_ITERATIONS: int = 10
    RECURSION_LIMIT: int = 25

    # LangSmith (optional)
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: Optional[str] = "govgig-ai"

    # Authentication
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours
    LOGIN_LOCKOUT_MINUTES: int = (
        15  # Lock account for this long after max failed attempts
    )
    LOGIN_MAX_ATTEMPTS: int = 5  # Max failed login attempts before lockout
    RATE_LIMIT_MAX_REQUESTS: int = 10  # Per-user request limit (query endpoint)
    RATE_LIMIT_WINDOW_SECONDS: int = 60  # Sliding window for rate limit
    COOKIE_SECRET: Optional[str] = None  # ECS secret for session management
    ADMIN_API_KEY: Optional[str] = None  # Admin API key for operational endpoints

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:3001"]

    # Logging
    LOG_LEVEL: str = "INFO"

    # Sovereign-AI guardrail integration (optional)
    SOVEREIGN_GUARD_ENABLED: bool = False
    SOVEREIGN_GUARD_BASE_URL: str = "http://localhost:8000"
    SOVEREIGN_GUARD_DETECT_PATH: str = "/detect"
    SOVEREIGN_GUARD_TIMEOUT_SECONDS: float = 3.0
    SOVEREIGN_GUARD_FAIL_OPEN: bool = True
    SOVEREIGN_GUARD_BLOCK_MODE: str = "soft"  # soft|hard
    SOVEREIGN_GUARD_AUTH_TOKEN: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_bool(cls, value):
        """Accept common environment strings for DEBUG."""
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            truthy = {"1", "true", "yes", "on", "debug", "dev", "development"}
            falsy = {"0", "false", "no", "off", "release", "prod", "production"}
            if normalized in truthy:
                return True
            if normalized in falsy:
                return False
        raise ValueError("DEBUG must be a boolean-like value")

    @field_validator("SOVEREIGN_GUARD_BLOCK_MODE", mode="before")
    @classmethod
    def parse_sovereign_block_mode(cls, value):
        """Normalize Sovereign block mode."""
        if value is None:
            return "soft"
        normalized = str(value).strip().lower()
        if normalized in {"soft", "hard"}:
            return normalized
        raise ValueError("SOVEREIGN_GUARD_BLOCK_MODE must be either 'soft' or 'hard'")

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        """Accept JSON array string from ECS (Terraform jsonencode) or list."""
        if value is None:
            return ["http://localhost:3000", "http://localhost:3001"]
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return ["http://localhost:3000", "http://localhost:3001"]
            if s.startswith("["):
                try:
                    parsed = json.loads(s)
                    return list(parsed) if isinstance(parsed, list) else [str(parsed)]
                except json.JSONDecodeError:
                    pass
            return [origin.strip() for origin in s.split(",") if origin.strip()]
        return ["http://localhost:3000", "http://localhost:3001"]

    @property
    def database_url(self) -> str:
        """Construct database URL"""
        base = f"postgresql://{self.PG_USER}:{self.PG_PASSWORD}@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DB}"
        if self.PG_SSLMODE != "disable":
            base += f"?sslmode={self.PG_SSLMODE}"
        return base


# Global settings instance
settings = Settings()
