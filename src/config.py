"""Configuration management for GovGig AI Backend"""

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
    PG_USER: str = "postgres"
    PG_PASSWORD: str
    PG_POOL_MIN: int = 2
    PG_POOL_MAX: int = 10
    
    # Database Tables
    PG_DENSE_TABLE: str = "embeddings_dense"
    REGULATIONS_NAMESPACE: str = "public-regulations"

    # Vector Search
    DENSE_TOP_K: int = 10
    SPARSE_TOP_K: int = 10
    HYBRID_DENSE_WEIGHT: float = 0.7
    HYBRID_SPARSE_WEIGHT: float = 0.3

    # Advanced RAG
    RRF_K: int = 60                               # Reciprocal Rank Fusion constant
    RERANKER_MODEL: str = "gpt-4o-mini"           # Model used for LLM reranking
    RERANKER_ENABLED: bool = True                 # Set False to skip LLM rerank, use RRF only
    RAG_TOKEN_LIMIT: int = 3200                   # Max tokens assembled into context for faster speed
    RETRIEVAL_TOP_K: int = 6                      # Primary retrieval size for regulation_search path
    REFLECTION_THRESHOLD: float = 0.35            # Heuristic confidence threshold before self-healing
    SELF_HEALING_SEARCH_K: int = 3                # Per expanded query search depth
    SELF_HEALING_MAX_QUERIES: int = 1             # Max expanded queries to execute
    SELF_HEALING_MAX_DOCS: int = 4                # Max additional docs added from self-healing
    MAX_DOC_CHARS_FOR_SYNTHESIS: int = 1200       # Per-document content trim before prompt assembly
    PILOT_SAFE_MODE: bool = True                  # Enforce evidence/citation guardrails
    PILOT_MIN_DOCS: int = 3                       # Minimum docs required for grounded answer
    PILOT_MIN_TOP_SCORE: float = 0.30             # Minimum normalized top doc score
    PILOT_MIN_AVG_SCORE: float = 0.20             # Minimum normalized average score

    # LLM Models (separate concerns)
    # MODEL_NAME:       used for the DataRetrieval tool-selector fallback
    # SYNTHESIZER_MODEL: used by _synthesize_response — gpt-4o-mini is 3-5x faster
    SYNTHESIZER_MODEL: str = "gpt-4o-mini"        # Fast synthesis model
    
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
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:3001"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
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
    
    @property
    def database_url(self) -> str:
        """Construct database URL"""
        return f"postgresql://{self.PG_USER}:{self.PG_PASSWORD}@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DB}"
    
    @property
    def async_database_url(self) -> str:
        """Construct async database URL"""
        return f"postgresql+asyncpg://{self.PG_USER}:{self.PG_PASSWORD}@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DB}"


# Global settings instance
settings = Settings()
