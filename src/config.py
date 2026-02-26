"""Configuration management for GovGig AI Backend"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os


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

    # Vector Search
    DENSE_TOP_K: int = 10
    SPARSE_TOP_K: int = 10
    HYBRID_DENSE_WEIGHT: float = 0.7
    HYBRID_SPARSE_WEIGHT: float = 0.3

    # Advanced RAG
    RRF_K: int = 60                               # Reciprocal Rank Fusion constant
    RERANKER_MODEL: str = "gpt-4o-mini"           # Model used for LLM reranking
    RERANKER_ENABLED: bool = True                 # Set False to skip LLM rerank, use RRF only
    RAG_TOKEN_LIMIT: int = 4000                   # Max tokens assembled into context for faster speed

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
