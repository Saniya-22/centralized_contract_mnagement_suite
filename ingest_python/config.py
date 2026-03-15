import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------
# Configuration & Constants (Environment-Driven)
# ---------------------------------------------------------

NAMESPACE: str = os.getenv("REGULATIONS_NAMESPACE", os.getenv("NAMESPACE", "public-regulations"))
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_ENDPOINT: str = os.getenv("EMBEDDING_ENDPOINT", "https://api.openai.com/v1/embeddings")

# Unified chunk size regime (strategy doc Part 2.6)
TARGET_CHUNK_TOKENS: int = int(os.getenv("TARGET_CHUNK_TOKENS", "450"))
MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", "650"))
MIN_CHUNK_TOKENS: int = int(os.getenv("MIN_CHUNK_TOKENS", "100"))
MIN_ANCHOR_TOKENS: int = int(os.getenv("MIN_ANCHOR_TOKENS", "50"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "0"))
CLAUSE_SPLIT_TRIGGER_TOKENS: int = int(os.getenv("CLAUSE_SPLIT_TRIGGER_TOKENS", "400"))
KEEP_STANDALONE_ANCHOR_CHUNKS: bool = os.getenv("KEEP_STANDALONE_ANCHOR_CHUNKS", "true").lower() == "true"

# Backwards compatibility for code still using old names
CHUNK_SIZE: int = MAX_CHUNK_TOKENS
TARGET_CHUNK_MIN_TOKENS: int = MIN_CHUNK_TOKENS
TARGET_CHUNK_MAX_TOKENS: int = TARGET_CHUNK_TOKENS

MAX_CONCURRENT_PDFS: int = int(os.getenv("MAX_CONCURRENT_PDFS", 2))
EMBED_RATE_DELAY: float = float(os.getenv("EMBED_RATE_DELAY", 0.1))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 20))
INCLUDE_FILES_RAW: str = os.getenv("INCLUDE_FILES", "")

OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
DATABASE_URL: str = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("Missing required environment variable: DATABASE_URL")
PG_DENSE_TABLE: str = os.getenv('PG_DENSE_TABLE', 'embeddings_dense')
PG_SPARSE_TABLE: str = os.getenv('PG_SPARSE_TABLE', 'embeddings_sparse')
PG_SSLMODE: Optional[str] = os.getenv('PG_SSLMODE')

USE_STEMMING: bool = os.getenv("USE_STEMMING", "false").lower() == "true"
USE_STOPWORDS: bool = os.getenv("USE_STOPWORDS", "true").lower() == "true"

if not OPENAI_API_KEY:
    raise ValueError("Missing required environment variable: OPENAI_API_KEY")

# ---------------------------------------------------------
# Path Configurations
# ---------------------------------------------------------

from pathlib import Path

SPECIFICATIONS_DIR_RAW: str = os.getenv("SPECIFICATIONS_DIR", "./specifications")
SPECIFICATIONS_DIR: Path = Path(SPECIFICATIONS_DIR_RAW)

if not SPECIFICATIONS_DIR.exists():
    # Fallback to local dev path relative to the script's directory
    SPECIFICATIONS_DIR = Path(__file__).parent.parent / "specifications"
