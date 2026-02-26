import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------
# Configuration & Constants (Environment-Driven)
# ---------------------------------------------------------

NAMESPACE: str = os.getenv("NAMESPACE", "public-regulations")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_ENDPOINT: str = os.getenv("EMBEDDING_ENDPOINT", "https://api.openai.com/v1/embeddings")

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))

MAX_CONCURRENT_PDFS: int = int(os.getenv("MAX_CONCURRENT_PDFS", 2))
EMBED_RATE_DELAY: float = float(os.getenv("EMBED_RATE_DELAY", 0.1))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 20))

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