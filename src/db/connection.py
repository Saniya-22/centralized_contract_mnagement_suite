"""Database connection management with pooling"""

import psycopg2
from psycopg2 import pool, OperationalError
from psycopg2.extras import RealDictCursor
from contextlib import asynccontextmanager, contextmanager
from typing import Generator, Optional, AsyncGenerator
import logging
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.config import settings

logger = logging.getLogger(__name__)


class DatabaseConnectionPool:
    """Singleton connection pool for PostgreSQL"""
    
    _instance: Optional['DatabaseConnectionPool'] = None
    _pool: Optional[pool.ThreadedConnectionPool] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Initialize lazily on first get_connection() call so imports and tests
        # do not require immediate DB connectivity.
        pass
    
    def _initialize_pool(self):
        """Initialize the connection pool with retries"""
        from tenacity import retry, stop_after_attempt, wait_fixed
        
        @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
        def do_init():
            return psycopg2.pool.ThreadedConnectionPool(
                minconn=settings.PG_POOL_MIN,
                maxconn=settings.PG_POOL_MAX,
                host=settings.PG_HOST,
                port=settings.PG_PORT,
                database=settings.PG_DB,
                user=settings.PG_USER,
                password=settings.PG_PASSWORD,
                cursor_factory=RealDictCursor
            )

        try:
            self._pool = do_init()
            logger.info(
                f"Database connection pool initialized: "
                f"{settings.PG_POOL_MIN}-{settings.PG_POOL_MAX} connections"
            )
        except Exception as e:
            logger.error(f"Failed to initialize database pool after retries: {e}")
            raise
    
    def get_connection(self):
        """Get a connection from the pool"""
        if self._pool is None:
            self._initialize_pool()
        
        try:
            conn = self._pool.getconn()
            logger.debug("Connection acquired from pool")
            return conn
        except OperationalError as e:
            logger.error(f"Failed to get connection from pool: {e}")
            raise
    
    def return_connection(self, conn):
        """Return a connection to the pool"""
        if self._pool is not None and conn is not None:
            try:
                self._pool.putconn(conn)
                logger.debug("Connection returned to pool")
            except Exception as e:
                # "trying to put unkeyed connection" is cosmetic — the connection
                # is still usable.  Never close it here; that poisons the pool.
                logger.debug(f"putconn warning (non-fatal): {e}")
    
    def close_all(self):
        """Close all connections in the pool"""
        if self._pool is not None:
            self._pool.closeall()
            logger.info("All database connections closed")
            self._pool = None


# Global pool instance
_db_pool = DatabaseConnectionPool()


@contextmanager
def get_db_connection() -> Generator:
    """Context manager for database connections.
    
    Usage:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM table")
    """
    conn = None
    try:
        conn = _db_pool.get_connection()
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            _db_pool.return_connection(conn)


class CheckpointerManager:
    """Manages the lifecycle of the LangGraph Postgres checkpointer."""
    
    _pool: Optional[AsyncConnectionPool] = None
    _checkpointer: Optional[AsyncPostgresSaver] = None

    @classmethod
    async def get_checkpointer(cls) -> AsyncPostgresSaver:
        """Initialize and return the checkpointer."""
        if cls._checkpointer is None:
            logger.info("Initializing LangGraph Postgres checkpointer")
            cls._pool = AsyncConnectionPool(
                conninfo=settings.database_url,
                max_size=settings.PG_POOL_MAX,
                kwargs={"autocommit": True}
            )
            cls._checkpointer = AsyncPostgresSaver(cls._pool)
            # Ensure tables are created
            await cls._checkpointer.setup()
            logger.info("LangGraph Postgres checkpointer initialized")
        return cls._checkpointer

    @classmethod
    async def close(cls):
        """Close the checkpointer pool."""
        if cls._pool:
            await cls._pool.close()
            cls._pool = None
            cls._checkpointer = None
            logger.info("LangGraph Postgres checkpointer pool closed")


from fastapi.concurrency import run_in_threadpool

async def execute_in_db(func, *args, **kwargs):
    """Execute a database function safely in the thread pool."""
    return await run_in_threadpool(func, *args, **kwargs)


def close_db_pool():
    """Close the database connection pool"""
    _db_pool.close_all()


# Test connection on import
def test_connection():
    """Test database connection"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                logger.info("Database connection test successful")
                return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False
