"""DB Migration: Add pgvector HNSW + GIN FTS indexes.

These two indexes are the single biggest remaining latency fix:

  HNSW index  → pgvector query: ~3-8s (full table scan) → ~30-80ms
  GIN index   → ts_rank_cd FTS: ~1-4s → ~20-50ms

Why HNSW over IVFFlat?
  - IVFFlat requires a fixed number of lists (clusters), tuned to row count.
    If data grows, recall degrades without re-index.
  - HNSW is a graph-based ANN structure that adapts dynamically.
  - Better recall at same speed. Recommended by pgvector maintainers.

CONCURRENTLY avoids table locks — the API stays live during indexing.
Build time: ~5-60 min depending on table size (1k rows → seconds, 500k → minutes).

Usage:
    cd backend_python
    python3 scripts/migrate_add_indexes.py [--dry-run]
"""

import sys
import os
import argparse
import logging
import psycopg2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Index definitions ─────────────────────────────────────────────────────────

TABLE = settings.PG_DENSE_TABLE

INDEXES = [
    {
        "name": "idx_embeddings_hnsw_cosine",
        "description": "HNSW approximate nearest-neighbour index (cosine distance)",
        "check_sql": """
            SELECT 1 FROM pg_indexes
            WHERE tablename = %(table)s
              AND indexname  = 'idx_embeddings_hnsw_cosine'
        """,
        # CONCURRENTLY cannot run inside a transaction block
        # m=16, ef_construction=64 are solid defaults for 1536-dim OpenAI embeddings
        "create_sql": f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_hnsw_cosine
            ON {TABLE}
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """,
    },
    {
        "name": "idx_embeddings_search_vector_gin",
        "description": "GIN index on search_vector for ts_rank_cd FTS",
        "check_sql": """
            SELECT 1 FROM pg_indexes
            WHERE tablename = %(table)s
              AND indexname  = 'idx_embeddings_search_vector_gin'
        """,
        "create_sql": f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_search_vector_gin
            ON {TABLE}
            USING gin(search_vector)
        """,
    },
]


def _connect():
    """Open a raw psycopg2 connection (not from pool) so we can set autocommit."""
    return psycopg2.connect(
        host=settings.PG_HOST,
        port=settings.PG_PORT,
        database=settings.PG_DB,
        user=settings.PG_USER,
        password=settings.PG_PASSWORD,
    )


def _index_exists(conn, check_sql: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(check_sql, {"table": table.split(".")[-1]})
        return cur.fetchone() is not None


def _table_row_count(conn, table: str) -> int:
    with conn.cursor() as cur:
        # Fast approximate count from pg_class
        cur.execute(
            "SELECT reltuples::bigint FROM pg_class WHERE relname = %s",
            (table.split(".")[-1],),
        )
        row = cur.fetchone()
        return row[0] if row else 0


def run_migration(dry_run: bool = False):
    logger.info(
        "Connecting to PostgreSQL at %s:%s/%s",
        settings.PG_HOST,
        settings.PG_PORT,
        settings.PG_DB,
    )
    conn = _connect()

    try:
        approx_rows = _table_row_count(conn, TABLE)
        logger.info("Table '%s' has approximately %s rows", TABLE, f"{approx_rows:,}")
        if approx_rows > 100_000:
            logger.warning(
                "Large table! HNSW index build may take 10-60 minutes. "
                "CONCURRENTLY means the API stays live throughout."
            )

        for idx in INDEXES:
            logger.info("── Checking: %s (%s)", idx["name"], idx["description"])

            if _index_exists(conn, idx["check_sql"], TABLE):
                logger.info("   ✅ Already exists — skipping.")
                continue

            logger.info("   ⏳ Creating index (CONCURRENTLY)...")
            if dry_run:
                logger.info(
                    "   [DRY RUN] Would execute:\n%s", idx["create_sql"].strip()
                )
                continue

            # CONCURRENTLY requires a connection that has NEVER started a transaction.
            # We cannot set autocommit on conn because the row-count query above has
            # already opened an implicit transaction on it. Open a brand-new connection.
            idx_conn = _connect()
            try:
                idx_conn.autocommit = True  # must be set before any statement
                with idx_conn.cursor() as cur:
                    cur.execute(idx["create_sql"])
                logger.info("   ✅ Index '%s' created successfully.", idx["name"])
            finally:
                idx_conn.close()

        logger.info("Migration complete. Verifying indexes…")
        for idx in INDEXES:
            exists = _index_exists(conn, idx["check_sql"], TABLE)
            status = "✅ present" if exists else "❌ MISSING"
            logger.info("  %s — %s", idx["name"], status)

    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add HNSW + GIN indexes to embeddings_dense"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print SQL without executing"
    )
    args = parser.parse_args()

    run_migration(dry_run=args.dry_run)
