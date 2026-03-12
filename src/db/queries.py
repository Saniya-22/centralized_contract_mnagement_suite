"""SQL queries for vector search and document retrieval.

Implements a 3-stage hybrid retrieval pipeline:
  1. Dense (pgvector cosine) search
  2. Full-Text (ts_rank_cd) search
  3. Reciprocal Rank Fusion (RRF) to merge results
Plus direct clause reference lookup for exact FAR/DFARS/EM385 lookups.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from psycopg2.extras import RealDictCursor
import logging

from src.db.connection import get_db_connection
from src.config import settings
import json
import hashlib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Section title substrings that indicate matrix/meta content to exclude from retrieval
META_SECTION_TITLE_SUBSTRINGS = (
    "matrix",
    "appendix",
    "solicitation provisions and contract clauses",
    "key:",
    "matrix notes",
)


class VectorQueries:
    """Database queries for vector operations."""
    _column_cache: Dict[str, set[str]] = {}

    @staticmethod
    def _table_parts(table_name: str) -> Tuple[str, str]:
        """Split a table reference into schema and table components."""
        if "." in table_name:
            schema, table = table_name.split(".", 1)
            return schema, table
        return "public", table_name

    @classmethod
    def _get_table_columns(cls, table_name: str) -> set[str]:
        """Fetch and cache column names for a configured table."""
        if table_name in cls._column_cache:
            return cls._column_cache[table_name]

        schema, table = cls._table_parts(table_name)
        sql = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s;
        """

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(sql, (schema, table))
                cols = {row["column_name"] for row in cursor.fetchall()}
                cls._column_cache[table_name] = cols
                return cols

    @classmethod
    def _has_column(cls, column_name: str, table_name: Optional[str] = None) -> bool:
        table_name = table_name or settings.PG_DENSE_TABLE
        return column_name in cls._get_table_columns(table_name)

    @classmethod
    def _optional_projection_sql(cls, table_name: Optional[str] = None) -> str:
        """Return resilient SELECT projection for optional legacy columns."""
        table_name = table_name or settings.PG_DENSE_TABLE
        has_chunk_index = cls._has_column("chunk_index", table_name)
        has_source_file = cls._has_column("source_file", table_name)
        chunk_expr = "chunk_index" if has_chunk_index else "NULL::int AS chunk_index"
        source_expr = (
            "source_file"
            if has_source_file
            else "COALESCE(metadata->>'source', '') AS source_file"
        )
        return f"{chunk_expr}, {source_expr}"

    # ─── Dense (pgvector) Search ──────────────────────────────────────────────

    @staticmethod
    def dense_search(
        query_embedding: List[float],
        k: int = None,
        regulation_type: Optional[str] = None,
        namespace: str = settings.REGULATIONS_NAMESPACE,
    ) -> List[Dict[str, Any]]:
        """Perform cosine-similarity vector search on embeddings_dense.

        Args:
            query_embedding: OpenAI dense embedding vector
            k: Number of results to return
            regulation_type: Optional filter ('FAR', 'DFARS', 'EM385')
            namespace: Row-level namespace filter (mirrors JS 'public-regulations')

        Returns:
            List of rows with similarity score and chunk_type='dense'
        """
        k = k or settings.DENSE_TOP_K

        optional_projection = VectorQueries._optional_projection_sql()
        sql = f"""
        SELECT
            id, namespace, text, metadata, {optional_projection},
            (1 - (embedding <=> %s::vector)) AS similarity
        FROM {settings.PG_DENSE_TABLE}
        WHERE 1=1
          AND (%s IS NULL OR namespace LIKE %s || '%%')
          AND (%s IS NULL OR metadata->>'source' = %s)
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """

        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql, (
                        query_embedding,
                        namespace, namespace,
                        regulation_type, regulation_type,
                        query_embedding,
                        k,
                    ))
                    results = cursor.fetchall()
                    logger.info(f"Dense search returned {len(results)} results")
                    return [
                        {**dict(r), "similarity": float(r["similarity"]), "chunk_type": "dense"}
                        for r in results
                    ]
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            raise

    # ─── Full-Text Search ─────────────────────────────────────────────────────

    @staticmethod
    def _build_or_tsquery(text: str) -> Optional[str]:
        """Convert a query string into a PostgreSQL OR tsquery expression.

        Mirrors JS _buildOrTsquery: splits on whitespace, drops single-char
        tokens, joins survivors with '|'.
        """
        tokens = [t for t in re.split(r"\W+", text) if len(t) > 1]
        return " | ".join(tokens) if tokens else None

    @staticmethod
    def fts_search(
        query_text: str,
        k: int = None,
        regulation_type: Optional[str] = None,
        namespace: str = settings.REGULATIONS_NAMESPACE,
    ) -> List[Dict[str, Any]]:
        """Full-text search using PostgreSQL ts_rank_cd + search_vector column.

        Mirrors JS ftsSearch(). Uses OR tsquery so partial matches score well.

        Args:
            query_text: Raw user query string
            k: Number of results to return
            regulation_type: Optional filter ('FAR', 'DFARS', 'EM385')
            namespace: Row-level namespace filter

        Returns:
            List of rows with ts_rank_cd score and chunk_type='fts'
        """
        k = k or settings.SPARSE_TOP_K
        or_expr = VectorQueries._build_or_tsquery(query_text)
        if not or_expr:
            return []

        optional_projection = VectorQueries._optional_projection_sql()
        sql = f"""
        SELECT
            id, namespace, text, metadata, {optional_projection},
            ts_rank_cd(search_vector, query, 32) AS rank
        FROM {settings.PG_DENSE_TABLE},
             to_tsquery('english', %s) AS query
        WHERE search_vector @@ query
          AND (%s IS NULL OR namespace LIKE %s || '%%')
          AND (%s IS NULL OR metadata->>'source' = %s)
        ORDER BY rank DESC
        LIMIT %s;
        """

        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql, (
                        or_expr,
                        namespace, namespace,
                        regulation_type, regulation_type,
                        k,
                    ))
                    results = cursor.fetchall()
                    logger.info(f"FTS search returned {len(results)} results")
                    return [
                        {**dict(r), "similarity": float(r["rank"]), "chunk_type": "fts"}
                        for r in results
                    ]
        except Exception as e:
            logger.error(f"FTS search failed: {e}")
            raise

    # ─── Reciprocal Rank Fusion ───────────────────────────────────────────────

    @staticmethod
    def _reciprocal_rank_fusion(
        dense_chunks: List[Dict[str, Any]],
        fts_chunks: List[Dict[str, Any]],
        rrf_k: int = None,
    ) -> List[Dict[str, Any]]:
        """Merge two ranked lists using Reciprocal Rank Fusion.

        Score formula: sum(1 / (rrf_k + rank)) for every list the doc appears in.
        Mirrors JS _reciprocalRankFusion().

        Args:
            dense_chunks: Results from dense_search (already ordered by similarity)
            fts_chunks:   Results from fts_search (already ordered by ts_rank_cd)
            rrf_k:        RRF constant k (default settings.RRF_K = 60)

        Returns:
            Merged list sorted by rrf_score DESC, with retrieval_methods tag
        """
        rrf_k = rrf_k if rrf_k is not None else settings.RRF_K
        score_map: Dict[int, float] = {}
        chunk_map: Dict[int, Dict[str, Any]] = {}

        for rank, chunk in enumerate(dense_chunks):
            cid = chunk["id"]
            score_map[cid] = score_map.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = {**chunk, "retrieval_methods": ["dense"]}

        for rank, chunk in enumerate(fts_chunks):
            cid = chunk["id"]
            score_map[cid] = score_map.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = {**chunk, "retrieval_methods": ["fts"]}
            else:
                chunk_map[cid]["retrieval_methods"].append("fts")

        fused = [
            {**chunk_map[cid], "rrf_score": score_map[cid]}
            for cid in score_map
        ]
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        logger.info(f"RRF fused {len(fused)} unique chunks")
        return fused

    @staticmethod
    def _is_meta_chunk(chunk: Dict[str, Any]) -> bool:
        """True if chunk is matrix/meta content (Matrix Notes, Appendix, etc.) to exclude from retrieval."""
        meta = chunk.get("metadata") or {}
        title = (meta.get("section_title") or "").lower()
        if any(sub in title for sub in META_SECTION_TITLE_SUBSTRINGS):
            return True
        # Fallback: check text snippet for matrix-style headers
        text = (chunk.get("text") or chunk.get("content") or "")[:300].lower()
        if "matrix notes" in text and ("key:" in text or "provision or clause" in text):
            return True
        return False

    @staticmethod
    def _section_number(chunk: Dict[str, Any]) -> Optional[str]:
        """Extract section_number (e.g. 52.211-10) from chunk metadata for boosting."""
        meta = chunk.get("metadata") or {}
        return meta.get("section_number") or meta.get("part") or None

    # ─── Hybrid Search (public API used by VectorSearchTool) ─────────────────

    @staticmethod
    def hybrid_search(
        query_embedding: List[float],
        query_text: str,
        k: int = None,
        regulation_type: Optional[str] = None,
        namespace: str = settings.REGULATIONS_NAMESPACE,
        exclude_meta_sections: bool = True,
        preferred_section_prefixes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run dense + FTS searches in parallel and merge with RRF.

        Optionally excludes matrix/meta chunks and boosts operational clause numbers
        (e.g. for mobilization / clauses-to-review queries).

        Args:
            query_embedding: Dense embedding for vector search
            query_text: Raw text for FTS search
            k: Candidates to retrieve from each search leg (final count returned)
            regulation_type: Optional filter ('FAR', 'DFARS', 'EM385')
            namespace: DB namespace
            exclude_meta_sections: If True, filter out Matrix/Appendix/meta chunks
            preferred_section_prefixes: Optional list of clause prefixes to boost (e.g. 52.211, 52.232)

        Returns:
            RRF-merged list sorted by score, with final_score alias for compat
        """
        k = k or settings.DENSE_TOP_K
        k_retrieve = k * 2 if exclude_meta_sections else k  # overfetch so filtering leaves enough

        import concurrent.futures

        dense_chunks = []
        fts_chunks = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(
                VectorQueries.dense_search,
                query_embedding, k=k_retrieve, regulation_type=regulation_type, namespace=namespace
            )
            future_fts = executor.submit(
                VectorQueries.fts_search,
                query_text, k=k_retrieve, regulation_type=regulation_type, namespace=namespace
            )

            try:
                dense_chunks = future_dense.result()
            except Exception as e:
                logger.error(f"Hybrid search dense leg failed: {e}")

            try:
                fts_chunks = future_fts.result()
            except Exception as e:
                logger.error(f"Hybrid search FTS leg failed: {e}")

        fused = VectorQueries._reciprocal_rank_fusion(dense_chunks, fts_chunks)

        if exclude_meta_sections:
            fused = [doc for doc in fused if not VectorQueries._is_meta_chunk(doc)]
            logger.info(f"After meta filter: {len(fused)} chunks")

        # Alias rrf_score → final_score; optionally boost preferred clause prefixes
        for doc in fused:
            doc["final_score"] = doc["rrf_score"]
            doc.setdefault("content", doc.get("text", ""))
            doc.setdefault("source_file", doc.get("metadata", {}).get("source", ""))
            if preferred_section_prefixes:
                sn = VectorQueries._section_number(doc)
                if sn and any(sn.startswith(prefix) for prefix in preferred_section_prefixes):
                    doc["final_score"] = doc["rrf_score"] * 1.4  # boost operational clauses

        fused.sort(key=lambda x: x["final_score"], reverse=True)
        return fused[:k]

    # ─── Direct Clause Text Lookup ────────────────────────────────────────────

    @staticmethod
    def direct_clause_search(
        clause_num: str,
        k: int = 10,
        namespace: str = settings.REGULATIONS_NAMESPACE,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """ILIKE text search for an exact clause number string.

        Mirrors JS directClauseSearch(). Ordered by how early the match appears
        in the text so title chunks rank first.

        Args:
            clause_num: Clause number string, e.g. '52.236-2'
            k: Max rows to return
            namespace: DB namespace
            source: Optional source filter ('FAR', 'DFARS', 'EM385')

        Returns:
            Matching rows with similarity=1.0 and chunk_type='direct'
        """
        optional_projection = VectorQueries._optional_projection_sql()
        sql = f"""
        SELECT
            id, namespace, text, metadata, {optional_projection},
            position(lower(%s) in lower(text)) AS match_pos
        FROM {settings.PG_DENSE_TABLE}
        WHERE text ILIKE %s
          AND (%s IS NULL OR namespace LIKE %s || '%%')
          AND (%s IS NULL OR metadata->>'source' = %s)
        ORDER BY match_pos ASC
        LIMIT %s;
        """
        pattern = f"%{clause_num}%"

        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql, (
                        clause_num,
                        pattern,
                        namespace, namespace,
                        source, source,
                        k,
                    ))
                    results = cursor.fetchall()
                    logger.info(f"Direct clause search for '{clause_num}' returned {len(results)} results")
                    return [
                        {**dict(r), "similarity": 1.0, "chunk_type": "direct"}
                        for r in results
                    ]
        except Exception as e:
            logger.error(f"Direct clause search failed: {e}")
            raise

    # ─── Clause Reference Lookup ──────────────────────────────────────────────

    @staticmethod
    def get_clause_by_reference(clause_reference: str) -> Dict[str, Any]:
        """Retrieve a regulation clause by its human-readable reference.

        Mirrors JS getClauseByReference(). Strategy:
          1. Parse reference with regex to extract source + clause number
          2. Try ILIKE direct text match (fastest, most exact)
          3. Fall back to FTS if no direct match
          4. Fall back to full hybrid pipeline as last resort

        Args:
            clause_reference: e.g. 'FAR 52.236-2', 'DFARS 252.204-7012',
                              'EM 385 Section 05.A'

        Returns:
            {found: bool, clause: dict|None, context: str}
        """
        # Try regex parse
        match = re.search(
            r'(FAR|DFARS|EM\s*385)\s*([\d]+[\.\-][\d\-]+)',
            clause_reference,
            re.IGNORECASE,
        )

        if not match:
            # Regex failed → try FTS fallback
            logger.info(
                f"Clause regex failed for '{clause_reference}', trying FTS fallback"
            )
            fts_results = VectorQueries.fts_search(
                clause_reference, k=5, namespace=settings.REGULATIONS_NAMESPACE
            )
            if fts_results:
                primary = fts_results[0]
                return {
                    "found": True,
                    "clause": {
                        "reference": clause_reference,
                        "source": primary.get("metadata", {}).get("source"),
                        "part": primary.get("metadata", {}).get("part"),
                        "text": primary.get("text", ""),
                        "related_clauses": primary.get("metadata", {}).get("clause_references", []),
                    },
                    "context": (
                        f"**Source: {primary.get('metadata', {}).get('source', 'Unknown')}**\n\n"
                        f"{primary.get('text', '')}"
                    ),
                }
            return {
                "found": False,
                "clause": None,
                "context": (
                    f"Could not parse clause reference: '{clause_reference}'. "
                    "Expected format: 'FAR 52.236-2' or 'DFARS 252.204-7012'"
                ),
            }

        source_raw = match.group(1).upper().replace(" ", "")
        source = "EM385" if source_raw.startswith("EM") else source_raw
        clause_num = match.group(2)

        logger.info(f"Parsed clause reference: source={source}, clause_num={clause_num}")

        # Stage 1: direct ILIKE search
        direct_results = VectorQueries.direct_clause_search(
            clause_num, k=10, namespace=settings.REGULATIONS_NAMESPACE, source=source
        )
        if direct_results:
            context_parts = []
            for chunk in direct_results[:5]:
                src = chunk.get("metadata", {}).get("source", "Unknown")
                part = chunk.get("metadata", {}).get("part", "")
                part_str = f" Part {part}" if part else ""
                context_parts.append(f"**Source: {src}{part_str}**\n\n{chunk.get('text', '')}")

            primary = direct_results[0]
            return {
                "found": True,
                "confidence": 1.0,
                "clause": {
                    "reference": clause_reference,
                    "source": primary.get("metadata", {}).get("source"),
                    "part": primary.get("metadata", {}).get("part"),
                    "text": primary.get("text", ""),
                    "related_clauses": primary.get("metadata", {}).get("clause_references", []),
                },
                "context": "\n\n---\n\n".join(context_parts),
            }

        # Stage 2: fall back to full hybrid pipeline
        logger.info(
            f"Direct clause search found nothing (0 matches) for '{clause_num}', "
            "falling back to hybrid fuzzy search"
        )
        from src.tools.llm_tools import get_embedding  # avoid circular import

        query_text = f"{source} {clause_num}"
        embedding = get_embedding(query_text)
        
        # Ensure normalization for the filter
        reg_type = source.strip().upper() if source else None
        
        fused_results = VectorQueries.hybrid_search(
            query_embedding=embedding,
            query_text=query_text,
            k=5,
            regulation_type=reg_type,
            namespace=settings.REGULATIONS_NAMESPACE,
        )

        if not fused_results:
            return {
                "found": False,
                "confidence": 0.0,
                "clause": None,
                "context": f"No results found for '{clause_reference}'.",
            }

        context_parts = []
        for chunk in fused_results:
            src = chunk.get("metadata", {}).get("source", "Unknown")
            part = chunk.get("metadata", {}).get("part", "")
            part_str = f" Part {part}" if part else ""
            context_parts.append(f"**Source: {src}{part_str}**\n\n{chunk.get('text', chunk.get('content', ''))}")

        # For fallback, found=False indicates it's NOT an exact match
        primary = fused_results[0]
        return {
            "found": False,
            "confidence": float(primary.get("rrf_score", 0.05)), # Low relative score
            "clause": {
                "reference": clause_reference,
                "source": primary.get("metadata", {}).get("source"),
                "part": primary.get("metadata", {}).get("part"),
                "text": primary.get("text", primary.get("content", "")),
                "related_clauses": primary.get("metadata", {}).get("clause_references", []),
            },
            "context": "\n\n---\n\n".join(context_parts),
            "fuzzy_results": fused_results
        }

    # ─── Legacy helpers (kept for backward compatibility) ─────────────────────

    @staticmethod
    def get_document_by_id(doc_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific document by ID."""
        optional_projection = VectorQueries._optional_projection_sql()
        sql = f"""
        SELECT id, text AS content, metadata, {optional_projection}
        FROM {settings.PG_DENSE_TABLE}
        WHERE id = %s;
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql, (doc_id,))
                    result = cursor.fetchone()
                    return dict(result) if result else None
        except Exception as e:
            logger.error(f"Get document by ID failed: {e}")
            raise

    @staticmethod
    def get_surrounding_chunks(
        chunk_index: int,
        source_file: str,
        window: int = 2,
    ) -> List[Dict[str, Any]]:
        """Get surrounding chunks for context window expansion."""
        has_chunk_index = VectorQueries._has_column("chunk_index")
        has_source_file = VectorQueries._has_column("source_file")
        if not has_chunk_index or not has_source_file:
            logger.warning(
                "Surrounding chunk lookup skipped because chunk_index/source_file "
                "columns are not present in %s",
                settings.PG_DENSE_TABLE,
            )
            return []

        sql = f"""
        SELECT id, text AS content, metadata, chunk_index, source_file
        FROM {settings.PG_DENSE_TABLE}
        WHERE source_file = %s
          AND chunk_index BETWEEN %s AND %s
        ORDER BY chunk_index;
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql, (
                        source_file,
                        chunk_index - window,
                        chunk_index + window,
                    ))
                    results = cursor.fetchall()
                    return [dict(r) for r in results]
        except Exception as e:
            logger.error(f"Get surrounding chunks failed: {e}")
            raise

    # ─── API Response Caching ────────────────────────────────────────────────

    @staticmethod
    def _cache_hash(query: str, cot: bool = True, cache_scope: Optional[str] = None) -> str:
        """Build a stable cache hash; optionally scope by user/thread context."""
        key = f"{query.strip().lower()}|cot:{cot}"
        if cache_scope:
            key += f"|scope:{cache_scope.strip().lower()}"
        return hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def init_cache_table():
        """Ensure the api_response_cache table exists."""
        sql = """
        CREATE TABLE IF NOT EXISTS api_response_cache (
            query_hash TEXT PRIMARY KEY,
            query_text TEXT,
            response_data JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP WITH TIME ZONE
        );
        CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON api_response_cache (expires_at);
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql)
                    logger.info("API response cache table verified")
        except Exception as e:
            logger.error(f"Failed to init cache table: {e}")

    @staticmethod
    def get_cached_response(
        query: str,
        cot: bool = True,
        cache_scope: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a cached response if valid and not expired."""
        query_hash = VectorQueries._cache_hash(query, cot, cache_scope)

        sql = """
        SELECT response_data
        FROM api_response_cache
        WHERE query_hash = %s
          AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP);
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql, (query_hash,))
                    result = cursor.fetchone()
                    if result:
                        logger.info(f"Cache hit for query: {query[:50]}...")
                        return result["response_data"]
            return None
        except Exception as e:
            logger.error(f"Cache lookup failed: {e}")
            return None

    @staticmethod
    def set_cached_response(
        query: str,
        response: Dict[str, Any],
        cot: bool = True,
        ttl_hours: int = 4,
        cache_scope: Optional[str] = None,
    ):
        """Store a response in the cache with a TTL."""
        query_hash = VectorQueries._cache_hash(query, cot, cache_scope)
        expires_at = datetime.now() + timedelta(hours=ttl_hours)

        sql = """
        INSERT INTO api_response_cache (query_hash, query_text, response_data, expires_at)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (query_hash) DO UPDATE
        SET response_data = EXCLUDED.response_data,
            expires_at = EXCLUDED.expires_at,
            created_at = CURRENT_TIMESTAMP;
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (
                        query_hash,
                        query,
                        json.dumps(response),
                        expires_at
                    ))
                    logger.debug(f"Cached response for query: {query[:50]}...")
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")

    # ─── User Feedback ──────────────────────────────────────────────────────────

    @staticmethod
    def init_user_feedback_table():
        """Ensure the user_feedback table exists."""
        sql = """
        CREATE TABLE IF NOT EXISTS user_feedback (
            id                  SERIAL PRIMARY KEY,
            user_id             UUID NOT NULL,
            query_id            UUID NOT NULL,
            feedback_response   TEXT NOT NULL CHECK (feedback_response IN ('good', 'bad')),
            created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_user_feedback_user_id ON user_feedback (user_id);
        CREATE INDEX IF NOT EXISTS idx_user_feedback_query_id ON user_feedback (query_id);
        CREATE INDEX IF NOT EXISTS idx_user_feedback_created_at ON user_feedback (created_at);
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql)
                    logger.info("User feedback table verified")
        except Exception as e:
            logger.error(f"Failed to init user_feedback table: {e}")

    @staticmethod
    def insert_user_feedback(user_id: str, query_id: str, feedback_response: str) -> bool:
        """Insert a feedback row. user_id and query_id must be valid UUID strings."""
        sql = """
        INSERT INTO user_feedback (user_id, query_id, feedback_response)
        VALUES (%s::uuid, %s::uuid, %s);
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (user_id, query_id, feedback_response))
                    logger.debug(f"Feedback recorded: query_id={query_id}, response={feedback_response}")
                    return True
        except Exception as e:
            logger.error(f"Failed to insert feedback: {e}")
            return False

    @staticmethod
    def get_feedback_summary() -> Dict[str, int]:
        """Return total, positive, and negative feedback counts."""
        sql = """
        SELECT
            COUNT(*)::int AS feedback_total,
            COUNT(*) FILTER (WHERE feedback_response = 'good')::int AS feedback_positive,
            COUNT(*) FILTER (WHERE feedback_response = 'bad')::int AS feedback_negative
        FROM user_feedback;
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql)
                    row = cursor.fetchone()
                    if row:
                        return dict(row)
            return {"feedback_total": 0, "feedback_positive": 0, "feedback_negative": 0}
        except Exception as e:
            logger.error(f"Failed to get feedback summary: {e}")
            return {"feedback_total": 0, "feedback_positive": 0, "feedback_negative": 0}

    # ─── Auth (users, login, lockout) ──────────────────────────────────────────

    @staticmethod
    def init_auth_tables():
        """Ensure users and auth_audit_log tables exist."""
        sql = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            full_name VARCHAR(255) NOT NULL,
            user_id UUID UNIQUE DEFAULT gen_random_uuid(),
            email VARCHAR(255) NOT NULL UNIQUE,
            hashed_password VARCHAR(255) NOT NULL,
            is_locked BOOLEAN DEFAULT FALSE,
            lock_until TIMESTAMP WITH TIME ZONE,
            failed_login_attempts INTEGER DEFAULT 0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        ALTER TABLE users ADD COLUMN IF NOT EXISTS user_id UUID UNIQUE DEFAULT gen_random_uuid();
        CREATE INDEX IF NOT EXISTS idx_users_user_id ON users (user_id);
        CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);
        CREATE INDEX IF NOT EXISTS idx_users_lock ON users (is_locked, lock_until);
        CREATE TABLE IF NOT EXISTS auth_audit_log (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            email VARCHAR(255),
            event_type VARCHAR(50) NOT NULL,
            details JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_auth_audit_user ON auth_audit_log (user_id);
        CREATE INDEX IF NOT EXISTS idx_auth_audit_created ON auth_audit_log (created_at);
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql)
                    logger.info("Auth tables (users, auth_audit_log) created/verified")
        except Exception as e:
            logger.error(f"Failed to init auth tables: {e}")

    @staticmethod
    def auth_tables_exist() -> bool:
        """Return True if users and auth_audit_log tables exist in the database."""
        sql = """
        SELECT COUNT(*) AS n
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name IN ('users', 'auth_audit_log');
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql)
                    row = cursor.fetchone()
                    return row and row.get("n") == 2
        except Exception as e:
            logger.error(f"Auth tables check failed: {e}")
            return False

    @staticmethod
    def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
        """Return user row by email or None."""
        sql = "SELECT id, user_id, full_name, email, hashed_password, is_locked, lock_until, failed_login_attempts FROM users WHERE email = %s;"
        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql, (email.strip().lower(),))
                    row = cursor.fetchone()
                    return dict(row) if row else None
        except Exception as e:
            logger.error(f"get_user_by_email failed: {e}")
            return None

    @staticmethod
    def create_user(full_name: str, email: str, hashed_password: str) -> Optional[Dict[str, Any]]:
        """Insert user; return dict with id and user_id (UUID) or None on duplicate/error."""
        sql = """
        INSERT INTO users (full_name, email, hashed_password)
        VALUES (%s, %s, %s)
        RETURNING id, user_id;
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql, (full_name.strip(), email.strip().lower(), hashed_password))
                    row = cursor.fetchone()
                    return dict(row) if row else None
        except Exception as e:
            logger.error(f"create_user failed: {e}")
            raise

    @staticmethod
    def update_login_success(user_id: int) -> None:
        """Reset failed_login_attempts and lock_until after successful login."""
        sql = """
        UPDATE users SET failed_login_attempts = 0, is_locked = FALSE, lock_until = NULL, updated_at = CURRENT_TIMESTAMP WHERE id = %s;
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (user_id,))
        except Exception as e:
            logger.error(f"update_login_success failed: {e}")

    @staticmethod
    def update_login_failure(user_id: int, lock_until: Optional[datetime] = None) -> bool:
        """Increment failed_login_attempts; if lock_until set, set lock. Returns True if account is now locked."""
        if lock_until is not None:
            sql = """
            UPDATE users SET failed_login_attempts = 5, is_locked = TRUE, lock_until = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s;
            """
            try:
                with get_db_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(sql, (lock_until, user_id))
                return True
            except Exception as e:
                logger.error(f"update_login_failure (lock) failed: {e}")
                return False
        sql = """
        UPDATE users SET failed_login_attempts = failed_login_attempts + 1, updated_at = CURRENT_TIMESTAMP WHERE id = %s;
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (user_id,))
            return False
        except Exception as e:
            logger.error(f"update_login_failure failed: {e}")
            return False

    @staticmethod
    def log_auth_audit(user_id: Optional[int], email: Optional[str], event_type: str, details: Optional[Dict] = None) -> None:
        """Insert into auth_audit_log."""
        sql = "INSERT INTO auth_audit_log (user_id, email, event_type, details) VALUES (%s, %s, %s, %s);"
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (user_id, email, event_type, json.dumps(details or {})))
        except Exception as e:
            logger.error(f"log_auth_audit failed: {e}")

    # ─── Chat History (existing schema: session_id, user_id, role, content) ──────

    @staticmethod
    def init_chat_history_table():
        """Ensure chat_history table exists (same schema as setupChatHistory.sql)."""
        sql = """
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(255) NOT NULL,
            user_id VARCHAR(255),
            role VARCHAR(20) NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history (session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history (user_id);
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql)
                    logger.info("Chat history table created/verified")
        except Exception as e:
            logger.error(f"Failed to init chat_history table: {e}")

    @staticmethod
    def get_chat_history(session_id: str, user_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Return chat history for a session (thread). Optionally scope by user_id. Format: [{role, text}, ...]."""
        if user_id:
            sql = """
            SELECT role, content FROM chat_history
            WHERE session_id = %s AND (user_id IS NULL OR user_id = %s)
            ORDER BY created_at ASC LIMIT %s;
            """
            params = (session_id, str(user_id), limit)
        else:
            sql = """
            SELECT role, content FROM chat_history
            WHERE session_id = %s
            ORDER BY created_at ASC LIMIT %s;
            """
            params = (session_id, limit)
        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql, params)
                    rows = cursor.fetchall()
                    return [{"role": r["role"], "text": r["content"]} for r in rows]
        except Exception as e:
            logger.error(f"get_chat_history failed: {e}")
            return []

    @staticmethod
    def insert_chat_message(session_id: str, user_id: Optional[str], role: str, content: str) -> None:
        """Insert one message into chat_history. role must be 'user' or 'assistant'."""
        if role not in ("user", "assistant"):
            logger.warning(f"insert_chat_message: invalid role {role}, skipping")
            return
        sql = """
        INSERT INTO chat_history (session_id, user_id, role, content)
        VALUES (%s, %s, %s, %s);
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (session_id, str(user_id) if user_id else None, role, content or ""))
        except Exception as e:
            logger.error(f"insert_chat_message failed: {e}")

    @staticmethod
    def list_chat_threads(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """List threads (conversations) for a user. Returns thread_id, updated_at, preview (first user message, truncated)."""
        sql = """
        SELECT
            t.session_id,
            t.updated_at,
            (
                SELECT LEFT(h2.content, 80)
                FROM chat_history h2
                WHERE h2.session_id = t.session_id
                  AND (h2.user_id IS NULL OR h2.user_id = %s)
                  AND h2.role = 'user'
                ORDER BY h2.created_at ASC
                LIMIT 1
            ) AS preview
        FROM (
            SELECT session_id, MAX(created_at) AS updated_at
            FROM chat_history
            WHERE user_id IS NULL OR user_id = %s
            GROUP BY session_id
        ) t
        ORDER BY t.updated_at DESC
        LIMIT %s;
        """
        params = (str(user_id), str(user_id), limit)
        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql, params)
                    rows = cursor.fetchall()
                    return [
                        {
                            "thread_id": r["session_id"],
                            "updated_at": r["updated_at"].isoformat() if hasattr(r["updated_at"], "isoformat") else str(r["updated_at"]),
                            "preview": (r["preview"] or "").strip() or None,
                        }
                        for r in rows
                    ]
        except Exception as e:
            logger.error(f"list_chat_threads failed: {e}")
            return []

    # ─── Query Analytics ──────────────────────────────────────────────────────

    @staticmethod
    def init_analytics_table():
        """Ensure the query_analytics table exists."""
        sql = """
        CREATE TABLE IF NOT EXISTS query_analytics (
            id                  SERIAL PRIMARY KEY,
            query_text          TEXT NOT NULL,
            query_hash          TEXT NOT NULL,
            user_id             TEXT,
            thread_id           TEXT,
            intent              TEXT,
            regulation_types    TEXT[],
            confidence          FLOAT,
            quality_score       FLOAT,
            citation_coverage   FLOAT,
            groundedness_score  FLOAT,
            evidence_score      FLOAT,
            low_confidence      BOOLEAN,
            doc_count           INT,
            reflection_triggered BOOLEAN DEFAULT FALSE,
            was_cached          BOOLEAN DEFAULT FALSE,
            latency_ms          INT,
            error_count         INT DEFAULT 0,
            errors              TEXT[],
            source              TEXT DEFAULT 'rest',
            created_at          TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_analytics_created ON query_analytics (created_at);
        CREATE INDEX IF NOT EXISTS idx_analytics_intent ON query_analytics (intent);
        CREATE INDEX IF NOT EXISTS idx_analytics_quality ON query_analytics (quality_score);
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql)
                    logger.info("Query analytics table verified")
        except Exception as e:
            logger.error(f"Failed to init analytics table: {e}")

    @staticmethod
    def log_query_analytics(data: Dict[str, Any]):
        """Insert a single analytics row. Fire-and-forget — never raises."""
        query_text = data.get("query_text", "")
        query_hash = hashlib.sha256(query_text.strip().lower().encode()).hexdigest()

        sql = """
        INSERT INTO query_analytics (
            query_text, query_hash, user_id, thread_id,
            intent, regulation_types, confidence,
            quality_score, citation_coverage, groundedness_score, evidence_score,
            low_confidence, doc_count, reflection_triggered,
            was_cached, latency_ms, error_count, errors, source
        ) VALUES (
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s, %s, %s
        );
        """
        try:
            qm = data.get("quality_metrics") or {}
            # Detect if reflection was triggered from agent_path
            agent_path = data.get("agent_path", [])
            reflection_triggered = any(
                "expand" in step.lower() or "re-search" in step.lower() or "healing" in step.lower()
                for step in agent_path
            ) if agent_path else False

            errors_list = data.get("errors", [])

            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (
                        query_text,
                        query_hash,
                        data.get("user_id"),
                        data.get("thread_id"),
                        data.get("intent"),
                        data.get("regulation_types", []),
                        data.get("confidence"),
                        qm.get("quality_score"),
                        qm.get("citation_coverage"),
                        qm.get("groundedness_score"),
                        qm.get("evidence_score"),
                        data.get("low_confidence"),
                        data.get("doc_count", 0),
                        reflection_triggered,
                        data.get("was_cached", False),
                        data.get("latency_ms"),
                        len(errors_list),
                        errors_list if errors_list else None,
                        data.get("source", "rest"),
                    ))
                    logger.debug(f"Analytics logged for query: {query_text[:50]}...")
        except Exception as e:
            logger.error(f"Failed to log analytics: {e}")

    @staticmethod
    def get_analytics_summary(hours: int = 24) -> Dict[str, Any]:
        """Return aggregate stats for the last N hours."""
        sql = """
        SELECT
            COUNT(*)                                        AS total_queries,
            COUNT(*) FILTER (WHERE was_cached)              AS cached_queries,
            COUNT(*) FILTER (WHERE intent = 'regulation_search') AS regulation_searches,
            COUNT(*) FILTER (WHERE intent = 'clause_lookup')     AS clause_lookups,
            COUNT(*) FILTER (WHERE intent = 'out_of_scope')      AS out_of_scope,
            COUNT(*) FILTER (WHERE reflection_triggered)         AS reflection_count,
            COUNT(*) FILTER (WHERE low_confidence)               AS low_confidence_count,
            COUNT(*) FILTER (WHERE error_count > 0)              AS error_queries,
            ROUND(AVG(quality_score)::numeric, 4)                AS avg_quality_score,
            ROUND(AVG(citation_coverage)::numeric, 4)            AS avg_citation_coverage,
            ROUND(AVG(groundedness_score)::numeric, 4)           AS avg_groundedness,
            ROUND(AVG(latency_ms)::numeric, 0)                  AS avg_latency_ms,
            ROUND(PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY latency_ms)::numeric, 0) AS p50_latency_ms,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms)::numeric, 0) AS p95_latency_ms,
            ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms)::numeric, 0) AS p99_latency_ms
        FROM query_analytics
        WHERE created_at > NOW() - INTERVAL '%s hours';
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql, (hours,))
                    row = cursor.fetchone()
                    if row:
                        result = dict(row)
                        # Convert Decimal types to float for JSON serialization
                        for key, val in result.items():
                            if val is not None and not isinstance(val, (int, float, str, bool)):
                                result[key] = float(val)
                        result["period_hours"] = hours
                        # Add feedback counts (all-time)
                        feedback = VectorQueries.get_feedback_summary()
                        result.update(feedback)
                        return result
                    out = {"total_queries": 0, "period_hours": hours}
                    out.update(VectorQueries.get_feedback_summary())
                    return out
        except Exception as e:
            logger.error(f"Failed to get analytics summary: {e}")
            return {"error": str(e), "period_hours": hours}
