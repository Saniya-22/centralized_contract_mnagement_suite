"""Vector search tools for regulatory document retrieval.

Exposes two LangChain tools:
  - search_regulations: hybrid dense+FTS search with RRF and GPT-4o-mini reranking
  - get_clause_by_reference: direct lookup of a specific FAR/DFARS/EM385 clause
"""

import logging
from typing import List, Dict, Any, Optional, Literal

from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field

from src.db.queries import VectorQueries
from src.tools.llm_tools import get_embedding
from src.services.reranker import rerank
from src.config import settings

logger = logging.getLogger(__name__)

# ── Token budget helper ────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Simple word-count based token estimator (mirrors JS _estimateTokens)."""
    word_count = len((text or "").split())
    return int(word_count * 1.3)


def _apply_token_budget(chunks: List[Dict], token_limit: int) -> List[Dict]:
    """Trim the chunk list so total estimated tokens stay within budget."""
    current = 0
    final: List[Dict] = []
    for chunk in chunks:
        tokens = _estimate_tokens(chunk.get("text") or chunk.get("content") or "")
        if current + tokens > token_limit:
            continue  # skip oversized chunk, try smaller ones (mirrors JS)
        final.append(chunk)
        current += tokens
    return final


# ── Input schemas ──────────────────────────────────────────────────────────────

class VectorSearchInput(BaseModel):
    """Input schema for search_regulations tool."""
    query: str = Field(description="Natural language search query about regulations")
    k: int = Field(default=10, ge=1, le=50, description="Number of results to return (1-50)")
    regulation_type: Optional[Literal["FAR", "DFARS", "EM385"]] = Field(
        default=None,
        description="Filter by specific regulation type: FAR, DFARS, or EM385"
    )
    search_mode: Literal["hybrid", "dense"] = Field(
        default="hybrid",
        description="'hybrid' uses dense+FTS+RRF (recommended), 'dense' uses vector only"
    )


class ClauseReferenceInput(BaseModel):
    """Input schema for get_clause_by_reference tool."""
    clause_reference: str = Field(
        description=(
            "Clause reference string, e.g. 'FAR 52.236-2', "
            "'DFARS 252.204-7012', 'EM 385 Section 05.A'"
        )
    )


# ── Tools ──────────────────────────────────────────────────────────────────────

class VectorSearchTool:
    """Wraps the regulations search tool for binding to LangChain agents."""

    @staticmethod
    @tool(args_schema=VectorSearchInput)
    def search_regulations(
        query: str,
        k: int = 10,
        regulation_type: Optional[str] = None,
        search_mode: str = "hybrid",
    ) -> List[Dict[str, Any]]:
        """Search regulatory documents (FAR, DFARS, EM385) using advanced hybrid retrieval.

        Uses a 3-stage pipeline:
          1. Dense vector search (cosine similarity via pgvector)
          2. Full-Text search (ts_rank_cd via PostgreSQL)
          3. Reciprocal Rank Fusion to merge + GPT-4o-mini reranking for precision

        Use this to find requirements, clauses, safety rules, or any compliance text.

        Examples:
            - "What are the requirements for small business set-asides?"
            - "Safety requirements for excavation work"
            - "Cost accounting standards for defense contracts"
        """
        try:
            logger.info(
                f"[VectorSearch] query='{query[:60]}...', k={k}, "
                f"type={regulation_type}, mode={search_mode}"
            )

            # Generate dense embedding
            query_embedding = get_embedding(query)

            # Run appropriate search
            if search_mode == "hybrid":
                fused = VectorQueries.hybrid_search(
                    query_embedding=query_embedding,
                    query_text=query,
                    k=k,
                    regulation_type=regulation_type,
                )
            else:
                fused = VectorQueries.dense_search(
                    query_embedding=query_embedding,
                    k=k,
                    regulation_type=regulation_type,
                )
                for doc in fused:
                    doc["rrf_score"] = doc.get("similarity", 0.0)
                    doc["final_score"] = doc["rrf_score"]

            # Rerank with GPT-4o-mini
            reranked = rerank(query, fused)

            # Apply token budget
            budgeted = _apply_token_budget(reranked, settings.RAG_TOKEN_LIMIT)

            # Format for agent consumption
            formatted: List[Dict[str, Any]] = []
            for idx, doc in enumerate(budgeted, 1):
                meta = doc.get("metadata") or {}
                formatted.append({
                    "rank": idx,
                    "content": doc.get("content") or doc.get("text", ""),
                    "source": doc.get("source_file") or meta.get("source", ""),
                    "regulation_type": meta.get("source", meta.get("regulation_type", "Unknown")),
                    "section": meta.get("part", meta.get("section", "N/A")),
                    "chunk_index": doc.get("chunk_index"),
                    "score": float(doc.get("rerank_score") or doc.get("rrf_score") or doc.get("final_score") or 0),
                    "retrieval_methods": doc.get("retrieval_methods", []),
                    "metadata": meta,
                })

            logger.info(
                f"[VectorSearch] Found {len(formatted)} results. "
                f"Token budget used: {_apply_token_budget.__name__} applied."
            )
            return formatted

        except Exception as exc:
            logger.error(f"[VectorSearch] Failed: {exc}", exc_info=True)
            return [{"error": str(exc), "message": "Search failed. Please try again."}]

    @staticmethod
    @tool(args_schema=ClauseReferenceInput)
    def get_clause_by_reference(clause_reference: str) -> Dict[str, Any]:
        """Retrieve the full text of a specific regulation clause by reference number.

        Use this when the user asks for an exact clause like:
          - 'What is FAR 52.236-2?'
          - 'Show me DFARS 252.204-7012'
          - 'Get EM 385 Section 05.A'

        The tool tries a direct text match first, then falls back to semantic search.
        """
        try:
            logger.info(f"[ClauseLookup] Looking up: '{clause_reference}'")
            result = VectorQueries.get_clause_by_reference(clause_reference)
            return result
        except Exception as exc:
            logger.error(f"[ClauseLookup] Failed: {exc}", exc_info=True)
            return {
                "found": False,
                "clause": None,
                "context": f"Error retrieving clause '{clause_reference}': {exc}",
            }

    def as_langchain_tools(self) -> List[StructuredTool]:
        """Return all tools as a list for binding to an agent LLM."""
        return [self.search_regulations, self.get_clause_by_reference]

    def as_langchain_tool(self) -> StructuredTool:
        """Return primary search tool (backward compatibility)."""
        return self.search_regulations

    def get_tool_definition(self) -> Dict[str, Any]:
        """Get primary tool definition for manual registration."""
        return {
            "name": "search_regulations",
            "description": self.search_regulations.description,
            "parameters": VectorSearchInput.model_json_schema(),
        }
