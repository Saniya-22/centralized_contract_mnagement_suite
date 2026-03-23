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


def _dedup_by_id(chunks: List[Dict]) -> List[Dict]:
    seen = set()
    out: List[Dict] = []
    for c in chunks:
        cid = c.get("id") or c.get("doc_id")
        if cid and cid in seen:
            continue
        if cid:
            seen.add(cid)
        out.append(c)
    return out


def _extract_refs_from_docs(
    docs: List[Dict[str, Any]], max_docs: int = 8
) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for d in (docs or [])[:max_docs]:
        meta = d.get("metadata") or {}
        for r in meta.get("clause_references") or []:
            if isinstance(r, dict) and r.get("type") and r.get("clause"):
                refs.append(r)
    return refs


def _extract_section_numbers(
    docs: List[Dict[str, Any]], max_docs: int = 8
) -> List[str]:
    out: List[str] = []
    for d in (docs or [])[:max_docs]:
        meta = d.get("metadata") or {}
        sn = meta.get("section_number")
        if sn:
            out.append(str(sn))
    # de-dup preserve order
    seen = set()
    uniq = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _build_context_prioritized(
    primary: List[Dict[str, Any]],
    anchors: List[Dict[str, Any]],
    refs: List[Dict[str, Any]],
    token_limit: int,
) -> List[Dict[str, Any]]:
    """Priority: primary, then anchors, then refs. If over limit, drop refs first."""
    primary = _dedup_by_id(primary)
    anchors = _dedup_by_id(anchors)
    refs = _dedup_by_id(refs)

    # Avoid duplicating anchors already present in primary.
    primary_ids = {d.get("id") for d in primary if d.get("id")}
    anchors = [a for a in anchors if not (a.get("id") and a.get("id") in primary_ids)]

    # Avoid duplicating refs already present.
    taken_ids = set(primary_ids) | {a.get("id") for a in anchors if a.get("id")}
    refs = [r for r in refs if not (r.get("id") and r.get("id") in taken_ids)]

    def _accumulate(
        cands: List[Dict[str, Any]], current: int, out: List[Dict[str, Any]]
    ) -> int:
        for c in cands:
            txt = c.get("text") or c.get("content") or ""
            t = _estimate_tokens(txt)
            if current + t > token_limit:
                continue
            out.append(c)
            current += t
        return current

    out: List[Dict[str, Any]] = []
    current = 0
    current = _accumulate(primary, current, out)
    current = _accumulate(anchors, current, out)
    current = _accumulate(refs, current, out)
    return out


# ── Input schemas ──────────────────────────────────────────────────────────────


class VectorSearchInput(BaseModel):
    """Input schema for search_regulations tool."""

    query: str = Field(description="Natural language search query about regulations")
    k: int = Field(
        default=10, ge=1, le=50, description="Number of results to return (1-50)"
    )
    regulation_type: Optional[Literal["FAR", "DFARS", "EM385"]] = Field(
        default=None,
        description="Filter by specific regulation type: FAR, DFARS, or EM385",
    )
    search_mode: Literal["hybrid", "dense"] = Field(
        default="hybrid",
        description="'hybrid' uses dense+FTS+RRF (recommended), 'dense' uses vector only",
    )
    exclude_meta_sections: bool = Field(
        default=True,
        description="Exclude matrix/notes/appendix meta chunks from results (recommended)",
    )
    preferred_section_prefixes: Optional[List[str]] = Field(
        default=None,
        description="Optional clause number prefixes to boost (e.g. 52.211, 52.232 for mobilization)",
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
        exclude_meta_sections: bool = True,
        preferred_section_prefixes: Optional[List[str]] = None,
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

            # Generate dense embedding (used for primary retrieval and ref expansion ranking)
            query_embedding = get_embedding(query)

            # Run appropriate search
            if search_mode == "hybrid":
                fused = VectorQueries.hybrid_search(
                    query_embedding=query_embedding,
                    query_text=query,
                    k=k,
                    regulation_type=regulation_type,
                    exclude_meta_sections=exclude_meta_sections,
                    preferred_section_prefixes=preferred_section_prefixes,
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

            # v2: Anchor fetch + ranked reference expansion (top 2)
            section_numbers = _extract_section_numbers(reranked, max_docs=8)
            anchor_rows = VectorQueries.get_anchor_chunks_for_sections(
                section_numbers=section_numbers,
                namespace=settings.REGULATIONS_NAMESPACE,
                source=regulation_type,
                limit=25,
            )
            anchors = [
                {
                    **r,
                    "content": r.get("text", ""),
                    "retrieval_methods": (r.get("retrieval_methods") or [])
                    + ["anchor_lookup"],
                }
                for r in anchor_rows
            ]

            refs = _extract_refs_from_docs(reranked, max_docs=8)
            ref_rows = VectorQueries.resolve_reference_chunks(
                query_embedding=query_embedding,
                refs=refs,
                namespace=settings.REGULATIONS_NAMESPACE,
                source=regulation_type,
                exclude_ids=[d.get("id") for d in reranked if d.get("id")],
                per_ref_limit=5,
                total_limit=20,
            )
            ref_ranked = [
                {
                    **r,
                    "content": r.get("text", ""),
                    "retrieval_methods": (r.get("retrieval_methods") or [])
                    + ["ref_expand"],
                }
                for r in ref_rows
            ]
            # Expand a bit more than top-2 to improve completeness for clause-heavy answers,
            # while still keeping context tight under the token budget.
            ref_top4 = ref_ranked[:4]

            # v2: prioritized context builder within token limit
            budgeted = _build_context_prioritized(
                primary=reranked,
                anchors=anchors,
                refs=ref_top4,
                token_limit=settings.RAG_TOKEN_LIMIT,
            )

            # Format for agent consumption
            formatted: List[Dict[str, Any]] = []
            for idx, doc in enumerate(budgeted, 1):
                meta = doc.get("metadata") or {}
                formatted.append(
                    {
                        "rank": idx,
                        "content": doc.get("content") or doc.get("text", ""),
                        "source": doc.get("source_file") or meta.get("source", ""),
                        "regulation_type": meta.get(
                            "source", meta.get("regulation_type", "Unknown")
                        ),
                        "section": meta.get("part", meta.get("section", "N/A")),
                        "chunk_index": doc.get("chunk_index"),
                        "score": float(
                            doc.get("rerank_score")
                            or doc.get("rrf_score")
                            or doc.get("final_score")
                            or 0
                        ),
                        "retrieval_methods": doc.get("retrieval_methods", []),
                        "metadata": meta,
                    }
                )

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
