from typing import List, Dict, Any
import logging
import asyncio
from .critique import RetrievalCritique
from .expansion import QueryExpansion

logger = logging.getLogger(__name__)


class ReflectionManager:
    """Coordinates the ReflectionRAG (Critique + Healing) process."""

    def __init__(
        self, threshold: float = 0.7, max_queries: int = 2, max_docs: int = 10
    ):
        self.critique = RetrievalCritique(threshold=threshold)
        self.expansion = QueryExpansion()
        self.max_queries = max(1, max_queries)
        self.max_docs = max(1, max_docs)

    def check_quality(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Determines if retrieval results are good enough or need healing."""
        return self.critique.evaluate(query, documents)

    async def heal_search(
        self, query: str, fail_reason: str, search_func
    ) -> List[Dict[str, Any]]:
        """Executes the self-healing process by expanding queries and re-searching."""
        expanded_queries = await self.expansion.expand(query, fail_reason)
        expanded_queries = expanded_queries[: self.max_queries]

        if not expanded_queries:
            return []

        logger.info(
            f"Self-healing: Re-searching with expanded queries: {expanded_queries}"
        )

        # In parallel, execute the new searches
        tasks = [search_func(q) for q in expanded_queries]
        results_list = await asyncio.gather(*tasks)

        # Flatten and return new results
        new_docs = []
        seen = set()
        for res in results_list:
            if isinstance(res, list):
                for doc in res:
                    key = (
                        doc.get("source"),
                        doc.get("section"),
                        (doc.get("content") or "")[:160],
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    new_docs.append(doc)
                    if len(new_docs) >= self.max_docs:
                        return new_docs

        return new_docs
