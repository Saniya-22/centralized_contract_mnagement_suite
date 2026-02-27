from typing import List, Dict, Any
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import settings

logger = logging.getLogger(__name__)

class QueryExpansion:
    """Generates expanded/refined queries when retrieval fails (Self-Healing)."""
    
    def __init__(self):
        # Use a fast model for healing to minimize latency hit
        self.llm = ChatOpenAI(
            model=settings.RERANKER_MODEL, # Usually gpt-4o-mini
            api_key=settings.OPENAI_API_KEY,
            temperature=0.2,
            max_tokens=100
        )

    async def expand(self, query: str, fail_reason: str) -> List[str]:
        """Generates broader search terms or a better technical query."""
        logger.info(f"Expanding query: '{query}' due to: {fail_reason}")
        
        system_prompt = (
            "You are a regulatory search expert for FAR, DFARS, and EM385. "
            "A search for a user query failed to find relevant results or returned results from the wrong regulation. "
            "Provide 2 alternative, technically precise search queries optimized for vector search "
            "that stay strictly within the domain of the specific regulation mentioned (FAR, DFARS, or EM385). "
            "Format: Return only the queries separated by a newline. No numbering."
        )
        
        user_prompt = f"Original Query: {query}\nFailure Reason: {fail_reason}"
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = await self.llm.ainvoke(messages)
            expanded_queries = [q.strip() for q in response.content.split('\n') if q.strip()]
            return expanded_queries[:2]
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return []
