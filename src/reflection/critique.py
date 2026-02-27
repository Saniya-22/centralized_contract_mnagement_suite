"""Retrieval quality critique for ReflectionRAG.

Evaluates retrieved documents using fast, heuristic-based checks:
  1. Score threshold check
  2. Regulation-type alignment (does the result match what was asked?)
  3. Keyword overlap for borderline scores
"""

import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Matches FAR, DFARS, EM385 / EM 385 in a query
_REG_PATTERN = re.compile(r"\b(FAR|DFARS|EM\s*385)\b", re.IGNORECASE)

_REG_NORM = {
    "FAR": "FAR",
    "DFARS": "DFARS",
    "EM385": "EM385",
    "EM 385": "EM385",
}


def _normalise_reg(raw: str) -> str:
    key = re.sub(r"\s+", " ", raw.strip().upper())
    return _REG_NORM.get(key, key)


class RetrievalCritique:
    """Evaluates the quality of retrieved documents for reflection.

    Primary goal: Fast, heuristic-based filtering (Latency-Neutral).
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def evaluate(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluates retrieval quality based on scores, regulation alignment,
        and content heuristics.

        Returns:
            Dict containing 'passed' (bool), 'score' (float), and 'reason' (str).
        """
        if not documents:
            return {
                "passed": False,
                "score": 0.0,
                "reason": "No documents retrieved."
            }

        # ── 1. Score check ────────────────────────────────────────────────
        top_doc = documents[0]
        score = float(top_doc.get("score") or top_doc.get("rerank_score") or 0.0)

        # Normalise: rerank scores are 0-10, RRF scores are ~0.03
        normalized_score = score / 10.0 if score > 1.0 else score

        # ── 2. Regulation-type alignment ──────────────────────────────────
        # If the query mentions a specific regulation (FAR, DFARS, EM385),
        # check whether the top results actually come from that regulation.
        query_reg = self._extract_query_regulation(query)
        if query_reg:
            matching_docs = 0
            check_count = min(len(documents), 5)  # check top-5
            for doc in documents[:check_count]:
                doc_reg = self._get_doc_regulation(doc)
                if doc_reg and doc_reg == query_reg:
                    matching_docs += 1

            alignment_ratio = matching_docs / check_count
            if alignment_ratio < 0.2:
                # Almost none of the top results match the asked regulation
                logger.info(
                    f"Critique: Regulation mismatch — query asks for {query_reg}, "
                    f"but only {matching_docs}/{check_count} docs match"
                )
                return {
                    "passed": False,
                    "score": normalized_score,
                    "reason": (
                        f"Regulation type mismatch: query asks for {query_reg} "
                        f"but top results are from other regulations."
                    )
                }

        # ── 3. Score threshold ────────────────────────────────────────────
        passed = normalized_score >= self.threshold

        # ── 4. Keyword overlap rescue for borderline scores ───────────────
        if not passed and normalized_score > 0.3:
            keywords = [w.lower() for w in query.split() if len(w) > 3]
            content = (top_doc.get("content") or "").lower()
            overlap = sum(1 for k in keywords if k in content)
            overlap_ratio = overlap / len(keywords) if keywords else 0

            if overlap_ratio > 0.6:
                logger.info(
                    f"Critique: Borderline score {normalized_score:.2f} "
                    f"passed via keyword overlap {overlap_ratio:.2f}"
                )
                passed = True

        reason = "High confidence match found." if passed else "Low retrieval confidence."
        return {
            "passed": passed,
            "score": normalized_score,
            "reason": reason,
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_query_regulation(query: str) -> Optional[str]:
        """Extract the regulation type the user is asking about."""
        match = _REG_PATTERN.search(query)
        if match:
            return _normalise_reg(match.group(1))
        return None

    @staticmethod
    def _get_doc_regulation(doc: Dict[str, Any]) -> Optional[str]:
        """Extract the regulation type from a retrieved document."""
        # Try explicit field first
        reg = doc.get("regulation_type")
        if reg and reg not in ("Unknown", "N/A", ""):
            return _normalise_reg(reg)

        # Fall back to metadata.source
        meta = doc.get("metadata") or {}
        source = meta.get("source", "")
        if source:
            return _normalise_reg(source)

        # Fall back to the source field
        source_field = doc.get("source", "")
        if source_field:
            match = _REG_PATTERN.search(source_field)
            if match:
                return _normalise_reg(match.group(1))

        return None
