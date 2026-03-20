"""GPT-4o-mini reranker service.

Mirrors the JS rag.service.js _rerankOpenAI() logic.
Sends top-N retrieved chunks to a fast LLM for relevance reordering,
then returns chunks sorted by rerank_score DESC.
Falls back to rrf_score / similarity sort on any error.
"""

import json
import logging
import time
from typing import List, Dict, Any

from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=settings.OPENAI_API_KEY)


def rerank(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rerank retrieved chunks using GPT-4o-mini as a relevance scorer.

    Identical in spirit to the JS _rerankOpenAI():
      - Builds a numbered preview of each chunk (first 300 chars)
      - Asks the model to assign a 0-10 relevance score per chunk index
      - Applies scores and re-sorts the list

    Args:
        query:  The original user query string
        chunks: RRF-fused chunks to be reranked (any order)

    Returns:
        Chunks sorted by rerank_score DESC.
        If GPT-4o-mini call fails, returns chunks sorted by rrf_score / similarity.
    """
    if not chunks:
        return chunks

    # ── RERANKER_ENABLED guard ──────────────────────────────────────────────────
    # Set RERANKER_ENABLED=False in .env to skip the LLM reranker and use
    # RRF score only (saves ~800-1500ms per search request).
    if not settings.RERANKER_ENABLED:
        logger.info("[Reranker] Disabled (RERANKER_ENABLED=False). Using RRF scores.")
        return chunks

    # ── Cap input to top 12 chunks so strong clauses at 9–12 can surface ───────
    # (Latency trade-off: 12 is a good default for quality without blowing prompt size.)
    chunks = chunks[:12]

    # Build numbered preview (metadata-rich to improve clause/title ranking)
    numbered_previews = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata") or {}
        src = (
            meta.get("source") or chunk.get("source_file") or chunk.get("source") or ""
        )
        section_number = (
            meta.get("section_number") or meta.get("part") or meta.get("section") or ""
        )
        section_title = meta.get("section_title") or meta.get("title") or ""
        text = chunk.get("text") or chunk.get("content") or ""
        text_preview = text[:400].replace("\n", " ")
        header = " | ".join(
            [
                p
                for p in [
                    str(src).strip(),
                    str(section_number).strip(),
                    str(section_title).strip(),
                ]
                if p
            ]
        )
        if header:
            numbered_previews.append(f"[{i}] {header}\n{text_preview}")
        else:
            numbered_previews.append(f"[{i}] {text_preview}")
    previews_block = "\n\n".join(numbered_previews)

    try:
        _t0 = time.perf_counter()
        response = _client.chat.completions.create(
            model=settings.RERANKER_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a relevance scoring system. "
                        "Given a query and numbered document excerpts, "
                        'return ONLY a JSON object with a key "scores" '
                        "whose value is an array of objects with "
                        '"index" (integer) and "score" (0-10 float, '
                        "where 10 = highly relevant, 0 = irrelevant). "
                        "No explanation, just the JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": f'Query: "{query}"\n\nDocuments:\n{previews_block}',
                },
            ],
        )

        raw = response.choices[0].message.content or ""
        _rerank_ms = time.perf_counter() - _t0
        logger.info(
            f"[Reranker] API call took {_rerank_ms:.2f}s for {len(chunks)} chunks"
        )
        parsed = json.loads(raw)

        # Support either {"scores": [...]} or {"results": [...]} or bare list
        score_items = parsed.get("scores") or parsed.get("results") or []
        if not isinstance(score_items, list):
            raise ValueError(f"Unexpected reranker response shape: {parsed}")

        score_map: Dict[int, float] = {
            item["index"]: float(item["score"])
            for item in score_items
            if isinstance(item, dict)
            and isinstance(item.get("index"), int)
            and isinstance(item.get("score"), (int, float))
        }

        reranked = [
            {**chunk, "rerank_score": score_map.get(i, 0.0)}
            for i, chunk in enumerate(chunks)
        ]
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        logger.info(
            f"[Reranker] Reranked {len(chunks)} chunks via {settings.RERANKER_MODEL}"
        )
        for i, c in enumerate(reranked[:5]):
            preview = (c.get("text") or c.get("content") or "")[:80].replace("\n", " ")
            logger.debug(
                f"[Reranker]   {i + 1}. rerank_score={c['rerank_score']:.1f} \"{preview}...\""
            )

        return reranked

    except Exception as exc:
        logger.warning(
            f"[Reranker] GPT reranking failed ({exc}), falling back to rrf_score sort"
        )
        return sorted(
            chunks,
            key=lambda x: x.get("rrf_score") or x.get("similarity") or 0.0,
            reverse=True,
        )
