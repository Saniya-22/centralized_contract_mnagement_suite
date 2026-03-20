"""LLM utility functions for embeddings and text processing"""

from openai import OpenAI
from typing import List, Dict, Any
import tiktoken
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=settings.OPENAI_API_KEY)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def get_embedding(text: str, model: str = None) -> List[float]:
    """Generate embedding for text using OpenAI.

    Args:
        text: Text to embed
        model: Embedding model to use (defaults to settings.EMBEDDING_MODEL)

    Returns:
        Embedding vector as list of floats
    """
    model = model or settings.EMBEDDING_MODEL

    try:
        import time as _time

        _t0 = _time.perf_counter()
        response = client.embeddings.create(model=model, input=text)
        _elapsed = _time.perf_counter() - _t0
        embedding = response.data[0].embedding
        logger.info(
            f"[Embedding] text-embedding API took {_elapsed:.2f}s "
            f"for {len(text)}-char input"
        )
        return embedding

    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise


def get_embeddings_batch(texts: List[str], model: str = None) -> List[List[float]]:
    """Generate embeddings for multiple texts in a single API call.

    Args:
        texts: List of texts to embed
        model: Embedding model to use

    Returns:
        List of embedding vectors
    """
    model = model or settings.EMBEDDING_MODEL

    try:
        response = client.embeddings.create(model=model, input=texts)
        embeddings = [item.embedding for item in response.data]
        logger.info(f"Generated {len(embeddings)} embeddings in batch")
        return embeddings

    except Exception as e:
        logger.error(f"Failed to generate batch embeddings: {e}")
        raise


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for
        model: Model to use for tokenization

    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Token counting failed: {e}, using estimate")
        # Rough estimate: ~4 characters per token
        return len(text) // 4


def format_documents(documents: List[Dict[str, Any]], max_tokens: int = 4000) -> str:
    """Format retrieved documents into a context string.

    Args:
        documents: List of document dictionaries
        max_tokens: Maximum tokens to include

    Returns:
        Formatted string with document contents
    """
    formatted_parts = []
    total_tokens = 0
    skipped = 0

    for idx, doc in enumerate(documents, 1):
        raw_content = doc.get("content", "") or ""
        trimmed_content = raw_content[: settings.MAX_DOC_CHARS_FOR_SYNTHESIS]
        if len(raw_content) > len(trimmed_content):
            trimmed_content += "..."

        # Format single document
        doc_text = f"""--- Document {idx} ---
Source: {doc.get('source', 'Unknown')}
Regulation: {doc.get('regulation_type', 'Unknown')}
Section: {doc.get('section', 'N/A')}
Score: {doc.get('score', 0):.3f}

Content:
{trimmed_content}

"""

        # Check token limit
        doc_tokens = count_tokens(doc_text)
        if total_tokens + doc_tokens > max_tokens:
            skipped += 1
            continue

        formatted_parts.append(doc_text)
        total_tokens += doc_tokens

    result = "\n".join(formatted_parts)
    logger.info(
        f"Formatted {len(formatted_parts)} documents ({total_tokens} tokens), "
        f"skipped {skipped} due to token budget"
    )
    return result


def truncate_text(text: str, max_tokens: int = 1000, model: str = "gpt-4") -> str:
    """Truncate text to fit within token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed
        model: Model for tokenization

    Returns:
        Truncated text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)

        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        truncated_text = encoding.decode(truncated_tokens)
        logger.info(f"Truncated text from {len(tokens)} to {max_tokens} tokens")
        return truncated_text + "..."

    except Exception as e:
        logger.error(f"Truncation failed: {e}")
        # Fallback: character-based truncation
        char_limit = max_tokens * 4
        return text[:char_limit] + "..."
