#!/usr/bin/env python3
"""Intrinsic coherence metrics for chunk quality.

Computes within-chunk coherence (sentence-level embeddings → mean pairwise cosine)
and optional boundary alignment. Use to evaluate chunk quality after ingestion.

Usage:
  python scripts/chunk_coherence_report.py
  python scripts/chunk_coherence_report.py --sample 3000 --worst 30
  python scripts/chunk_coherence_report.py --full
  python scripts/chunk_coherence_report.py --namespace-prefix public-regulations --no-boundary

Requires: OPENAI_API_KEY, PG_* in .env. Uses EMBEDDING_MODEL (default text-embedding-3-small).
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from typing import Any

import psycopg
from dotenv import dotenv_values

# Optional: numpy for cosine. Fallback to manual dot/norm.
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# OpenAI for sentence embeddings (batch)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def _env(key: str, default: str = "") -> str:
    val = os.getenv(key)
    return val if val is not None else default


def _load_db_config() -> dict[str, str]:
    env_file = dotenv_values(".env")
    return {
        "host": _env("PG_HOST", env_file.get("PG_HOST", "localhost")),
        "port": _env("PG_PORT", env_file.get("PG_PORT", "5432")),
        "dbname": _env("PG_DB", env_file.get("PG_DB", "")),
        "user": _env("PG_USER", env_file.get("PG_USER", "")),
        "password": _env("PG_PASSWORD", env_file.get("PG_PASSWORD", "")),
    }


def _print_kv(key: str, value: object) -> None:
    print(f"{key:<36} {value}")


# --- Sentence splitting (min length to avoid noise) ---
_MIN_SENTENCE_CHARS = 10
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences; filter very short fragments."""
    if not (text or text.strip()):
        return []
    parts = _SENTENCE_END_RE.split(text.strip())
    return [s.strip() for s in parts if len(s.strip()) >= _MIN_SENTENCE_CHARS]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if _HAS_NUMPY:
        va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1e-12
    norm_b = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (norm_a * norm_b)


def mean_pairwise_cosine(embeddings: list[list[float]]) -> float:
    """Mean pairwise cosine similarity among vectors. 0-1 scale (cosine is -1..1; we clamp)."""
    n = len(embeddings)
    if n <= 1:
        return 1.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            total += (sim + 1.0) / 2.0  # map [-1,1] to [0,1]
            count += 1
    return total / count if count else 1.0


def sentence_aligned_end(text: str) -> bool:
    """True if chunk text ends at a sentence boundary (. ! ? " ' or newline)."""
    if not text or not text.strip():
        return True
    t = text.strip()
    if not t:
        return True
    return t[-1] in ".!?\"'\n" or t.endswith(".") or t.endswith("?") or t.endswith("!")


def fetch_embeddings_batch(
    client: OpenAI, model: str, texts: list[str], batch_size: int = 50
) -> list[list[float]]:
    """Call OpenAI embeddings API in batches. Returns list of embedding vectors."""
    import time
    # Avoid empty strings (API can reject)
    texts = [t if t.strip() else " " for t in texts]
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        order = sorted(resp.data, key=lambda x: x.index)
        all_embeddings.extend([order[j].embedding for j in range(len(order))])
        if i + batch_size < len(texts):
            time.sleep(0.15)  # gentle rate limit
    return all_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk intrinsic coherence report (sentence-level mean pairwise cosine + boundary alignment)"
    )
    parser.add_argument(
        "--namespace-prefix",
        default=_env("REGULATIONS_NAMESPACE", _env("NAMESPACE", "public-regulations")),
        help="Namespace prefix filter",
    )
    parser.add_argument(
        "--dense-table",
        default=_env("PG_DENSE_TABLE", "embeddings_dense"),
        help="Dense table name",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=3000,
        help="Number of chunks to sample for coherence (default 3000). Use --full for no sampling.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run on all chunks (no sampling). Can be slow for large indexes.",
    )
    parser.add_argument(
        "--worst",
        type=int,
        default=20,
        help="Number of worst-coherence chunks to list (default 20)",
    )
    parser.add_argument(
        "--no-boundary",
        action="store_true",
        help="Skip boundary alignment check",
    )
    parser.add_argument(
        "--embedding-model",
        default=_env("EMBEDDING_MODEL", "text-embedding-3-small"),
        help="OpenAI embedding model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Sentence embedding batch size (default 50)",
    )
    args = parser.parse_args()

    if not _HAS_NUMPY:
        print("Warning: numpy not installed. Using pure-Python cosine (slower).", file=sys.stderr)

    if OpenAI is None:
        print("Error: openai package required. pip install openai", file=sys.stderr)
        sys.exit(1)

    api_key = _env("OPENAI_API_KEY", dotenv_values(".env").get("OPENAI_API_KEY", ""))
    if not api_key:
        print("Error: OPENAI_API_KEY not set in .env or environment.", file=sys.stderr)
        sys.exit(1)

    cfg = _load_db_config()
    if not cfg["dbname"] or not cfg["user"]:
        print("Error: Missing DB config. Set PG_DB and PG_USER in .env", file=sys.stderr)
        sys.exit(1)

    ns_like = f"{args.namespace_prefix}%"
    table = args.dense_table
    conninfo = (
        f"host={cfg['host']} port={cfg['port']} dbname={cfg['dbname']} "
        f"user={cfg['user']} password={cfg['password']}"
    )

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM {table} WHERE namespace LIKE %s",
                (ns_like,),
            )
            total_in_db = cur.fetchone()[0]
            if total_in_db == 0:
                print("No chunks found in namespace.")
                return

            if args.full:
                limit_sql = ""
                params: tuple = (ns_like,)
                sample_size = total_in_db
            else:
                sample_size = min(args.sample, total_in_db)
                limit_sql = " ORDER BY random() LIMIT %s"
                params = (ns_like, sample_size)

            cur.execute(
                f"SELECT id, text, metadata FROM {table} WHERE namespace LIKE %s{limit_sql}",
                params,
            )
            rows = cur.fetchall()

    # Build list of chunks with id, text, source
    chunks: list[dict[str, Any]] = []
    for row in rows:
        chunk_id, text, meta = row[0], row[1] or "", row[2] or {}
        source = (meta.get("source") or meta.get("regulation_type") or "UNKNOWN") if isinstance(meta, dict) else "UNKNOWN"
        chunks.append({"id": chunk_id, "text": text, "source": source})

    print("\n== Chunk Coherence Report ==")
    _print_kv("Namespace prefix", args.namespace_prefix)
    _print_kv("Dense table", table)
    _print_kv("Total chunks in namespace", total_in_db)
    _print_kv("Sample size", len(chunks))
    _print_kv("Embedding model", args.embedding_model)

    # 1) Boundary alignment (no API)
    if not args.no_boundary:
        aligned = sum(1 for c in chunks if sentence_aligned_end(c["text"]))
        pct = (aligned / len(chunks)) * 100.0 if chunks else 0.0
        print("\n-- Boundary alignment --")
        _print_kv("Chunks with sentence-aligned end", f"{aligned} ({pct:.1f}%)")

    # 2) Sentence split and collect all sentences for batching
    chunk_sentences: list[list[str]] = []
    for c in chunks:
        sents = split_sentences(c["text"])
        chunk_sentences.append(sents)

    # Sanity: how many chunks have 2+ sentences (get real coherence); rest get 1.0 by convention
    n_single = sum(1 for s in chunk_sentences if len(s) <= 1)
    n_multi = sum(1 for s in chunk_sentences if len(s) > 1)
    print("\n-- Sentence split (sanity) --")
    _print_kv("Chunks with 0-1 sentences (coherence=1.0 by convention)", f"{n_single} ({100.0 * n_single / len(chunks):.1f}%)")
    _print_kv("Chunks with 2+ sentences (coherence computed)", f"{n_multi} ({100.0 * n_multi / len(chunks):.1f}%)")

    # Only multi-sentence chunks get embeddings; coherence_scores will hold only their computed values
    coherence_scores: list[float] = []
    chunk_to_embed_indices: list[tuple[int, list[int]]] = []  # (chunk_idx, [sent indices in flat list])
    flat_sentences: list[str] = []
    sent_index = 0
    for i, sents in enumerate(chunk_sentences):
        if len(sents) <= 1:
            pass  # no embedding; score_by_chunk_index[i] = 1.0 later
        else:
            indices = list(range(sent_index, sent_index + len(sents)))
            chunk_to_embed_indices.append((i, indices))
            flat_sentences.extend(sents)
            sent_index += len(sents)

    if not flat_sentences:
        print("\n-- Within-chunk coherence --")
        print("No multi-sentence chunks in sample; nothing to embed. All scores are 1.0 by convention.")
        print("\nReport complete.")
        return

    # 3) Batch embed all sentences
    client = OpenAI(api_key=api_key)
    print("\nEmbedding sentences (batches of %s)..." % args.batch_size)
    all_embeddings = fetch_embeddings_batch(client, args.embedding_model, flat_sentences, args.batch_size)

    # 4) Per-chunk coherence for multi-sentence chunks
    for chunk_idx, sent_indices in chunk_to_embed_indices:
        embs = [all_embeddings[j] for j in sent_indices]
        coh = mean_pairwise_cosine(embs)
        coherence_scores.append(coh)

    # Map chunk index -> score (single-sentence = 1.0; multi-sentence = coherence_scores in order)
    score_by_chunk_index: dict[int, float] = {}
    for i, sents in enumerate(chunk_sentences):
        if len(sents) <= 1:
            score_by_chunk_index[i] = 1.0
    for mc_idx, (chunk_idx, _) in enumerate(chunk_to_embed_indices):
        score_by_chunk_index[chunk_idx] = coherence_scores[mc_idx]

    all_scores = [score_by_chunk_index[i] for i in range(len(chunks))]
    avg_coh = sum(all_scores) / len(all_scores) if all_scores else 0.0
    sorted_scores = sorted(all_scores)
    p50 = sorted_scores[len(sorted_scores) // 2] if sorted_scores else 0.0
    p90 = sorted_scores[int(len(sorted_scores) * 0.9)] if sorted_scores else 0.0
    low = sum(1 for s in all_scores if s < 0.4) / len(all_scores) * 100.0 if all_scores else 0.0
    very_low = sum(1 for s in all_scores if s < 0.25) / len(all_scores) * 100.0 if all_scores else 0.0

    print("\n-- Within-chunk coherence (mean pairwise cosine, 0-1) --")
    _print_kv("Avg coherence", f"{avg_coh:.3f}")
    _print_kv("p50", f"{p50:.3f}")
    _print_kv("p90", f"{p90:.3f}")
    _print_kv("Low (<0.4)", f"{low:.1f}%")
    _print_kv("Very low (<0.25)", f"{very_low:.1f}%")

    # 5) Worst chunks
    indexed = [(i, score_by_chunk_index[i]) for i in range(len(chunks))]
    indexed.sort(key=lambda x: x[1])
    worst_n = min(args.worst, len(indexed))
    print(f"\n-- Worst {worst_n} chunks (id / source / coherence / preview) --")
    for i in range(worst_n):
        idx, score = indexed[i]
        c = chunks[idx]
        preview = (c["text"][:120] + "...") if len(c["text"]) > 120 else c["text"]
        preview = preview.replace("\n", " ")
        print(f"  [{i+1}] id={c['id']} source={c['source']} coherence={score:.3f}")
        print(f"      {preview}")

    print("\nReport complete.")


if __name__ == "__main__":
    main()
