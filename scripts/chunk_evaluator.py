#!/usr/bin/env python3
"""Unified chunk quality evaluation and scoring."""

import argparse
import os
import re
import random
from typing import Dict, Any

import psycopg
import tiktoken
from dotenv import dotenv_values


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


def calculate_technical_score(metrics: Dict[str, Any]) -> float:
    """Calculates a 0-100 technical score based on raw metrics."""
    score = 100.0

    # 1. Penalty for duplicates (max -20)
    dup_ratio = metrics.get("duplicate_ratio", 0.0)
    score -= min(dup_ratio * 100, 20.0)

    # 2. Penalty for tiny non-anchor chunks (max -30)
    # These are usually noise or fragmented sentences.
    tiny_ratio = metrics.get("tiny_non_anchor_ratio", 0.0)
    score -= min(tiny_ratio * 100 * 2, 30.0)

    # 3. Penalty for oversized chunks (max -10)
    # These might exceed context windows or lead to diluted embeddings.
    oversized_ratio = metrics.get("oversized_ratio", 0.0)
    score -= min(oversized_ratio * 100 * 5, 10.0)

    # 4. Penalty for missing metadata (max -10)
    missing_meta_ratio = metrics.get("missing_metadata_ratio", 0.0)
    score -= min(missing_meta_ratio * 100, 10.0)

    return max(0.0, score)


def main():
    parser = argparse.ArgumentParser(description="Chunk Quality Evaluator")
    parser.add_argument("--ns", default="public-regulations", help="Namespace prefix")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of chunks to sample for semantic review",
    )
    args = parser.parse_args()

    cfg = _load_db_config()
    ns_like = f"{args.ns}%"
    enc = tiktoken.get_encoding("cl100k_base")
    anchor_re = re.compile(
        r"^\s*(?:FAR|DFARS)?\s*(?:52\.\d{3}-\d+|252\.\d{3}-\d+|\d{2,3}\.\d{3}(?:-\d+)?)\b",
        re.IGNORECASE,
    )

    conninfo = f"host={cfg['host']} port={cfg['port']} dbname={cfg['dbname']} user={cfg['user']} password={cfg['password']}"

    try:
        with psycopg.connect(conninfo) as conn:
            with conn.cursor() as cur:
                # 1. Basic Counts
                cur.execute(
                    "SELECT COUNT(*) FROM embeddings_dense WHERE namespace LIKE %s",
                    (ns_like,),
                )
                total_chunks = cur.fetchone()[0]
                if total_chunks == 0:
                    print("No chunks found in namespace.")
                    return

                # 2. Duplicate Detection
                cur.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT md5(text) FROM embeddings_dense WHERE namespace LIKE %s GROUP BY 1 HAVING COUNT(*) > 1
                    ) t
                """,
                    (ns_like,),
                )
                dup_groups = cur.fetchone()[0]

                # 3. Token Analysis
                cur.execute(
                    "SELECT text, metadata FROM embeddings_dense WHERE namespace LIKE %s",
                    (ns_like,),
                )
                rows = cur.fetchall()

                tiny_non_anchor = 0
                oversized = 0
                missing_source = 0
                all_tokens = []

                for text, metadata in rows:
                    if not metadata or not metadata.get("source"):
                        missing_source += 1

                    tokens = len(enc.encode(text or ""))
                    all_tokens.append(tokens)

                    if tokens < 80:
                        first_line = (text or "").strip().splitlines()
                        is_anchor = bool(first_line and anchor_re.match(first_line[0]))
                        if not is_anchor:
                            tiny_non_anchor += 1

                    if tokens > 800:
                        oversized += 1

                # 4. Final Metrics
                metrics = {
                    "total_chunks": total_chunks,
                    "duplicate_ratio": dup_groups / total_chunks,
                    "tiny_non_anchor_ratio": tiny_non_anchor / total_chunks,
                    "oversized_ratio": oversized / total_chunks,
                    "missing_metadata_ratio": missing_source / total_chunks,
                    "avg_tokens": sum(all_tokens) / total_chunks,
                }

                t_score = calculate_technical_score(metrics)

                # 5. Sampling for Semantic Review
                sample = random.sample(rows, min(args.sample_size, total_chunks))

                print(f"\n## Chunk Quality Report: {args.ns} ##")
                print(f"Total Chunks: {total_chunks}")
                print(f"Technical Score: {t_score:.2f}/100")
                print("-" * 30)
                print(f"Avg Tokens: {metrics['avg_tokens']:.1f}")
                print(
                    f"Tiny Non-Anchor Chunks: {tiny_non_anchor} ({metrics['tiny_non_anchor_ratio']*100:.2f}%)"
                )
                print(
                    f"Duplicate Groups: {dup_groups} ({metrics['duplicate_ratio']*100:.2f}%)"
                )
                print(
                    f"Oversized Chunks (>800): {oversized} ({metrics['oversized_ratio']*100:.2f}%)"
                )
                print("-" * 30)
                print(f"\n### Semantic Sample (Top {len(sample)}) ###")
                for i, (text, meta) in enumerate(sample):
                    source = meta.get("source", "UNKNOWN") if meta else "UNKNOWN"
                    preview = (text[:200] + "...") if len(text) > 200 else text
                    print(
                        f"\n[{i+1}] Source: {source} | Tokens: {len(enc.encode(text))}"
                    )
                    print(f"Content: {preview.replace(chr(10), ' ')}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
