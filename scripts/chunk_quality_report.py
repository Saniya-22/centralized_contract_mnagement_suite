#!/usr/bin/env python3
"""Chunk quality diagnostics for the regulations vector index.

Usage:
  python scripts/chunk_quality_report.py
  python scripts/chunk_quality_report.py --namespace-prefix public-regulations
  python scripts/chunk_quality_report.py --max-token-rows 20000
"""

from __future__ import annotations

import argparse
import math
import os
import re
from typing import Iterable

import psycopg
import tiktoken
from dotenv import dotenv_values


def _pct(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    idx = max(0, min(len(values) - 1, math.ceil((p / 100.0) * len(values)) - 1))
    return float(values[idx])


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


def _count_query(cur: psycopg.Cursor, sql: str, params: Iterable | None = None) -> int:
    cur.execute(sql, params or ())
    row = cur.fetchone()
    return int(row[0]) if row else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk quality diagnostics")
    parser.add_argument(
        "--namespace-prefix",
        default=_env("REGULATIONS_NAMESPACE", _env("NAMESPACE", "public-regulations")),
        help="Namespace prefix filter (default: REGULATIONS_NAMESPACE or 'public-regulations')",
    )
    parser.add_argument(
        "--dense-table",
        default=_env("PG_DENSE_TABLE", "embeddings_dense"),
        help="Dense table name",
    )
    parser.add_argument(
        "--sparse-table",
        default=_env("PG_SPARSE_TABLE", "embeddings_sparse"),
        help="Sparse table name",
    )
    parser.add_argument(
        "--min-chunk-tokens",
        type=int,
        default=int(_env("MIN_CHUNK_TOKENS", "80")),
        help="Tiny-chunk threshold",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=int(_env("CHUNK_SIZE", "800")),
        help="Oversized-chunk threshold",
    )
    parser.add_argument(
        "--max-token-rows",
        type=int,
        default=20000,
        help="Max rows to tokenize exactly for token-distribution stats",
    )
    args = parser.parse_args()

    cfg = _load_db_config()
    if not cfg["dbname"] or not cfg["user"]:
        raise SystemExit("Missing DB config. Set PG_DB and PG_USER in env/.env.")

    ns_like = f"{args.namespace_prefix}%"
    enc = tiktoken.get_encoding("cl100k_base")

    conninfo = (
        f"host={cfg['host']} port={cfg['port']} dbname={cfg['dbname']} "
        f"user={cfg['user']} password={cfg['password']}"
    )
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            anchor_re = re.compile(
                r"^\s*(?:FAR|DFARS)?\s*(?:52\.\d{3}-\d+|252\.\d{3}-\d+|\d{2,3}\.\d{3}(?:-\d+)?)\b",
                re.IGNORECASE,
            )
            print("\n== Chunk Quality Report ==")
            _print_kv("Namespace prefix", args.namespace_prefix)
            _print_kv("Dense table", args.dense_table)
            _print_kv("Sparse table", args.sparse_table)

            print("\n-- Table Coverage --")
            dense_total = _count_query(cur, f"SELECT COUNT(*) FROM {args.dense_table}")
            sparse_total = _count_query(
                cur, f"SELECT COUNT(*) FROM {args.sparse_table}"
            )
            dense_ns = _count_query(
                cur,
                f"SELECT COUNT(*) FROM {args.dense_table} WHERE namespace LIKE %s",
                (ns_like,),
            )
            sparse_ns = _count_query(
                cur,
                f"SELECT COUNT(*) FROM {args.sparse_table} WHERE namespace LIKE %s",
                (ns_like,),
            )
            _print_kv("Dense rows (all)", dense_total)
            _print_kv("Sparse rows (all)", sparse_total)
            _print_kv("Dense rows (namespace)", dense_ns)
            _print_kv("Sparse rows (namespace)", sparse_ns)

            print("\n-- Source Breakdown (Dense) --")
            cur.execute(
                f"""
                SELECT COALESCE(metadata->>'source', 'UNKNOWN') AS source, COUNT(*)
                FROM {args.dense_table}
                WHERE namespace LIKE %s
                GROUP BY 1
                ORDER BY 2 DESC
                """,
                (ns_like,),
            )
            for source, count in cur.fetchall():
                _print_kv(f"source={source}", count)

            print("\n-- Data Integrity --")
            dup_groups = _count_query(
                cur,
                f"""
                SELECT COUNT(*) FROM (
                  SELECT md5(text), COUNT(*)
                  FROM {args.dense_table}
                  WHERE namespace LIKE %s
                  GROUP BY 1
                  HAVING COUNT(*) > 1
                ) t
                """,
                (ns_like,),
            )
            missing_source = _count_query(
                cur,
                f"""
                SELECT COUNT(*)
                FROM {args.dense_table}
                WHERE namespace LIKE %s
                  AND COALESCE(metadata->>'source','') = ''
                """,
                (ns_like,),
            )
            _print_kv("Duplicate text hash groups", dup_groups)
            _print_kv("Missing metadata.source", missing_source)

            print("\n-- Dense/Sparse ID Consistency --")
            sparse_without_dense = _count_query(
                cur,
                f"""
                SELECT COUNT(*)
                FROM {args.sparse_table} s
                LEFT JOIN {args.dense_table} d ON d.id = s.id
                WHERE s.namespace LIKE %s
                  AND d.id IS NULL
                """,
                (ns_like,),
            )
            dense_without_sparse = _count_query(
                cur,
                f"""
                SELECT COUNT(*)
                FROM {args.dense_table} d
                LEFT JOIN {args.sparse_table} s ON s.id = d.id
                WHERE d.namespace LIKE %s
                  AND s.id IS NULL
                """,
                (ns_like,),
            )
            _print_kv("Sparse rows without dense pair", sparse_without_dense)
            _print_kv("Dense rows without sparse pair", dense_without_sparse)

            print("\n-- Character Length Stats (Dense) --")
            cur.execute(
                f"""
                SELECT
                  ROUND(AVG(char_length(text))::numeric,1) AS avg_len,
                  MIN(char_length(text)) AS min_len,
                  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY char_length(text)) AS p50,
                  PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY char_length(text)) AS p90,
                  PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY char_length(text)) AS p99,
                  MAX(char_length(text)) AS max_len
                FROM {args.dense_table}
                WHERE namespace LIKE %s
                """,
                (ns_like,),
            )
            avg_len, min_len, p50, p90, p99, max_len = cur.fetchone()
            _print_kv("avg chars", avg_len)
            _print_kv("min chars", min_len)
            _print_kv("p50 chars", int(p50) if p50 is not None else 0)
            _print_kv("p90 chars", int(p90) if p90 is not None else 0)
            _print_kv("p99 chars", int(p99) if p99 is not None else 0)
            _print_kv("max chars", max_len)

            print("\n-- Token Stats (Exact, Dense) --")
            cur.execute(
                f"""
                SELECT text
                FROM {args.dense_table}
                WHERE namespace LIKE %s
                LIMIT %s
                """,
                (ns_like, args.max_token_rows),
            )
            texts = [row[0] for row in cur.fetchall()]
            token_counts = sorted(len(enc.encode(t or "")) for t in texts)
            if not token_counts:
                _print_kv("rows tokenized", 0)
            else:
                tiny = sum(1 for t in token_counts if t < args.min_chunk_tokens)
                oversized = sum(1 for t in token_counts if t > args.max_chunk_tokens)
                tiny_anchor = 0
                tiny_non_anchor = 0
                for t in texts:
                    tc = len(enc.encode(t or ""))
                    if tc >= args.min_chunk_tokens:
                        continue
                    first_line = (t or "").strip().splitlines()
                    is_anchor = bool(first_line and anchor_re.match(first_line[0]))
                    if is_anchor:
                        tiny_anchor += 1
                    else:
                        tiny_non_anchor += 1
                _print_kv("rows tokenized", len(token_counts))
                _print_kv("avg tokens", round(sum(token_counts) / len(token_counts), 1))
                _print_kv("min tokens", token_counts[0])
                _print_kv("p50 tokens", int(_pct(token_counts, 50)))
                _print_kv("p90 tokens", int(_pct(token_counts, 90)))
                _print_kv("p99 tokens", int(_pct(token_counts, 99)))
                _print_kv("max tokens", token_counts[-1])
                _print_kv(
                    f"tiny chunks (<{args.min_chunk_tokens})",
                    f"{tiny} ({(tiny / len(token_counts)) * 100:.2f}%)",
                )
                _print_kv(
                    f"tiny anchor chunks (<{args.min_chunk_tokens})",
                    f"{tiny_anchor} ({(tiny_anchor / len(token_counts)) * 100:.2f}%)",
                )
                _print_kv(
                    f"tiny non-anchor chunks (<{args.min_chunk_tokens})",
                    f"{tiny_non_anchor} ({(tiny_non_anchor / len(token_counts)) * 100:.2f}%)",
                )
                _print_kv(
                    f"oversized chunks (>{args.max_chunk_tokens})",
                    f"{oversized} ({(oversized / len(token_counts)) * 100:.2f}%)",
                )

            print("\n-- Embedding Model Mix (Dense metadata.embedding_model) --")
            cur.execute(
                f"""
                SELECT COALESCE(metadata->>'embedding_model', 'UNKNOWN') AS embedding_model, COUNT(*)
                FROM {args.dense_table}
                WHERE namespace LIKE %s
                GROUP BY 1
                ORDER BY 2 DESC
                """,
                (ns_like,),
            )
            for model, count in cur.fetchall():
                _print_kv(f"embedding_model={model}", count)

            print("\nReport complete.")


if __name__ == "__main__":
    main()
