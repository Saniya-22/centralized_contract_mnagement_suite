#!/usr/bin/env python3
"""Promote an indexed namespace to a target namespace with quality gates.

Default mode is dry-run. Use --apply to execute.

Examples:
  python scripts/promote_index.py --source public-regulations-pilot --target public-regulations
  python scripts/promote_index.py --source public-regulations-pilot --target public-regulations --apply
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
from dataclasses import dataclass

import psycopg
import tiktoken
from dotenv import dotenv_values


ANCHOR_RE = re.compile(
    r"^\s*(?:FAR|DFARS)?\s*(?:52\.\d{3}-\d+|252\.\d{3}-\d+|\d{2,3}\.\d{3}(?:-\d+)?)\b",
    re.IGNORECASE,
)


@dataclass
class GateResult:
    ok: bool
    message: str


def env(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v if v is not None else default


def load_db():
    f = dotenv_values(".env")
    return {
        "host": env("PG_HOST", f.get("PG_HOST", "localhost")),
        "port": env("PG_PORT", f.get("PG_PORT", "5432")),
        "dbname": env("PG_DB", f.get("PG_DB", "")),
        "user": env("PG_USER", f.get("PG_USER", "")),
        "password": env("PG_PASSWORD", f.get("PG_PASSWORD", "")),
    }


def pct(part: int, whole: int) -> float:
    return (part / whole * 100.0) if whole else 0.0


def print_kv(k: str, v: object) -> None:
    print(f"{k:<40} {v}")


def fetch_count(cur: psycopg.Cursor, sql: str, params=()) -> int:
    cur.execute(sql, params)
    row = cur.fetchone()
    return int(row[0]) if row else 0


def run_gates(
    cur: psycopg.Cursor,
    *,
    source: str,
    dense_table: str,
    sparse_table: str,
    min_chunk_tokens: int,
    max_chunk_tokens: int,
    max_tiny_non_anchor_pct: float,
    max_oversized_pct: float,
) -> list[GateResult]:
    ns_like = f"{source}%"
    gates: list[GateResult] = []
    enc = tiktoken.get_encoding("cl100k_base")

    dense = fetch_count(
        cur, f"SELECT COUNT(*) FROM {dense_table} WHERE namespace LIKE %s", (ns_like,)
    )
    sparse = fetch_count(
        cur, f"SELECT COUNT(*) FROM {sparse_table} WHERE namespace LIKE %s", (ns_like,)
    )
    gates.append(
        GateResult(
            ok=(dense > 0 and sparse > 0),
            message=f"Source rows present (dense={dense}, sparse={sparse})",
        )
    )
    gates.append(
        GateResult(
            ok=(dense == sparse),
            message=f"Dense/Sparse counts match (dense={dense}, sparse={sparse})",
        )
    )

    sparse_without_dense = fetch_count(
        cur,
        f"""
        SELECT COUNT(*)
        FROM {sparse_table} s
        LEFT JOIN {dense_table} d ON d.id = s.id
        WHERE s.namespace LIKE %s
          AND d.id IS NULL
        """,
        (ns_like,),
    )
    dense_without_sparse = fetch_count(
        cur,
        f"""
        SELECT COUNT(*)
        FROM {dense_table} d
        LEFT JOIN {sparse_table} s ON s.id = d.id
        WHERE d.namespace LIKE %s
          AND s.id IS NULL
        """,
        (ns_like,),
    )
    gates.append(
        GateResult(
            ok=(sparse_without_dense == 0 and dense_without_sparse == 0),
            message=(
                "No orphan IDs "
                f"(sparse_without_dense={sparse_without_dense}, dense_without_sparse={dense_without_sparse})"
            ),
        )
    )

    cur.execute(
        f"""
        SELECT COALESCE(metadata->>'embedding_model', 'UNKNOWN') AS embedding_model, COUNT(*)
        FROM {dense_table}
        WHERE namespace LIKE %s
        GROUP BY 1
        ORDER BY 2 DESC
        """,
        (ns_like,),
    )
    model_rows = cur.fetchall()
    model_count = len(model_rows)
    model_details = (
        ", ".join([f"{m}:{c}" for m, c in model_rows]) if model_rows else "none"
    )
    gates.append(
        GateResult(
            ok=(model_count == 1), message=f"Single embedding model ({model_details})"
        )
    )

    # Token quality gates
    cur.execute(f"SELECT text FROM {dense_table} WHERE namespace LIKE %s", (ns_like,))
    texts = [r[0] or "" for r in cur.fetchall()]
    token_counts = [len(enc.encode(t)) for t in texts]
    tiny_non_anchor = 0
    oversized = 0
    for t, tok in zip(texts, token_counts):
        if tok > max_chunk_tokens:
            oversized += 1
        if tok < min_chunk_tokens:
            first = t.strip().splitlines()
            is_anchor = bool(first and ANCHOR_RE.match(first[0]))
            if not is_anchor:
                tiny_non_anchor += 1

    tiny_non_anchor_pct = pct(tiny_non_anchor, len(token_counts))
    oversized_pct = pct(oversized, len(token_counts))
    gates.append(
        GateResult(
            ok=(tiny_non_anchor_pct <= max_tiny_non_anchor_pct),
            message=(
                f"Tiny non-anchor pct <= {max_tiny_non_anchor_pct:.2f} "
                f"(actual={tiny_non_anchor_pct:.2f})"
            ),
        )
    )
    gates.append(
        GateResult(
            ok=(oversized_pct <= max_oversized_pct),
            message=(
                f"Oversized pct <= {max_oversized_pct:.2f} "
                f"(actual={oversized_pct:.2f})"
            ),
        )
    )

    return gates


def promote(
    cur: psycopg.Cursor,
    *,
    source: str,
    target: str,
    dense_table: str,
    sparse_table: str,
    archive_existing: bool,
) -> tuple[int, int]:
    source_like = f"{source}%"
    target_like = f"{target}%"

    archive_prefix = f"{target}-archive-{dt.datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    if archive_existing:
        cur.execute(
            f"""
            UPDATE {dense_table}
               SET namespace = regexp_replace(namespace, '^' || %s, %s),
                   id = regexp_replace(id, '^' || %s, %s)
             WHERE namespace LIKE %s
            """,
            (target, archive_prefix, target, archive_prefix, target_like),
        )
        cur.execute(
            f"""
            UPDATE {sparse_table}
               SET namespace = regexp_replace(namespace, '^' || %s, %s),
                   id = regexp_replace(id, '^' || %s, %s)
             WHERE namespace LIKE %s
            """,
            (target, archive_prefix, target, archive_prefix, target_like),
        )
    else:
        cur.execute(
            f"DELETE FROM {sparse_table} WHERE namespace LIKE %s", (target_like,)
        )
        cur.execute(
            f"DELETE FROM {dense_table} WHERE namespace LIKE %s", (target_like,)
        )

    cur.execute(
        f"""
        INSERT INTO {dense_table} (id, namespace, text, metadata, embedding)
        SELECT
          regexp_replace(id, '^' || %s, %s) AS id,
          regexp_replace(namespace, '^' || %s, %s) AS namespace,
          text, metadata, embedding
        FROM {dense_table}
        WHERE namespace LIKE %s
        ON CONFLICT (id) DO UPDATE
          SET namespace = EXCLUDED.namespace,
              text = EXCLUDED.text,
              metadata = EXCLUDED.metadata,
              embedding = EXCLUDED.embedding
        """,
        (source, target, source, target, source_like),
    )
    dense_promoted = cur.rowcount

    cur.execute(
        f"""
        INSERT INTO {sparse_table} (id, namespace, text, metadata, embedding)
        SELECT
          regexp_replace(id, '^' || %s, %s) AS id,
          regexp_replace(namespace, '^' || %s, %s) AS namespace,
          text, metadata, embedding
        FROM {sparse_table}
        WHERE namespace LIKE %s
        ON CONFLICT (id) DO UPDATE
          SET namespace = EXCLUDED.namespace,
              text = EXCLUDED.text,
              metadata = EXCLUDED.metadata,
              embedding = EXCLUDED.embedding
        """,
        (source, target, source, target, source_like),
    )
    sparse_promoted = cur.rowcount
    return dense_promoted, sparse_promoted


def main() -> None:
    p = argparse.ArgumentParser(
        description="Promote indexed namespace with quality gates"
    )
    p.add_argument("--source", required=True, help="Source namespace prefix")
    p.add_argument("--target", required=True, help="Target namespace prefix")
    p.add_argument("--dense-table", default=env("PG_DENSE_TABLE", "embeddings_dense"))
    p.add_argument(
        "--sparse-table", default=env("PG_SPARSE_TABLE", "embeddings_sparse")
    )
    p.add_argument(
        "--min-chunk-tokens", type=int, default=int(env("MIN_CHUNK_TOKENS", "80"))
    )
    p.add_argument(
        "--max-chunk-tokens", type=int, default=int(env("CHUNK_SIZE", "800"))
    )
    p.add_argument("--max-tiny-non-anchor-pct", type=float, default=2.0)
    p.add_argument("--max-oversized-pct", type=float, default=1.0)
    p.add_argument(
        "--archive-existing",
        action="store_true",
        help="Archive current target rows before promotion",
    )
    p.add_argument(
        "--apply", action="store_true", help="Execute promotion (default: dry-run)"
    )
    args = p.parse_args()

    cfg = load_db()
    if not cfg["dbname"] or not cfg["user"]:
        raise SystemExit("Missing PG_DB/PG_USER in env/.env")

    conninfo = (
        f"host={cfg['host']} port={cfg['port']} dbname={cfg['dbname']} "
        f"user={cfg['user']} password={cfg['password']}"
    )
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            print("\n== Index Promotion ==")
            print_kv("Source", args.source)
            print_kv("Target", args.target)
            print_kv("Mode", "APPLY" if args.apply else "DRY-RUN")
            print_kv("Archive existing target", args.archive_existing)

            print("\n-- Quality Gates --")
            gates = run_gates(
                cur,
                source=args.source,
                dense_table=args.dense_table,
                sparse_table=args.sparse_table,
                min_chunk_tokens=args.min_chunk_tokens,
                max_chunk_tokens=args.max_chunk_tokens,
                max_tiny_non_anchor_pct=args.max_tiny_non_anchor_pct,
                max_oversized_pct=args.max_oversized_pct,
            )
            failed = [g for g in gates if not g.ok]
            for g in gates:
                print_kv("PASS" if g.ok else "FAIL", g.message)

            if failed:
                print("\nPromotion blocked: one or more quality gates failed.")
                raise SystemExit(2)

            if not args.apply:
                print(
                    "\nDry-run complete. All gates passed. Re-run with --apply to promote."
                )
                return

            print("\n-- Executing Promotion --")
            dense_promoted, sparse_promoted = promote(
                cur,
                source=args.source,
                target=args.target,
                dense_table=args.dense_table,
                sparse_table=args.sparse_table,
                archive_existing=args.archive_existing,
            )
            conn.commit()
            print_kv("Dense rows promoted", dense_promoted)
            print_kv("Sparse rows promoted", sparse_promoted)
            print("\nPromotion complete.")


if __name__ == "__main__":
    main()
