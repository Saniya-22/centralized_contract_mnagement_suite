#!/usr/bin/env python3
"""One-time deduplication of embeddings_dense (and sync embeddings_sparse) by text hash.

Keeps one row per md5(text); deletes the rest from both tables so dense/sparse stay in sync.
Run chunk_quality_report.py first to see "Duplicate text hash groups" count.

Usage:
  python scripts/dedup_embeddings.py --dry-run   # default: only report what would be deleted
  python scripts/dedup_embeddings.py --execute   # actually delete duplicate rows

Requires: PG_* in .env (same as chunk_quality_report).
"""

from __future__ import annotations

import argparse
import os
import sys

import psycopg
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate embeddings by text hash (keep one per md5(text), delete rest)."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete rows. Default is --dry-run (report only).",
    )
    parser.add_argument(
        "--namespace-prefix",
        default=_env("REGULATIONS_NAMESPACE", _env("NAMESPACE", "public-regulations")),
        help="Namespace prefix filter (default: public-regulations)",
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
    args = parser.parse_args()

    cfg = _load_db_config()
    if not cfg["dbname"] or not cfg["user"]:
        print(
            "Error: Missing DB config. Set PG_DB and PG_USER in .env", file=sys.stderr
        )
        sys.exit(1)

    ns_like = f"{args.namespace_prefix}%"
    dense_table = args.dense_table
    sparse_table = args.sparse_table
    conninfo = (
        f"host={cfg['host']} port={cfg['port']} dbname={cfg['dbname']} "
        f"user={cfg['user']} password={cfg['password']}"
    )

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            # IDs to delete: for each md5(text) group, keep one (min id), drop the rest
            cur.execute(
                f"""
                WITH dup AS (
                    SELECT id,
                           ROW_NUMBER() OVER (PARTITION BY md5(text) ORDER BY id) AS rn
                    FROM {dense_table}
                    WHERE namespace LIKE %s
                )
                SELECT id FROM dup WHERE rn > 1
                """,
                (ns_like,),
            )
            ids_to_delete = [row[0] for row in cur.fetchall()]

    if not ids_to_delete:
        print(
            "No duplicate rows found (one row per md5(text) already). Nothing to delete."
        )
        return

    n = len(ids_to_delete)
    print(
        f"Duplicate rows to remove: {n} (keeping one per md5(text) in namespace {args.namespace_prefix!r})"
    )
    print(f"Tables: {dense_table}, {sparse_table}")

    if not args.execute:
        print("\n[DRY-RUN] No rows deleted. Run with --execute to apply.")
        return

    # Delete in batches to avoid huge IN lists; delete sparse first then dense
    batch_size = 2000
    deleted_sparse = 0
    deleted_dense = 0
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            for i in range(0, n, batch_size):
                batch = ids_to_delete[i : i + batch_size]
                cur.execute(
                    f"DELETE FROM {sparse_table} WHERE id = ANY(%s)",
                    (batch,),
                )
                deleted_sparse += cur.rowcount
            conn.commit()
        with conn.cursor() as cur:
            for i in range(0, n, batch_size):
                batch = ids_to_delete[i : i + batch_size]
                cur.execute(
                    f"DELETE FROM {dense_table} WHERE id = ANY(%s)",
                    (batch,),
                )
                deleted_dense += cur.rowcount
            conn.commit()

    print(
        f"\n[EXECUTE] Deleted: {deleted_sparse} from {sparse_table}, {deleted_dense} from {dense_table}."
    )
    print("Run chunk_quality_report.py again to verify duplicate groups are 0.")


if __name__ == "__main__":
    main()
