#!/usr/bin/env python3
"""Delete all public-regulations% rows from embeddings_dense and embeddings_sparse, then run full re-ingest.

Option A (wipe and re-ingest). Run from project root with .env set (DATABASE_URL, OPENAI_API_KEY).

Usage:
  python scripts/clear_regulations_index_and_reingest.py
  python scripts/clear_regulations_index_and_reingest.py --dry-run   # only print what would be deleted
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Project root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env():
    from dotenv import dotenv_values

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        return dotenv_values(env_path)
    return {}


def main():
    ap = argparse.ArgumentParser(
        description="Clear regulations index and run full re-ingest"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report row counts and skip DELETE + ingest",
    )
    args = ap.parse_args()

    env = load_env()
    db_url = env.get("DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not db_url:
        print(
            "ERROR: DATABASE_URL not set. Set it in .env or environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import psycopg
    except ImportError:
        print(
            "ERROR: psycopg not installed. pip install psycopg[binary]", file=sys.stderr
        )
        sys.exit(1)

    dense_table = env.get("PG_DENSE_TABLE", "embeddings_dense")
    sparse_table = env.get("PG_SPARSE_TABLE", "embeddings_sparse")
    namespace_like = "public-regulations%"

    conninfo = db_url.replace("postgresql+asyncpg://", "postgresql://").split("?")[0]
    if "postgresql://" not in conninfo:
        conninfo = "postgresql://" + conninfo

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM {dense_table} WHERE namespace LIKE %s",
                (namespace_like,),
            )
            dense_count = cur.fetchone()[0]
            cur.execute(
                f"SELECT COUNT(*) FROM {sparse_table} WHERE namespace LIKE %s",
                (namespace_like,),
            )
            sparse_count = cur.fetchone()[0]
    print(
        f"Current index: {dense_count} dense rows, {sparse_count} sparse rows (namespace LIKE '{namespace_like}')"
    )

    if args.dry_run:
        print("Dry-run: skipping DELETE and re-ingest.")
        return

    print("Deleting from embeddings_sparse and embeddings_dense...")
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {sparse_table} WHERE namespace LIKE %s", (namespace_like,)
            )
            sparse_deleted = cur.rowcount
            cur.execute(
                f"DELETE FROM {dense_table} WHERE namespace LIKE %s", (namespace_like,)
            )
            dense_deleted = cur.rowcount
        conn.commit()
    print(f"Deleted: {sparse_deleted} sparse, {dense_deleted} dense.")

    ingest_dir = PROJECT_ROOT / "ingest_python"
    if not (ingest_dir / "pipeline.py").exists():
        print(f"ERROR: {ingest_dir / 'pipeline.py'} not found.", file=sys.stderr)
        sys.exit(1)

    env_run = os.environ.copy()
    env_run.update(env)
    env_run.pop("PG_SSLMODE", None)
    if "?" in db_url and "ssl=" in db_url:
        pass
    else:
        env_run["DATABASE_URL"] = db_url

    python_exe = sys.executable
    venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
    if venv_python.exists():
        python_exe = str(venv_python)
    print("Running full re-ingest (ingest_python/pipeline.py)...")
    result = subprocess.run(
        [python_exe, "pipeline.py"],
        cwd=ingest_dir,
        env=env_run,
    )
    if result.returncode != 0:
        print("Re-ingest failed.", file=sys.stderr)
        sys.exit(result.returncode)
    print("Re-ingest complete.")


if __name__ == "__main__":
    main()
