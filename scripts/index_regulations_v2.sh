#!/usr/bin/env bash
set -euo pipefail

# One-command indexing for Chunking Strategy v2.
# - Uses existing DB schema (no new tables)
# - Does NOT delete any existing namespaces
# - Ingests into REGULATIONS_NAMESPACE (defaults to public-regulations-v2)
#
# Prereqs:
# - repo root has venv/ (or set VENV_PYTHON)
# - .env or environment provides DATABASE_URL and OPENAI_API_KEY

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PYTHON="${VENV_PYTHON:-$ROOT_DIR/venv/bin/python}"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "ERROR: venv python not found at '$VENV_PYTHON'"
  echo "Create venv first (see README), or set VENV_PYTHON=/path/to/python"
  exit 1
fi

export REGULATIONS_NAMESPACE="${REGULATIONS_NAMESPACE:-public-regulations-v2}"

echo ">> Using REGULATIONS_NAMESPACE=$REGULATIONS_NAMESPACE"
echo ">> Installing ingestion deps (Docling included)"
$VENV_PYTHON -m pip install -q -r ingest_python/requirements.txt

echo ">> Running ingestion pipeline"
cd ingest_python
$VENV_PYTHON pipeline.py
