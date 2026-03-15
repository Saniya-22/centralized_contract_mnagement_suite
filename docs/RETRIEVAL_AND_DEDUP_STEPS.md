# Retrieval config and duplicate-chunk cleanup

Summary of changes made so you can verify results before doing boundary alignment or coherence filter.

---

## 1) Retrieval config (done)

**What was changed**

- **`src/config.py`** (defaults):
  - `DENSE_TOP_K`: 10 → **15** (more candidates from hybrid search before rerank)
  - `RETRIEVAL_TOP_K`: 6 → **10** (more chunks passed to synthesis)
  - `SELF_HEALING_MAX_QUERIES`: 1 → **2** (up to 2 expanded queries when reflection triggers)
- **`.env`** (override):
  - `RETRIEVAL_TOP_K=10`
  - `DENSE_TOP_K=15`
  - `SELF_HEALING_MAX_QUERIES=2`

**How to verify**

- Run a few regulation queries (e.g. from `queries.txt` or the app). You should see up to 10 retrieved docs and slightly better coverage. No re-ingest or DB change needed.

---

## 2) Duplicate chunks (script provided)

**What was added**

- **`scripts/dedup_embeddings.py`**  
  One-time dedup by `md5(text)`: for each duplicate group it keeps one row (smallest `id`) and deletes the rest from both **embeddings_dense** and **embeddings_sparse** so the two tables stay in sync.

**How to use**

1. **Check current duplicate count**
   ```bash
   python3 scripts/chunk_quality_report.py
   ```
   Look at **Duplicate text hash groups** in the output.

2. **Dry-run (see how many would be removed, no deletes)**
   ```bash
   python3 scripts/dedup_embeddings.py --dry-run
   ```
   Or just:
   ```bash
   python3 scripts/dedup_embeddings.py
   ```

3. **Apply dedup**
   ```bash
   python3 scripts/dedup_embeddings.py --execute
   ```

4. **Verify**
   ```bash
   python3 scripts/chunk_quality_report.py
   ```
   **Duplicate text hash groups** should be 0 (or lower than before).

**Options**

- `--namespace-prefix` — default `public-regulations` (same as chunk_quality_report).
- `--dense-table` / `--sparse-table` — default from env `PG_DENSE_TABLE` / `PG_SPARSE_TABLE`.

**Note:** This script does not change the ingest pipeline. Future full re-ingests can still create duplicates if the pipeline produces identical text for multiple chunks. To avoid that long-term, a dedup step can be added inside `ingest_python/pipeline.py` (e.g. before writing to DB, drop chunks whose `md5(text)` already exists in the batch).

---

## Next steps (after you check)

- If retrieval and duplicate count look good: proceed to **boundary alignment** (chunk split at sentence boundaries) and/or **coherence filter** (backfill coherence score, filter in retrieval) when you’re ready.
