# Technical Specification: Ingestion & Robust Chunking Strategy

This document specifies the chunking and ingestion strategy implemented in the GovGig AI system (Chunking Strategy v2).

---

## 1. Document Structure Analysis

The ingestion pipeline is optimized for federal regulatory documents (FAR, DFARS, EM 385-1-1).

### 1.1 FAR / DFARS Structure
- **Clause Identifiers:** Exact matches for clause numbers like `52.202-1`, `52.236-11`, or `252.204-7012`.
- **Hierarchical Markers:** The pipeline recognizes sub-clause boundaries at line-starts: `(a)`, `(b)`, `(1)`, `(2)`, `(i)`, `(ii)`.
- **Chunking Logic:** Splitting occurs primarily on clause/section boundaries. For oversized blocks, the pipeline splits on sub-clause markers to maintain requirement integrity.

### 1.2 EM 385-1-1 Structure
- **Citations:** Uses chapter-paragraph format like `1-1.` or `10-3.b`.
- **Sub-items:** recognized at `a.`, `b.`, or `(1)`, `(2)` for deeper levels.
- **Section Boundaries:** Lines starting with `\d+-\d+\.` are treated as primary boundaries.

### 1.3 Table Processing
- **Docling Integration:** PDFs are converted using Docling to preserve table structure.
- **Dual Storage:** Table content is stored as both flattened text for semantic search and structured JSON for precision retrieval.

---

## 2. Robust Chunking Strategy (v2)

### 2.1 Core Principles
1. **Meaning-First, Size-Second:** Chunks are defined by document structure (clauses/sub-clauses) rather than arbitrary token counts.
2. **Anchor Enrichment:** Each section emits a high-quality "anchor chunk" containing the title and initial context, marked with `is_anchor=true`.
3. **Integrity Guard:** Legal units containing strong obligations (`shall`, `must`, `required`) are kept intact when possible.
4. **Minimal Overlap:** Overlap is used only as a fallback when splitting long paragraphs that lack natural structure.

### 2.2 Token Gates & Size Regimes
- **Target Chunk Size:** 450 tokens.
- **Maximum Chunk Size:** 650 tokens (hard cap).
- **Minimum Chunk Size:** 100 tokens (non-anchors) or 50 tokens (anchors).
- **Consolidation:** Chunks below the minimum threshold are merged with adjacent chunks within the same section to reduce retrieval noise.

### 2.3 Metadata Schema
Each chunk stored in the `JSONB` metadata column includes:
- `is_anchor`: Boolean identifying anchor chunks.
- `clause_references`: Extracted FAR/DFARS/EM385 references for improved cross-linking.
- `hierarchy_struct`: Original document hierarchy path.
- `type`: Set to `"table"` for dedicated table chunks.
- `table_text` / `table_structured`: Structured content for Docling-extracted tables.

---

## 3. Operations & Maintenance

### 3.1 Re-ingestion Workflow
To re-ingest regulatory data while maintaining production availability:

1. **Ingest to pilot namespace:**
   ```bash
   # Set NAMESPACE in ingest_python/config.py or env
   export NAMESPACE=public-regulations-v2
   python ingest_python/pipeline.py
   ```
2. **Promotion:** Once verified, promote the namespace using `scripts/promote_index.py` or update the backend `REGULATIONS_NAMESPACE` in `src/config.py`.

### 3.2 Idempotency
The pipeline uses SHA-256 file hashing to avoid redundant processing of unchanged PDFs. If a file hash matches an existing record in the database, the file is skipped during ingestion.
