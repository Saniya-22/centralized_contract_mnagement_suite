# Deep Analysis: Ingestion Documents & Robust Chunking Strategy

**Status:** Analysis complete. **No ingestion/index code has been changed.** Implementation and index removal/promotion will proceed only after your approval.

---

## Part 1 — Deep Analysis of Real Documents (Extracted from PDFs)

Text was extracted from `specifications/` PDFs (FAR_52, DFARS_201_to_253, EM 385-1-1 March 2024) and inspected for structure.

### 1.1 FAR (FAR_52.pdf)

- **Early pages:** Table of contents — clause numbers like `52.202-1`, `52.203-2`, `52.236-11` with titles; indentation by hierarchy.
- **Body pages:** Full clause text appears as:
  - **Clause start:** Line like `52.215-2 Audit and Records-Negotiation.` then prescription text, then:
  - **(a), (b), (c)** at line start (with leading spaces); each can contain multiple sentences.
  - **(1), (2), (3), (4)** under (c); **(i), (ii), (iii), (iv)** under (1).
- **Pattern:** Sub-clause boundaries are **line-start** `(a)`, `(b)`, `(1)`, `(2)`, `(i)`, `(ii)`. Current pipeline splits only on **next clause number** (52.xxx), not on (a)(b)(1)(2), so one large clause is one block and gets split by paragraph/token only → mid-requirement cuts.

### 1.2 DFARS (DFARS_201_to_253.pdf)

- **Structure:** Section numbers like `201.602`, `201.602-2`, `201.603-2`; then **(d)**, **(1)**, **(i)(ii)(iii)(iv)**, **(2)**, **(3)**, **(a)(b)**. Same (a)(b)(1)(2)(i)(ii) hierarchy as FAR.
- **Clause references in text:** e.g. `252.201-7000` mentioned in body; actual clause text in other PDFs (Appendix etc.).
- **Conclusion:** Same need as FAR — use (a)(b)(1)(2) at line start as secondary split boundaries within a section.

### 1.3 EM 385-1-1 (15 March 2024)

- **Numbering in this PDF:** Uses **`1-1.`**, **`1-2.`**, **`1-3.`** (chapter-paragraph), not `01.A.01` in the sampled pages. Example: `1-1. References.`, `1-2. Definitions.`, `1-3. Personnel Required Qualification/Training.`
- **Sub-items:** Lowercase letters with period: `a.`, `b.`, `c.`; then `(1)`, `(2)`; then `(a)`, `(b)` for deeper levels. Inline citations like `(1-1.a)`, `(1-2.b)`.
- **Manual says:** "Citations start with chapter number, then paragraph number, followed by a letter... e.g. 10-3.b means chapter 10, paragraph 3, subparagraph b."
- **Conclusion:** Section boundaries = lines starting with `\d+-\d+\.` (e.g. 1-1., 1-2.). Pipeline’s EM385 section regex already matches this. Optional: support `01.A.01` if it appears in other chapters; secondary split on `a.` `b.` `(1)` `(2)` for long sections.

### 1.4 Tables

- Pipeline detects table-like lines (tabs, 4+ spaces, pipes) and puts them in `content['tables']`; only `content['text']` is used for section extraction. So **table content is not currently chunked**. Recommendation: append table text into the main text stream (e.g. with a `[Table]` delimiter) before section parsing so it is included in chunks.

---

## Part 2 — Robust Chunking Strategy (Proposed)

### 2.1 Principles

1. **Chunk = one retrievable unit of obligation** — one clause, sub-clause, or one clear requirement; size is a guardrail.
2. **Structure-first, size-second** — boundaries from document structure; split at token limit only when necessary, and then at sub-structure.
3. **Source-specific boundaries** — FAR/DFARS vs EM385 vs Appendix.
4. **Short standalone requirements stay separate** — do not merge lines that are clearly a clause/section start (e.g. "(a) Effective date.").
5. **Minimal overlap** — only at natural boundaries when a long clause is split; no large duplicate spans.
6. **Single size regime** — one target range, one max cap, one min (with anchor exception).

### 2.2 FAR / DFARS

- **Primary boundary:** Line starting a new clause (52.xxx-x, 252.xxx-x) or appendix section (A-1, Rule 1). *Keep current behaviour.*
- **Secondary boundary (new):** Within a clause block, split on **line-start sub-clause** patterns:
  - `^\s*\([a-z]\)\s`  → (a), (b), (c)
  - `^\s*\(\d+\)\s`   → (1), (2), (3)
  - Optionally `^\s*\([ivx]+\)\s` for (i), (ii)
- **Size:** Target 350–550 tokens per chunk; hard max 650. Chunks &lt; 100 tokens: keep if first line matches clause/section number (anchor); else merge only with **next** chunk if combined &lt; 550.
- **Overlap:** Default 0; allow 1–2 sentences only when splitting a long (a) or (1) across two chunks.

### 2.3 EM385

- **Primary boundary:** Section start `\d+-\d+\.` (1-1., 1-2.). Already supported in section extraction. Add **clause-boundary split for EM385** in `_split_on_clause_boundaries`: when source is EM385, split on line-start `\d+-\d+\.` so each 1-x. block is separate before token-based split.
- **Secondary (optional):** For blocks &gt; max_tokens, split on `a.` `b.` `(1)` `(2)` at line start.
- **Anchor:** Treat line starting with `\d+-\d+\.` as anchor (don’t force-merge tiny).
- **01.A.01:** Add support in section regex and clause split if that pattern appears in other EM385 chapters; not present in sampled pages.

### 2.4 Appendices (FAR/DFARS)

- A-1, Rule 1 = section start (existing). Within a rule, use (1), (2), (3) at line start as secondary boundaries. Same size/merge rules as main FAR/DFARS.

### 2.5 Tables

- Append `content['tables']` (or flattened table text) into `content['text']` before `extract_structured_sections` so table content is included in sections and chunks.

### 2.6 Single Size / Merge Regime

- **Target chunk:** 350–550 tokens (e.g. 450).
- **Max chunk:** 650 tokens.
- **Min chunk:** 50 for anchor (clause/section number line); 100 for non-anchor. Below 100 and not anchor → merge with **next** only if combined &lt; 550.
- **Overlap:** 0 by default; max 50 tokens when splitting long sub-clause.
- Use one set of constants (no dual TARGET_CHUNK vs CHUNK_SIZE).

---

## Part 3 — Implementation Steps (After Your Approval)

1. **ingest_python/config.py** — Unify: e.g. `TARGET_CHUNK_TOKENS=450`, `MAX_CHUNK_TOKENS=650`, `MIN_CHUNK_TOKENS=100`, `MIN_ANCHOR_TOKENS=50`, `CHUNK_OVERLAP=0` (or 50 for split-only).
2. **ingest_python/pipeline.py**  
   - Add secondary split on (a)(b)(1)(2) line-start for FAR/DFARS in `_split_on_clause_boundaries` or a new helper used when block &gt; max_tokens.  
   - For EM385: in `_split_on_clause_boundaries`, allow source EM385 and split on `^\d+-\d+\.`; extend anchor regex for 1-x.  
   - Append table text to page text before section extraction.  
   - Use single size/merge constants; adjust merge logic to “merge with next” and anchor exception.
3. **ingest_python/parsing/rules.py** — No change required for 1-1.; optionally add rule for 01.A.01 if needed.
4. **Tests** — Run `chunk_quality_report` on a small re-ingest (one PDF) to validate chunk sizes and boundaries before full run.

---

## Part 4 — Index Removal & Promotion (For Your Approval)

Retrieval uses **namespace LIKE `public-regulations%`** (e.g. `public-regulations-far`, `public-regulations-dfars`, `public-regulations-em385`). Ingest writes to `{NAMESPACE}-{source}`; default NAMESPACE = `public-regulations`.

### Option A — Direct replace (simplest)

1. **Back up (optional):** Export row counts or copy table if you need a rollback.
2. **Remove existing index:**  
   ```sql
   DELETE FROM embeddings_sparse WHERE namespace LIKE 'public-regulations%';
   DELETE FROM embeddings_dense   WHERE namespace LIKE 'public-regulations%';
   ```
   Or run a small script that runs the same (with your DB connection from .env).
3. **Re-ingest** with updated chunking (same NAMESPACE `public-regulations`):  
   From project root: `cd ingest_python && python pipeline.py` (with .env and OPENAI_API_KEY, DATABASE_URL set). This repopulates `public-regulations-far`, `public-regulations-dfars`, `public-regulations-em385`.
4. **Quality check:** Run `python scripts/chunk_quality_report.py` and fix any config if needed.
5. **Application:** Point app at same DB; no promotion step (you replaced in place).

### Option B — Pilot then promote (safer, with gates)

1. **Ingest to pilot namespace:** Set `REGULATIONS_NAMESPACE=public-regulations-pilot` in .env (or env for the run). Run `cd ingest_python && python pipeline.py`. This creates `public-regulations-pilot-far`, `-dfars`, `-em385`.
2. **Run quality gates (dry-run):**  
   ```bash
   python scripts/promote_index.py --source public-regulations-pilot-far    --target public-regulations-far    --apply  # repeat for dfars, em385; or use one source/target if script supports wildcard
   ```  
   Promotion script expects one source/target; run once per source (far, dfars, em385). First run **without** `--apply` to see gate results.
3. **Remove current production index (optional if using --archive-existing):**  
   ```sql
   DELETE FROM embeddings_sparse WHERE namespace LIKE 'public-regulations-%' AND namespace NOT LIKE 'public-regulations-pilot%';
   DELETE FROM embeddings_dense   WHERE namespace LIKE 'public-regulations-%' AND namespace NOT LIKE 'public-regulations-pilot%';
   ```
   Or delete only `public-regulations-far`, `public-regulations-dfars`, `public-regulations-em385`. If you use `--archive-existing` in the next step, the script can rename the current target namespaces to archive instead of you deleting manually.
4. **Promote (one run for all sources):** The script uses namespace prefix; all pilot namespaces are copied to target in one go:  
   ```bash
   python scripts/promote_index.py --source public-regulations-pilot --target public-regulations --archive-existing --apply
   ```
   This copies `public-regulations-pilot-far` → `public-regulations-far`, `public-regulations-pilot-dfars` → `public-regulations-dfars`, `public-regulations-pilot-em385` → `public-regulations-em385`, and with `--archive-existing` renames existing `public-regulations-*` to `public-regulations-archive-YYYYMMDDHHMMSS` before copy. Run without `--apply` first to see gate results.
5. **Application:** Uses same `public-regulations*`; retrieval continues to work.

### Important notes

- **Do not** delete or overwrite namespaces until you have confirmed the new ingest (and optionally pilot) looks correct (chunk_quality_report, spot-check queries).
- **Backup:** If you have production data, take a DB snapshot or export before DELETE.
- **Promote script:** `--apply` is required to execute; without it, only dry-run. Use `--archive-existing` if you want to keep the old index under an archive namespace instead of hard delete.

---

## Part 5 — Confirmation Checklist (Before You Approve)

- [ ] I have read the deep analysis (Part 1) and the proposed strategy (Part 2).
- [ ] I approve the implementation steps (Part 3) and want to proceed with code changes.
- [ ] I have chosen Option A (direct replace) or Option B (pilot then promote).
- [ ] I understand that existing index rows for `public-regulations%` will be removed (Option A) or replaced via promotion (Option B).
- [ ] I will run re-ingestion (and optionally promotion) after approval; no automatic delete or promotion will be done without my go-ahead.

---

**Next:** After you approve, implementation will: (1) apply config and pipeline changes for chunking, (2) provide a small script or exact commands for index removal (and promotion if Option B), and (3) leave the actual DELETE and re-ingest/promotion runs to you (or to a follow-up step you confirm).
