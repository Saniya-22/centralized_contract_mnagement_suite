# Retrieval Fix for 31.205-33 (Professional and Consultant Service Costs) — Discussion

**Goal:** When users ask about "professional services" or "consultant services", retrieval should surface FAR 31.205-33 (and related clauses like 52.237-3) so the answer can cite them from the KB instead of relying on model memory.

---

## Why 31.205-33 Might Not Be Surfacing Now

1. **No boost for Part 31:** Right now we only boost **mobilization-style** queries with prefixes like `52.211`, `52.232`, `52.236`, etc. There is no similar boost for **FAR Part 31** (cost principles) or for "professional services"–type queries.
2. **Semantic competition:** For a query like "What is the FAR clause that relates to professional services?", the embedding and FTS search may rank **52.237-3** (Continuity of Services), **52.222-41** (Service Contract Labor Standards), and **52.212-4/5** (Commercial Items) higher than **31.205-33**, because the words "professional" and "services" appear in many 52.xxx clauses. So the 31.205-33 chunk may be in the candidate set but **below the top-k** after RRF + rerank.
3. **Section number in metadata:** Ingest stores `section_number` (e.g. `31.205-33`, `31.203`) in chunk metadata. The existing boost logic in `hybrid_search` uses `preferred_section_prefixes`: if a chunk’s `section_number` starts with one of those prefixes, its score is multiplied by 1.4. So we already have the **mechanism**; we just don’t pass a list that includes `31.205` for professional-services queries.

---

## What Needs to Change (Retrieval Fix)

### 1. Add a “professional services” prefix list and trigger

**Where:** `src/agents/data_retrieval.py`

- Define a small list, e.g.  
  `PROFESSIONAL_SERVICES_PREFIXES = ["31.205", "52.237"]`  
  - `31.205` → 31.205-33 (Professional and Consultant Service Costs)  
  - `52.237` → 52.237-3 (Continuity of Services) and related service clauses  

- Extend **query-based detection** so that when the user query clearly asks about professional/consultant services, we pass this list as `preferred_section_prefixes`. Options:
  - **Option A:** Extend `_preferred_clause_prefixes_for_query(query)` so that, in addition to mobilization triggers, it checks for phrases like `"professional services"`, `"consultant services"`, `"professional and consultant"`, `"FAR clause professional"`, etc. If matched, return `PROFESSIONAL_SERVICES_PREFIXES` (and **not** the mobilization list for that query).
  - **Option B:** Separate helper, e.g. `_preferred_prefixes_professional_services(query)`, and in `_do_regulation_search` call both; if professional-services triggers, pass that list (or merge with mobilization only if you ever want both in one query — usually not needed).

So: **same pattern as mobilization**, but a second set of triggers and a second list of prefixes.

### 2. Pass the list into the search tool

**Where:** `_do_regulation_search` in `data_retrieval.py`

- We already pass `preferred_section_prefixes` when `_preferred_clause_prefixes_for_query(query)` returns non-null (mobilization).  
- After adding professional-services detection, we need to set `tool_args["preferred_section_prefixes"]` to either:
  - the **mobilization** list, or  
  - the **professional services** list,  
  depending on which trigger matched (or define a simple priority, e.g. professional services over mobilization if both match).

No change needed in `vector_search.py` or `queries.py`: they already accept and use `preferred_section_prefixes` for the 1.4x boost.

### 3. Ensure FAR Part 31 is in the index

- Retrieval fix only helps if chunks containing **31.205-33** exist in `embeddings_dense` / FTS.  
- We’ve already seen **FAR 31.203** in a previous response, so FAR 31 is likely indexed.  
- If not, add the FAR Part 31 PDF to `specifications/` and re-run ingest.  
- Optional: a one-off SQL check that some chunk has `metadata->>'section_number' = '31.205-33'` (or similar) to confirm.

---

## Summary of Code Changes (Retrieval Only)

| File | Change |
|------|--------|
| `src/agents/data_retrieval.py` | Add `PROFESSIONAL_SERVICES_PREFIXES`; in query detection (e.g. inside or alongside `_preferred_clause_prefixes_for_query`), detect "professional services" / "consultant services" and return this list; in `_do_regulation_search`, set `preferred_section_prefixes` from this when triggered (with clear priority vs mobilization). |
| `src/db/queries.py` | No change (already supports `preferred_section_prefixes` and boosts by `section_number`). |
| `src/tools/vector_search.py` | No change (already passes `preferred_section_prefixes` to `hybrid_search`). |

---

## Risks / Edge Cases

- **Overlap with mobilization:** If a query somehow matched both "mobilization" and "professional services", we need a rule (e.g. prefer one list or merge). Merging both lists could over-boost. Prefer: **mutually exclusive** (e.g. if professional-services trigger fires, use only that list; else use mobilization list).
- **False positives:** Generic queries containing "professional" or "consultant" might get the boost even when the user isn’t asking about cost principles. Mitigation: use phrases like "professional services", "consultant services", "FAR clause professional" so we don’t trigger on every "professional" mention.
- **31.205-33 still not in top-k:** If the chunk is very low in the fused list, a 1.4x boost might not be enough to bring it into the top 5–6. Options: (a) slightly increase `k` for this intent, or (b) overfetch then filter/boost (already done for meta filtering). We can tune later if needed.

---

## Conclusion

**Retrieval fix** = add professional-services triggers + `PROFESSIONAL_SERVICES_PREFIXES` in data_retrieval, and pass that list into the existing `preferred_section_prefixes` path. No change in DB or vector_search logic. Confirm FAR 31 (and 31.205-33) is in the index. This is a small, low-risk change consistent with the existing mobilization boost.
