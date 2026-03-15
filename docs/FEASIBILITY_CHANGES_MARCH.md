# Feasibility & Risk: Pre–March End Changes

**Purpose:** Discuss whether the proposed changes are feasible and what issues could arise before implementing.

---

## 1. Retrieval-error handling

**What:** In the orchestrator synthesizer, before using `documents`, treat any doc that has an `"error"` key as a failed-retrieval signal. If all docs are error dicts (or the only doc is an error dict), treat as “no documents” and return the same fallback message + low_confidence as when `documents` is empty.

**Feasibility:** Yes. One place to change: `_synthesize_response` in `src/agents/orchestrator.py`. After computing `documents` from state (and offsets), add:

- Filter out any item where `doc.get("error")` is truthy.
- If the filtered list is empty, take the existing “no documents” path (fallback message, quality_metrics, low_confidence).

**Risks / issues:**

- **False positive:** If a normal chunk ever had a metadata field named `"error"` with a value, it would be dropped. Current pipeline does not put `"error"` on real chunks; only the vector_search tool returns `{"error": ..., "message": ...}` on exception. So risk is low.
- **Partial failure:** If we have e.g. 3 real docs + 1 error doc (e.g. from a future code path that merges results), we’d drop only the error doc and synthesize on the 3. That’s desired.
- **Backward compatibility:** No API or state shape change. Existing clients and tests unaffected.

**Verdict:** Feasible, low risk. Safe to add before demo.

---

## 2. Clause-first format (clause lookup)

**What:** For `query_intent == "clause_lookup"` and when we have clause content, change the synthesizer prompt so the model is asked to: (1) start with the exact clause text or a key excerpt, then (2) add a short “What this means” in plain language.

**Feasibility:** Yes. Change only in `src/agents/prompts.py` in `get_synthesizer_prompt()`, inside the existing `if intent == "clause_lookup" and clause_ref:` branch. Add 2–3 lines to the `intent_guidance` string (e.g. “Start with the exact clause text or a key excerpt from the retrieved document; then add a short ‘What this means’ in plain language.”).

**Risks / issues:**

- **Prompt creep:** Slightly longer prompt. Minor; no functional risk.
- **Model behavior:** The model might sometimes shorten or paraphrase the “exact” clause. We’re not enforcing strict formatting in code, so output can vary. Still an improvement over no clause text.
- **When clause not found:** Direct clause lookup can return “found”: false with fuzzy results. In that case we still have one or more docs; the same prompt applies. No need to special-case “no clause” in this change.

**Verdict:** Feasible, low risk. Safe to add before demo.

---

## 3. 31.205-33 + “which one would you like more detail on?”

**What:** (A) For queries that mention “professional services” or “consultant services”, ensure the answer can include FAR 31.205-33 and (B) when multiple clauses are relevant, list them with titles and end with something like “Which one would you like more detail on?”

**Feasibility:**

- **(A) 31.205-33:** No hardcoding of 31.205-33 in code today. It will only appear if retrieval returns a chunk that contains it (e.g. from FAR_31). So either:
  - **Option 1:** Rely on retrieval (query expansion / better ranking). That’s a retrieval-tuning or intent-strategy change (medium effort).
  - **Option 2:** In the synthesizer prompt for “professional services”–type queries, add one line: “If relevant, include FAR 31.205-33 (Professional and Consultant Service Costs).” The model can then mention it even if that chunk wasn’t in the top‑k; but that could be **ungrounded** if 31.205-33 wasn’t retrieved. So Option 2 has a grounding risk unless we only add it when we know we have FAR 31 in the index and we’re okay with the model “reaching” for it when it wasn’t in the retrieved set.
- **(B) “Which one?”:** Easy. In the same prompt (e.g. for regulation_search or a new “multiple clauses” branch), add: “When multiple clauses are relevant, list each with its number and title; end with a single line asking which one the user would like more detail on.”

**Risks / issues:**

- **31.205-33 without retrieval:** If we prompt the model to “include 31.205-33” and that chunk wasn’t retrieved, the model might still write something about it (from training). That can be useful but is **not fully grounded** in your KB. For a compliance product, some teams prefer to only cite retrieved docs. So: either we only add the “which one?” part (safe), or we add 31.205-33 via prompt and accept possible ungrounded mention, or we do retrieval work so 31.205-33 actually surfaces (best long term).
- **“Which one?” on every answer:** If we add “list clauses and ask which one” to the default regulation_search branch, every multi-doc answer might end with that question. That could feel repetitive for queries that aren’t “give me a list of clauses.” So we’d want to restrict it (e.g. only when the query is about “professional services” or “which clause” or “what clauses apply”) so we don’t change behavior for all queries.

**Verdict:**  
- “Which one?” (B): Feasible and low risk if we limit it to specific intents or query phrases.  
- 31.205-33 (A): Feasible via prompt only, but with grounding risk; better long term to fix retrieval. For March end, we can add (B) and a cautious prompt hint for (A) only when we already have multiple clause-like docs.

---

## 4. Summary

| Change                     | Feasible? | Risk              | Suggested for March end |
|----------------------------|-----------|-------------------|--------------------------|
| Retrieval-error handling   | Yes       | Low               | Yes                      |
| Clause-first format        | Yes       | Low               | Yes                      |
| “Which one?” line          | Yes       | Low (if scoped)   | Yes                      |
| 31.205-33 in answer        | Yes*      | Grounding concern | Optional / cautious       |

*31.205-33: best done by retrieval; prompt-only mention is possible but can be ungrounded.

**Conclusion:** Changes 1, 2, and 3(B) scoped to relevant queries are feasible and unlikely to cause issues. We can implement 1 and 2 with high confidence; 3 with “which one?” and optional 31.205-33 hint can be added with minimal risk if we scope and document the grounding trade-off.
