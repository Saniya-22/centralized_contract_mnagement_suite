# System Correctness Verification

**Date:** 2025-03-03  
**Scope:** End-to-end flow, state contract, RAG pipeline, reflection, and API response shape.

---

## 1. Executive Summary

| Area | Status | Notes |
|------|--------|--------|
| **State flow (Router → Data Retrieval → Synthesizer)** | ✅ Verified | Classifier output passed correctly; delta-only returns; OUT_OF_SCOPE doc clearing |
| **Reflection (critique + self-healing)** | ✅ Verified | Tests pass; regulation mismatch and healing logic correct |
| **Query classifier** | ✅ Verified | 23 tests pass; intent/confidence/clause extraction correct |
| **Vector search tool** | ✅ Verified | Schema and mocked execution tests pass |
| **Sovereign guard** | ✅ Verified | URL build, verdict parsing, fail-open/closed behavior |
| **Checkpointer import** | ✅ Fixed | Lazy import so tests run without `langgraph-checkpoint-postgres` |
| **Unit test suite (no app)** | ✅ 44 passed | query_classifier, reflection, vector_search, sovereign_guard |
| **API / Orchestrator tests** | ⚠️ Env-dependent | Require Python 3.11+ and optional checkpoint package |

---

## 2. State and Data Flow

### 2.1 Router → State

- **Router** (`_route_query`) sets:
  - `query_intent` = `intent.value` (e.g. `"clause_lookup"`, `"regulation_search"`, `"out_of_scope"`)
  - `detected_clause_ref` = `classification.clause_reference`
  - `detected_reg_type` = `classification.regulation_type`
  - `next_agent` = `"data_retrieval"` or `"end"`
  - For OUT_OF_SCOPE: `generated_response` = refusal message
- **Data Retrieval** reads `state.get("query_intent")`, `state.get("detected_clause_ref")`, `state.get("detected_reg_type")`.
- **Intent comparison:** `QueryIntent` is a `str` Enum; `state["query_intent"]` is the same string, so `query_intent == QueryIntent.CLAUSE_LOOKUP` is correct.

### 2.2 Data Retrieval → Delta

- Agent returns **only deltas** (e.g. `retrieved_documents`, `tool_calls`, `regulation_types_used`, `agent_path`).
- `GovGigState` uses `Annotated[List, operator.add]` for list fields, so the graph reducer merges deltas correctly.
- No full state overwrite; contract is correct.

### 2.3 OUT_OF_SCOPE and Document Leakage

- When router sets `next_agent="end"`, the graph goes to END without running Data Retrieval or Synthesizer.
- **run_async** (REST API) explicitly clears documents for OUT_OF_SCOPE:
  - `is_out_of_scope` = agent_path contains `"out_of_scope"` or response starts with refusal prefix.
  - Return uses `"documents": [] if is_out_of_scope else result.get("retrieved_documents", [])`.
- So the API never returns previous-session docs for out-of-scope queries.

### 2.4 Synthesizer Inputs

- Reads `state.get("retrieved_documents", [])`.
- No-docs path returns `_safe_fallback_message()` and `low_confidence: True`.
- Quality metrics (`_assess_answer_quality`) use evidence summary, citation coverage, groundedness; clause lookup gets relaxed `min_docs` (1 vs 2).

---

## 3. Reflection and Self-Healing

- **RetrievalCritique:** Score normalization (rerank 0–10 vs RRF ~0.01–0.05), regulation-type alignment, keyword-overlap rescue. Covered by tests.
- **ReflectionManager.heal_search:** Calls `expand()` then runs `search_func` per expanded query; flattens and dedupes by (source, section, content prefix). `max_docs` cap applied.
- **Data retrieval:** On critique failure, clears docs when mismatch is detected, then calls `heal_search` with `_search_wrapper` that passes `regulation_type` filter. Correct.

---

## 4. RAG Pipeline Correctness

- **Hybrid search:** Dense + FTS in parallel, RRF merge, optional rerank, token budget. Order of operations and field names (`content`, `source`, `score`, etc.) match what the synthesizer and critique expect.
- **Clause lookup:** Direct ILIKE → FTS fallback → hybrid fallback; `found` and `fuzzy_results` semantics correct.
- **Token budget:** Applied after rerank; respects `RAG_TOKEN_LIMIT`.

---

## 5. Test Results (Current Run)

- **tests/test_reflection.py:** 7 passed  
- **tests/test_query_classifier.py:** 23 passed  
- **tests/test_vector_search.py:** 3 passed (mocked)  
- **tests/test_sovereign_guard.py:** 11 passed  
- **Total:** 44 passed (no DB, no live API)

**Tests not run in this verification (require Python 3.11+ and/or deps):**

- `test_api.py` – imports FastAPI app and orchestrator (Python 3.9 fails on `dict[str, float | bool]`).
- `test_orchestrator.py` – same Python version requirement.
- `test_data_retrieval.py` – depends on orchestrator/agent stack.

To run the full suite:

```bash
# Recommended: Python 3.11+
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# If using persistence:
pip install langgraph-checkpoint-postgres
pytest tests/ -v
```

---

## 6. Change Made During Verification

- **src/db/connection.py:** Checkpointer import made **lazy** inside `CheckpointerManager.get_checkpointer()`. This allows importing `src.db.connection` (and thus running query_classifier, vector_search, and reflection tests) without installing `langgraph-checkpoint-postgres`. The checkpointer is only loaded when the API starts and calls `await CheckpointerManager.get_checkpointer()`.

---

## 7. Conclusion

Core behavior is **correct** for:

- Router → Data Retrieval → Synthesizer state and delta flow  
- OUT_OF_SCOPE handling and document clearing in the REST response  
- Reflection critique and self-healing  
- Query classifier and vector search tool contracts  

For full system tests (API, orchestrator, data_retrieval), use Python 3.11+ and install optional checkpoint dependencies as above.
