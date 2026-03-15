# Intent, Retrieval, and Synthesis Design (Roadmap)

This document outlines a **structural** approach to making the GovGig AI assistant more robust: intent classification, retrieval strategy by intent, synthesis rules, and evaluation. It is intended as a roadmap rather than a one-off patch.

---

## 0. Robust solution (evidence-based, not query-specific)

**Principle:** Correctness should not depend on matching specific query phrases (e.g. "daily reports"). The system should **only claim the regs answer the question when the retrieved documents explicitly support that**. Otherwise, direct the user to the contract and Contracting Officer.

**How it’s implemented:**

1. **Universal sourcing rule (every response)**  
   The synthesizer prompt includes a **Critical — Sourcing** block that applies to **all** intents:
   - Only cite a regulation as answering the question if it **explicitly** addresses what was asked.
   - If the retrieved documents do **not** contain a direct requirement that answers the user’s specific question (e.g. a specific frequency, schedule, or project-specific procedure), state that clearly and recommend checking the contract and consulting the CO. Do not over-claim.
   - When citing a clause, be precise about what it requires and whom it applies to; do not use a clause about one type of requirement to answer a different type.

2. **Low-evidence reinforcement**  
   When retrieved evidence strength is limited (`evidence_avg < 0.55` and not clause_lookup), the prompt adds an explicit note: if the documents do not explicitly answer the user’s question, state that and recommend contract/CO. This reinforces the rule when retrieval is weak.

3. **Contract/CO branch as emphasis only**  
   Queries that match "how often", "daily report", "when do I submit", etc. still get the **contract/CO intent branch** in the prompt for extra emphasis. The **primary** behavior comes from the universal rule, so the system behaves correctly even for phrasings we didn’t add to the trigger list.

**Result:** Any question whose specific answer (e.g. "how often", "when", "what schedule") is not explicitly in the retrieved text gets a "check contract and CO" style answer instead of an over-cited, wrong clause. No dependency on a single query or keyword list for correctness.

---

## 0.1 Robust vs quick fixes (no over-engineering)

**Principle:** Prefer one structural rule over many one-off triggers. Avoid adding a new pattern per failing query.

| Problem | Quick fix (avoid) | Robust fix (use) |
|--------|--------------------|-------------------|
| **In-scope vs out-of-scope** | Add every new phrase to an OOS list or keyword list. | **Layer 1.5:** Minimal set of product/UI phrases only (e.g. "document generator", "upload the specifications"). **Layer 4 (semantic):** Main boundary — improve centroids so regulation vs non-regulation is learned. Layer 3.5 (question frame + acquisition hook) catches conditional / long-tail in-scope questions. |
| **“If I have X, can I be paid?”** | Add "concurrent delay", "compensable" as one-off keywords. | **Widen Layer 3.5:** Question frame includes "if i", "if we"; acquisition hook includes "compensable", "concurrent", "specifications", "vendor". One change to the frame, all conditional/compliance questions benefit. |
| **“Write letter + clause X”** | Special-case “serial letter” or “DFARS 252.236-7000” in synthesis. | **Synthesis order:** If `is_document_request` is True, use the document_request branch **first** (before clause_lookup). So “write/draft” + clause always gets guidance + structure; clause content is still retrieved and cited, but not dumped as the main answer. |
| **Product/feature questions** | Add each product phrase to Layer 1.5. | Keep Layer 1.5 **small and fixed**: only clear product/UI phrases. Rely on Layer 4 semantics for the rest; improve centroids if needed. |

**Result:** Fewer trigger lists, one question-frame layer for in-scope boundary, one synthesis precedence rule for document vs clause. No extra LLM calls, no new subsystems.

---

## 1. Intent layer

**Goal:** Classify each query so we can choose the right retrieval and synthesis behavior.

| Intent | Description | Example queries |
|--------|-------------|-----------------|
| **clause_lookup** | User wants the text or summary of a specific clause (e.g. FAR 52.236-2, DFARS 252.204-7012). | "Show me DFARS 252.204-7012", "What is FAR 36.204?" |
| **regulation_search** | User asks a general regulatory question; answer comes from one or more regs. | "Which FAR clause applies to differing site conditions?", "What is a concurrent delay?" |
| **procedural** | User asks "what do I do" / next steps; answer should be numbered steps. | "We ran into UXO, what do I do?", "First 10 steps as PM post-award." |
| **contract_co_reference** | Question is about frequency, schedule, or project-specific procedures that are often **not** mandated by a single FAR/DFARS clause—specified in contract or by CO. | "How often do I need to send in daily reports?", "When do I submit X?" |
| **document_request** | User asks for a draft (letter, REA, form). We give guidance and structure, not a full draft. | "Write me a serial letter…", "Draft an REA." |
| **out_of_scope** | Not answerable from FAR/DFARS/EM385 or system capability. | "Who founded GovGig?", "What is INDOPACOM?" |

**Current state:** The **5-layer (5-tier) classifier** in `src/tools/query_classifier.py` is the backbone of routing and retrieval:

| Layer | What it does | Output intent | Confidence |
|-------|----------------|---------------|------------|
| 1 | Regex: exact clause reference (FAR 52.236-2, etc.) | clause_lookup | 1.0 |
| 2 | Regex: regulation source only (FAR, DFARS, …) | regulation_search | 0.8 |
| 3 | Word-boundary keyword match (large regulation list) | regulation_search | 0.6 |
| 3.5 | Question frame + acquisition/construction hook | regulation_search | 0.55 |
| 4 | Semantic embeddings vs centroids | regulation_search or out_of_scope | 0.82+ |
| 5 | Micro-LLM fallback (messy clause extraction) | clause_lookup | 0.9 |

So **“5 tier” = 5 (actually 6) waterfall layers**, not 5 intent categories. The classifier today outputs **3 intents**: `clause_lookup`, `regulation_search`, `out_of_scope`. These are written to state (`query_intent`, `detected_clause_ref`, `detected_reg_type`) and drive:

- **Routing:** OUT_OF_SCOPE → end (refuse); else → data_retrieval.
- **Retrieval:** clause_lookup + detected_clause_ref → direct clause fetch; regulation_search → hybrid search (+ optional LLM tool-selector when ambiguous).
- **Quality:** low_confidence overrides use `is_clause_lookup` and evidence.

**Procedural** and **contract_co** are now **classifier flags**: the classifier sets `is_procedural` and `is_contract_co` on every in-scope result (single source of truth in `query_classifier.py`). The router writes them to state; the synthesizer uses state. So the whole pipeline is intent-aware for procedural and contract/CO.

**How to expand intents:** Two options:

1. **New main intent (changes routing):** Add to `QueryIntent` in `query_classifier.py` (e.g. `DOCUMENT_REQUEST = "document_request"`). In the classifier, add a layer or condition that returns it (e.g. keywords: "draft", "write me", "generate"). In the orchestrator router, map it to `next_agent`. In `get_synthesizer_prompt`, add `elif intent == "document_request":` with the right structure (guidance only, no full draft).

2. **New flag (format only, like procedural/contract_co):** Add `is_document_request: bool = False` to `ClassificationResult`, compute it in `classify_query()` with a trigger list, add the field to state and router delta, and in the synthesizer prompt add a branch when `state.get("is_document_request", False)`. Routing stays the same; only answer format changes.

Use (1) when the new intent should change **routing**. Use (2) when it only changes **answer format**.

---

## 2. Retrieval strategy by intent

**Goal:** Retrieve the right kind and amount of content for each intent.

| Intent | Retrieval behavior (target) |
|--------|----------------------------|
| **clause_lookup** | Prefer exact clause/section; fewer chunks, high precision. |
| **regulation_search** | Broader search; multiple clauses/sections; rank by relevance. |
| **procedural** | Same as regulation_search but ensure we pull clauses that govern the procedure (e.g. differing site conditions, REA, notification). |
| **contract_co_reference** | Optional: still search for "reporting" or "daily report" so we can cite any related clauses, but **do not** require a single hit to answer. Synthesizer will say "no single reg; check contract/CO" when appropriate. |
| **document_request** | Same as regulation_search; focus on clauses that inform the document (e.g. excusable delay, REA). |
| **out_of_scope** | No retrieval; refuse and suggest in-scope examples. |

**Current state:** Retrieval is largely intent-agnostic. Clause lookup benefits from detected clause ref (e.g. exact section filter). No separate retrieval path yet for contract/CO queries; the main fix is in **synthesis** (see below).

---

## 3. Synthesis rules

**Goal:** Answer in the right format and with the right sourcing rules.

- **Clause lookup:** Lead with clause text or key excerpt; then "what this means"; applicability and practical note.
- **Regulation search:** Key requirements, applicability, practical note; cite regs; no over-claiming.
- **Procedural:** Start with "Recommended steps:" or "Steps to take:"; numbered steps; no "Key Requirements" heading; optional closing line to consult contract/legal.
- **Contract/CO reference:** If the question is about something often contract-specific (e.g. daily report frequency):
  - State clearly when the retrieved regs **do not** mandate the specific thing asked.
  - Recommend checking the contract, contract schedule, or specs, and consulting the Contracting Officer.
  - If citing a clause (e.g. subcontracting reporting), be precise: say what that clause actually requires and that it may be different from what the user asked; do not over-claim.
- **Document request:** Provide structure and guidance; cite relevant clauses; do not generate full letter/REA/form text.
- **Out of scope:** Polite refusal; suggest example in-scope questions.

**Current state:** Prompts in `src/agents/prompts.py` implement these branches (clause lookup, mobilization, procedural, **contract/CO**, default regulation search). Procedural post-processing in the orchestrator replaces "Key Requirements" with "Recommended steps" when the model still outputs it.

---

## 4. Evaluation and iteration

**Goal:** Track behavior over time and avoid regressions.

- **Gold set:** A small set of queries with expected behavior (see `docs/TEST_QUERIES_REFERENCE.md`):
  - Intent/path (clause_lookup, regulation_search, procedural, contract_co, out_of_scope, document_request).
  - For contract/CO queries (e.g. "How often do I need to send in daily reports?"): expect answer to state that frequency is typically contract/CO-specific and to recommend checking the contract and CO; no over-citation of unrelated reporting clauses.
  - For procedural: expect "Recommended steps" / "Steps to take" and numbered steps.
- **What to log:** For each test run, record: query, intent/path, whether reflection fired, and (optionally) a short verdict (format correct, contract/CO guidance present, no over-claim).
- **Regression runs:** Run the smoke subset (or full list) after changes; diff or compare key answers (e.g. daily reports, differing site conditions, clause lookups).

---

## 5. Implemented so far

- **Universal sourcing rule** (`src/agents/prompts.py`): A **Critical — Sourcing** block is included in **every** synthesizer prompt. It requires: only cite a reg as answering the question if it explicitly does; if the docs don’t contain a direct answer to the user’s specific question, state that and recommend contract/CO; no over-claiming. This is the core of the robust, evidence-based behavior.
- **Low-evidence reinforcement**: When `evidence_summary.avg_norm < 0.55` (and intent is not clause_lookup), the prompt adds a line telling the model that evidence is limited and to recommend contract/CO if the docs don’t explicitly answer. The orchestrator passes `evidence_summary` into `get_synthesizer_prompt()`.
- **Contract/CO branch** (emphasis only): For queries matching `is_contract_co_query()`, the prompt adds extra contract/CO guidance. Correctness does not depend on this branch; the universal rule applies to all queries.
- **Procedural branch:** "Recommended steps" / "Steps to take"; no "Key Requirements"; post-processing in orchestrator.
- **Low-confidence overrides:** In orchestrator, the low-confidence notice is suppressed when evidence/citation strength is high (e.g. clause lookup with good evidence; evidence_avg ≥ 0.70 and citation_coverage ≥ 0.80).

---

## 6. Suggested next steps

1. **Add contract/CO to test reference:** In `TEST_QUERIES_REFERENCE.md`, mark "How often do I need to send in daily reports?" (and similar) as expecting **contract/CO guidance** (state no single reg; check contract and CO).
2. **Optional: intent in state:** Set `query_intent` (or a dedicated flag) to `contract_co_reference` when `is_contract_co_query()` is true, so retrieval or routing can depend on it later.
3. **Regression script:** Run `scripts/run_test_queries.py` on a small subset (e.g. smoke + daily reports + one clause lookup) and check that (a) daily reports gets contract/CO guidance, (b) procedural gets steps format, (c) clause lookups still return clause text.
4. **Expand contract/CO triggers:** If more "frequency" or "when do I" questions appear, add phrases to `_CONTRACT_CO_TRIGGERS` in `prompts.py` (and optionally to the classifier) so they get the same synthesis guidance.
