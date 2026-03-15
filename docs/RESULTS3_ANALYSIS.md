# results3.txt — Analysis

Run: 109 queries from `queries.txt`. Summary and issues.

---

## 1. Path distribution

| Path               | Count | %    |
|--------------------|-------|------|
| regulation_search  | 94    | 86%  |
| other (out of scope) | 12  | 11%  |
| clause_lookup     | 3     | 3%   |

- **clause_lookup** [6, 25, 85]: "Write the KO a serial...", "Add DFARS 252.236-7000...", "What is FAR 36.204?" — clause ref detected, 1 doc, conf 1.0.
- **other**: [1–4, 8, 11, 12, 62, 66, 87, 94, 109] — refused with in-scope suggestions.

---

## 2. What’s working well

- **Contract/CO (frequency)**
  - [46] "How often do I need to send in daily reports?" — Correct: no single reg; check contract/CO. No over-cite of 52.219-9.
  - [13] "How often should the superintendent update drawings?" — Same: not mandated, contract/CO.
  - [37] "How often can a general contractor submit billing?" — Same.
- **Procedural (Recommended steps)**
  - [7] UXO at Camp Pendleton, [31] REA vs change order, [34] safety incident, [35] daily rate, [43] prime not paying, [47] duct bank, [61] draft REA, [67] set-aside, [91] first 10 steps PM, [102] fire at site, [107] oil tank — all use **Recommended steps** and numbered steps.
- **Clause lookups**
  - [85] FAR 36.204, [25] DFARS 252.236-7000 — clause-focused answers.
- **Honest “not in docs”**
  - [80] "What is SIOP?" — States SIOP not in retrieved docs; no fake cite.
- **Out-of-scope refusals**
  - [1–4] Navy/GAO/Puget/Founder, [8] document generator working, [11] INDOPACOM, [12] OCONUS, [66] Export to Word, [94] upload specs, [109] "Test these queries" — all correctly refused.

---

## 3. Issues and fixes

### 3.1 [9] "Why is the document generator still under construction?"

- **Current:** path=regulation_search; answer says “check contract/CO” (wrong context).
- **Cause:** Product/feature question was not classified out of scope.
- **Fix (done):** Layer 1.5 in classifier — `document generator` (+ similar) → OUT_OF_SCOPE. After redeploy, [9] should be path=other and get the standard refusal.

### 3.2 [62] "If I have concurrent compensable and non-compensable days of delay, can I be paid...?"

- **Current:** path=other, refused.
- **Issue:** False negative. Valid FAR/regulations question (concurrent delay, compensable).
- **Fix:** Add regulation keywords or question-frame hook for “concurrent”, “compensable”, “delay”, “paid” so it stays in-scope (e.g. in Layer 3 or 3.5). Then retrieval + synthesis can answer from delay/REA clauses.

### 3.3 [87] "What do we do if a vendor cannot match the specifications?"

- **Current:** path=other, refused.
- **Issue:** Arguably in-scope (specs, compliance, substitution). Likely missed by keyword list.
- **Fix:** Consider adding “specifications”, “vendor”, “match” (or similar) to regulation keywords / acquisition hook so it gets regulation_search when appropriate.

### 3.4 [6] "Write the KO a serial about the shutdown..."

- **Current:** path=clause_lookup, 1 doc (DFARS 252.236-7000); answer is price breakdown.
- **Issue:** User asked for a serial letter (document request) + shutdown/impacts. Clause lookup picked up “252.236-7000” and returned clause content; answer is off-topic for “serial about shutdown”.
- **Fix:** Document_request should take precedence when query asks to “write/draft” something and also mentions a clause. Option: in classifier, if both clause ref and document_request triggers match, set is_document_request=True and still do retrieval (regulation_search or clause_lookup), but synthesizer uses document_request branch (guidance + structure, not full draft; include relevant clauses for the letter).

### 3.5 [65] "Can you recommend me a Government contracts attorney?"

- **Current:** regulation_search; answer politely declines to recommend.
- **Note:** Behaviour is reasonable. Could optionally add out-of-scope pattern for “recommend … attorney” to get consistent refusal message; low priority.

### 3.6 Low confidence notice on many answers

- Many regulation_search answers have low confidence and the “verify cited clauses” notice (e.g. conf &lt; 0.55 or quality thresholds).
- Expected when evidence is weak; reduces over-claiming. Optional: tune thresholds if notice appears too often on solid answers.

### 3.7 [98] "What is a PPI?"

- Answer defines PPI as “Protected Personally Identifiable Information”. In construction/gov contracting, PPI often means “Pre-Purchase Inspection”. So possible wrong sense.
- Fix: Either improve retrieval for “PPI” in contract context or add a short note in synthesis that “PPI” can have multiple meanings and cite which one the regs use.

---

## 4. Summary table

| Category                    | Count / status |
|----------------------------|----------------|
| Total queries               | 109            |
| Correct out-of-scope refuse | 11 (1,2,3,4,8,11,12,66,94,109; 87 borderline) |
| Wrong out-of-scope (refused but in-scope) | 1–2 ([62] clear; [87] possible) |
| Wrong in-scope (product Q treated as reg) | 1 ([9]; fix in place) |
| Contract/CO frequency       | Working ([46], [13], [37]) |
| Procedural (Recommended steps) | Working (many) |
| Document request / clause mix | 1 ([6]); document_request precedence could help |

---

## 5. Recommended next steps

1. **Re-run after Layer 1.5** — Confirm [9] becomes path=other and gets refusal.
2. **Reduce false refusals** — Add triggers for [62] (concurrent delay, compensable) and optionally [87] (specifications, vendor) so they get regulation_search.
3. **Document request + clause** — When “write/draft” and a clause both appear, prefer document_request guidance (structure + which clauses to use) over raw clause_lookup answer.
4. **Optional** — Add out-of-scope pattern for “recommend … attorney”; clarify PPI in synthesis when context is construction/contracting.
