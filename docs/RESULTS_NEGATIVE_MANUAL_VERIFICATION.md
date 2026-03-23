# Manual Verification: Response vs Previous Response

**Source:** `results_negative_update.xlsx`  
**Method:** Human review of each row; which column (Response or Previous Response) is better for the query (more on-topic, more complete, or more accurate). No use of “Our system is still expanding” as automatic penalty; no use of User Feedback.

**Rows verified:** 2–65 (64 rows with content). Rows 66+ in the Excel are empty.

---

## Summary

| Verdict | Count |
|--------|--------|
| **Response better** | **44** |
| **Previous better** | **11** |
| **Tie** | **9** |

**Conclusion:** For the verified set, **Response** is better more often than **Previous Response** (44 vs 11), with 9 ties.

---

## When Previous Response is better (11 rows)

- **Row 8:** Document generator status — Previous correctly says “under construction.”
- **Row 9:** Construction scheduling risks — Previous has more scheduling-risk clauses (e.g. FAR 36.208, 36.515).
- **Row 11:** Commissioning pre-award — Previous addresses pre-award evaluation (FAR 36); Response is off-topic.
- **Row 14:** Why REA not change order — Previous gives clearer comparison (initiation, circumstances).
- **Row 17:** Safety incident, government needs from GC — Previous has EM 385-1-1, ENG Form 3394, OSHA.
- **Row 27:** Clauses before mobilization — Previous more construction-focused (labor, wage, PLA).
- **Row 45:** Protest filing deadline — Previous has agency/GAO/Court plus FAR 52.233-2.
- **Row 49:** Rephrase to include FAR/DFAR/EM 385 — Previous has EM385 25-8 utility; Response is generic FAR list.
- **Row 56:** What is PPI — Previous defines Past Performance Information (FAR 15.305).
- **Row 63:** FAR clause for differing site conditions — Previous includes full clause text (a)(b)(c)(d).
- **Row 64:** FAR clause professional services — Previous has 52.222-41 with explanation; Response is hedged.

---

## Per-row verdicts

Full list: **`results_negative_manual_verdicts.csv`**  
Columns: `Row`, `Query_snippet`, `Verdict` (Response | Previous | Tie), `Note`.

---

## Files

- **`results_negative_manual_verify.txt`** — Export of Query + Response + Previous Response (truncated) for each row.
- **`results_negative_manual_verdicts.csv`** — Manual verdict and short note per row.
- **`RESULTS_NEGATIVE_MANUAL_VERIFICATION.md`** — This summary.
