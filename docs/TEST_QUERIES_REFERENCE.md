# Test Queries Reference

Use this list for manual testing or to drive automated smoke tests. **Expected** indicates how the current system should behave: **clause_lookup**, **regulation_search**, **out_of_scope**, or **document_request**. For document requests (write/draft letter, REA, RFI), the system routes to the **letter-drafting agent** and returns a **full draft** (grounded in retrieved regulations) with a disclaimer; previously this was guidance-only.

---

## Out of scope (expect refusal + suggested examples)

| # | Query | Note |
|---|--------|------|
| 1 | What bases are within Navy Region NW? | Not FAR/DFARS/EM385 |
| 2 | Who is the founder of govgig? | Not regulatory |
| 3 | What did the GAO state on the status of the nation's four major naval shipyards? | GAO/report, not in KB |
| 4 | Provide me with detailed history of Puget Sound naval shipyard | History, not regs |
| 5 | Is the document generator working yet? | System/feature question |
| 6 | Why is the document generator still under construction? | System/feature question |
| 7 | What is INDOPACOM? | DoD term, not in regs KB |
| 8 | What does OCONUS mean? | Term, may get out_of_scope |
| 9 | How do I access the process guidance agent? | System/feature |
| 10 | Why can't I access tools? | System/feature |
| 11 | Can you recommend me a Government contracts attorney? | Out of scope |
| 12 | Export to a word document | Feature not supported |
| 13 | What is a serial letter? | May get regulation_search if “serial” + “letter” trigger; else definition |

---

## Document / letter request (full draft via letter-drafting agent)

When the user asks to write/draft a letter, REA, RFI, or similar document, the system routes to the **letter-drafting agent** and returns a **full draft** (header, body, citations from retrieved regs, disclaimer). Expect `agent_path` to include "LetterDrafter".

| # | Query | Expected |
|---|--------|----------|
| 14 | Write the KO a serial about the shutdown, notifying them of potential impacts, include excusable delay clauses, and DFARS 252.236-7000 | regulation_search + full letter draft (LetterDrafter) |
| 15 | We are building a barracks project at camp Pendleton and have uncovered unexploded ordnance. What do I do and write me a serial letter notifying the KO. Include FAR excusable delay, differing site conditions | regulation_search (UXO/EM385) + what to do + full letter draft |
| 16 | Write a serial letter notifying the ko of the wildfire impact | regulation_search + full letter draft |
| 17 | Draft a letter to the KO regarding 52.242-5 stating the prime contractor is not paying | regulation_search (52.242-5) + full letter draft |
| 18 | Draft me an REA for compensable delay | regulation_search + full REA draft |
| 19 | Write me an RFI, in standard NAVFAC form, for this issue | regulation_search + full RFI draft |
| 20 | Generate a government property tracking form | Out of scope or full draft (if document_request) |
| 21 | Generate a QC dashboard for project quality performance | Out of scope or guidance |
| 22 | Night-shift lighting levels below 50 foot-candles. Write the dual safety/quality stop-work order | regulation_search (EM385) + full draft (LetterDrafter) |
| 23 | Generate a QC checklist for conduit installation | Full draft or guidance |
| 24 | Generate a checklist for verifying masonry reinforcement placement | Full draft or guidance |
| 25 | Generate a QC inspection checklist for demolition activities | Full draft or guidance |
| 26 | Draft an Initial Phase inspection report verifying the first installation of Division 23 – HVAC components | Full draft or guidance |
| 27 | How should I structure the serial letter? | Out of scope (no reg context) or brief guidance |

---

## Clause lookup (direct clause fetch)

| # | Query | Expected |
|---|--------|----------|
| 28 | Which FAR clause applies to differing site conditions? | regulation_search or clause_lookup → 52.236-2 |
| 29 | What is the FAR clause that relates to professional services? | regulation_search (52.237-3, 52.222-41, etc.) |
| 30 | Show me the standard project cost ranges for a government solicitation | regulation_search → DFARS 236.204 |
| 31 | What is FAR 36.204? | clause_lookup |
| 32 | What is DFARS? | regulation_search or short definition |
| 33 | Show me DFARS 252.204-7012 | clause_lookup |

---

## Regulation search (hybrid retrieval + synthesis)

| # | Query | Expected |
|---|--------|----------|
| 34 | Core Federal Acquisition Regulations / FAR clauses / federal construction document list (long structured input) | regulation_search |
| 35 | Review the FAR or DFARS clauses and identify risks related to construction scheduling | regulation_search |
| 36 | What is a concurrent delay? | regulation_search |
| 37 | As a PM, analyze FAR clause interpretation during pre-award… | regulation_search |
| 38 | As a PM, analyze contract compliance during pre-award… | regulation_search |
| 39 | As a PM, analyze commissioning requirements during pre-award… | regulation_search |
| 40 | As a PM, analyze punchlist requirements during pre-award… | regulation_search |
| 41 | As a PM, analyze warranties requirements during pre-award… | regulation_search |
| 42 | As a PM, analyze inspections requirements during pre-award… | regulation_search |
| 43 | As a PM, analyze differing site conditions requirements during pre-award… | regulation_search |
| 44 | As a PM, analyze delays requirements during pre-award… | regulation_search |
| 45 | As a PM, analyze pricing adjustments during pre-award… | regulation_search |
| 46 | As a PM, analyze change orders during pre-award… | regulation_search |
| 47 | As a PM, analyze safety requirements during pre-award… | regulation_search |
| 48 | Add DFARS clause 252.236-7000 into the letter | regulation_search (clause content) + guidance |
| 49 | Write a delay letter to NAVFAC regarding outage disapproval, include default FAR clause | regulation_search + guidance |
| 50 | What clauses are covered by the Christian Doctrine? | regulation_search |
| 51 | Why would a general contractor submit an REA and not a change order? | regulation_search |
| 52 | Document use and possession per FAR for benefit occupancy | regulation_search |
| 53 | What is the difference between a change order and an REA? | regulation_search |
| 54 | If there is a safety incident, what does the government need from the GC? | regulation_search |
| 55 | How do I calculate my daily rate? | regulation_search (FAR cost principles) |
| 56 | What is allowable for Field Office Overhead? | regulation_search |
| 57 | How often can a general contractor submit billing? | regulation_search |
| 58 | Can the government designate anyone as a contracting officer representative? | regulation_search |
| 59 | What dollar value do I need to certify my REA? | regulation_search |
| 60 | If the government's COR is not qualified, what can the GC do? | regulation_search |
| 61 | Can the contracting officer adjust the schedule? | regulation_search |
| 62 | What counts as justification for the CO to change the schedule? | regulation_search |
| 63 | My prime isn't paying me. What can I do? | regulation_search (e.g. 52.242-5) |
| 64 | When can the contracting officer not change the schedule? | regulation_search |
| 65 | What should I verify regarding bonding requirements after award? | regulation_search |
| 66 | How often do I need to send in daily reports? | contract/CO guidance (no single reg; check contract and CO) |
| 67 | During excavation I encountered a duct bank not on government drawings. Can I get reimbursed? | regulation_search (differing site conditions) |
| 68 | If the CO and I do not agree on the legitimacy of a change order, what are the next steps? | regulation_search |
| 69 | How do you appeal an unsatisfactory CPARS? | regulation_search |
| 70 | What is the difference between termination for default and termination for convenience and impacts to the contractor? | regulation_search |
| 71 | What is the reasonable withholdings on an invoice? | regulation_search |
| 72 | What documents do I need for a novation agreement? | regulation_search |
| 73 | How do I confirm the contract scope aligns with the solicitation and award documents? | regulation_search |
| 74 | What is the difference between product substitution and product variance? | regulation_search |
| 75 | I have received an offer to buy my company. What impact on my government contracts? | regulation_search |
| 76 | What contract clauses should I review before project mobilization? | regulation_search (mobilization boost) |
| 77 | What is the difference between limited rights, government purpose rights, and unlimited rights? | regulation_search |
| 78 | What documents must I review immediately after federal construction contract award? | regulation_search |
| 79 | If I have concurrent compensable and non-compensable days of delay, can I be paid for the compensable days? | regulation_search |
| 80 | What FAR clauses can be the basis of an REA? | regulation_search |
| 81 | What data do I need to support an REA? | regulation_search |
| 82 | I think the Government has messed up the small business set aside. What should I do? | regulation_search |
| 83 | We were digging a footing and ran into an unexpected gas line not on the drawings. Now what? | regulation_search (differing site conditions) |
| 84 | Rephrase answer to include all applicable FAR/DFAR/and EM 385 requirements | Meta; depends on prior turn |
| 85 | What are my rights as a contractor during a debrief? | regulation_search |
| 86 | Identify the quality-related deliverables required in Division 00 before construction begins | regulation_search |
| 87 | What are the roles and responsibilities of a federal construction quality control manager? | regulation_search |
| 88 | What do we do if a vendor cannot match the specifications? | regulation_search |
| 89 | As a Superintendent, analyze inspections during claims & disputes phase… | regulation_search |
| 90 | As a Superintendent, analyze pricing adjustments during claims & disputes phase… | regulation_search |
| 91 | I was just issued a job as a PM, post award, for a federal DBB construction job through NAVFAC. Provide the first 10 steps | regulation_search |
| 92 | How do I upload submittal documents? | Out of scope (product feature) |
| 93 | Per EM385, when is a hot work permitted? | regulation_search (EM385) |
| 94 | I want to upload the specifications / I want to load the submittal registry | Out of scope |
| 95 | Is there a requirement to advise the government if a change has been made to a QC plan? | regulation_search |
| 96 | Are there QC program requirements in the FAR, DFAR, or EM385? | regulation_search |
| 97 | What is a PPI? | regulation_search or out_of_scope |
| 98 | As a PM, analyze submittals requirements during pre-award… | regulation_search |
| 99 | As a PM, analyze RFIs requirements during pre-award… | regulation_search |
| 100 | Where do I find my insurance information? | Out of scope |
| 101 | We had a fire overnight at the site, what should I do? | regulation_search (safety/incident) |
| 102 | As an Estimator, analyze RFIs during pre-award… | regulation_search |
| 103 | As a PM, analyze safety requirements during post-award… | regulation_search |
| 104 | We have discovered an oil tank that is not on the drawings, what do I do next? | regulation_search (differing site conditions) |
| 105 | What does the project manager do on a federal construction project? | regulation_search |
| 106 | Add additional information that the time and cost impacts are unknown at this time | Meta / follow-up |

---

## Quick smoke subset (run these first)

Use for automated or manual smoke test after changes:

1. What is FAR 36.204?  
2. Which FAR clause applies to differing site conditions?  
3. Show me DFARS 252.204-7012  
4. What are the standard project cost ranges for a government solicitation?  
5. What is the FAR clause that relates to professional services?  
6. How should I structure the serial letter?  
7. What is INDOPACOM?  
8. As a PM, analyze safety requirements during pre-award for federal construction and recommend actions.
