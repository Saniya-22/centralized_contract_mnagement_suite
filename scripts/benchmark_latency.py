"""E2E Latency Benchmark Suite — Phase Breakdown.

Shows per-phase timing for every query:
  [Embedding] text-embedding API took Xs
  [Reranker]  API call took Xs
  [Synthesizer] LLM generation took Xs
  Total wall-clock latency

SLA gates:  PASS < 4s  |  WARN < 8s  |  FAIL >= 8s
"""

import sys
import os
import re
import time
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.orchestrator import GovGigOrchestrator

QUERIES = [
    {
        "name": "1. Clause Lookup (Path A — no embed, no reranker)",
        "query": "Which FAR clause applies to differing site conditions?",
    },
    {
        "name": "2. General Safety Search (Path B — full pipeline)",
        "query": "What are the Buy American requirements for steel?",
    },
    {
        "name": "3. DFARS Cybersecurity Query (Path B)",
        "query": "Show me DFARS cybersecurity requirements",
    },
]


# ── Log capture ────────────────────────────────────────────────────────────────
class PhaseCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())

    def reset(self):
        self.records.clear()

    def first(self, keyword):
        for msg in self.records:
            if keyword in msg:
                return msg
        return None

    def parse_seconds(self, keyword):
        msg = self.first(keyword)
        if not msg:
            return None
        m = re.search(r"(\d+\.\d+)s", msg)
        return float(m.group(1)) if m else None


def _fmt(val):
    return f"{val:.2f}s" if val is not None else "  N/A "


def run_benchmarks():
    print("=" * 68)
    print("  GovGig AI — Latency Benchmark (phase breakdown)")
    print("=" * 68)

    handler = PhaseCapture()
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    t0 = time.perf_counter()
    orchestrator = GovGigOrchestrator()
    print(f"Orchestrator init: {time.perf_counter()-t0:.2f}s\n")

    rows = []
    for test in QUERIES:
        handler.reset()
        t_start = time.perf_counter()
        result = orchestrator.run_sync(test["query"])
        total = time.perf_counter() - t_start

        embed_s = handler.parse_seconds("[Embedding] text-embedding API took")
        rerank_s = handler.parse_seconds("[Reranker] API call took")
        synth_s = handler.parse_seconds("[Synthesizer] LLM generation took")

        docs = len(result.get("documents", []))
        agent_path = result.get("agent_path", [])
        path = (
            "clause_lookup"
            if any("clause_lookup" in p for p in agent_path)
            else "regulation_search"
        )
        reranker_fired = handler.first("[Reranker] API call took") is not None

        sla = "✅ PASS" if total < 4 else ("⚠️  WARN" if total < 8 else "🔴 FAIL")

        print(f"{'─' * 68}")
        print(f"  {test['name']}")
        print(f"  Query: '{test['query'][:70]}'")
        print("  ┌─────────────────────────────────────────┐")
        print(f"  │  Embedding (OpenAI embed-3-small):  {_fmt(embed_s):>7}  │")
        print("  │  pgvector Dense + FTS (parallel):    ≈ db   │")
        print(f"  │  Reranker (gpt-4o-mini, top-5):     {_fmt(rerank_s):>7}  │")
        print(f"  │  Synthesizer (gpt-4o-mini):          {_fmt(synth_s):>7}  │")
        print("  │  ──────────────────────────────────────── │")
        print(f"  │  Total wall-clock:                  {total:>6.2f}s  │  {sla}")
        print("  └─────────────────────────────────────────┘")
        print(
            f"  Path: {path} | Docs: {docs} | Reranker: {'fired' if reranker_fired else 'skipped'}"
        )

        rows.append(
            {
                "name": test["name"].split(".")[1].strip()[:32],
                "embed": embed_s,
                "rerank": rerank_s,
                "synth": synth_s,
                "total": total,
                "docs": docs,
                "sla": sla,
            }
        )

    # Summary table
    print(f"\n{'=' * 68}")
    print(f"  {'Test':<34} {'Embed':>6} {'Rerank':>7} {'Synth':>6} {'Total':>7}  SLA")
    print(f"  {'─' * 34} {'─' * 6} {'─' * 7} {'─' * 6} {'─' * 7}  ───")
    for r in rows:
        print(
            f"  {r['name']:<34} {_fmt(r['embed']):>6} {_fmt(r['rerank']):>7} "
            f"{_fmt(r['synth']):>6} {r['total']:>6.2f}s  {r['sla']}"
        )
    print("=" * 68)

    logging.getLogger().removeHandler(handler)


if __name__ == "__main__":
    run_benchmarks()
