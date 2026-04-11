#!/usr/bin/env python3
"""Run test queries against the GovGig API or the orchestrator (direct).

API mode (--api): login, then POST each query to POST /api/v1/query.
Direct mode (default): use orchestrator in-process (no server needed).

Requires: project venv and deps. For API mode: server running, API_BASE_URL, API_TEST_EMAIL, API_TEST_PASSWORD (or --api-url, --email, --password).

Usage:
  python scripts/run_test_queries.py                    # smoke subset, direct
  python scripts/run_test_queries.py --api              # smoke subset, via API
  python scripts/run_test_queries.py --list FILE -o out.txt
  python scripts/run_test_queries.py --api --list scripts/queries.txt -o results/results.txt
  python scripts/run_test_queries.py --api --list scripts/queries.txt -o results/results.csv --csv

Output: with -o only, query/path/docs/confidence and first 300 chars of answer. With --csv, full responses in CSV.
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Quick smoke subset (see docs/TEST_QUERIES_REFERENCE.md)
SMOKE_QUERIES = [
    "What is FAR 36.204?",
    "Which FAR clause applies to differing site conditions?",
    "Show me DFARS 252.204-7012",
    "What are the standard project cost ranges for a government solicitation?",
    "What is the FAR clause that relates to professional services?",
    "How should I structure the serial letter?",
    "What is INDOPACOM?",
    "As a Project Manager, analyze the safety requirements during the pre-award phase of a federal construction project and recommend actions to ensure compliance with contract requirements.",
]

EXTENDED_QUERIES = SMOKE_QUERIES + [
    "What is a concurrent delay?",
    "What is the difference between a change order and an REA?",
    "What clauses are covered by the Christian Doctrine?",
    "What is the reasonable withholdings on an invoice?",
    "Per EM385, when is a hot work permitted?",
    "What are redline drawings?",
    "How often should the superintendent update drawings?",
    "What is a serial letter?",
    "What is SIOP?",
]


def _path_summary(agent_path):
    if not agent_path:
        return "unknown"
    if any("clause_lookup" in p for p in agent_path):
        return "clause_lookup"
    if any("Clarifier:" in p for p in agent_path):
        return "clarifier"
    if any("regulation_search" in p for p in agent_path):
        return "regulation_search"
    return "other"


def _reflection_triggered(result):
    """Check explicit flag first; fall back to agent_path heuristic for API results."""
    if isinstance(result, dict) and "reflection_triggered" in result:
        return bool(result["reflection_triggered"])
    agent_path = (
        result if isinstance(result, list) else (result or {}).get("agent_path", [])
    )
    if not agent_path:
        return False
    return any(
        "Reflection: Low" in p or "Self-healing: Added" in p or "QualityGate:" in p
        for p in agent_path
    )


# ── API client ───────────────────────────────────────────────────────────────


def _get_token(base_url: str, email: str, password: str) -> str:
    url = f"{base_url.rstrip('/')}/api/v1/auth/login"
    data = json.dumps({"email": email, "password": password}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = {}
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            pass
        raise RuntimeError(f"Login failed ({e.code}): {body.get('detail', e.reason)}")
    token = body.get("access_token")
    if not token:
        raise RuntimeError("Login response missing access_token")
    return token


def _post_query(base_url: str, token: str, query: str) -> dict:
    url = f"{base_url.rstrip('/')}/api/v1/query"
    data = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = {}
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            pass
        raise RuntimeError(f"Query failed ({e.code}): {body.get('detail', e.reason)}")


_CSV_FIELDS = [
    "index",
    "query",
    "response",
    "path",
    "doc_count",
    "confidence",
    "mode",
    "reflection_triggered",
    "low_confidence",
    "show_banner",
]


def _row_from_result(index: int, query: str, result: dict) -> dict:
    """Build a single result row (for CSV) from API/orchestrator result."""
    agent_path = result.get("agent_path", [])
    path = _path_summary(agent_path)
    docs = len(result.get("documents", []))
    conf = result.get("confidence", "")
    if conf != "unknown" and conf is not None:
        conf = str(conf)
    else:
        conf = ""
    refl = _reflection_triggered(result)
    answer = (result.get("response") or "").strip()
    mode = (result.get("mode") or "").strip() or ""
    low = result.get("low_confidence")
    qm = result.get("quality_metrics") or {}
    banner = qm.get("show_banner")
    return {
        "index": index,
        "query": query,
        "response": answer,
        "path": path,
        "doc_count": docs,
        "confidence": conf,
        "mode": mode,
        "reflection_triggered": "yes" if refl else "no",
        "low_confidence": "yes" if low else "no",
        "show_banner": "yes" if banner else "no",
    }


def run_queries_via_api(
    queries,
    base_url: str,
    email: str,
    password: str,
    verbose=True,
    out_path=None,
    csv_mode=False,
):
    token = _get_token(base_url, email, password)
    lines = []
    rows = []
    for i, q in enumerate(queries, 1):
        try:
            result = _post_query(base_url, token, q)
            agent_path = result.get("agent_path", [])
            path = _path_summary(agent_path)
            docs = len(result.get("documents", []))
            conf = result.get("confidence", "unknown")
            refl = _reflection_triggered(result)
            answer = (result.get("response") or "").strip()
            preview = (answer[:300] + "…") if len(answer) > 300 else answer
            ref_str = " reflection_triggered=yes" if refl else ""
            line = f"[{i}] path={path} docs={docs} confidence={conf}{ref_str}\n  Q: {q[:80]}{'…' if len(q) > 80 else ''}\n  A: {preview}"
            if verbose:
                print(line)
                print()
            lines.append(line)
            if csv_mode:
                rows.append(_row_from_result(i, q, result))
            # Rate limit: API often 10/min
            if len(queries) > 15:
                time.sleep(6.5 / 10.0)
        except Exception as e:
            line = f"[{i}] ERROR: {q[:80]}\n  {type(e).__name__}: {e}"
            if verbose:
                print(line)
                print()
            lines.append(line)
            if csv_mode:
                rows.append(
                    {
                        "index": i,
                        "query": q,
                        "response": f"ERROR: {type(e).__name__}: {e}",
                        "path": "error",
                        "doc_count": 0,
                        "confidence": "",
                        "reflection_triggered": "no",
                        "low_confidence": "no",
                        "show_banner": "no",
                    }
                )

    if out_path:
        if csv_mode and rows:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
                w.writeheader()
                w.writerows(rows)
            print(f"Wrote {len(rows)} results (full responses) to {out_path}")
        else:
            with open(out_path, "w") as f:
                f.write("\n\n".join(lines))
            print(f"Wrote {len(lines)} results to {out_path}")
    return lines


# ── Direct (orchestrator) ─────────────────────────────────────────────────────


async def _run_one(orchestrator, q, i, verbose):
    result = await orchestrator.run_async(q)
    path = _path_summary(result.get("agent_path", []))
    docs = len(result.get("documents", []))
    conf = result.get("confidence", "unknown")
    answer = (result.get("response") or "").strip()
    preview = (answer[:300] + "…") if len(answer) > 300 else answer
    refl = _reflection_triggered(result)
    ref_str = " reflection_triggered=yes" if refl else ""
    line = f"[{i}] path={path} docs={docs} confidence={conf}{ref_str}\n  Q: {q[:80]}{'…' if len(q) > 80 else ''}\n  A: {preview}"
    if verbose:
        print(line)
        print()
    return line, result


def run_queries_direct(queries, verbose=True, out_path=None, csv_mode=False):
    from src.agents.orchestrator import GovGigOrchestrator

    orchestrator = GovGigOrchestrator()
    lines = []
    rows = []

    async def _run_all():
        nonlocal lines, rows
        for i, q in enumerate(queries, 1):
            try:
                line, result = await _run_one(orchestrator, q, i, verbose)
                lines.append(line)
                if csv_mode:
                    rows.append(_row_from_result(i, q, result))
            except Exception as e:
                line = f"[{i}] ERROR: {q[:80]}\n  {type(e).__name__}: {e}"
                if verbose:
                    print(line)
                    print()
                lines.append(line)
                if csv_mode:
                    rows.append(
                        {
                            "index": i,
                            "query": q,
                            "response": f"ERROR: {type(e).__name__}: {e}",
                            "path": "error",
                            "doc_count": 0,
                            "confidence": "",
                            "reflection_triggered": "no",
                            "low_confidence": "no",
                            "show_banner": "no",
                        }
                    )

    asyncio.run(_run_all())

    if out_path:
        if csv_mode and rows:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
                w.writeheader()
                w.writerows(rows)
            print(f"Wrote {len(rows)} results (full responses) to {out_path}")
        else:
            with open(out_path, "w") as f:
                f.write("\n\n".join(lines))
            print(f"Wrote {len(lines)} results to {out_path}")
    return lines


def main():
    ap = argparse.ArgumentParser(
        description="Run test queries (API or direct orchestrator)"
    )
    ap.add_argument(
        "--api",
        action="store_true",
        help="Hit API (login + POST /api/v1/query) instead of in-process orchestrator",
    )
    ap.add_argument(
        "--api-url",
        metavar="URL",
        default=os.environ.get("API_BASE_URL", "http://localhost:8000"),
        help="API base URL (default: API_BASE_URL or http://localhost:8000)",
    )
    ap.add_argument(
        "--email",
        metavar="EMAIL",
        default=os.environ.get("API_TEST_EMAIL"),
        help="Login email (or API_TEST_EMAIL)",
    )
    ap.add_argument(
        "--password",
        metavar="PASS",
        default=os.environ.get("API_TEST_PASSWORD"),
        help="Login password (or API_TEST_PASSWORD)",
    )
    ap.add_argument("--all", action="store_true", help="Run extended query list")
    ap.add_argument(
        "--list", metavar="FILE", help="Run queries from file (one per line)"
    )
    ap.add_argument("-o", "--out", metavar="FILE", help="Write results to file")
    ap.add_argument(
        "--csv", action="store_true", help="Write full responses in CSV (use with -o)"
    )
    ap.add_argument(
        "-q", "--quiet", action="store_true", help="Only print summary to --out"
    )
    args = ap.parse_args()

    if args.list:
        with open(args.list) as f:
            queries = [line.strip() for line in f if line.strip()]
    elif args.all:
        queries = EXTENDED_QUERIES
    else:
        queries = SMOKE_QUERIES

    print(
        f"Running {len(queries)} queries"
        + (" via API" if args.api else " (direct orchestrator)")
        + "…"
    )

    if args.api:
        if not args.email or not args.password:
            print(
                "Error: --api requires --email and --password (or set API_TEST_EMAIL and API_TEST_PASSWORD)",
                file=sys.stderr,
            )
            sys.exit(1)
        run_queries_via_api(
            queries,
            base_url=args.api_url,
            email=args.email,
            password=args.password,
            verbose=not args.quiet,
            out_path=args.out,
            csv_mode=args.csv,
        )
    else:
        run_queries_direct(
            queries, verbose=not args.quiet, out_path=args.out, csv_mode=args.csv
        )


if __name__ == "__main__":
    main()
