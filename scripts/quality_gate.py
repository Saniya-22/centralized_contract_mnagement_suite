#!/usr/bin/env python3
"""End-to-end answer quality gate for GovGig Python backend.

Runs a fixed query set against the orchestrator and enforces pass/fail
thresholds suitable for deploy pipelines.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.orchestrator import GovGigOrchestrator

# Keep this in sync with GovGigOrchestrator._safe_fallback_message()
# which triggers when no retrieved docs are available.
FALLBACK_PHRASE = (
    "The retrieved regulatory excerpts do not directly address this specific question"
)
CITATION_RE = re.compile(r"\b(FAR|DFARS|EM\s*385)\b\s*\d", re.IGNORECASE)


@dataclass
class QueryCase:
    id: str
    query: str
    expected_regulation: str | None = None
    must_contain_any: list[str] | None = None
    require_citation: bool = True
    max_seconds: float | None = None


@dataclass
class CaseResult:
    id: str
    query: str
    passed: bool
    reasons: list[str]
    latency_seconds: float
    fallback: bool
    has_citation: bool
    confidence: float | None
    docs_count: int
    regulation_types: list[str]
    response_preview: str


def _load_cases(path: Path) -> list[QueryCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[QueryCase] = []
    for item in raw:
        cases.append(
            QueryCase(
                id=item["id"],
                query=item["query"],
                expected_regulation=item.get("expected_regulation"),
                must_contain_any=item.get("must_contain_any"),
                require_citation=bool(item.get("require_citation", True)),
                max_seconds=float(item["max_seconds"])
                if item.get("max_seconds")
                else None,
            )
        )
    return cases


def _contains_any(response: str, keywords: list[str] | None) -> bool:
    if not keywords:
        return True
    lower = response.lower()
    return any(keyword.lower() in lower for keyword in keywords)


async def _run_case(
    orchestrator: GovGigOrchestrator,
    case: QueryCase,
    user_id: str,
    run_id: str,
) -> CaseResult:
    t0 = time.perf_counter()
    result = await orchestrator.run_async(
        case.query,
        {
            "history": [],
            "cot": False,
            "person_id": user_id,
            "thread_id": f"quality_gate:{run_id}:{case.id}",
        },
    )
    elapsed = time.perf_counter() - t0

    response = (result.get("response") or "").strip()
    docs = result.get("documents") or []
    regulation_types = result.get("regulation_types") or []

    fallback = FALLBACK_PHRASE in response
    has_citation = bool(CITATION_RE.search(response))

    reasons: list[str] = []
    if not response:
        reasons.append("empty_response")
    if fallback:
        reasons.append("fallback_response")
    if case.require_citation and not has_citation:
        reasons.append("missing_citation_marker")
    if not _contains_any(response, case.must_contain_any):
        reasons.append("missing_expected_keywords")
    if case.expected_regulation and case.expected_regulation not in regulation_types:
        reasons.append("unexpected_regulation_type")
    if case.max_seconds is not None and elapsed > case.max_seconds:
        reasons.append("latency_exceeded")

    return CaseResult(
        id=case.id,
        query=case.query,
        passed=not reasons,
        reasons=reasons,
        latency_seconds=elapsed,
        fallback=fallback,
        has_citation=has_citation,
        confidence=result.get("confidence"),
        docs_count=len(docs),
        regulation_types=regulation_types,
        response_preview=response[:400].replace("\n", " "),
    )


def _print_case_result(case_result: CaseResult) -> None:
    status = "PASS" if case_result.passed else "FAIL"
    reason_text = ",".join(case_result.reasons) if case_result.reasons else "-"
    print(
        f"[{status}] {case_result.id} | "
        f"lat={case_result.latency_seconds:.2f}s "
        f"docs={case_result.docs_count} "
        f"fallback={case_result.fallback} "
        f"citation={case_result.has_citation} "
        f"reasons={reason_text}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GovGig deploy quality gate")
    parser.add_argument(
        "--queries-file",
        default="scripts/quality_queries.json",
        help="Path to query cases JSON file",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.85,
        help="Minimum required pass ratio across all queries",
    )
    parser.add_argument(
        "--max-fallback-rate",
        type=float,
        default=0.20,
        help="Maximum allowed fallback response ratio",
    )
    parser.add_argument(
        "--min-citation-rate",
        type=float,
        default=0.85,
        help="Minimum required citation marker ratio",
    )
    parser.add_argument(
        "--max-avg-latency",
        type=float,
        default=12.0,
        help="Maximum allowed average query latency in seconds",
    )
    parser.add_argument(
        "--user-id",
        default="quality_gate_user",
        help="User id used for run context",
    )
    parser.add_argument(
        "--report-out",
        default="",
        help="Optional path to write JSON report",
    )
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    cases = _load_cases(Path(args.queries_file))
    if not cases:
        print("No query cases found.")
        return 2

    run_id = str(int(time.time()))
    orchestrator = GovGigOrchestrator()
    results: list[CaseResult] = []

    print(f"Running quality gate with {len(cases)} query cases")
    for case in cases:
        case_result = await _run_case(orchestrator, case, args.user_id, run_id)
        results.append(case_result)
        _print_case_result(case_result)

    total = len(results)
    passed = sum(1 for item in results if item.passed)
    fallback_count = sum(1 for item in results if item.fallback)
    citation_count = sum(1 for item in results if item.has_citation)
    avg_latency = sum(item.latency_seconds for item in results) / total

    pass_rate = passed / total
    fallback_rate = fallback_count / total
    citation_rate = citation_count / total

    gate_failures: list[str] = []
    if pass_rate < args.min_pass_rate:
        gate_failures.append(
            f"pass_rate {pass_rate:.2%} < min_pass_rate {args.min_pass_rate:.2%}"
        )
    if fallback_rate > args.max_fallback_rate:
        gate_failures.append(
            f"fallback_rate {fallback_rate:.2%} > max_fallback_rate {args.max_fallback_rate:.2%}"
        )
    if citation_rate < args.min_citation_rate:
        gate_failures.append(
            f"citation_rate {citation_rate:.2%} < min_citation_rate {args.min_citation_rate:.2%}"
        )
    if avg_latency > args.max_avg_latency:
        gate_failures.append(
            f"avg_latency {avg_latency:.2f}s > max_avg_latency {args.max_avg_latency:.2f}s"
        )

    print("-" * 96)
    print(
        "SUMMARY | "
        f"pass_rate={pass_rate:.2%} "
        f"fallback_rate={fallback_rate:.2%} "
        f"citation_rate={citation_rate:.2%} "
        f"avg_latency={avg_latency:.2f}s"
    )
    if gate_failures:
        print("QUALITY_GATE: FAIL")
        for failure in gate_failures:
            print(f" - {failure}")
        exit_code = 1
    else:
        print("QUALITY_GATE: PASS")
        exit_code = 0

    if args.report_out:
        report = {
            "summary": {
                "total": total,
                "passed": passed,
                "pass_rate": pass_rate,
                "fallback_rate": fallback_rate,
                "citation_rate": citation_rate,
                "avg_latency": avg_latency,
                "gate_failures": gate_failures,
                "exit_code": exit_code,
            },
            "results": [asdict(item) for item in results],
        }
        out_path = Path(args.report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written: {out_path}")

    return exit_code


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    exit_code = asyncio.run(_main_async(args))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
