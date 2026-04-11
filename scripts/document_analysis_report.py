#!/usr/bin/env python3
"""Analyze source PDFs for structure/chunking risk before indexing."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any

import fitz

import sys

sys.path.insert(0, "ingest_python")
import pipeline  # noqa: E402


def pct(vals: list[int], p: int) -> int:
    if not vals:
        return 0
    idx = int(round((p / 100) * (len(vals) - 1)))
    return vals[max(0, min(idx, len(vals) - 1))]


def analyze_pdf(pdf_path: Path) -> dict[str, Any]:
    filename = pdf_path.stem
    meta = pipeline.extract_metadata(filename, str(pdf_path))
    source = meta.get("source", "UNKNOWN")

    doc = fitz.open(str(pdf_path))
    full_text_parts = []
    for i, page in enumerate(doc):
        content = pipeline.extract_page_content(page, i + 1)
        full_text_parts.append(content.get("text", ""))
    doc.close()
    full_text = "\n\n".join(full_text_parts)

    sections = pipeline.extract_structured_sections(full_text, source)
    chunks = pipeline.create_section_aware_chunks(sections, source)
    fallback = False
    if not chunks:
        fallback = True
        chunks = [{"text": t} for t in pipeline.create_chunks(full_text)]

    token_counts = sorted(
        [pipeline.count_tokens(c.get("text", "")) for c in chunks if c.get("text")]
    )
    tiny = sum(1 for t in token_counts if t < 80)
    anchor_tiny = sum(
        1
        for c in chunks
        if c.get("text")
        and pipeline.count_tokens(c["text"]) < 80
        and pipeline._is_anchor_chunk(c["text"])
    )
    return {
        "file": pdf_path.name,
        "source": source,
        "pages": len(full_text_parts),
        "chars": len(full_text),
        "sections": len(sections),
        "chunks": len(token_counts),
        "tiny_pct": round((tiny / len(token_counts)) * 100, 2) if token_counts else 0.0,
        "anchor_tiny_pct": round((anchor_tiny / len(token_counts)) * 100, 2)
        if token_counts
        else 0.0,
        "avg_tok": round(mean(token_counts), 1) if token_counts else 0.0,
        "p50_tok": pct(token_counts, 50),
        "p90_tok": pct(token_counts, 90),
        "max_tok": max(token_counts) if token_counts else 0,
        "fallback": fallback,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Document analysis report for chunking readiness"
    )
    p.add_argument(
        "--spec-dir",
        default="specifications",
        help="Directory containing regulation PDFs",
    )
    p.add_argument("--csv-out", default="", help="Optional CSV output path")
    args = p.parse_args()

    spec_dir = Path(args.spec_dir)
    pdfs = sorted(spec_dir.rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found under {spec_dir}")

    rows = [analyze_pdf(p) for p in pdfs]

    print(
        "file|source|pages|sections|chunks|tiny%|anchor_tiny%|avg_tok|p50|p90|max|fallback"
    )
    for r in rows:
        print(
            f"{r['file']}|{r['source']}|{r['pages']}|{r['sections']}|{r['chunks']}|"
            f"{r['tiny_pct']}|{r['anchor_tiny_pct']}|{r['avg_tok']}|{r['p50_tok']}|"
            f"{r['p90_tok']}|{r['max_tok']}|{r['fallback']}"
        )

    total_chunks = sum(r["chunks"] for r in rows)
    weighted_tiny = (
        sum(r["chunks"] * r["tiny_pct"] for r in rows) / total_chunks
        if total_chunks
        else 0.0
    )
    fallback_docs = [r["file"] for r in rows if r["fallback"]]
    high_tiny_docs = [r["file"] for r in rows if r["tiny_pct"] > 20]
    print("\nSUMMARY")
    print(
        f"docs={len(rows)} total_chunks={total_chunks} weighted_tiny_pct={weighted_tiny:.2f}"
    )
    print(f"fallback_docs={fallback_docs}")
    print(f"high_tiny_docs(>20%)={high_tiny_docs}")

    if args.csv_out:
        fields = [
            "file",
            "source",
            "pages",
            "chars",
            "sections",
            "chunks",
            "tiny_pct",
            "anchor_tiny_pct",
            "avg_tok",
            "p50_tok",
            "p90_tok",
            "max_tok",
            "fallback",
        ]
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"csv_written={out_path}")


if __name__ == "__main__":
    main()
