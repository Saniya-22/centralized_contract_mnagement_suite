#!/usr/bin/env python3

import json
import asyncio
import aiohttp
import re
import hashlib
import logging
import tiktoken
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

from docling.document_converter import DocumentConverter

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# SQLAlchemy Core & Async IO
from sqlalchemy import MetaData, Table, Column, String, select, bindparam
from sqlalchemy.types import UserDefinedType
from sqlalchemy.dialects.postgresql import insert as pg_insert, JSONB
from sqlalchemy.ext.asyncio import create_async_engine, AsyncConnection, AsyncEngine
from tenacity import retry, stop_after_attempt, wait_exponential

from parsing.classifier import classify_line
from config import (
    NAMESPACE,
    EMBEDDING_MODEL,
    EMBEDDING_ENDPOINT,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    TARGET_CHUNK_TOKENS,
    CHUNK_OVERLAP,
    CLAUSE_SPLIT_TRIGGER_TOKENS,
    KEEP_STANDALONE_ANCHOR_CHUNKS,
    KEEP_INTACT_TOKEN_LIMIT,
    SUBCLAUSE_SPLIT_TOKEN_LIMIT,
    PARAGRAPH_SPLIT_TOKEN_LIMIT,
    MAX_CONCURRENT_PDFS,
    EMBED_RATE_DELAY,
    BATCH_SIZE,
    INCLUDE_FILES_RAW,
    OPENAI_API_KEY,
    DATABASE_URL,
    PG_DENSE_TABLE,
    PG_SPARSE_TABLE,
    PG_SSLMODE,
    USE_STEMMING,
    USE_STOPWORDS,
    SPECIFICATIONS_DIR,
)


class PGVector(UserDefinedType):
    """Custom type for PostgreSQL pgvector extension."""

    def get_col_spec(self, **kw):
        return "vector"

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                # Convert list [1, 2, 3] to Postgres vector string "[1, 2, 3]"
                return "[" + ",".join(map(str, value)) + "]"
            return value

        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
            if isinstance(value, str):
                # Convert "[1, 2, 3]" back to list
                return [float(x) for x in value.strip("[]").split(",")]
            return value

        return process


# ---------------------------------------------------------
# Text normalization (PDF extraction hardening)
# ---------------------------------------------------------

_LINESTART_MARKER_RE = re.compile(
    r"^\s*(?:"
    r"\([a-z]\)\s|"  # (a)
    r"\(\d+\)\s|"  # (1)
    r"\([ivx]+\)\s|"  # (i)
    r"\d{2,3}\.\d{3}(?:-\d+)?\s|"  # FAR/DFARS 52.212-4
    r"\d+-\d+\.\s|"  # EM385 1-1.
    r"[A-Z]-\d+\s|"  # Appendix A-1
    r"Rule\s+\d+\s"  # Appendix Rule 1
    r")",
    re.IGNORECASE,
)


def normalize_legal_text(text: str, source: str) -> str:
    """Conservatively normalize PDF-extracted legal text for chunking.

    Goals:
    - Reduce extraction artifacts that create oversized "sentences"
    - Preserve legal structure markers at line-start
    """
    if not text:
        return ""

    src = (source or "").upper()

    # Normalize common unicode whitespace/dashes that break regexes.
    t = text.replace("\u00a0", " ")  # NBSP
    t = (
        t.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    )  # – — minus

    # Normalize newlines.
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # Fix hyphenation across line breaks: "manufac-\ntured" -> "manufactured"
    t = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", t)

    # Collapse runs of spaces/tabs (but keep newlines for structure).
    t = re.sub(r"[ \t\f\v]+", " ", t)

    # Join "soft wraps" (single newline within a paragraph) into spaces, but preserve:
    # - blank lines (paragraph boundaries)
    # - explicit line-start markers like (a), (1), 52.xxx-x, 1-1., A-1, Rule 1
    lines = t.split("\n")
    out: List[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i].rstrip()
        if cur.strip() == "":
            out.append("")
            i += 1
            continue

        # Build a paragraph by consuming subsequent non-blank lines.
        para_parts = [cur.strip()]
        i += 1
        while i < len(lines) and lines[i].strip() != "":
            nxt_raw = lines[i].rstrip()
            nxt = nxt_raw.strip()

            # Preserve structure markers as their own lines.
            if _LINESTART_MARKER_RE.match(nxt):
                break

            prev = para_parts[-1]
            # Heuristic: join line if previous doesn't end sentence-ish and next looks like continuation.
            prev_end = prev[-1:] if prev else ""
            continuation = bool(re.match(r"^(?:[a-z0-9,;:\)\]\}])", nxt))
            not_sentence_end = prev_end not in ".!?:"

            if not_sentence_end and continuation:
                para_parts[-1] = f"{prev} {nxt}"
            else:
                para_parts.append(nxt)
            i += 1

        out.append("\n".join(para_parts) if src == "EM385" else " ".join(para_parts))

    # Collapse multiple blank lines to max two.
    t2 = "\n".join(out)
    t2 = re.sub(r"\n{3,}", "\n\n", t2)
    return t2.strip()


def _norm_for_dedupe(text: str) -> str:
    """Normalization for exact-dedupe hashing (whitespace-insensitive)."""
    if not text:
        return ""
    t = (
        text.replace("\u00a0", " ")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
    )
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # Make whitespace differences irrelevant while preserving visible characters.
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def dedupe_chunks_exact(
    chunks: List[Dict[str, Any]],
    *,
    scope: str = "section",
) -> List[Dict[str, Any]]:
    """Drop exact-duplicate chunk texts to avoid redundant embeddings/storage.

    scope:
      - "section": dedupe within (filename, section_number)
      - "document": dedupe within a document regardless of section
    """
    if not chunks:
        return []
    seen_by_key: Dict[Tuple[Optional[str], Optional[str]], set[str]] = {}
    out: List[Dict[str, Any]] = []
    for c in chunks:
        txt = c.get("text") or ""
        if not txt.strip():
            continue
        norm = _norm_for_dedupe(txt)
        h = hashlib.md5(norm.encode("utf-8")).hexdigest()
        if scope == "document":
            key = (None, None)
        else:
            key = (
                str(c.get("section_number") or ""),
                str(c.get("section_title") or ""),
            )
        bucket = seen_by_key.setdefault(key, set())
        if h in bucket:
            continue
        bucket.add(h)
        out.append(c)
    return out


# ---------------------------------------------------------
# SQL Table Definitions (SQLAlchemy Core)
# ---------------------------------------------------------

metadata_obj = MetaData()

# Define tables using SQLAlchemy Core for query building
dense_table = Table(
    PG_DENSE_TABLE,
    metadata_obj,
    Column("id", String, primary_key=True),
    Column("namespace", String),
    Column("text", String),
    Column("metadata", JSONB),
    Column("embedding", PGVector),
)

sparse_table = Table(
    PG_SPARSE_TABLE,
    metadata_obj,
    Column("id", String, primary_key=True),
    Column("namespace", String),
    Column("text", String),
    Column("metadata", JSONB),
    Column("embedding", JSONB),
)

# ---------------------------------------------------------
# SQL Statement Definitions (SQLAlchemy Core)
# ---------------------------------------------------------

# Build Dense Insert Statement (Idempotent Upsert)
stmt_insert_dense = pg_insert(dense_table).values(
    {
        "id": bindparam("id"),
        "namespace": bindparam("namespace"),
        "text": bindparam("text"),
        "metadata": bindparam("metadata"),
        "embedding": bindparam("embedding"),
    }
)
stmt_upsert_dense = stmt_insert_dense.on_conflict_do_update(
    index_elements=["id"],
    set_={
        "text": stmt_insert_dense.excluded.text,
        "metadata": stmt_insert_dense.excluded.metadata,
        "embedding": stmt_insert_dense.excluded.embedding,
    },
)

# Build Sparse Insert Statement (Idempotent Upsert)
stmt_insert_sparse = pg_insert(sparse_table).values(
    {
        "id": bindparam("id"),
        "namespace": bindparam("namespace"),
        "text": bindparam("text"),
        "metadata": bindparam("metadata"),
        "embedding": bindparam("embedding"),
    }
)
stmt_upsert_sparse = stmt_insert_sparse.on_conflict_do_update(
    index_elements=["id"],
    set_={
        "text": stmt_insert_sparse.excluded.text,
        "metadata": stmt_insert_sparse.excluded.metadata,
        "embedding": stmt_insert_sparse.excluded.embedding,
    },
)

# Build Hash Check Statement (Idempotency)
# Using strict form to avoid text = jsonb mismatch
stmt_hash_check = (
    select(dense_table.c.id)
    .where(
        dense_table.c.metadata["file_hash"].astext == bindparam("file_hash"),
        dense_table.c.namespace.like(bindparam("namespace_prefix")),
    )
    .limit(1)
)

# ---------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# NLTK Initialization
# ---------------------------------------------------------

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Tokenizer for OpenAI Embeddings
TOKENIZER = tiktoken.get_encoding("cl100k_base")
MAX_TOKEN_CAP = MAX_CHUNK_TOKENS

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------


def murmurhash3_32(data: str) -> int:
    """Calculates the 32-bit MurmurHash3 value of a string."""
    data_bytes = data.encode("utf-8")
    length = len(data_bytes)
    c1 = 0xCC9E2D51
    c2 = 0x1B873593
    h1 = 0
    nblocks = length // 4

    for i in range(nblocks):
        k1 = int.from_bytes(data_bytes[i * 4 : (i + 1) * 4], "little")
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & 0xFFFFFFFF
        h1 = (h1 * 5 + 0xE6546B64) & 0xFFFFFFFF

    h1 ^= length
    h1 ^= h1 >> 16
    h1 = (h1 * 0x85EBCA6B) & 0xFFFFFFFF
    h1 ^= h1 >> 13
    h1 = (h1 * 0xC2B2AE35) & 0xFFFFFFFF
    h1 ^= h1 >> 16

    if h1 & 0x80000000:
        h1 = -((~h1 + 1) & 0xFFFFFFFF)

    return h1


def file_hash(path: Union[str, Path]) -> str:
    """Computes the SHA-256 hash of a file for idempotency checks."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def encode_bm25(text: str) -> Dict[str, List[int]]:
    """Encodes text into a sparse BM25-compatible vector format."""
    if not text:
        return {"indices": [], "values": []}

    tokens = nltk.word_tokenize(text.lower())

    if USE_STOPWORDS:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    if USE_STEMMING:
        tokens = [stemmer.stem(t) for t in tokens]

    counts: Dict[str, int] = {}
    for term in tokens:
        counts[term] = counts.get(term, 0) + 1

    indices: List[int] = []
    values: List[int] = []
    for term, count in counts.items():
        indices.append(murmurhash3_32(term))
        values.append(count)

    return {"indices": indices, "values": values}


# ---------------------------------------------------------
# PDF Parsing (Docling)
# ---------------------------------------------------------


def _safe_model_dump(obj: Any) -> Any:
    """Best-effort conversion of pydantic-ish objects to JSON-serializable dicts."""
    if obj is None:
        return None
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return obj


def _table_structured_from_docling(table_item: Any) -> Dict[str, Any]:
    """Convert Docling TableItem -> {headers: [...], rows: [...]} + raw fallback."""
    data = getattr(table_item, "data", None)
    raw = _safe_model_dump(data) if data is not None else _safe_model_dump(table_item)
    table_cells = []
    num_rows = 0
    num_cols = 0
    if data is not None:
        table_cells = getattr(data, "table_cells", []) or []
        num_rows = int(getattr(data, "num_rows", 0) or 0)
        num_cols = int(getattr(data, "num_cols", 0) or 0)

    # Build a dense grid when possible.
    headers: List[str] = []
    rows: List[List[str]] = []
    if table_cells and num_rows > 0 and num_cols > 0:
        grid: List[List[str]] = [["" for _ in range(num_cols)] for _ in range(num_rows)]
        header_mask: List[List[bool]] = [
            [False for _ in range(num_cols)] for _ in range(num_rows)
        ]

        for cell in table_cells:
            try:
                r0 = int(getattr(cell, "start_row_offset_idx"))
                r1 = int(getattr(cell, "end_row_offset_idx"))
                c0 = int(getattr(cell, "start_col_offset_idx"))
                c1 = int(getattr(cell, "end_col_offset_idx"))
                txt = str(getattr(cell, "text", "") or "").strip()
                is_header = bool(getattr(cell, "column_header", False))
            except Exception:
                continue

            for r in range(max(0, r0), min(num_rows, r1 + 1)):
                for c in range(max(0, c0), min(num_cols, c1 + 1)):
                    if txt and not grid[r][c]:
                        grid[r][c] = txt
                    header_mask[r][c] = header_mask[r][c] or is_header

        # Pick header row: first row where any cell is marked column_header; else row 0.
        header_row_idx = 0
        for r in range(num_rows):
            if any(header_mask[r][c] for c in range(num_cols)):
                header_row_idx = r
                break

        headers = [grid[header_row_idx][c] or f"col_{c+1}" for c in range(num_cols)]
        for r in range(header_row_idx + 1, num_rows):
            if any((grid[r][c] or "").strip() for c in range(num_cols)):
                rows.append([grid[r][c] for c in range(num_cols)])

    return {
        "headers": headers,
        "rows": rows,
        "raw": raw,
        "num_rows": num_rows,
        "num_cols": num_cols,
    }


def _table_text_from_structured(structured: Dict[str, Any], max_rows: int = 30) -> str:
    """Flatten structured table into an embedding-friendly text string."""
    headers = structured.get("headers") or []
    rows = structured.get("rows") or []
    out: List[str] = []
    if headers:
        out.append(" | ".join(str(h) for h in headers))
    for r in rows[:max_rows]:
        out.append(" | ".join(str(x) for x in (r or [])))
    return "\n".join(out).strip()


def parse_pdf_with_docling(file_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Parse PDF via Docling. Returns (full_text, table_chunks)."""
    converter = DocumentConverter()
    result = converter.convert(file_path)

    doc = result.document

    # Prefer markdown export for reading order; fallback to concatenated text items.
    full_text = ""
    try:
        full_text = doc.export_to_markdown() or ""
    except Exception:
        full_text = ""

    if not full_text:
        texts = getattr(doc, "texts", None) or []
        parts: List[str] = []
        for item in texts:
            txt = getattr(item, "text", None)
            if txt:
                parts.append(str(txt))
        full_text = "\n".join(parts)

    # Build dedicated table chunks (dual storage)
    table_chunks: List[Dict[str, Any]] = []
    tables = getattr(doc, "tables", None) or []
    for idx, table in enumerate(tables):
        structured = _table_structured_from_docling(table)
        table_text = _table_text_from_structured(structured)
        if not table_text:
            # Last-resort: embed raw JSON-ish form.
            try:
                table_text = json.dumps(
                    structured.get("raw") or structured, ensure_ascii=False
                )[:4000]
            except Exception:
                table_text = str(structured.get("raw") or "")

        table_chunks.append(
            {
                "text": table_text,
                "metadata_overrides": {
                    "type": "table",
                    "table_text": table_text,
                    "table_structured": structured,
                    "table_index": idx,
                },
            }
        )

    return full_text, table_chunks


def extract_metadata(filename: str, file_path: str) -> Dict[str, Any]:
    """Extracts initial document metadata from filename and path."""
    metadata: Dict[str, Any] = {
        "filename": filename,
        "document_path": file_path,
        "classification": "regulation",
        "indexed_at": datetime.now().isoformat(),
    }

    if filename.startswith("FAR_"):
        metadata["source"] = "FAR"
        match = re.search(r"FAR_(\d+)", filename)
        if match:
            metadata["part"] = match.group(1)
    elif filename.startswith("DFARS_"):
        metadata["source"] = "DFARS"
        metadata["part"] = "201-253"
    elif "EM 385" in filename or "EM_385" in filename:
        metadata["source"] = "EM385"
        metadata["part"] = "1-1"

    return metadata


def extract_clause_references(text: str) -> List[Dict[str, str]]:
    """Regex-based extraction of regulatory clause references."""
    refs: List[Dict[str, str]] = []
    far_matches = re.findall(r"FAR\s+(\d+\.\d+(?:-\d+)?)", text, re.IGNORECASE)
    for match in far_matches:
        refs.append({"type": "FAR", "clause": match})

    dfars_matches = re.findall(r"DFARS\s+(\d+\.\d+(?:-\d+)?)", text, re.IGNORECASE)
    for match in dfars_matches:
        refs.append({"type": "DFARS", "clause": match})

    return refs


def extract_structured_sections(full_text: str, source: str) -> List[Dict[str, Any]]:
    """Parses flat text into hierarchical sections."""
    lines = full_text.split("\n")
    sections: List[Dict[str, Any]] = []
    current_part: Optional[str] = None
    current_subpart: Optional[str] = None
    current_chapter: Optional[str] = None
    current_section: Optional[Dict[str, Any]] = None
    current_appendix: Optional[str] = None
    current_appendix_part: Optional[str] = None

    for raw_line in lines:
        line = raw_line.rstrip()
        line_type = classify_line(line, source)

        if line_type in ["EMPTY", "FOOTER"]:
            continue

        if line_type == "PART":
            match = re.search(r"PART\s+(\d+)", line, re.IGNORECASE)
            if match:
                current_part = match.group(1)
                current_appendix = None
                current_appendix_part = None
            continue

        if line_type == "SUBPART":
            match = re.search(r"Subpart\s+(\d+\.\d+)", line, re.IGNORECASE)
            if match:
                current_subpart = match.group(1)
            continue

        if line_type == "CHAPTER":
            match = re.search(r"Chapter\s+(\d+)", line, re.IGNORECASE)
            if match:
                current_chapter = match.group(1)
            continue

        if line_type == "APPENDIX":
            match = re.search(r"APPENDIX\s+([A-Z])", line, re.IGNORECASE)
            if match:
                current_appendix = match.group(1)
                current_part = None
                current_subpart = None
                current_appendix_part = None
            continue

        if line_type == "APPENDIX_PART":
            match = re.search(r"Part\s+(\d+)", line, re.IGNORECASE)
            if match:
                current_appendix_part = match.group(1)
            continue

        if line_type == "APPENDIX_SECTION":
            if current_section:
                sections.append(current_section)

            # Match standard A-1 or Rule 1
            match = re.match(r"^([A-Z]-\d+|Rule\s+\d+)\s*(.*)", line, re.IGNORECASE)
            if match:
                sec_num = match.group(1)
                sec_title = match.group(2).strip()
                current_section = {
                    "section_number": sec_num,
                    "section_title": sec_title if sec_title else sec_num,
                    "hierarchy_struct": {
                        "appendix": current_appendix,
                        "appendix_part": current_appendix_part,
                    },
                    "hierarchy_path": [
                        x
                        for x in [
                            (
                                f"Appendix {current_appendix}"
                                if current_appendix
                                else None
                            ),
                            (
                                f"Part {current_appendix_part}"
                                if current_appendix_part
                                else None
                            ),
                        ]
                        if x
                    ],
                    "full_text": line + "\n",
                }
            continue

        if line_type == "SECTION":
            if current_section:
                sections.append(current_section)
            if source in ["FAR", "DFARS"]:
                match = re.match(r"^(\d{2,3}\.\d{3}(?:-\d+)?)\s+(.+)", line)
                if match:
                    current_section = {
                        "section_number": match.group(1),
                        "section_title": match.group(2).strip(),
                        "hierarchy_struct": {
                            "part": current_part,
                            "subpart": current_subpart,
                            "appendix": current_appendix,
                        },
                        "hierarchy_path": [
                            x
                            for x in [
                                f"PART {current_part}" if current_part else None,
                                (
                                    f"Subpart {current_subpart}"
                                    if current_subpart
                                    else None
                                ),
                                (
                                    f"Appendix {current_appendix}"
                                    if current_appendix
                                    else None
                                ),
                            ]
                            if x
                        ],
                        "full_text": line + "\n",
                    }
                    continue
            if source == "EM385":
                match = re.match(r"^(\d+-\d+)\.\s+(.+)", line)
                if match:
                    current_section = {
                        "section_number": match.group(1),
                        "section_title": match.group(2).strip(),
                        "hierarchy_struct": {"chapter": current_chapter},
                        "hierarchy_path": [
                            x
                            for x in [
                                (
                                    f"Chapter {current_chapter}"
                                    if current_chapter
                                    else None
                                )
                            ]
                            if x
                        ],
                        "full_text": line + "\n",
                    }
                    continue

        if current_section:
            current_section["full_text"] += line + "\n"

    if current_section:
        sections.append(current_section)
    return sections


# ---------------------------------------------------------
# Chunking Functions
# ---------------------------------------------------------


def count_tokens(text: str) -> int:
    """Returns the number of tokens in a text string."""
    return len(TOKENIZER.encode(text))


_OBLIGATION_RE = re.compile(r"\b(shall|must|required|only\s+if)\b", re.IGNORECASE)
_CROSSREF_RE = re.compile(
    r"\b(see\s+paragraph\s+\([a-z0-9ivx]+\)|as\s+defined\s+in|FAR\s+\d+\.\d+(?:-\d+)?|DFARS\s+\d+\.\d+(?:-\d+)?)\b",
    re.IGNORECASE,
)
_ENUM_LINE_RE = re.compile(r"^\s*\(([a-z]|\d+|[ivx]+)\)\s+\S", re.IGNORECASE)


def should_force_keep(section_text: str) -> bool:
    """Section Integrity Guard: keep strong legal units intact when possible."""
    if not section_text:
        return False
    txt = section_text.strip()
    if _OBLIGATION_RE.search(txt):
        return True
    if _CROSSREF_RE.search(txt):
        return True
    # Enumerated conditions with substantive text: multiple enum lines.
    enum_hits = 0
    for line in txt.splitlines():
        if _ENUM_LINE_RE.match(line):
            enum_hits += 1
            if enum_hits >= 2:
                return True
    return False


def _split_em385_sublevels(text: str) -> List[str]:
    """Split EM385 blocks on sub-level patterns (1-1.a, a., (1), (a))."""
    if not text:
        return []
    lines = text.splitlines()
    start_re = re.compile(
        r"^\s*(?:\d+-\d+\.[a-z]\b|[a-z]\.\s|\(\d+\)\s|\([a-z]\)\s)",
        re.IGNORECASE,
    )
    blocks: List[str] = []
    current: List[str] = []
    for line in lines:
        if start_re.match(line) and current:
            blocks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [b for b in blocks if b]


def _split_anchor_and_remainder(
    section_text: str, desired_anchor_tokens: int = 150
) -> Tuple[Optional[str], str]:
    """Create anchor chunk = header + next 1–2 paragraphs (or token target)."""
    if not section_text:
        return None, ""
    paragraphs = [
        p.strip() for p in re.split(r"\n\s*\n+", section_text.strip()) if p.strip()
    ]
    if not paragraphs:
        return None, section_text.strip()
    # Always include the first paragraph (usually contains the clause/section header line).
    anchor_parts = [paragraphs[0]]
    # Add up to one more paragraph, then optionally expand until token target.
    for p in paragraphs[1:3]:
        if count_tokens("\n\n".join(anchor_parts + [p])) > desired_anchor_tokens:
            break
        anchor_parts.append(p)
    anchor_text = "\n\n".join(anchor_parts).strip()
    remainder = "\n\n".join(paragraphs[len(anchor_parts) :]).strip()
    return anchor_text if anchor_text else None, remainder


def _ensure_max_chunk_tokens(text: str) -> List[str]:
    """Enforce MAX_TOKEN_CAP with minimal splitting (paragraph → sentence → token)."""
    if not text:
        return []
    if count_tokens(text) <= MAX_TOKEN_CAP:
        return [text]
    # Reuse existing create_chunks() which already does paragraph/sentence/token fallback.
    return create_chunks(text)


_ANCHOR_RE = re.compile(
    r"^\s*(?:FAR|DFARS)?\s*(?:52\.\d{3}-\d+|252\.\d{3}-\d+|\d{2,3}\.\d{3}(?:-\d+)?)\b",
    re.IGNORECASE,
)
_EM385_ANCHOR_RE = re.compile(r"^\s*\d+-\d+\.", re.MULTILINE)


def _is_anchor_chunk(text: str) -> bool:
    """Clause-title like anchor chunks are allowed to stay small."""
    first_line = (text or "").strip().splitlines()
    if not first_line:
        return False
    return bool(_ANCHOR_RE.match(first_line[0]))


def _is_em385_anchor(text: str) -> bool:
    """EM385 section start (e.g. 1-1., 1-2.) as anchor."""
    first_line = (text or "").strip().splitlines()
    if not first_line:
        return False
    return bool(_EM385_ANCHOR_RE.match(first_line[0]))


def _keep_anchor_standalone(text: str) -> bool:
    """Keep clause/section-start chunks standalone (FAR/DFARS/EM385)."""
    return KEEP_STANDALONE_ANCHOR_CHUNKS and (
        _is_anchor_chunk(text) or _is_em385_anchor(text)
    )


def _tail_overlap_by_tokens(parts: List[str], token_budget: int) -> List[str]:
    """Keep the tail of `parts` whose total token count fits the overlap budget."""
    if token_budget <= 0 or not parts:
        return []
    overlap: List[str] = []
    total = 0
    for part in reversed(parts):
        t = count_tokens(part)
        if total + t > token_budget:
            break
        overlap.insert(0, part)
        total += t
    return overlap


def _merge_small_text_chunks(
    chunks: List[str], min_tokens: int, max_tokens: int
) -> List[str]:
    """Merge undersized chunks into adjacent chunks to reduce retrieval noise."""
    if not chunks:
        return []
    merged: List[str] = []
    i = 0
    while i < len(chunks):
        cur = chunks[i]
        cur_tokens = count_tokens(cur)
        if cur_tokens >= min_tokens or _keep_anchor_standalone(cur):
            merged.append(cur)
            i += 1
            continue

        # Prefer merging tiny tail into previous chunk.
        if merged:
            prev = merged[-1]
            if count_tokens(prev) + cur_tokens <= max_tokens:
                merged[-1] = f"{prev}\n\n{cur}"
                i += 1
                continue

        # Otherwise merge forward with next chunk when possible.
        if i + 1 < len(chunks):
            nxt = chunks[i + 1]
            if cur_tokens + count_tokens(nxt) <= max_tokens:
                merged.append(f"{cur}\n\n{nxt}")
                i += 2
                continue

        merged.append(cur)
        i += 1
    return merged


def _split_on_subclause_boundaries(text: str) -> List[str]:
    """Split FAR/DFARS block on line-start (a), (b), (1), (2), (i), (ii)."""
    if not text or count_tokens(text) <= MAX_TOKEN_CAP:
        return [text] if text else []
    lines = text.splitlines()
    subclause_start = re.compile(r"^\s*\(([a-z]|\d+|[ivx]+)\)\s")
    blocks: List[str] = []
    current: List[str] = []
    for line in lines:
        if subclause_start.match(line) and current:
            blocks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [b for b in blocks if b]


def _split_on_clause_boundaries(text: str, source: str) -> List[str]:
    """Pre-split on clause/section boundaries. FAR/DFARS: clause numbers; EM385: 1-1., 1-2."""
    source = (source or "").upper()
    if count_tokens(text) < CLAUSE_SPLIT_TRIGGER_TOKENS:
        return [text]

    if source == "EM385":
        lines = text.splitlines()
        blocks = []
        current = []
        em385_section_start = re.compile(r"^\s*\d+-\d+\.\s")
        for line in lines:
            if em385_section_start.match(line) and current:
                blocks.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            blocks.append("\n".join(current).strip())
        return [b for b in blocks if b]

    if source not in {"FAR", "DFARS"}:
        return [text]

    lines = text.splitlines()
    blocks: List[str] = []
    current: List[str] = []

    for line in lines:
        if _ANCHOR_RE.match(line) and current:
            blocks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        blocks.append("\n".join(current).strip())

    return [b for b in blocks if b]


def _passes_quality_gate(text: str) -> bool:
    """Reject chunks that are mostly noise (headers/footers/artifacts)."""
    if not text:
        return False
    tokens = count_tokens(text)
    if tokens < 8:
        return False

    stripped = text.strip()
    alnum = sum(ch.isalnum() for ch in stripped)
    printable = sum(ch.isprintable() and not ch.isspace() for ch in stripped) or 1
    alnum_ratio = alnum / printable
    if alnum_ratio < 0.35:
        return False

    words = re.findall(r"[A-Za-z0-9]{2,}", stripped.lower())
    if not words:
        return False
    unique_ratio = len(set(words)) / len(words)
    return unique_ratio >= 0.08


def _merge_chunk_records_within_section(
    chunk_records: List[Dict[str, Any]],
    min_tokens: int,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    """Global consolidation pass to reduce tiny chunks after section chunking."""
    if not chunk_records:
        return []
    merged: List[Dict[str, Any]] = []
    i = 0
    while i < len(chunk_records):
        cur = dict(chunk_records[i])
        cur_text = cur.get("text", "")
        cur_tokens = count_tokens(cur_text)
        if (
            cur.get("is_anchor")
            or cur_tokens >= min_tokens
            or _keep_anchor_standalone(cur_text)
        ):
            merged.append(cur)
            i += 1
            continue

        # Try merging tiny chunk into previous chunk in same section.
        if merged:
            prev = merged[-1]
            same_section = prev.get("section_number") == cur.get(
                "section_number"
            ) and prev.get("section_title") == cur.get("section_title")
            if (
                same_section
                and not prev.get("is_anchor")
                and count_tokens(prev["text"]) + cur_tokens <= max_tokens
            ):
                prev["text"] = f"{prev['text']}\n\n{cur_text}"
                i += 1
                continue

        # Or merge forward with next chunk in same section.
        if i + 1 < len(chunk_records):
            nxt = dict(chunk_records[i + 1])
            same_section = nxt.get("section_number") == cur.get(
                "section_number"
            ) and nxt.get("section_title") == cur.get("section_title")
            if (
                same_section
                and not nxt.get("is_anchor")
                and cur_tokens + count_tokens(nxt.get("text", "")) <= max_tokens
            ):
                nxt["text"] = f"{cur_text}\n\n{nxt['text']}"
                chunk_records[i + 1] = nxt
                i += 1
                continue

        merged.append(cur)
        i += 1
    return merged


def create_chunks(text: str) -> List[str]:
    """Breaks a text block into smaller chunks based on token count with overlap."""
    if not text:
        return []

    # Initial split by paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_tokens = 0

    for i, para in enumerate(paragraphs):
        para_tokens = count_tokens(para)

        # If a single paragraph is larger than the cap, we must split it by sentences
        if para_tokens > MAX_TOKEN_CAP:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent_tokens = count_tokens(sent)
                if current_tokens + sent_tokens > TARGET_CHUNK_TOKENS and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    overlap_sentences = _tail_overlap_by_tokens(
                        current_chunk, CHUNK_OVERLAP
                    )
                    current_chunk = overlap_sentences
                    current_tokens = sum(count_tokens(x) for x in overlap_sentences)

                if sent_tokens > MAX_TOKEN_CAP:
                    logger.warning(
                        f"Oversized sentence ({sent_tokens} tokens) found. Force-splitting."
                    )
                    # Try meaning-preserving splits before token-window fallback.
                    # 1) Split on line-start style markers that may have been flattened into a single line.
                    marker_parts = re.split(r"\s(?=\([a-z]|\d+|[ivx]+\)\s)", sent)
                    marker_parts = [p.strip() for p in marker_parts if p.strip()]
                    if len(marker_parts) > 1 and all(
                        count_tokens(p) <= MAX_TOKEN_CAP for p in marker_parts
                    ):
                        for p in marker_parts:
                            chunks.append(p)
                        continue

                    # 2) Split on semicolons / enumerator separators (common in legal text).
                    semi_parts = re.split(r";\s+", sent)
                    semi_parts = [p.strip() for p in semi_parts if p.strip()]
                    if len(semi_parts) > 1 and all(
                        count_tokens(p) <= MAX_TOKEN_CAP for p in semi_parts
                    ):
                        for p in semi_parts:
                            chunks.append(p)
                        continue

                    # 3) Last resort: token-window splitting.
                    tokens = TOKENIZER.encode(sent)
                    for j in range(0, len(tokens), TARGET_CHUNK_TOKENS):
                        chunk_tokens = tokens[j : j + TARGET_CHUNK_TOKENS]
                        chunks.append(TOKENIZER.decode(chunk_tokens))
                else:
                    current_chunk.append(sent)
                    current_tokens += sent_tokens
            continue

        split_threshold = min(MAX_TOKEN_CAP, TARGET_CHUNK_TOKENS)
        if current_tokens + para_tokens > split_threshold:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))

            # Carry over tail paragraphs until overlap token budget is reached.
            overlap_chunk = _tail_overlap_by_tokens(current_chunk, CHUNK_OVERLAP)
            overlap_tokens = sum(count_tokens(x) for x in overlap_chunk)
            current_chunk = overlap_chunk + [para]
            current_tokens = overlap_tokens + para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    merged = _merge_small_text_chunks(chunks, MIN_CHUNK_TOKENS, MAX_TOKEN_CAP)
    return merged


def create_section_aware_chunks(
    sections: List[Dict[str, Any]],
    source: str,
) -> List[Dict[str, Any]]:
    """Chunking v2: meaning-first gates (800/900/1200) + anchors + integrity guard."""
    all_chunks: List[Dict[str, Any]] = []
    src = (source or "").upper()

    for section in sections:
        section_text = (section.get("full_text") or "").strip()
        if not section_text:
            continue

        force_keep = should_force_keep(section_text)
        section_tokens = count_tokens(section_text)

        # Anchor enrichment: emit an anchor chunk (header + 1–2 paragraphs).
        anchor_text, remainder = _split_anchor_and_remainder(
            section_text, desired_anchor_tokens=150
        )
        if anchor_text:
            for part in _ensure_max_chunk_tokens(anchor_text):
                all_chunks.append(
                    {
                        "text": part,
                        "section_number": section.get("section_number"),
                        "section_title": section.get("section_title"),
                        "hierarchy_struct": section.get("hierarchy_struct"),
                        "hierarchy_path": section.get("hierarchy_path"),
                        "is_anchor": True,
                    }
                )

        # Use remainder for the main body chunking (avoid duplicating anchor content).
        body_text = remainder if remainder else ""

        # If no remainder (tiny section), we're done.
        if not body_text:
            continue

        # Section Integrity Guard: keep as one unit (then enforce max cap minimally).
        if force_keep:
            for part in _ensure_max_chunk_tokens(body_text):
                all_chunks.append(
                    {
                        "text": part,
                        "section_number": section.get("section_number"),
                        "section_title": section.get("section_title"),
                        "hierarchy_struct": section.get("hierarchy_struct"),
                        "hierarchy_path": section.get("hierarchy_path"),
                    }
                )
            continue

        # Meaning-first gate: keep intact for <=800, or <=900 (no subclause split),
        # then enforce max-cap if needed.
        if (
            section_tokens <= KEEP_INTACT_TOKEN_LIMIT
            or section_tokens <= SUBCLAUSE_SPLIT_TOKEN_LIMIT
        ):
            for part in _ensure_max_chunk_tokens(body_text):
                all_chunks.append(
                    {
                        "text": part,
                        "section_number": section.get("section_number"),
                        "section_title": section.get("section_title"),
                        "hierarchy_struct": section.get("hierarchy_struct"),
                        "hierarchy_path": section.get("hierarchy_path"),
                    }
                )
            continue

        # For >900 tokens: split by subclause/sublevel boundaries first.
        split_blocks = _split_on_clause_boundaries(body_text, source)
        for block in split_blocks:
            block_tokens = count_tokens(block)

            if block_tokens <= SUBCLAUSE_SPLIT_TOKEN_LIMIT:
                for part in _ensure_max_chunk_tokens(block):
                    all_chunks.append(
                        {
                            "text": part,
                            "section_number": section.get("section_number"),
                            "section_title": section.get("section_title"),
                            "hierarchy_struct": section.get("hierarchy_struct"),
                            "hierarchy_path": section.get("hierarchy_path"),
                        }
                    )
                continue

            if src in {"FAR", "DFARS"}:
                sub_blocks = _split_on_subclause_boundaries(
                    block
                )  # (a)(1)(i) at line start
            elif src == "EM385":
                sub_blocks = _split_em385_sublevels(block)
            else:
                sub_blocks = [block]

            for sub_block in sub_blocks:
                st = count_tokens(sub_block)
                # If still large, paragraph/sentence/token split only as needed to satisfy max cap.
                if st > PARAGRAPH_SPLIT_TOKEN_LIMIT:
                    parts = create_chunks(sub_block)
                else:
                    parts = _ensure_max_chunk_tokens(sub_block)

                parts = _merge_small_text_chunks(parts, MIN_CHUNK_TOKENS, MAX_TOKEN_CAP)
                for chunk_text in parts:
                    all_chunks.append(
                        {
                            "text": chunk_text,
                            "section_number": section.get("section_number"),
                            "section_title": section.get("section_title"),
                            "hierarchy_struct": section.get("hierarchy_struct"),
                            "hierarchy_path": section.get("hierarchy_path"),
                        }
                    )

    # Consolidate tiny chunks within sections, but never merge anchors.
    merged = _merge_chunk_records_within_section(
        all_chunks,
        min_tokens=MIN_CHUNK_TOKENS,
        max_tokens=MAX_TOKEN_CAP,
    )
    return merged


# ---------------------------------------------------------
# Embedding and Persistence Logic
# ---------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def _call_embedding_api(session: aiohttp.ClientSession, batch: List[str]):
    """Internal helper with retry logic for embedding API calls."""
    async with session.post(
        EMBEDDING_ENDPOINT,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": EMBEDDING_MODEL, "input": batch},
        timeout=30,
    ) as response:
        if response.status == 200:
            return await response.json()
        else:
            error_text = await response.text()
            logger.error(f"OpenAI API Error {response.status}: {error_text}")
            raise RuntimeError(f"API Error {response.status}: {error_text}")


async def generate_embeddings(
    chunks: List[Dict[str, Any]], session: aiohttp.ClientSession, filename: str
) -> tuple[List[Dict[str, Any]], List[Optional[List[float]]]]:
    """Generates dense vector embeddings, with a guardrail for token limits."""
    final_chunks: List[Dict[str, Any]] = []
    embeddings: List[Optional[List[float]]] = []

    for i, chunk in enumerate(chunks):
        tokens = count_tokens(chunk["text"])
        if tokens > MAX_TOKEN_CAP:
            logger.info(
                f"  Guardrail: {filename} chunk {i} exceeds limit ({tokens} tokens). Re-chunking."
            )
            sub_texts = create_chunks(chunk["text"])
            for sub_text in sub_texts:
                final_chunks.append({**chunk, "text": sub_text})
        else:
            final_chunks.append(chunk)

    texts = [c["text"] for c in final_chunks]
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        logger.info(
            f"  Requesting embeddings for batch {i//BATCH_SIZE + 1} ({len(batch)} items)"
        )
        try:
            result = await _call_embedding_api(session, batch)
            for item in result["data"]:
                embeddings.append(item["embedding"])
        except Exception as e:
            logger.error(f"Failed embedding generation after retries: {e}")
            embeddings.extend([None] * len(batch))

        if i + BATCH_SIZE < len(texts):
            await asyncio.sleep(EMBED_RATE_DELAY)

    return final_chunks, embeddings


async def store_chunks(
    chunks: List[Dict[str, Any]],
    embeddings: List[Optional[List[float]]],
    base_metadata: Dict[str, Any],
    conn: AsyncConnection,
) -> None:
    """Upserts chunk data into the database using SQLAlchemy-generated SQL."""
    dense_params: List[Dict[str, Any]] = []
    sparse_params: List[Dict[str, Any]] = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        if embedding is None:
            continue

        # Debug: Verify embedding dimension
        EXPECTED_EMBED_DIM = 1536
        if embedding and len(embedding) != EXPECTED_EMBED_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"expected {EXPECTED_EMBED_DIM}, got {len(embedding)}"
            )

        text = chunk["text"]
        if not _passes_quality_gate(text):
            continue
        token_count = count_tokens(text)
        if token_count < MIN_CHUNK_TOKENS and not _keep_anchor_standalone(text):
            continue

        refs = extract_clause_references(text)

        metadata: Dict[str, Any] = {
            **base_metadata,
            "document_id": base_metadata["filename"],
            "chunk_index": i,
            "chunk_tokens": token_count,
            "embedding_model": EMBEDDING_MODEL,
            "clause_references": refs,
            "relevant_text": text[:500],
            "section_number": chunk.get("section_number"),
            "section_title": chunk.get("section_title"),
            "hierarchy_struct": chunk.get("hierarchy_struct"),
            "hierarchy_path": chunk.get("hierarchy_path"),
            "regulation_type": base_metadata.get("source"),
        }

        # Chunking v2: carry additional per-chunk metadata.
        if chunk.get("is_anchor"):
            metadata["is_anchor"] = True
        overrides = chunk.get("metadata_overrides") or {}
        if overrides:
            metadata.update(overrides)

        # Source-specific namespace for better filtering
        source = base_metadata.get("source", "Unknown").lower()
        chunk_namespace = f"{NAMESPACE}-{source}"

        chunk_id = f"{chunk_namespace}-{base_metadata['filename']}-{i}"

        dense_params.append(
            {
                "id": chunk_id,
                "namespace": chunk_namespace,
                "text": text,
                "metadata": metadata,
                "embedding": embedding,
            }
        )

        sparse_emb = encode_bm25(text)
        sparse_params.append(
            {
                "id": chunk_id,
                "namespace": chunk_namespace,
                "text": text,
                "metadata": metadata,
                "embedding": sparse_emb,
            }
        )

    if not dense_params:
        return

    if len(dense_params) != len(sparse_params):
        raise ValueError(
            f"Dense/sparse payload mismatch for {base_metadata['filename']}: "
            f"{len(dense_params)} vs {len(sparse_params)}"
        )

    try:
        # Execute batch natively with SQLAlchemy
        await conn.execute(stmt_upsert_dense, dense_params)
        await conn.execute(stmt_upsert_sparse, sparse_params)
    except Exception as e:
        logger.exception(
            f"Database bulk insert failure for {base_metadata['filename']}: {e}"
        )
        raise


# ---------------------------------------------------------
# High-Level Orchestration
# ---------------------------------------------------------


async def process_pdf(
    file_path: str, session: aiohttp.ClientSession, engine: AsyncEngine
) -> Dict[str, Any]:
    """Orchestrates the full ingestion pipeline for a single PDF file."""
    filename: str = Path(file_path).stem
    logger.info(f">> Processing file: {filename}")

    try:
        fhash: str = file_hash(file_path)

        # Use engine.begin() for automatic transaction management
        async with engine.begin() as conn:
            # Check for existing hash using SQLAlchemy selection
            res = await conn.execute(
                stmt_hash_check,
                {"file_hash": fhash, "namespace_prefix": f"{NAMESPACE}%"},
            )
            if res.first():
                logger.info(f"  {filename}: File already processed. Skipping.")
                return {"filename": filename, "skipped": True}

        metadata: Dict[str, Any] = extract_metadata(filename, file_path)
        metadata["file_hash"] = fhash
        full_text, table_chunks = parse_pdf_with_docling(file_path)

        source = metadata.get("source", "UNKNOWN")
        normalized_text = normalize_legal_text(full_text, source)
        sections = extract_structured_sections(normalized_text, source)
        chunks = create_section_aware_chunks(sections, source)
        before = len(chunks)
        chunks = dedupe_chunks_exact(chunks, scope="section")
        if len(chunks) != before:
            logger.info(
                f"  {filename}: Dropped {before - len(chunks)} exact-duplicate chunks (pre-embed)."
            )
        # Add Docling tables as dedicated chunks (dual storage in metadata)
        if table_chunks:
            for t in table_chunks:
                chunks.append(
                    {
                        "text": t.get("text", ""),
                        "section_number": None,
                        "section_title": "Table",
                        "hierarchy_struct": {},
                        "hierarchy_path": [],
                        "metadata_overrides": t.get("metadata_overrides") or {},
                    }
                )
        if not chunks:
            logger.warning(
                f"  {filename}: Structured section parsing returned no chunks. Using raw fallback chunking."
            )
            chunks = [{"text": t} for t in create_chunks(normalized_text)]
        logger.info(f"  {filename}: Extracted {len(chunks)} chunks.")

        # Generate embeddings with token-limit guardrail
        chunks, embeddings = await generate_embeddings(chunks, session, filename)

        async with engine.begin() as conn:
            await store_chunks(chunks, embeddings, metadata, conn)

        logger.info(f"  {filename}: Pipeline complete.")
        return {"filename": filename, "chunks": len(chunks), "success": True}

    except Exception as e:
        logger.exception(f"Processing failed for {file_path}")
        return {"filename": filename, "error": str(e), "success": False}


async def main() -> None:
    """Main entry point for discovery and parallel processing of regulatory PDFs."""
    if not SPECIFICATIONS_DIR.exists():
        logger.error(
            f"FATAL: Specifications source directory not found: {SPECIFICATIONS_DIR}"
        )
        return

    pdf_files: List[str] = [str(p) for p in SPECIFICATIONS_DIR.rglob("*.pdf")]

    include_files = {
        name.strip().lower() for name in INCLUDE_FILES_RAW.split(",") if name.strip()
    }
    if include_files:
        pdf_files = [p for p in pdf_files if Path(p).stem.lower() in include_files]
        logger.info(
            f"INCLUDE_FILES active. Processing {len(pdf_files)} file(s): "
            f"{', '.join(sorted(include_files))}"
        )

    logger.info(f"Found {len(pdf_files)} PDF files to process.")

    # Database Config
    db_url = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    if PG_SSLMODE == "require" and "ssl=" not in db_url:
        db_url += "?ssl=require"

    engine = create_async_engine(db_url, pool_size=MAX_CONCURRENT_PDFS + 1)

    try:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_PDFS)

        async def bounded_process(pdf_file: str) -> Dict[str, Any]:
            async with semaphore:
                return await process_pdf(pdf_file, session, engine)

        async with aiohttp.ClientSession() as session:
            tasks = [bounded_process(pdf) for pdf in pdf_files]
            await asyncio.gather(*tasks)

    finally:
        await engine.dispose()
    logger.info("Pipeline execution complete.")


if __name__ == "__main__":
    asyncio.run(main())
