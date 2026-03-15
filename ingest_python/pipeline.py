#!/usr/bin/env python3

import os
import json
import asyncio
import aiohttp
import asyncpg
import fitz
import re
import hashlib
import logging
import warnings
import tiktoken
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# SQLAlchemy Core & Async IO
from sqlalchemy import MetaData, Table, Column, String, select, bindparam, cast, text
from sqlalchemy.types import UserDefinedType
from sqlalchemy.dialects.postgresql import insert as pg_insert, JSONB
from sqlalchemy.ext.asyncio import create_async_engine, AsyncConnection, AsyncEngine

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

warnings.filterwarnings('ignore')

from config import *

# ---------------------------------------------------------
# SQL Table Definitions (SQLAlchemy Core)
# ---------------------------------------------------------

metadata_obj = MetaData()

# Define tables using SQLAlchemy Core for query building
dense_table = Table(
    PG_DENSE_TABLE, metadata_obj,
    Column("id", String, primary_key=True),
    Column("namespace", String),
    Column("text", String),
    Column("metadata", JSONB),
    Column("embedding", PGVector),
)

sparse_table = Table(
    PG_SPARSE_TABLE, metadata_obj,
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
stmt_insert_dense = pg_insert(dense_table).values({
    "id": bindparam("id"),
    "namespace": bindparam("namespace"),
    "text": bindparam("text"),
    "metadata": bindparam("metadata"),
    "embedding": bindparam("embedding")
})
stmt_upsert_dense = stmt_insert_dense.on_conflict_do_update(
    index_elements=["id"],
    set_={
        "text": stmt_insert_dense.excluded.text,
        "metadata": stmt_insert_dense.excluded.metadata,
        "embedding": stmt_insert_dense.excluded.embedding
    }
)

# Build Sparse Insert Statement (Idempotent Upsert)
stmt_insert_sparse = pg_insert(sparse_table).values({
    "id": bindparam("id"),
    "namespace": bindparam("namespace"),
    "text": bindparam("text"),
    "metadata": bindparam("metadata"),
    "embedding": bindparam("embedding")
})
stmt_upsert_sparse = stmt_insert_sparse.on_conflict_do_update(
    index_elements=["id"],
    set_={
        "text": stmt_insert_sparse.excluded.text,
        "metadata": stmt_insert_sparse.excluded.metadata,
        "embedding": stmt_insert_sparse.excluded.embedding
    }
)

# Build Hash Check Statement (Idempotency)
# Using strict form to avoid text = jsonb mismatch
stmt_hash_check = select(dense_table.c.id).where(
    dense_table.c.metadata["file_hash"].astext == bindparam("file_hash"),
    dense_table.c.namespace.like(bindparam("namespace_prefix"))
).limit(1)

# ---------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# NLTK Initialization
# ---------------------------------------------------------

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Tokenizer for OpenAI Embeddings
TOKENIZER = tiktoken.get_encoding("cl100k_base")
MAX_TOKEN_CAP = MAX_CHUNK_TOKENS

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

def murmurhash3_32(data: str) -> int:
    """Calculates the 32-bit MurmurHash3 value of a string."""
    data_bytes = data.encode('utf-8')
    length = len(data_bytes)
    c1 = 0xcc9e2d51
    c2 = 0x1b873593
    h1 = 0
    nblocks = length // 4

    for i in range(nblocks):
        k1 = int.from_bytes(data_bytes[i*4:(i+1)*4], 'little')
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & 0xFFFFFFFF
        h1 = (h1 * 5 + 0xe6546b64) & 0xFFFFFFFF

    h1 ^= length
    h1 ^= h1 >> 16
    h1 = (h1 * 0x85ebca6b) & 0xFFFFFFFF
    h1 ^= h1 >> 13
    h1 = (h1 * 0xc2b2ae35) & 0xFFFFFFFF
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
# PDF Extraction Functions
# ---------------------------------------------------------

def extract_text_from_block(block: Dict[str, Any]) -> str:
    """Helper to extract text content from a PyMuPDF text block."""
    text = ""
    if 'lines' in block:
        for line in block['lines']:
            if 'text' in line:
                text += line['text'] + '\n'
            elif 'spans' in line:
                for span in line['spans']:
                    if 'text' in span:
                        text += span['text']
                text += '\n'
    return text

def extract_page_content(page: fitz.Page, page_num: int) -> Dict[str, Any]:
    """Extracts text and identifies potential tables from a single PDF page."""
    content: Dict[str, Any] = {
        'pageNum': page_num,
        'text': '',
        'tables': [],
        'hasImages': False
    }

    try:
        text_dict = page.get_text('dict')
        if 'blocks' in text_dict:
            for block in text_dict['blocks']:
                if block.get('type') == 0:
                    block_text = extract_text_from_block(block)
                    if block_text.strip():
                        content['text'] += block_text
                elif block.get('type') == 1:
                    content['hasImages'] = True

        lines = content['text'].split('\n')
        table_lines: List[str] = []
        for line in lines:
            # Improved table detection: pipes OR multiple spaces/tabs (columnar alignment)
            is_table_row = (line.count('\t') >= 1 or 
                           len(re.findall(r'\s{4,}', line)) >= 1 or
                           line.count('|') >= 2)
            
            if is_table_row:
                table_lines.append(line)
            elif len(table_lines) >= 3:
                content['tables'].append('\n'.join(table_lines))
                table_lines = []
            else:
                table_lines = []

        if len(table_lines) >= 3:
            content['tables'].append('\n'.join(table_lines))

    except Exception as e:
        logger.warning(f"Page {page_num}: extraction failed, attempting backup - {e}")
        try:
            content['text'] = page.get_text()
        except Exception as e2:
            logger.error(f"Page {page_num}: backup extraction failed - {e2}")

    return content

def extract_metadata(filename: str, file_path: str) -> Dict[str, Any]:
    """Extracts initial document metadata from filename and path."""
    metadata: Dict[str, Any] = {
        'filename': filename,
        'document_path': file_path,
        'classification': 'regulation',
        'indexed_at': datetime.now().isoformat()
    }

    if filename.startswith('FAR_'):
        metadata['source'] = 'FAR'
        match = re.search(r'FAR_(\d+)', filename)
        if match:
            metadata['part'] = match.group(1)
    elif filename.startswith('DFARS_'):
        metadata['source'] = 'DFARS'
        metadata['part'] = '201-253'
    elif 'EM 385' in filename or 'EM_385' in filename:
        metadata['source'] = 'EM385'
        metadata['part'] = '1-1'

    return metadata

def extract_clause_references(text: str) -> List[Dict[str, str]]:
    """Regex-based extraction of regulatory clause references."""
    refs: List[Dict[str, str]] = []
    far_matches = re.findall(r'FAR\s+(\d+\.\d+(?:-\d+)?)', text, re.IGNORECASE)
    for match in far_matches:
        refs.append({'type': 'FAR', 'clause': match})

    dfars_matches = re.findall(r'DFARS\s+(\d+\.\d+(?:-\d+)?)', text, re.IGNORECASE)
    for match in dfars_matches:
        refs.append({'type': 'DFARS', 'clause': match})

    return refs

from parsing.classifier import classify_line

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
            match = re.search(r'PART\s+(\d+)', line, re.IGNORECASE)
            if match:
                current_part = match.group(1)
                current_appendix = None
                current_appendix_part = None
            continue

        if line_type == "SUBPART":
            match = re.search(r'Subpart\s+(\d+\.\d+)', line, re.IGNORECASE)
            if match:
                current_subpart = match.group(1)
            continue

        if line_type == "CHAPTER":
            match = re.search(r'Chapter\s+(\d+)', line, re.IGNORECASE)
            if match:
                current_chapter = match.group(1)
            continue

        if line_type == "APPENDIX":
            match = re.search(r'APPENDIX\s+([A-Z])', line, re.IGNORECASE)
            if match:
                current_appendix = match.group(1)
                current_part = None
                current_subpart = None
                current_appendix_part = None
            continue

        if line_type == "APPENDIX_PART":
            match = re.search(r'Part\s+(\d+)', line, re.IGNORECASE)
            if match:
                current_appendix_part = match.group(1)
            continue

        if line_type == "APPENDIX_SECTION":
            if current_section:
                sections.append(current_section)
            
            # Match standard A-1 or Rule 1
            match = re.match(r'^([A-Z]-\d+|Rule\s+\d+)\s*(.*)', line, re.IGNORECASE)
            if match:
                sec_num = match.group(1)
                sec_title = match.group(2).strip()
                current_section = {
                    "section_number": sec_num,
                    "section_title": sec_title if sec_title else sec_num,
                    "hierarchy_struct": {
                        "appendix": current_appendix,
                        "appendix_part": current_appendix_part
                    },
                    "hierarchy_path": [
                        x for x in [
                            f"Appendix {current_appendix}" if current_appendix else None,
                            f"Part {current_appendix_part}" if current_appendix_part else None
                        ] if x
                    ],
                    "full_text": line + "\n"
                }
            continue

        if line_type == "SECTION":
            if current_section:
                sections.append(current_section)
            if source in ["FAR", "DFARS"]:
                match = re.match(r'^(\d{2,3}\.\d{3}(?:-\d+)?)\s+(.+)', line)
                if match:
                    current_section = {
                        "section_number": match.group(1),
                        "section_title": match.group(2).strip(),
                        "hierarchy_struct": {
                            "part": current_part,
                            "subpart": current_subpart,
                            "appendix": current_appendix
                        },
                        "hierarchy_path": [
                            x for x in [
                                f"PART {current_part}" if current_part else None,
                                f"Subpart {current_subpart}" if current_subpart else None,
                                f"Appendix {current_appendix}" if current_appendix else None
                            ] if x
                        ],
                        "full_text": line + "\n"
                    }
                    continue
            if source == "EM385":
                match = re.match(r'^(\d+-\d+)\.\s+(.+)', line)
                if match:
                    current_section = {
                        "section_number": match.group(1),
                        "section_title": match.group(2).strip(),
                        "hierarchy_struct": {"chapter": current_chapter},
                        "hierarchy_path": [
                            x for x in [f"Chapter {current_chapter}" if current_chapter else None] if x
                        ],
                        "full_text": line + "\n"
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
    return KEEP_STANDALONE_ANCHOR_CHUNKS and (_is_anchor_chunk(text) or _is_em385_anchor(text))

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

def _merge_small_text_chunks(chunks: List[str], min_tokens: int, max_tokens: int) -> List[str]:
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
        if cur_tokens >= min_tokens or _keep_anchor_standalone(cur_text):
            merged.append(cur)
            i += 1
            continue

        # Try merging tiny chunk into previous chunk in same section.
        if merged:
            prev = merged[-1]
            same_section = (
                prev.get("section_number") == cur.get("section_number")
                and prev.get("section_title") == cur.get("section_title")
            )
            if same_section and count_tokens(prev["text"]) + cur_tokens <= max_tokens:
                prev["text"] = f"{prev['text']}\n\n{cur_text}"
                i += 1
                continue

        # Or merge forward with next chunk in same section.
        if i + 1 < len(chunk_records):
            nxt = dict(chunk_records[i + 1])
            same_section = (
                nxt.get("section_number") == cur.get("section_number")
                and nxt.get("section_title") == cur.get("section_title")
            )
            if same_section and cur_tokens + count_tokens(nxt.get("text", "")) <= max_tokens:
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
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
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
            
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent_tokens = count_tokens(sent)
                if (
                    current_tokens + sent_tokens > TARGET_CHUNK_TOKENS
                    and current_chunk
                ):
                    chunks.append(" ".join(current_chunk))
                    overlap_sentences = _tail_overlap_by_tokens(current_chunk, CHUNK_OVERLAP)
                    current_chunk = overlap_sentences
                    current_tokens = sum(count_tokens(x) for x in overlap_sentences)
                
                if sent_tokens > MAX_TOKEN_CAP:
                    logger.warning(f"Oversized sentence ({sent_tokens} tokens) found. Force-splitting.")
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

def create_section_aware_chunks(sections: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """Processes sections into chunks; structure-first (clause/subclause), then size."""
    all_chunks: List[Dict[str, Any]] = []
    src = (source or "").upper()
    for section in sections:
        text = section["full_text"].strip()
        if not text:
            continue

        split_blocks = _split_on_clause_boundaries(text, source)

        for block in split_blocks:
            block_tokens = count_tokens(block)
            if block_tokens <= MAX_TOKEN_CAP:
                sub_blocks = [block]
            elif src in {"FAR", "DFARS"}:
                sub_blocks = _split_on_subclause_boundaries(block)
            else:
                sub_blocks = [block]

            for sub_block in sub_blocks:
                st = count_tokens(sub_block)
                if st <= MAX_TOKEN_CAP:
                    block_chunks = [sub_block]
                else:
                    block_chunks = create_chunks(sub_block)
                block_chunks = _merge_small_text_chunks(block_chunks, MIN_CHUNK_TOKENS, MAX_TOKEN_CAP)

                for chunk_text in block_chunks:
                    all_chunks.append({
                        "text": chunk_text,
                        "section_number": section["section_number"],
                        "section_title": section["section_title"],
                        "hierarchy_struct": section["hierarchy_struct"],
                        "hierarchy_path": section["hierarchy_path"],
                    })
    return _merge_chunk_records_within_section(
        all_chunks,
        min_tokens=MIN_CHUNK_TOKENS,
        max_tokens=MAX_TOKEN_CAP,
    )

# ---------------------------------------------------------
# Embedding and Persistence Logic
# ---------------------------------------------------------

from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
async def _call_embedding_api(session: aiohttp.ClientSession, batch: List[str]):
    """Internal helper with retry logic for embedding API calls."""
    async with session.post(
        EMBEDDING_ENDPOINT,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBEDDING_MODEL, "input": batch},
        timeout=30
    ) as response:
        if response.status == 200:
            return await response.json()
        else:
            error_text = await response.text()
            logger.error(f"OpenAI API Error {response.status}: {error_text}")
            raise RuntimeError(f"API Error {response.status}: {error_text}")

async def generate_embeddings(chunks: List[Dict[str, Any]], session: aiohttp.ClientSession, filename: str) -> tuple[List[Dict[str, Any]], List[Optional[List[float]]]]:
    """Generates dense vector embeddings, with a guardrail for token limits."""
    final_chunks: List[Dict[str, Any]] = []
    embeddings: List[Optional[List[float]]] = []

    for i, chunk in enumerate(chunks):
        tokens = count_tokens(chunk["text"])
        if tokens > MAX_TOKEN_CAP:
            logger.info(f"  Guardrail: {filename} chunk {i} exceeds limit ({tokens} tokens). Re-chunking.")
            sub_texts = create_chunks(chunk["text"])
            for sub_text in sub_texts:
                final_chunks.append({**chunk, "text": sub_text})
        else:
            final_chunks.append(chunk)

    texts = [c["text"] for c in final_chunks]
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        logger.info(f"  Requesting embeddings for batch {i//BATCH_SIZE + 1} ({len(batch)} items)")
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

async def store_chunks(chunks: List[Dict[str, Any]], embeddings: List[Optional[List[float]]], 
                      base_metadata: Dict[str, Any], conn: AsyncConnection) -> None:
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

        # Source-specific namespace for better filtering
        source = base_metadata.get("source", "Unknown").lower()
        chunk_namespace = f"{NAMESPACE}-{source}"

        chunk_id = f"{chunk_namespace}-{base_metadata['filename']}-{i}"
        
        dense_params.append({
            "id": chunk_id,
            "namespace": chunk_namespace,
            "text": text,
            "metadata": metadata,
            "embedding": embedding
        })
        
        sparse_emb = encode_bm25(text)
        sparse_params.append({
            "id": chunk_id,
            "namespace": chunk_namespace,
            "text": text,
            "metadata": metadata,
            "embedding": sparse_emb
        })

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
        logger.exception(f"Database bulk insert failure for {base_metadata['filename']}: {e}")
        raise

# ---------------------------------------------------------
# High-Level Orchestration
# ---------------------------------------------------------

async def process_pdf(file_path: str, session: aiohttp.ClientSession, engine: AsyncEngine) -> Dict[str, Any]:
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
                {"file_hash": fhash, "namespace_prefix": f"{NAMESPACE}%"}
            )
            if res.first():
                logger.info(f"  {filename}: File already processed. Skipping.")
                return {"filename": filename, "skipped": True}

        doc: fitz.Document = fitz.open(file_path)
        metadata: Dict[str, Any] = extract_metadata(filename, file_path)
        metadata["file_hash"] = fhash

        full_text: str = ""
        for i, page in enumerate(doc):
            content = extract_page_content(page, i + 1)
            full_text += content["text"] + "\n\n"
            for table in content.get("tables", []):
                if table.strip():
                    full_text += "\n\n[Table]\n" + table.strip() + "\n\n"

        source = metadata.get("source", "UNKNOWN")
        sections = extract_structured_sections(full_text, source)
        chunks = create_section_aware_chunks(sections, source)
        if not chunks:
            logger.warning(f"  {filename}: Structured section parsing returned no chunks. Using raw fallback chunking.")
            chunks = [{"text": t} for t in create_chunks(full_text)]
        logger.info(f"  {filename}: Extracted {len(chunks)} chunks.")

        # Generate embeddings with token-limit guardrail
        chunks, embeddings = await generate_embeddings(chunks, session, filename)
        
        async with engine.begin() as conn:
            await store_chunks(chunks, embeddings, metadata, conn)
        
        doc.close()
        logger.info(f"  {filename}: Pipeline complete.")
        return {"filename": filename, "chunks": len(chunks), "success": True}

    except Exception as e:
        logger.exception(f"Processing failed for {file_path}")
        return {"filename": filename, "error": str(e), "success": False}

async def main() -> None:
    """Main entry point for discovery and parallel processing of regulatory PDFs."""
    if not SPECIFICATIONS_DIR.exists():
        logger.error(f"FATAL: Specifications source directory not found: {SPECIFICATIONS_DIR}")
        return

    pdf_files: List[str] = [str(p) for p in SPECIFICATIONS_DIR.rglob("*.pdf")]

    include_files = {
        name.strip().lower()
        for name in INCLUDE_FILES_RAW.split(",")
        if name.strip()
    }
    if include_files:
        pdf_files = [
            p for p in pdf_files
            if Path(p).stem.lower() in include_files
        ]
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
