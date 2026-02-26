from fpdf import FPDF
import os

class RepoSummaryPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'GovGig AI - Repository Summary', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

def generate_summary():
    pdf = RepoSummaryPDF()
    pdf.add_page()

    # 1. Project Overview
    pdf.chapter_title('1. Project Overview')
    pdf.chapter_body(
        "GovGig AI is a Python-based backend system designed for regulatory document Retrieval-Augmented Generation (RAG). "
        "It leverages LangGraph for multi-agent orchestration and FastAPI for serving REST and WebSocket APIs. "
        "The system provides high-precision answers to regulatory queries (FAR, DFARS, EM385) by combining "
        "advanced retrieval techniques with LLM synthesis."
    )

    # 2. Technical Architecture & Working Flow
    pdf.chapter_title('2. Technical Architecture & Working Flow')
    pdf.chapter_body(
        "The application follows a structured request-response cycle managed by LangGraph:\n\n"
        "A. Routing & Intent Classification:\n"
        "   - Every query is first processed by a deterministic QueryClassifier.\n"
        "   - It identifies if the user is looking for a specific clause (CLAUSE_LOOKUP) or general "
        "information (REGULATION_SEARCH).\n"
        "   - This classification allows for 'Fast Dispatch', often bypassing expensive LLM tool-selection calls.\n\n"
        "B. Advanced Data Retrieval (Multi-Stage Pipeline):\n"
        "   - Stage 1: Dense Vector Search using pgvector and OpenAI embeddings (semantic similarity).\n"
        "   - Stage 2: Sparse Full-Text Search (FTS) using PostgreSQL ts_rank_cd (lexical matching).\n"
        "   - Stage 3: Reciprocal Rank Fusion (RRF) to merge results from both dense and sparse sources.\n"
        "   - Stage 4: GPT-4o-mini Reranking to ensure the most relevant chunks are prioritized for the LLM.\n\n"
        "C. Response Synthesis:\n"
        "   - A synthesizer node takes the top reranked documents and generates a concise response with proper "
        "citations, ensuring accuracy and grounding in the regulatory text."
    )

    # 3. Data Ingestion Pipeline (ingest_python)
    pdf.chapter_title('3. Data Ingestion Pipeline')
    pdf.chapter_body(
        "The ingestion pipeline (pipeline.py) handles the transformation of raw PDFs into searchable vectors:\n"
        "   - Extraction: Uses PyMuPDF (fitz) for high-fidelity text extraction.\n"
        "   - Structured Parsing: Identifies PARTS, SUBPARTS, and SECTIONS to maintain document hierarchy.\n"
        "   - Section-Aware Chunking: Breaks text into chunks while preserving context and metadata (hierarchy path).\n"
        "   - Dual Storage: Chunks are stored in 'dense' (embeddings) and 'sparse' (BM25-compatible) tables for hybrid search."
    )

    # 4. Tech Stack & Infrastructure
    pdf.chapter_title('4. Tech Stack & Infrastructure')
    pdf.chapter_body(
        "- Core: Python 3.11+, FastAPI (API Layer), LangGraph (Workflow orchestration)\n"
        "- AI/ML: LangChain, OpenAI Models (GPT-4o for complex tasks, GPT-4o-mini for speed and reranking)\n"
        "- Database: PostgreSQL with pgvector extension (idempotent storage with hash-based caching)\n"
        "- Quality: Pytest for unified testing, Black/Ruff for code standards"
    )

    # 5. Key File Structure
    pdf.chapter_title('5. Key File Structure')
    pdf.chapter_body(
        "- src/agents/orchestrator.py: Defines the StateGraph and routing logic.\n"
        "- src/agents/data_retrieval.py: Implements the intent-based retrieval strategies.\n"
        "- src/tools/vector_search.py: Wraps the hybrid search and reranking tools.\n"
        "- ingest_python/pipeline.py: The entry point for the document ingestion system."
    )

    output_path = "repository_summary.pdf"
    pdf.output(output_path)
    print(f"Enhanced PDF summary generated: {output_path}")

if __name__ == "__main__":
    generate_summary()
