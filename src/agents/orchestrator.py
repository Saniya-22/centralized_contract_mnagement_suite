"""Main orchestrator using LangGraph for multi-agent coordination"""

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Literal, AsyncIterator, Dict, Any
import logging
from datetime import datetime
import re

from src.state.graph_state import GovGigState
from src.agents.data_retrieval import DataRetrievalAgent
from src.agents.prompts import get_synthesizer_prompt, get_letter_drafter_prompt, get_oos_response_prompt
from src.tools.llm_tools import format_documents
from src.tools.query_classifier import classify_query, QueryIntent
from src.services.sovereign_guard import SovereignGuard
from src.config import settings

logger = logging.getLogger(__name__)


class GovGigOrchestrator:
    """Main orchestrator for the GovGig multi-agent system."""
    # FAR/DFARS/EM385 with Part or clause numbers; standalone clause numbers (52.236-2, 252.204-7012); bracketed refs
    _CITATION_RE = re.compile(
        r"\b(FAR|DFARS|EM\s*385)\b\s*(?:Part\s+)?\d+(?:\.\d+)?(?:-\d+)?"
        r"|\b\d{2}\.\d{3}(?:-\d+)?\b"
        r"|\b252\.\d{3}-\d+\b"
        r"|\[[^\]]+\]",
        re.IGNORECASE,
    )
    _CLAIM_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
    _WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-]{2,}")
    _STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
        "have", "has", "had", "not", "but", "you", "your", "about", "into", "while",
        "where", "when", "what", "which", "who", "how", "they", "them", "their", "its",
        "can", "may", "must", "shall", "should", "would", "could", "than", "then",
        "there", "here", "also", "such", "each", "any", "all", "only", "other",
        "under", "over", "between", "within", "through", "into", "onto", "our", "out",
        "per", "via", "use", "using", "used", "being", "been", "more", "most",
    }
    _LOW_CONFIDENCE_LABEL = (
        "Our system is still expanding its coverage here. Where it's critical, we recommend confirming with your contract or CO.\n\n"
    )
    _SAFETY_REVIEW_LABEL = "Safety review notice: {reason}\n\n"
    
    def __init__(self, checkpointer=None):
        logger.info("Initializing GovGigOrchestrator")

        # Synthesizer: uses SYNTHESIZER_MODEL (default gpt-4o).
        self.synthesizer_llm = ChatOpenAI(
            model=settings.SYNTHESIZER_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=settings.TEMPERATURE,
            streaming=True,
            max_tokens=400,
        )
        # Letter drafter: same model, higher token limit for full drafts.
        self.letter_drafter_llm = ChatOpenAI(
            model=settings.SYNTHESIZER_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=settings.TEMPERATURE,
            streaming=False,
            max_tokens=1200,
        )

        # Initialize agents
        self.data_retrieval = DataRetrievalAgent()
        self.sovereign_guard = SovereignGuard()

        # Build and compile graph
        self.graph = self._build_graph()
        self.checkpointer = checkpointer
        self.app = self.graph.compile(checkpointer=checkpointer)

        logger.info(f"GovGigOrchestrator initialized successfully (persistence={'enabled' if checkpointer else 'disabled'})")

    @staticmethod
    def _normalize_score(raw_score: float) -> float:
        """Normalize heterogeneous score regimes into ~0..1 confidence."""
        score = float(raw_score or 0.0)
        
        # Cross-encoder range (0..10ish)
        if score > 1.0:
            return min(score / 10.0, 1.0)
            
        # RRF scores without a reranker are typically very small (< 0.05).
        # A rank 1 match in a single index is 1/(60+1) = ~0.0164.
        # We scale so Rank 1 ≈ 0.82; apply a small floor so weak docs don't over-penalize avg
        if 0 < score <= 0.05:
            return min(max(score * 50.0, 0.10), 1.0)
            
        # Already normalized 0..1 (e.g. from clause_lookup)
        return min(max(score, 0.0), 1.0)

    def _evidence_summary(self, documents: list[dict]) -> dict[str, float]:
        if not documents:
            return {"doc_count": 0.0, "top_norm": 0.0, "avg_norm": 0.0}
        norms = [
            self._normalize_score(doc.get("score") or doc.get("rerank_score") or 0.0)
            for doc in documents
        ]
        return {
            "doc_count": float(len(documents)),
            "top_norm": float(norms[0] if norms else 0.0),
            "avg_norm": float(sum(norms) / len(norms)) if norms else 0.0,
        }

    @classmethod
    def _extract_claim_units(cls, text: str) -> list[str]:
        if not text:
            return []
        units: list[str] = []
        for chunk in cls._CLAIM_SPLIT_RE.split(text):
            snippet = chunk.strip()
            if not snippet:
                continue
            if snippet.startswith("#"):
                continue
            if len(snippet) < 30:
                continue
            if len(snippet.split()) < 6:
                continue
            units.append(snippet)
        return units

    @classmethod
    def _citation_coverage(cls, text: str) -> tuple[float, bool]:
        """Returns (score, has_claims). When no claim units, score=0.5 (neutral) and has_claims=False."""
        claims = cls._extract_claim_units(text)
        if not claims:
            return (0.5, False)  # Not measured → neutral; don't fail on citation_bar
        cited = sum(1 for claim in claims if cls._CITATION_RE.search(claim))
        return (cited / len(claims), True)

    @classmethod
    def _content_tokens(cls, text: str) -> set[str]:
        if not text:
            return set()
        tokens = {t for t in cls._WORD_RE.findall(text.lower()) if t not in cls._STOPWORDS}
        return tokens

    @classmethod
    def _groundedness_score(cls, response_text: str, documents: list[dict]) -> float:
        response_tokens = cls._content_tokens(response_text)
        if not response_tokens:
            return 0.0
        corpus = []
        for doc in documents:
            content = doc.get("content") or doc.get("text") or ""
            if content:
                corpus.append(str(content))
        doc_tokens = cls._content_tokens(" ".join(corpus))
        if not doc_tokens:
            return 0.0
        overlap = response_tokens & doc_tokens
        union = response_tokens | doc_tokens
        # Jaccard: overlap/union so generic verbosity (many words not in docs) gets lower score
        return len(overlap) / len(union) if union else 0.0

    def _assess_answer_quality(
        self,
        response_text: str,
        documents: list[dict],
        evidence: dict[str, float],
        state: GovGigState,
    ) -> dict[str, float | bool]:
        citation_coverage, has_claims = self._citation_coverage(response_text)
        groundedness = self._groundedness_score(response_text, documents)
        evidence_avg = float(evidence.get("avg_norm", 0.0))
        is_clause_lookup = (
            state.get("query_intent") == QueryIntent.CLAUSE_LOOKUP.value
            or bool(state.get("detected_clause_ref"))
        )
        is_procedural = bool(state.get("is_procedural", False))
        min_docs = 1.0 if is_clause_lookup else 2.0

        # Procedural/analytical: reweight toward evidence + grounding; citation bar lower
        citation_bar = 0.35 if is_procedural else 0.42
        if is_procedural:
            quality_score = (
                0.45 * evidence_avg
                + 0.40 * groundedness
                + 0.15 * citation_coverage
            )
            quality_bar = 0.50
        else:
            quality_score = (
                0.40 * evidence_avg
                + 0.35 * groundedness
                + 0.25 * citation_coverage
            )
            quality_bar = 0.52

        # Soft guardrails: when we didn't measure citation (no claim units), don't fail on citation_bar
        low_confidence = bool(
            evidence.get("doc_count", 0.0) < min_docs
            or evidence_avg < 0.18
            or groundedness < 0.42
            or (has_claims and citation_coverage < citation_bar)
            or quality_score < quality_bar
        )
        # Don't over-apply: high evidence = trust the answer, avoid notice
        if evidence_avg >= 0.80:
            low_confidence = False
        elif evidence_avg >= 0.75:
            low_confidence = False
        elif is_clause_lookup and evidence.get("doc_count", 0) >= 1 and evidence_avg >= 0.70:
            low_confidence = False
        elif evidence_avg >= 0.70 and citation_coverage >= 0.80:
            low_confidence = False
        elif evidence_avg >= 0.62 and groundedness >= 0.48:
            # Solid evidence + decent grounding: skip notice (reduces false low-confidence for analytical)
            low_confidence = False
        elif state.get("is_document_request") and evidence.get("doc_count", 0) >= 2 and evidence_avg >= 0.55:
            # Letter/REA/RFI draft with enough docs: avoid notice (drafts are less citation-dense by design)
            low_confidence = False

        return {
            "citation_coverage": round(citation_coverage, 4),
            "groundedness_score": round(groundedness, 4),
            "evidence_score": round(evidence_avg, 4),
            "quality_score": round(quality_score, 4),
            "low_confidence": low_confidence,
        }

    def _with_low_confidence_label(self, response_text: str, low_confidence: bool) -> str:
        if not low_confidence:
            return response_text
        if response_text.startswith(self._LOW_CONFIDENCE_LABEL):
            return response_text
        return f"{self._LOW_CONFIDENCE_LABEL}{response_text}"

    def _with_safety_review_label(self, response_text: str, reason: str) -> str:
        reason_text = (reason or "Automated guardrails flagged policy risk.").strip()
        label = self._SAFETY_REVIEW_LABEL.format(reason=reason_text)
        if response_text.startswith(label):
            return response_text
        return f"{label}{response_text}"

    @staticmethod
    def _safe_fallback_message() -> str:
        return (
            "I don’t have sufficient high-confidence evidence in the retrieved regulatory text "
            "to provide a reliable answer. Please refine the query with regulation type/section "
            "(e.g., FAR/DFARS/EM385 clause), and I’ll return a fully cited response."
        )

    @staticmethod
    def _safe_blocked_message() -> str:
        return (
            "I can’t provide that response because automated safety guardrails flagged it. "
            "Please rephrase with a narrower, regulation-focused request."
        )
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(GovGigState)

        # Add nodes
        workflow.add_node("router", self._route_query)
        workflow.add_node("data_retrieval", self.data_retrieval.run)
        workflow.add_node("letter_drafter", self._draft_letter)
        workflow.add_node("synthesizer", self._synthesize_response)

        # Set entry point
        workflow.set_entry_point("router")

        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self._determine_next_agent,
            {
                "data_retrieval": "data_retrieval",
                "synthesizer": "synthesizer",
                "end": END
            }
        )

        # After data_retrieval: route to letter_drafter (document request) or synthesizer
        workflow.add_conditional_edges(
            "data_retrieval",
            self._after_retrieval_routing,
            {
                "letter_drafter": "letter_drafter",
                "synthesizer": "synthesizer",
            }
        )

        # Both letter_drafter and synthesizer go to END
        workflow.add_edge("letter_drafter", END)
        workflow.add_edge("synthesizer", END)

        logger.info("LangGraph workflow built")
        return workflow
    
    async def _route_query(self, state: GovGigState) -> Dict[str, Any]:
        """Route the query using the deterministic QueryClassifier."""
        import time as _time
        _t0_node = _time.perf_counter()
        
        query = state["query"]
        logger.info(f"[Router] Classifying query: '{query[:100]}...'")

        classification = await classify_query(query)
        intent         = classification.intent

        # Map intent to graph routing decision.
        # If user asked for a document draft (letter/REA/RFI), send to data_retrieval even when intent is OUT_OF_SCOPE.
        next_agent = "end" if (intent == QueryIntent.OUT_OF_SCOPE and not classification.is_document_request) else "data_retrieval"

        elapsed = _time.perf_counter() - _t0_node
        logger.info(f"[Telemetry] Node 'router' completed in {elapsed:.2f}s")

        logger.info(
            f"[Router] intent={intent.value}, confidence={classification.confidence:.2f}, "
            f"next={next_agent}, clause_ref={classification.clause_reference}"
        )

        delta: Dict[str, Any] = {
            "next_agent":          next_agent,
            "reasoning":           f"QueryClassifier: intent={intent.value} confidence={classification.confidence:.2f}",
            "query_intent":        intent.value,
            "detected_clause_ref": classification.clause_reference,
            "detected_reg_type":   classification.regulation_type,
            "is_procedural":        classification.is_procedural,
            "is_contract_co":       classification.is_contract_co,
            "is_document_request":   classification.is_document_request,
            "document_request_type": getattr(classification, "document_request_type", None),
            "agent_path": [
                f"Router: intent={intent.value} conf={classification.confidence:.2f} "
                + (f"ref='{classification.clause_reference}' " if classification.clause_reference else "")
                + (f" proc={classification.is_procedural} co={classification.is_contract_co} doc={classification.is_document_request} " if (classification.is_procedural or classification.is_contract_co or classification.is_document_request) else "")
                + f"→ {next_agent}"
            ],
        }

        if intent == QueryIntent.OUT_OF_SCOPE and not classification.is_document_request:
            # Direct LLM call: answer briefly from general knowledge + polite scope notification
            try:
                messages = [
                    SystemMessage(content=get_oos_response_prompt(state)),
                    HumanMessage(content=query),
                ]
                oos_response = await self.synthesizer_llm.ainvoke(messages)
                delta["generated_response"] = (oos_response.content or "").strip()
            except Exception as e:
                logger.warning(f"[Router] OOS LLM call failed: {e}, using fallback message")
                delta["generated_response"] = (
                    "This question is outside my main area. I'm specialized in FAR, DFARS, EM385, OSHA, and related regulatory frameworks—ask me about those for more accurate answers."
                )

        return delta
    
    def _determine_next_agent(
        self, 
        state: GovGigState
    ) -> Literal["data_retrieval", "synthesizer", "end"]:
        """Determine the next agent based on routing decision."""
        next_agent = state.get('next_agent', 'data_retrieval')
        
        # For Phase 1, we only have data_retrieval implemented
        if next_agent == 'data_retrieval':
            return "data_retrieval"
        elif next_agent == 'end':
            return "end"
        else:
            # Other agents not yet implemented, go directly to synthesizer
            logger.warning(f"Agent {next_agent} not implemented, going to synthesizer")
            return "synthesizer"

    def _after_retrieval_routing(
        self, state: GovGigState
    ) -> Literal["letter_drafter", "synthesizer"]:
        """Route to letter_drafter when user asked for a draft; otherwise synthesizer."""
        if state.get("is_document_request"):
            return "letter_drafter"
        return "synthesizer"

    def _draft_letter(self, state: GovGigState) -> Dict[str, Any]:
        """Produce a full letter/document draft from retrieved documents. Used when is_document_request is True."""
        import time as _time
        _t0_node = _time.perf_counter()

        new_agent_path: list = []
        new_errors: list = []

        try:
            logger.info("Letter drafter: producing full draft")

            all_documents = state.get("retrieved_documents", [])
            offsets = state.get("run_offsets") or {}
            start_idx = int(offsets.get("retrieved_documents", 0) or 0)
            documents = all_documents[start_idx:] if start_idx > 0 else all_documents
            documents = [d for d in documents if not d.get("error")]

            if not documents:
                new_agent_path.append("LetterDrafter: No documents — returning fallback")
                return {
                    "generated_response": self._safe_fallback_message(),
                    "quality_metrics": {
                        "citation_coverage": 0.0,
                        "groundedness_score": 0.0,
                        "evidence_score": 0.0,
                        "quality_score": 0.0,
                        "low_confidence": True,
                    },
                    "low_confidence": True,
                    "agent_path": new_agent_path,
                }

            new_agent_path.append(
                f"LetterDrafter: Processing {len(documents)} retrieved documents"
            )
            evidence = self._evidence_summary(documents)
            formatted_docs = format_documents(documents, max_tokens=settings.RAG_TOKEN_LIMIT)
            system_prompt = get_letter_drafter_prompt(state, documents)
            user_message = (
                f"Retrieved Documents:\n{formatted_docs}\n\n"
                f"User Request: {state['query']}"
            )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
            response = self.letter_drafter_llm.invoke(
                messages, config={"tags": ["letter_drafter_token"]}
            )
            final_response = response.content or ""
            quality_metrics = self._assess_answer_quality(
                final_response, documents, evidence, state
            )

            guard_verdict = self.sovereign_guard.evaluate_response(
                response_text=final_response,
                query=state.get("query", ""),
                documents=documents,
            )
            if guard_verdict:
                quality_metrics["sovereign_guard"] = guard_verdict
                guard_action = str(guard_verdict.get("action") or "allow").lower()
                guard_should_block = bool(guard_verdict.get("should_block"))
                guard_reason = guard_verdict.get("reason") or "Automated guardrails flagged policy risk."

                if guard_action in {"warn", "block"} or guard_should_block:
                    quality_metrics["low_confidence"] = True

                if guard_should_block:
                    block_mode = str(settings.SOVEREIGN_GUARD_BLOCK_MODE or "soft").lower()
                    if block_mode == "hard":
                        final_response = self._safe_blocked_message()
                        new_agent_path.append(
                            "LetterDrafter: Sovereign guard BLOCK applied (hard mode)"
                        )
                    else:
                        final_response = self._with_safety_review_label(
                            final_response, str(guard_reason)
                        )
                        new_agent_path.append(
                            "LetterDrafter: Sovereign guard BLOCK applied (soft mode)"
                        )
                elif guard_action == "warn":
                    final_response = self._with_safety_review_label(
                        final_response, str(guard_reason)
                    )
                    new_agent_path.append("LetterDrafter: Sovereign guard WARN applied")
                else:
                    new_agent_path.append("LetterDrafter: Sovereign guard ALLOW")

            final_response = self._with_low_confidence_label(
                final_response, bool(quality_metrics["low_confidence"])
            )

            elapsed = _time.perf_counter() - _t0_node
            logger.info(f"[Telemetry] Node 'letter_drafter' completed in {elapsed:.2f}s")
            new_agent_path.append(
                f"LetterDrafter: quality score={quality_metrics['quality_score']:.2f}, "
                f"citation_coverage={quality_metrics['citation_coverage']:.2f}"
            )
            new_agent_path.append(
                f"LetterDrafter: Generated {len(final_response)}-char draft"
            )

            return {
                "generated_response": final_response,
                "confidence_score": float(evidence.get("avg_norm", 0.0)),
                "quality_metrics": quality_metrics,
                "low_confidence": bool(quality_metrics["low_confidence"]),
                "agent_path": new_agent_path,
            }

        except Exception as exc:
            logger.error(f"Letter drafter failed: {exc}", exc_info=True)
            new_errors.append(f"[LetterDrafter] {exc}")
            new_agent_path.append(f"LetterDrafter: ERROR — {exc}")
            return {
                "generated_response": "An error occurred while generating the draft. Please try again.",
                "quality_metrics": {
                    "citation_coverage": 0.0,
                    "groundedness_score": 0.0,
                    "evidence_score": 0.0,
                    "quality_score": 0.0,
                    "low_confidence": True,
                },
                "low_confidence": True,
                "agent_path": new_agent_path,
                "errors": new_errors,
            }

    def _synthesize_response(self, state: GovGigState) -> Dict[str, Any]:
        """Synthesize the final response from retrieved documents."""
        import time as _time
        _t0_node = _time.perf_counter()
        
        new_agent_path: list = []
        new_errors:     list = []

        try:
            logger.info("Synthesizing final response")

            # Read only the current-turn documents when checkpoint state is reused.
            all_documents = state.get("retrieved_documents", [])
            offsets = state.get("run_offsets") or {}
            start_idx = int(offsets.get("retrieved_documents", 0) or 0)
            documents = all_documents[start_idx:] if start_idx > 0 else all_documents

            # Treat retrieval failures as no documents (vector_search returns error dicts on failure)
            raw_count = len(documents)
            documents = [d for d in documents if not d.get("error")]
            if raw_count and not documents:
                new_agent_path.append("Synthesizer: Retrieved docs contained errors — treating as no documents")

            new_agent_path.append(
                f"Synthesizer: Processing {len(documents)} retrieved documents"
            )

            if not documents:
                logger.warning("No documents retrieved, providing fallback response")
                new_agent_path.append("Synthesizer: No documents — returning fallback")
                return {
                    "generated_response": self._safe_fallback_message(),
                    "quality_metrics": {
                        "citation_coverage": 0.0,
                        "groundedness_score": 0.0,
                        "evidence_score": 0.0,
                        "quality_score": 0.0,
                        "low_confidence": True,
                    },
                    "low_confidence": True,
                    "agent_path": new_agent_path,
                }

            evidence = self._evidence_summary(documents)
            new_agent_path.append(
                "Synthesizer: evidence "
                f"docs={int(evidence['doc_count'])}, "
                f"top={evidence['top_norm']:.2f}, avg={evidence['avg_norm']:.2f}"
            )

            formatted_docs = format_documents(documents, max_tokens=settings.RAG_TOKEN_LIMIT)
            system_prompt   = get_synthesizer_prompt(state, documents, evidence_summary=evidence)

            user_message = (
                f"Retrieved Documents:\n{formatted_docs}\n\n"
                f"User Query: {state['query']}"
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
            # Use a tag to distinguish this final user-facing LLM call from internal ones (like self-healing)
            response = self.synthesizer_llm.invoke(messages, config={"tags": ["synthesizer_token"]})
            quality_metrics = self._assess_answer_quality(response.content, documents, evidence, state)
            final_response = response.content
            # For procedural queries, ensure "Key Requirements" is replaced with "Recommended steps"
            if state.get("is_procedural", False) and "**Key Requirements**" in final_response and re.search(r"1\.\s+\*\*", final_response):
                final_response = re.sub(
                    r"\*\*Key Requirements\*\*:?\s*",
                    "**Recommended steps:** ",
                    final_response,
                    count=1,
                )
            hard_block_applied = False

            guard_verdict = self.sovereign_guard.evaluate_response(
                response_text=final_response,
                query=state.get("query", ""),
                documents=documents,
            )
            if guard_verdict:
                quality_metrics["sovereign_guard"] = guard_verdict
                guard_action = str(guard_verdict.get("action") or "allow").lower()
                guard_should_block = bool(guard_verdict.get("should_block"))
                guard_reason = guard_verdict.get("reason") or "Automated guardrails flagged policy risk."

                if guard_action in {"warn", "block"} or guard_should_block:
                    quality_metrics["low_confidence"] = True

                if guard_should_block:
                    block_mode = str(settings.SOVEREIGN_GUARD_BLOCK_MODE or "soft").lower()
                    if block_mode == "hard":
                        final_response = self._safe_blocked_message()
                        hard_block_applied = True
                        new_agent_path.append(
                            "Synthesizer: Sovereign guard BLOCK applied (hard mode)"
                        )
                    else:
                        final_response = self._with_safety_review_label(
                            final_response, str(guard_reason)
                        )
                        new_agent_path.append(
                            "Synthesizer: Sovereign guard BLOCK applied (soft mode)"
                        )
                elif guard_action == "warn":
                    final_response = self._with_safety_review_label(
                        final_response, str(guard_reason)
                    )
                    new_agent_path.append("Synthesizer: Sovereign guard WARN applied")
                else:
                    new_agent_path.append("Synthesizer: Sovereign guard ALLOW")

            if not hard_block_applied:
                final_response = self._with_low_confidence_label(
                    final_response,
                    bool(quality_metrics["low_confidence"]),
                )

            elapsed = _time.perf_counter() - _t0_node
            logger.info(f"[Telemetry] Node 'synthesizer' completed in {elapsed:.2f}s")

            logger.info(f"Response synthesized: {len(final_response)} characters")
            new_agent_path.append(
                f"Synthesizer: quality score={quality_metrics['quality_score']:.2f}, "
                f"citation_coverage={quality_metrics['citation_coverage']:.2f}, "
                f"groundedness={quality_metrics['groundedness_score']:.2f}"
            )
            if quality_metrics["low_confidence"]:
                new_agent_path.append("Synthesizer: low-confidence label applied")
            new_agent_path.append(
                f"Synthesizer: Generated {len(final_response)}-char response"
            )

            return {
                "generated_response": final_response,
                "confidence_score":   float(evidence.get("avg_norm", 0.0)),
                "quality_metrics":    quality_metrics,
                "low_confidence":     bool(quality_metrics["low_confidence"]),
                "agent_path":         new_agent_path,
            }

        except Exception as exc:
            logger.error(f"Synthesis failed: {exc}", exc_info=True)
            new_errors.append(f"[Synthesizer] Synthesis failed: {exc}")
            new_agent_path.append(f"Synthesizer: ERROR — {exc}")
            return {
                "generated_response": "An error occurred while generating the response. Please try again.",
                "quality_metrics": {
                    "citation_coverage": 0.0,
                    "groundedness_score": 0.0,
                    "evidence_score": 0.0,
                    "quality_score": 0.0,
                    "low_confidence": True,
                },
                "low_confidence": True,
                "agent_path": new_agent_path,
                "errors":     new_errors,
            }

    
    async def run_async(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Asynchronous non-streaming run for REST API."""
        context = context or {}

        # Configuration for persistence
        config = {"configurable": {"thread_id": context.get("thread_id", "default_thread")}}

        # ── Snapshot pre-existing accumulated list lengths ──────────────
        # When a thread_id is reused the checkpointer loads the previous
        # run's state.  operator.add appends new items, so after ainvoke
        # the result lists contain *old + new* entries.  We record the
        # old lengths here so we can slice them out of the response below.
        prev_state: Dict[str, Any] = {}
        if self.checkpointer is not None:
            try:
                snapshot = await self.app.aget_state(config)
                prev_state = snapshot.values if snapshot and snapshot.values else {}
            except Exception:
                prev_state = {}

        _acc_lists = {
            "agent_path": len(prev_state.get("agent_path", [])),
            "errors": len(prev_state.get("errors", [])),
            "thought_process": len(prev_state.get("thought_process", [])),
            "retrieved_documents": len(prev_state.get("retrieved_documents", [])),
            "regulation_types_used": len(prev_state.get("regulation_types_used", [])),
        }

        initial_state: GovGigState = {
            "messages": [],
            "query": query,
            "person_id": context.get("person_id"),
            "current_date": context.get("current_date", datetime.now().strftime("%A, %B %d, %Y")),
            "chat_history": context.get("history", []),
            "cot_enabled": context.get("cot", False),
            "run_offsets": _acc_lists,

            # Classifier fields
            "query_intent": None,
            "detected_clause_ref": None,
            "detected_reg_type": None,
            "is_procedural": None,
            "is_contract_co": None,
            "is_document_request": None,
            "document_request_type": None,

            "retrieved_documents": [],
            "tool_calls": [],
            "thought_process": [],
            "agent_path": [],
            "quality_metrics": None,
            "low_confidence": None,
            "ui_action": None,
            "regulation_types_used": [],
            "errors": []
        }

        # Run graph asynchronously
        result = await self.app.ainvoke(initial_state, config=config)

        # ── Slice accumulated lists to current-run only ────────────────
        agent_path    = result.get("agent_path", [])[_acc_lists["agent_path"]:]
        errors        = result.get("errors", [])[_acc_lists["errors"]:]
        thought_proc  = result.get("thought_process", [])[_acc_lists["thought_process"]:]
        documents     = result.get("retrieved_documents", [])[_acc_lists["retrieved_documents"]:]
        reg_types     = result.get("regulation_types_used", [])[_acc_lists["regulation_types_used"]:]

        response_text = result.get("generated_response") or ""

        # Force-clear documents for OUT_OF_SCOPE
        is_out_of_scope = (
            any("out_of_scope" in entry.lower() for entry in agent_path[-3:])
            or response_text.startswith("This question doesn't appear to be about")
        )

        return {
            "response":        response_text,
            "documents":       [] if is_out_of_scope else documents,
            "confidence":      result.get("confidence_score"),
            "quality_metrics": result.get("quality_metrics"),
            "low_confidence":  result.get("low_confidence"),
            "agent_path":      agent_path,
            "thought_process": thought_proc if result.get("cot_enabled") else None,
            "regulation_types": reg_types,
            "ui_action":       result.get("ui_action"),
            "errors":          errors,
        }

    async def run(
        self, 
        query: str, 
        context: Dict[str, Any] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Run the orchestrator and stream results.
        
        Args:
            query: User query
            context: Additional context (person_id, history, cot, etc.)
            
        Yields:
            Events during execution
        """
        context = context or {}

        logger.info(f"Starting orchestrator run for query: {query[:100]}...")

        # Configuration for persistence
        config = {"configurable": {"thread_id": context.get("thread_id", "default_thread")}}
        prev_state: Dict[str, Any] = {}
        if self.checkpointer is not None:
            try:
                snapshot = await self.app.aget_state(config)
                prev_state = snapshot.values if snapshot and snapshot.values else {}
            except Exception:
                prev_state = {}
        _acc_lists = {
            "agent_path": len(prev_state.get("agent_path", [])),
            "errors": len(prev_state.get("errors", [])),
            "thought_process": len(prev_state.get("thought_process", [])),
            "retrieved_documents": len(prev_state.get("retrieved_documents", [])),
            "regulation_types_used": len(prev_state.get("regulation_types_used", [])),
        }

        # Initialize state
        initial_state: GovGigState = {
            "messages": [],
            "query": query,
            "person_id": context.get("person_id"),
            "current_date": context.get("current_date", datetime.now().strftime("%A, %B %d, %Y")),
            "chat_history": context.get("history", []),
            "cot_enabled": context.get("cot", False),
            "run_offsets": _acc_lists,

            # Classifier fields
            "query_intent": None,
            "detected_clause_ref": None,
            "detected_reg_type": None,
            "is_procedural": None,
            "is_contract_co": None,
            "is_document_request": None,
            "document_request_type": None,

            "retrieved_documents": [],
            "tool_calls": [],
            "thought_process": [],
            "agent_path": [],
            "quality_metrics": None,
            "low_confidence": None,
            "ui_action": None,
            "regulation_types_used": [],
            "errors": []
        }
        
        accumulated_docs = []
        accumulated_response = ""
        final_complete_data = None
        
        try:
            # Stream events from graph using astream_events to catch tokens
            async for event in self.app.astream_events(initial_state, config=config, version="v1"):
                kind = event["event"]

                # Handle node completion (steps)
                if kind == "on_chain_end":
                    # Capture documents as soon as data_retrieval finishes
                    if event["name"] == "data_retrieval":
                        output = event["data"].get("output") or {}
                        node_docs = output.get("retrieved_documents", [])
                        if node_docs:
                            accumulated_docs.extend(node_docs)

                    # Filter for the final state of the whole graph (top-level LangGraph chain)
                    # This ensures we only send ONE 'complete' event at the very end.
                    if event["name"] == "LangGraph" and event["data"].get("output"):
                        final_state = event["data"]["output"]
                        
                        # Use accumulated docs if the state doesn't have them yet (common in streaming)
                        docs_to_send = final_state.get('retrieved_documents', [])[_acc_lists["retrieved_documents"]:]
                        if not docs_to_send:
                            docs_to_send = accumulated_docs

                        # Ensure response is never empty: use state first, then streamed tokens (for DB persist & UI)
                        response_text = final_state.get('generated_response') or accumulated_response or ""
                        final_complete_data = {
                            "response": response_text,
                            "documents": docs_to_send,
                            "confidence": final_state.get('confidence_score'),
                            "quality_metrics": final_state.get('quality_metrics'),
                            "low_confidence": final_state.get('low_confidence'),
                            "agent_path": final_state.get("agent_path", [])[_acc_lists["agent_path"]:],
                            "thought_process": final_state.get('thought_process', [])[_acc_lists["thought_process"]:] if final_state.get('cot_enabled') else None,
                            "regulation_types": final_state.get('regulation_types_used', [])[_acc_lists["regulation_types_used"]:],
                            "ui_action": final_state.get('ui_action'),
                            "errors": final_state.get('errors', [])[_acc_lists["errors"]:],
                        }
                        # We don't yield yet, we wait for the stream to naturally progress or finish

                elif kind == "on_chat_model_stream":
                    # Capture tokens ONLY from the synthesizer (final response)
                    # This prevents tokens from self-healing/expansion leaking into the UI
                    tags = event.get("tags", [])
                    if "synthesizer_token" in tags:
                        content = event["data"]["chunk"].content
                        if content:
                            accumulated_response += content
                            yield {
                                "type": "token",
                                "data": content
                            }

            # After the stream loop finishes, yield the collected complete event
            if final_complete_data:
                yield {
                    "type": "complete",
                    "data": final_complete_data
                }
            
            logger.info("Orchestrator run completed successfully")
            
        except Exception as e:
            logger.error(f"Orchestrator run failed: {e}", exc_info=True)
            yield {
                "type": "error",
                "data": {
                    "error": str(e),
                    "message": "An error occurred during processing"
                }
            }
    
    def run_sync(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Synchronous run for testing."""
        context = context or {}

        # Configuration for persistence
        config = {"configurable": {"thread_id": context.get("thread_id", "default_thread")}}
        prev_state: Dict[str, Any] = {}
        if self.checkpointer is not None:
            try:
                snapshot = self.app.get_state(config)
                prev_state = snapshot.values if snapshot and snapshot.values else {}
            except Exception:
                prev_state = {}
        _acc_lists = {
            "agent_path": len(prev_state.get("agent_path", [])),
            "errors": len(prev_state.get("errors", [])),
            "thought_process": len(prev_state.get("thought_process", [])),
            "retrieved_documents": len(prev_state.get("retrieved_documents", [])),
            "regulation_types_used": len(prev_state.get("regulation_types_used", [])),
        }

        initial_state: GovGigState = {
            "messages": [],
            "query": query,
            "person_id": context.get("person_id"),
            "current_date": context.get("current_date", datetime.now().strftime("%A, %B %d, %Y")),
            "chat_history": context.get("history", []),
            "cot_enabled": context.get("cot", False),
            "run_offsets": _acc_lists,

            # Classifier fields
            "query_intent": None,
            "detected_clause_ref": None,
            "detected_reg_type": None,
            "is_procedural": None,
            "is_contract_co": None,
            "is_document_request": None,
            "document_request_type": None,

            "retrieved_documents": [],
            "tool_calls": [],
            "thought_process": [],
            "agent_path": [],
            "quality_metrics": None,
            "low_confidence": None,
            "ui_action": None,
            "regulation_types_used": [],
            "errors": []
        }

        result = self.app.invoke(initial_state, config=config)
        agent_path = result.get("agent_path", [])[_acc_lists["agent_path"]:]
        errors = result.get("errors", [])[_acc_lists["errors"]:]
        thought_proc = result.get("thought_process", [])[_acc_lists["thought_process"]:]
        documents = result.get("retrieved_documents", [])[_acc_lists["retrieved_documents"]:]
        reg_types = result.get("regulation_types_used", [])[_acc_lists["regulation_types_used"]:]

        return {
            "response":        result.get("generated_response"),
            "documents":       documents,
            "confidence":      result.get("confidence_score"),
            "quality_metrics": result.get("quality_metrics"),
            "low_confidence":  result.get("low_confidence"),
            "agent_path":      agent_path,
            "thought_process": thought_proc if result.get("cot_enabled") else None,
            "regulation_types": reg_types,
            "ui_action":       result.get("ui_action"),
            "errors":          errors,
        }
