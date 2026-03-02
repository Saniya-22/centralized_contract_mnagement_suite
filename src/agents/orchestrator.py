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
from src.agents.prompts import get_synthesizer_prompt
from src.tools.llm_tools import format_documents
from src.tools.query_classifier import classify_query, QueryIntent
from src.services.sovereign_guard import SovereignGuard
from src.config import settings

logger = logging.getLogger(__name__)


class GovGigOrchestrator:
    """Main orchestrator for the GovGig multi-agent system."""
    _CITATION_RE = re.compile(r"(\b(FAR|DFARS|EM\s*385)\b\s*\d+|\[[^\]]+\])", flags=re.IGNORECASE)
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
        "Low confidence notice: Retrieved evidence may be incomplete for parts of this answer. "
        "Please verify the cited clauses before final use.\n\n"
    )
    _SAFETY_REVIEW_LABEL = "Safety review notice: {reason}\n\n"
    
    def __init__(self, checkpointer=None):
        logger.info("Initializing GovGigOrchestrator")

        # Synthesizer: use gpt-4o-mini (SYNTHESIZER_MODEL) instead of gpt-4o.
        self.synthesizer_llm = ChatOpenAI(
            model=settings.SYNTHESIZER_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=settings.TEMPERATURE,
            streaming=True,
            max_tokens=400,
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
        if score > 1.0:
            return min(score / 10.0, 1.0)
        if score <= 0.1:
            return min(score * 20.0, 1.0)
        return min(score, 1.0)

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
    def _citation_coverage(cls, text: str) -> float:
        claims = cls._extract_claim_units(text)
        if not claims:
            return 1.0
        cited = sum(1 for claim in claims if cls._CITATION_RE.search(claim))
        return cited / len(claims)

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
        overlap = len(response_tokens & doc_tokens)
        return overlap / len(response_tokens)

    def _assess_answer_quality(
        self,
        response_text: str,
        documents: list[dict],
        evidence: dict[str, float],
        state: GovGigState,
    ) -> dict[str, float | bool]:
        citation_coverage = self._citation_coverage(response_text)
        groundedness = self._groundedness_score(response_text, documents)
        evidence_avg = float(evidence.get("avg_norm", 0.0))
        is_clause_lookup = (
            state.get("query_intent") == QueryIntent.CLAUSE_LOOKUP.value
            or bool(state.get("detected_clause_ref"))
        )
        min_docs = 1.0 if is_clause_lookup else 2.0
        quality_score = (
            0.40 * evidence_avg
            + 0.35 * groundedness
            + 0.25 * citation_coverage
        )

        # Soft guardrails: annotate low confidence instead of blocking.
        low_confidence = bool(
            evidence.get("doc_count", 0.0) < min_docs
            or evidence_avg < 0.20
            or groundedness < 0.45
            or citation_coverage < 0.45
            or quality_score < 0.55
        )

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
        workflow.add_node("synthesizer", self._synthesize_response)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self._determine_next_agent,
            {
                "data_retrieval": "data_retrieval",
                "synthesizer": "synthesizer",  # Direct to synthesizer for simple queries
                "end": END
            }
        )
        
        # Data retrieval always goes to synthesizer
        workflow.add_edge("data_retrieval", "synthesizer")
        
        # Synthesizer always ends
        workflow.add_edge("synthesizer", END)
        
        logger.info("LangGraph workflow built")
        return workflow
    
    def _route_query(self, state: GovGigState) -> Dict[str, Any]:
        """Route the query using the deterministic QueryClassifier."""
        import time as _time
        _t0_node = _time.perf_counter()
        
        query = state["query"]
        logger.info(f"[Router] Classifying query: '{query[:100]}...'")

        classification = classify_query(query)
        intent         = classification.intent

        # Map intent to graph routing decision
        next_agent = "end" if intent == QueryIntent.OUT_OF_SCOPE else "data_retrieval"

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
            "agent_path": [
                f"Router: intent={intent.value} conf={classification.confidence:.2f} "
                + (f"ref='{classification.clause_reference}' " if classification.clause_reference else "")
                + f"→ {next_agent}"
            ],
        }

        if intent == QueryIntent.OUT_OF_SCOPE:
            delta["generated_response"] = (
                "I can only answer questions related to government acquisition regulations "
                "(FAR, DFARS, EM385, OSHA, etc.). Please provide a relevant query."
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
    
    def _synthesize_response(self, state: GovGigState) -> Dict[str, Any]:
        """Synthesize the final response from retrieved documents."""
        import time as _time
        _t0_node = _time.perf_counter()
        
        new_agent_path: list = []
        new_errors:     list = []

        try:
            logger.info("Synthesizing final response")

            # Read accumulated documents from state (read-only — no mutation)
            documents = state.get('retrieved_documents', [])
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
            system_prompt   = get_synthesizer_prompt(state, documents)

            user_message = (
                f"Retrieved Documents:\n{formatted_docs}\n\n"
                f"User Query: {state['query']}\n\n"
                "Answer very concisely with citations. Get straight to the point."
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
            response = self.synthesizer_llm.invoke(messages)
            quality_metrics = self._assess_answer_quality(response.content, documents, evidence, state)
            final_response = response.content
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

            avg_score = (
                sum(doc.get('score', 0) for doc in documents) / len(documents)
                if documents else 0.0
            )

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
                "confidence_score":   float(avg_score),
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

        initial_state: GovGigState = {
            "messages": [],
            "query": query,
            "person_id": context.get("person_id"),
            "current_date": context.get("current_date", datetime.now().strftime("%A, %B %d, %Y")),
            "chat_history": context.get("history", []),
            "cot_enabled": context.get("cot", False),

            # Classifier fields
            "query_intent": None,
            "detected_clause_ref": None,
            "detected_reg_type": None,

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

        # Configuration for persistence
        config = {"configurable": {"thread_id": context.get("thread_id", "default_thread")}}

        # Run graph asynchronously
        result = await self.app.ainvoke(initial_state, config=config)

        return {
            "response":        result.get("generated_response"),
            "documents":       result.get("retrieved_documents", []),
            "confidence":      result.get("confidence_score"),
            "quality_metrics": result.get("quality_metrics"),
            "low_confidence":  result.get("low_confidence"),
            "agent_path":      result.get("agent_path", []),
            "thought_process": result.get("thought_process", []) if result.get("cot_enabled") else None,
            "regulation_types": result.get("regulation_types_used", []),
            "ui_action":       result.get("ui_action"),
            "errors":          result.get("errors", []),
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
        
        # Initialize state
        initial_state: GovGigState = {
            "messages": [],
            "query": query,
            "person_id": context.get("person_id"),
            "current_date": context.get("current_date", datetime.now().strftime("%A, %B %d, %Y")),
            "chat_history": context.get("history", []),
            "cot_enabled": context.get("cot", False),
            
            # Classifier fields
            "query_intent": None,
            "detected_clause_ref": None,
            "detected_reg_type": None,
            
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
        
        logger.info(f"Starting orchestrator run for query: {query[:100]}...")
        
        # Configuration for persistence
        config = {"configurable": {"thread_id": context.get("thread_id", "default_thread")}}
        
        try:
            # Stream events from graph using astream_events to catch tokens
            async for event in self.app.astream_events(initial_state, config=config, version="v1"):
                kind = event["event"]

                # Handle node completion (steps)
                if kind == "on_chain_end" and event["name"] == "LangGraph":
                    # This is the final state after the graph finishes
                    final_state = event["data"]["output"]
                    yield {
                        "type": "complete",
                        "data": {
                            "response": final_state.get('generated_response'),
                            "documents": final_state.get('retrieved_documents', []),
                            "confidence": final_state.get('confidence_score'),
                            "quality_metrics": final_state.get('quality_metrics'),
                            "low_confidence": final_state.get('low_confidence'),
                            "agent_path": final_state.get('agent_path', []),
                            "thought_process": final_state.get('thought_process', []) if final_state.get('cot_enabled') else None,
                            "regulation_types": final_state.get('regulation_types_used', []),
                            "ui_action": final_state.get('ui_action'),
                            "errors": final_state.get('errors', [])
                        }
                    }

                elif kind == "on_chat_model_stream":
                    # Capture tokens from any streaming model (e.g., synthesizer)
                    content = event["data"]["chunk"].content
                    if content:
                        yield {
                            "type": "token",
                            "data": content
                        }

                elif kind == "on_chain_end":
                    # Handle node boundaries (like the old 'step' event)
                    # We only care about nodes we defined
                    node_name = event.get("name")
                    if node_name in ["router", "data_retrieval", "synthesizer"]:
                        yield {
                            "type": "step",
                            "node": node_name,
                            "data": event["data"]["output"]
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

        initial_state: GovGigState = {
            "messages": [],
            "query": query,
            "person_id": context.get("person_id"),
            "current_date": context.get("current_date", datetime.now().strftime("%A, %B %d, %Y")),
            "chat_history": context.get("history", []),
            "cot_enabled": context.get("cot", False),

            # Classifier fields
            "query_intent": None,
            "detected_clause_ref": None,
            "detected_reg_type": None,

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

        # Configuration for persistence
        config = {"configurable": {"thread_id": context.get("thread_id", "default_thread")}}

        # Run graph synchronously — delta returns + operator.add reducers ensure
        # each document is accumulated exactly once, so no dedup needed here.
        result = self.app.invoke(initial_state, config=config)

        return {
            "response":        result.get("generated_response"),
            "documents":       result.get("retrieved_documents", []),
            "confidence":      result.get("confidence_score"),
            "quality_metrics": result.get("quality_metrics"),
            "low_confidence":  result.get("low_confidence"),
            "agent_path":      result.get("agent_path", []),
            "thought_process": result.get("thought_process", []) if result.get("cot_enabled") else None,
            "regulation_types": result.get("regulation_types_used", []),
            "ui_action":       result.get("ui_action"),
            "errors":          result.get("errors", []),
        }
