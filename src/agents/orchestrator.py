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
from src.config import settings

logger = logging.getLogger(__name__)


class GovGigOrchestrator:
    """Main orchestrator for the GovGig multi-agent system."""
    
    def __init__(self, checkpointer=None):
        logger.info("Initializing GovGigOrchestrator")

        # Synthesizer: use gpt-4o-mini (SYNTHESIZER_MODEL) instead of gpt-4o.
        self.synthesizer_llm = ChatOpenAI(
            model=settings.SYNTHESIZER_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=settings.TEMPERATURE,
            streaming=True,
            max_tokens=800,
        )

        # Initialize agents
        self.data_retrieval = DataRetrievalAgent()

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

    @staticmethod
    def _has_citation_markers(text: str) -> bool:
        if not text:
            return False
        return bool(
            re.search(r"\b(FAR|DFARS|EM\s*385)\b\s*\d", text, flags=re.IGNORECASE)
            or re.search(r"\[[^\]]+\]", text)
        )

    @staticmethod
    def _safe_fallback_message() -> str:
        return (
            "I don’t have sufficient high-confidence evidence in the retrieved regulatory text "
            "to provide a reliable answer. Please refine the query with regulation type/section "
            "(e.g., FAR/DFARS/EM385 clause), and I’ll return a fully cited response."
        )

    def _evidence_thresholds_for_state(self, state: GovGigState) -> tuple[int, float, float]:
        """Resolve evidence thresholds with intent-aware guardrail tuning."""
        min_docs = int(settings.PILOT_MIN_DOCS)
        min_top = float(settings.PILOT_MIN_TOP_SCORE)
        min_avg = float(settings.PILOT_MIN_AVG_SCORE)

        query_intent = state.get("query_intent")
        detected_clause_ref = state.get("detected_clause_ref")
        is_clause_lookup = (
            query_intent == QueryIntent.CLAUSE_LOOKUP.value
            or bool(detected_clause_ref)
        )

        # Clause lookups frequently return one strong, exact match document.
        # Requiring >=3 docs for this path creates false negative fallbacks.
        if is_clause_lookup:
            min_docs = 1

        return min_docs, min_top, min_avg
    
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
        if intent == QueryIntent.OUT_OF_SCOPE:
            next_agent = "end"
        else:
            next_agent = "data_retrieval"

        elapsed = _time.perf_counter() - _t0_node
        logger.info(f"[Telemetry] Node 'router' completed in {elapsed:.2f}s")

        logger.info(
            f"[Router] intent={intent.value}, next={next_agent}, "
            f"clause_ref={classification.clause_reference}"
        )

        delta: Dict[str, Any] = {
            "next_agent":           next_agent,
            "reasoning":            f"QueryClassifier: intent={intent.value}",
            "query_intent":         intent.value,
            "detected_clause_ref":  classification.clause_reference,
            "detected_reg_type":    classification.regulation_type,
            "agent_path":           [
                f"Router: intent={intent.value} "
                + (f"ref='{classification.clause_reference}' " if classification.clause_reference else "")
                + f"→ {next_agent}"
            ],
        }
        if intent == QueryIntent.OUT_OF_SCOPE:
            delta["generated_response"] = "I can only answer questions related to FAR, DFARS, or EM385 regulations. Please provide a relevant query."
            
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
                    "agent_path": new_agent_path,
                }

            evidence = self._evidence_summary(documents)
            new_agent_path.append(
                "Synthesizer: evidence "
                f"docs={int(evidence['doc_count'])}, "
                f"top={evidence['top_norm']:.2f}, avg={evidence['avg_norm']:.2f}"
            )
            if settings.PILOT_SAFE_MODE:
                min_docs, min_top, min_avg = self._evidence_thresholds_for_state(state)
                weak = (
                    evidence["doc_count"] < min_docs
                    or evidence["top_norm"] < min_top
                    or evidence["avg_norm"] < min_avg
                )
                if weak:
                    logger.warning("Pilot safe mode blocked low-evidence synthesis")
                    new_agent_path.append(
                        "Synthesizer: blocked due to low-evidence guardrail "
                        f"(need docs>={min_docs}, top>={min_top:.2f}, avg>={min_avg:.2f})"
                    )
                    return {
                        "generated_response": self._safe_fallback_message(),
                        "confidence_score": float(evidence["avg_norm"]),
                        "agent_path": new_agent_path,
                    }

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

            if settings.PILOT_SAFE_MODE and not self._has_citation_markers(response.content):
                logger.warning("Pilot safe mode rejected uncited synthesis output")
                new_agent_path.append("Synthesizer: blocked due to missing citation markers")
                return {
                    "generated_response": self._safe_fallback_message(),
                    "confidence_score": float(evidence["avg_norm"]),
                    "agent_path": new_agent_path,
                }
            
            elapsed = _time.perf_counter() - _t0_node
            logger.info(f"[Telemetry] Node 'synthesizer' completed in {elapsed:.2f}s")

            avg_score = (
                sum(doc.get('score', 0) for doc in documents) / len(documents)
                if documents else 0.0
            )

            logger.info(f"Response synthesized: {len(response.content)} characters")
            new_agent_path.append(
                f"Synthesizer: Generated {len(response.content)}-char response"
            )

            return {
                "generated_response": response.content,
                "confidence_score":   float(avg_score),
                "agent_path":         new_agent_path,
            }

        except Exception as exc:
            logger.error(f"Synthesis failed: {exc}", exc_info=True)
            new_errors.append(f"[Synthesizer] Synthesis failed: {exc}")
            new_agent_path.append(f"Synthesizer: ERROR — {exc}")
            return {
                "generated_response": "An error occurred while generating the response. Please try again.",
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
            "agent_path":      result.get("agent_path", []),
            "thought_process": result.get("thought_process", []) if result.get("cot_enabled") else None,
            "regulation_types": result.get("regulation_types_used", []),
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
                            "agent_path": final_state.get('agent_path', []),
                            "thought_process": final_state.get('thought_process', []) if final_state.get('cot_enabled') else None,
                            "regulation_types": final_state.get('regulation_types_used', []),
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
            "agent_path":      result.get("agent_path", []),
            "thought_process": result.get("thought_process", []) if result.get("cot_enabled") else None,
            "regulation_types": result.get("regulation_types_used", []),
            "errors":          result.get("errors", []),
        }
