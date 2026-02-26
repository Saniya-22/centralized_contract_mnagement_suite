"""Data Retrieval Agent for regulatory document search.

## LangGraph contract

Every public method registered as a LangGraph node returns a *delta dict*
(only the fields that changed), NOT the full mutated state. This is the correct
contract when fields are declared with Annotated[..., reducer] in GovGigState.

## Intent-based dispatch (latency optimisation)

The agent first checks `state["query_intent"]` set by the deterministic
QueryClassifier in the router node. When the intent is unambiguous:

  CLAUSE_LOOKUP      → directly calls get_clause_by_reference, no LLM call
  REGULATION_SEARCH  → directly calls search_regulations, no LLM call
  (unknown/ambiguous) → falls back to one gpt-4o tool-selector call

This removes the LLM tool-selector call (~1,500–3,000 ms) for the vast
majority of queries, cutting those queries from 5 LLM calls to 3.
"""

from typing import List, Dict, Any, Optional
import json
import logging

from src.agents.base import BaseAgent
from src.agents.prompts import get_data_retrieval_prompt
from src.state.graph_state import GovGigState
from src.tools.vector_search import VectorSearchTool
from src.tools.query_classifier import QueryIntent

logger = logging.getLogger(__name__)


class DataRetrievalAgent(BaseAgent):
    """Agent specialised in retrieving regulatory documents."""

    def __init__(self):
        super().__init__(name="DataRetrievalAgent")

        self.vector_search_tool = VectorSearchTool()
        all_tools = self.vector_search_tool.as_langchain_tools()

        # Only bound when we genuinely need LLM tool selection (ambiguous queries)
        self.llm_with_tools = self.llm.bind_tools(all_tools)

        logger.info(
            f"DataRetrievalAgent initialised with {len(all_tools)} tools: "
            + ", ".join(t.name for t in all_tools)
        )

    def get_system_prompt(self, state: GovGigState) -> str:
        return get_data_retrieval_prompt(state)

    # ── LangGraph node ─────────────────────────────────────────────────────────

    def run(self, state: GovGigState) -> Dict[str, Any]:
        """Execute document retrieval and return a *delta dict*.

        Dispatch priority:
          1. query_intent == CLAUSE_LOOKUP  → direct clause lookup, 0 LLM calls
          2. query_intent == REGULATION_SEARCH → direct search, 0 LLM calls
          3. unknown intent                 → 1 gpt-4o tool-selector call (fallback)

        Returns:
            dict with only the *new* values to be merged via operator.add reducers.
        """
        # ── Local delta accumulators ──────────────────────────────────────────
        new_docs:       List[Dict[str, Any]] = []
        new_tool_calls: List[Dict[str, Any]] = []
        new_reg_types:  List[str]            = []
        new_agent_path: List[str]            = []
        new_thoughts:   List[str]            = []
        new_errors:     List[str]            = []

        def _log(msg: str):
            entry = f"{self.name}: {msg}"
            logger.info(f"[{self.name}] {msg}")
            new_agent_path.append(entry)

        def _think(thought: str):
            if state.get("cot_enabled", False):
                new_thoughts.append(f"[{self.name}] {thought}")

        def _add_reg_type(reg_type: Optional[str]):
            if reg_type and reg_type not in new_reg_types:
                new_reg_types.append(reg_type)

        try:
            _log("Starting document retrieval")

            # ── Intent-based fast dispatch ────────────────────────────────────
            # Read classifier output written by the router node
            query_intent      = state.get("query_intent")
            detected_clause   = state.get("detected_clause_ref")
            detected_reg_type = state.get("detected_reg_type")

            if query_intent == QueryIntent.CLAUSE_LOOKUP and detected_clause:
                # ── Path A: Direct clause lookup (0 LLM calls) ───────────────
                _log(f"Intent=clause_lookup: direct lookup for '{detected_clause}'")
                _think(f"Clause reference detected: '{detected_clause}'. Using direct lookup.")
                new_docs, new_tool_calls, new_reg_types = self._do_clause_lookup(
                    detected_clause, new_docs, new_tool_calls, new_reg_types, _log
                )

            elif query_intent == QueryIntent.REGULATION_SEARCH:
                # ── Path B: Direct regulation search (0 LLM calls) ───────────
                _log("Intent=regulation_search: calling search_regulations directly")
                _think(f"Regulatory query detected. Searching with type filter: {detected_reg_type}")
                new_docs, new_tool_calls, new_reg_types = self._do_regulation_search(
                    query=state["query"],
                    regulation_type=detected_reg_type,
                    new_docs=new_docs,
                    new_tool_calls=new_tool_calls,
                    new_reg_types=new_reg_types,
                    _log=_log,
                    _think=_think,
                )

            else:
                # ── Path C: Ambiguous intent — LLM tool-selector (1 call) ────
                _log(f"Intent='{query_intent}' (ambiguous): using LLM tool-selector")
                _think("Query intent is ambiguous. Using LLM to select the right tool.")
                new_docs, new_tool_calls, new_reg_types = self._do_llm_dispatch(
                    state, new_docs, new_tool_calls, new_reg_types, _log, _think
                )

            _log(f"Completed retrieval: {len(new_docs)} new documents")

        except Exception as exc:
            msg = f"Data retrieval failed: {exc}"
            logger.error(f"[{self.name}] {msg}", exc_info=True)
            new_errors.append(f"[{self.name}] {msg}")
            new_agent_path.append(f"{self.name}: ERROR — {msg}")

        # ── Return ONLY the delta ─────────────────────────────────────────────
        delta: Dict[str, Any] = {
            "retrieved_documents":   new_docs,
            "tool_calls":            new_tool_calls,
            "regulation_types_used": new_reg_types,
            "agent_path":            new_agent_path,
        }
        if new_thoughts:
            delta["thought_process"] = new_thoughts
        if new_errors:
            delta["errors"] = new_errors
        return delta

    # ── Private dispatch helpers ───────────────────────────────────────────────

    def _do_clause_lookup(
        self,
        clause_reference: str,
        new_docs: list, new_tool_calls: list, new_reg_types: list,
        _log,
    ):
        """Direct clause lookup — no LLM involved."""
        result = self.vector_search_tool.get_clause_by_reference.invoke(
            {"clause_reference": clause_reference}
        )
        if result.get("found") and result.get("clause"):
            clause = result["clause"]
            doc = {
                "rank": 1,
                "content":         result.get("context", clause.get("text", "")),
                "source":          clause.get("source", ""),
                "regulation_type": clause.get("source", "Unknown"),
                "section":         clause.get("part", "N/A"),
                "chunk_index":     None,
                "score":           1.0,
                "retrieval_methods": ["clause_lookup"],
                "metadata":        clause,
            }
            new_docs.append(doc)
            if clause.get("source") and clause["source"] not in new_reg_types:
                new_reg_types.append(clause["source"])

        new_tool_calls.append({
            "agent": self.name,
            "tool":  "get_clause_by_reference",
            "args":  {"clause_reference": clause_reference},
            "found": result.get("found", False),
        })
        _log(f"Clause lookup: found={result.get('found')}")
        return new_docs, new_tool_calls, new_reg_types

    def _do_regulation_search(
        self,
        query: str,
        regulation_type: Optional[str],
        new_docs: list, new_tool_calls: list, new_reg_types: list,
        _log, _think,
    ):
        """Direct hybrid search — no LLM tool-selector involved."""
        tool_args = {"query": query, "k": 10, "search_mode": "hybrid"}
        if regulation_type:
            tool_args["regulation_type"] = regulation_type

        results = self.vector_search_tool.search_regulations.invoke(tool_args)

        new_docs.extend(results)
        new_tool_calls.append({
            "agent":        self.name,
            "tool":         "search_regulations",
            "args":         tool_args,
            "result_count": len(results),
        })
        for result in results:
            rt = result.get("regulation_type")
            if rt and rt not in new_reg_types:
                new_reg_types.append(rt)

        _log(f"Retrieved {len(results)} documents via search_regulations")
        if results:
            _think(
                f"Found {len(results)} relevant documents. "
                f"Top result from {results[0].get('regulation_type', 'Unknown')} "
                f"with score {results[0].get('score', 0):.3f}. "
                f"Methods: {results[0].get('retrieval_methods', [])}"
            )
        return new_docs, new_tool_calls, new_reg_types

    def _do_llm_dispatch(
        self,
        state: GovGigState,
        new_docs: list, new_tool_calls: list, new_reg_types: list,
        _log, _think,
    ):
        """Fallback: one LLM call to select the right tool for ambiguous queries."""
        messages = self._create_messages(state)
        response  = self.llm_with_tools.invoke(messages)

        if not response.tool_calls:
            _log("No tool calls made — agent provided direct response")
            if response.content:
                # Store content as a pseudo-document so synthesizer sees it
                new_docs.append({
                    "rank": 1, "content": response.content,
                    "source": "agent", "regulation_type": "N/A",
                    "section": "N/A", "chunk_index": None, "score": 0.5,
                    "retrieval_methods": ["direct_response"], "metadata": {},
                })
            return new_docs, new_tool_calls, new_reg_types

        _log(f"Processing {len(response.tool_calls)} LLM tool calls")
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            logger.info(f"LLM tool call: {tool_name} args={json.dumps(tool_args)}")

            if tool_name == "search_regulations":
                new_docs, new_tool_calls, new_reg_types = self._do_regulation_search(
                    query=tool_args.get("query", state["query"]),
                    regulation_type=tool_args.get("regulation_type"),
                    new_docs=new_docs,
                    new_tool_calls=new_tool_calls,
                    new_reg_types=new_reg_types,
                    _log=_log,
                    _think=_think,
                )
            elif tool_name == "get_clause_by_reference":
                new_docs, new_tool_calls, new_reg_types = self._do_clause_lookup(
                    clause_reference=tool_args.get("clause_reference", ""),
                    new_docs=new_docs,
                    new_tool_calls=new_tool_calls,
                    new_reg_types=new_reg_types,
                    _log=_log,
                )

        return new_docs, new_tool_calls, new_reg_types
