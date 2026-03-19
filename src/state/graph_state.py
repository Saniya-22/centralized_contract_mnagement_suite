"""State definitions for LangGraph multi-agent system"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph import MessagesState
import operator

from src.tools.query_classifier import QueryIntent


class GovGigState(MessagesState):
    """State for the GovGig multi-agent system.
    
    Extends MessagesState to include conversation history and custom fields.
    """
    
    # Input
    query: str
    person_id: Optional[str]
    current_date: str
    chat_history: List[Dict[str, Any]]
    # Offsets for accumulator fields when checkpoint state is reused across turns.
    run_offsets: Optional[Dict[str, int]]
    
    # Agent outputs
    retrieved_documents: Annotated[List[Dict[str, Any]], operator.add]
    analysis_result: Optional[str]
    generated_response: Optional[str]
    ui_action: Optional[Dict[str, Any]] # e.g. {"type": "navigate", "route": "/dashboard"}
    
    # Routing and control flow
    next_agent: Optional[str]
    reasoning: Optional[str]
    # Classifier output — set by the deterministic router, consumed by agents
    query_intent: Optional[str]        # QueryIntent enum value (str)
    detected_clause_ref: Optional[str] # e.g. "FAR 52.236-2" — skips tool-selector LLM
    detected_reg_type: Optional[str]   # e.g. "FAR", "DFARS", "EM385"
    is_procedural: Optional[bool]       # steps / what to do (from classifier)
    is_contract_co: Optional[bool]      # frequency/schedule → contract/CO (from classifier)
    is_document_request: Optional[bool]  # True only for letter-type; checklist/form use synthesis (from classifier)
    document_request_type: Optional[str]  # "letter" | "checklist" | "form" | None
    is_comparison: Optional[bool]              # "REA vs change order", "type 1 vs type 2" (from classifier)
    is_construction_lifecycle: Optional[bool]  # commissioning, punchlist, closeout (from classifier)
    is_schedule_risk: Optional[bool]           # schedule/delay risk analysis (from classifier)
    agent_path: Annotated[List[str], operator.add]  # Track agent execution path
    
    # Chain-of-Thought
    cot_enabled: bool
    thought_process: Annotated[List[str], operator.add]
    
    # Tool tracking
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]
    
    # Metadata
    mode: Optional[str]  # "grounded" | "copilot" | "refusal" | "clarify"
    confidence_score: Optional[float]
    quality_metrics: Optional[Dict[str, Any]]
    low_confidence: Optional[bool]
    regulation_types_used: Annotated[List[str], operator.add]

    # Reflection tracking (explicit flags — never inferred from strings)
    reflection_triggered: Optional[bool]
    reflection_retries: Optional[int]
    quality_gate_healing: Optional[bool]  # True only on the step that triggered healing; False on no-op passes
    
    # Error handling
    errors: Annotated[List[str], operator.add]
