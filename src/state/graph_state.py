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
    chat_history: List[Dict[str, str]]
    
    # Agent outputs
    retrieved_documents: Annotated[List[Dict[str, Any]], operator.add]
    analysis_result: Optional[str]
    generated_response: Optional[str]
    
    # Routing and control flow
    next_agent: Optional[str]
    reasoning: Optional[str]
    # Classifier output — set by the deterministic router, consumed by agents
    query_intent: Optional[str]        # QueryIntent enum value (str)
    detected_clause_ref: Optional[str] # e.g. "FAR 52.236-2" — skips tool-selector LLM
    detected_reg_type: Optional[str]   # e.g. "FAR", "DFARS", "EM385"
    agent_path: Annotated[List[str], operator.add]  # Track agent execution path
    
    # Chain-of-Thought
    cot_enabled: bool
    thought_process: Annotated[List[str], operator.add]
    
    # Tool tracking
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]
    
    # Metadata
    confidence_score: Optional[float]
    regulation_types_used: Annotated[List[str], operator.add]
    
    # Error handling
    errors: Annotated[List[str], operator.add]
