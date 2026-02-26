"""Base agent class with common functionality"""

from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List, Dict, Any
import logging

from src.config import settings
from src.state.graph_state import GovGigState

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, name: str, model_name: str = None, temperature: float = None):
        self.name = name
        self.model_name = model_name or settings.MODEL_NAME
        self.temperature = temperature if temperature is not None else settings.TEMPERATURE

        # Non-streaming LLM for synchronous .invoke() calls.
        # streaming=True must NOT be used with .invoke() — it opens a streaming
        # connection but reconstructs the response synchronously, adding latency
        # with zero benefit.  The WebSocket synthesizer uses self.streaming_llm.
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=settings.OPENAI_API_KEY,
            temperature=self.temperature,
            streaming=False,   # correct for .invoke() path
        )

        logger.info(f"Initialized {name} with model {self.model_name}")
    
    @abstractmethod
    def get_system_prompt(self, state: GovGigState) -> str:
        """Get the system prompt for this agent.
        
        Args:
            state: Current graph state
            
        Returns:
            System prompt string
        """
        pass
    
    @abstractmethod
    def run(self, state: GovGigState) -> Dict[str, Any]:
        """Execute the agent logic.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated graph state dictionary delta
        """
        pass
    
    def _log_step(self, state: GovGigState, message: str) -> Dict[str, Any]:
        """Log an agent step and return state delta."""
        logger.info(f"[{self.name}] {message}")
        
        # Return state delta for agent path
        return {"agent_path": [f"{self.name}: {message}"]}
    
    def _add_thought(self, state: GovGigState, thought: str) -> Dict[str, Any]:
        """Add a thought to the state (for CoT mode) and return state delta."""
        if state.get("cot_enabled", False):
            logger.debug(f"[{self.name}] Thought: {thought}")
            return {"thought_process": [f"[{self.name}] {thought}"]}
        return {}
    
    def _add_error(self, state: GovGigState, error: str) -> Dict[str, Any]:
        """Add an error to the state and return state delta."""
        logger.error(f"[{self.name}] Error: {error}")
        return {"errors": [f"[{self.name}] {error}"]}
    
    def _format_history(self, state: GovGigState) -> List:
        """Format chat history for LLM."""
        messages = []
        
        for msg in state.get("chat_history", []):
            role = msg.get("role")
            content = msg.get("content")
            
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        
        return messages
    
    def _create_messages(self, state: GovGigState, user_message: str = None) -> List:
        """Create message list for LLM invocation.
        
        Args:
            state: Current graph state
            user_message: Optional user message (defaults to state['query'])
            
        Returns:
            List of messages for LLM
        """
        messages = []
        
        # System prompt
        system_prompt = self.get_system_prompt(state)
        messages.append(SystemMessage(content=system_prompt))
        
        # Chat history
        messages.extend(self._format_history(state))
        
        # Current query
        query = user_message or state.get("query", "")
        if query:
            messages.append(HumanMessage(content=query))
        
        return messages
