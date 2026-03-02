import asyncio
import time
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.agents.orchestrator import GovGigOrchestrator
from src.state.graph_state import GovGigState

async def measure_latency(query: str):
    orchestrator = GovGigOrchestrator()
    print(f"\n>>> Query: {query}")
    
    start_time = time.perf_counter()
    
    # We'll use astream_events to capture node-level timing
    node_timings = {}
    last_event_time = start_time
    
    reflection_triggered = False
    
    async for event in orchestrator.app.astream_events(
        {
            "messages": [],
            "query": query,
            "chat_history": [],
            "retrieved_documents": [],
            "agent_path": [],
            "errors": []
        }, 
        version="v1"
    ):
        kind = event["event"]
        if kind == "on_chain_end":
            node_name = event.get("name")
            if node_name in ["router", "data_retrieval", "synthesizer"]:
                now = time.perf_counter()
                elapsed = now - last_event_time
                node_timings[node_name] = elapsed
                last_event_time = now
                
                # Check for reflection in data_retrieval output
                if node_name == "data_retrieval":
                    output = event["data"]["output"]
                    path = output.get("agent_path", [])
                    if any("Reflection: Low confidence" in p for p in path):
                        reflection_triggered = True

    total_time = time.perf_counter() - start_time
    
    print(f"Total Latency: {total_time:.2f}s")
    for node, duration in node_timings.items():
        print(f"  - {node:<15}: {duration:.2f}s")
    
    print(f"Reflection Triggered: {reflection_triggered}")
    return {
        "total": total_time,
        "nodes": node_timings,
        "reflection": reflection_triggered
    }

async def main():
    # 1. Simple unambiguous query (Fast path)
    await measure_latency("What is FAR 52.212-4?")
    
    # 2. Query likely to trigger reflection (Vague or low data)
    await measure_latency("Tell me about vague safety rules in EM-385 section 99")

if __name__ == "__main__":
    asyncio.run(main())
