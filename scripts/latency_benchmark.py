import asyncio
import time
import sys
import os
import logging
from typing import List, Dict, Any

# Configure logging to see internal step timings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project root to path
sys.path.append(os.getcwd())

from src.agents.orchestrator import GovGigOrchestrator

async def run_benchmark(query: str, label: str):
    orchestrator = GovGigOrchestrator()
    context = {"history": [], "cot": False}
    
    print(f"\n>>> Benchmarking {label}: '{query}'")
    
    start_time = time.perf_counter()
    first_token_time = None
    node_times = {}
    
    async for event in orchestrator.run(query, context):
        current_time = time.perf_counter() - start_time
        
        etype = event["type"]
        if etype == "token" and first_token_time is None:
            first_token_time = current_time
            print(f"  [TTFT] First token received at {first_token_time:.3f}s")
            
        elif etype == "step":
            node = event.get("node")
            node_times[node] = current_time
            print(f"  [Node] {node} finished at {current_time:.3f}s")
            
        elif etype == "complete":
            total_time = current_time
            print(f"  [Total] Response completed in {total_time:.3f}s")
            return {
                "label": label,
                "ttft": first_token_time,
                "total": total_time,
                "nodes": node_times
            }

async def main():
    queries = [
        ("FAR 52.232-1", "Fast Path (Clause Lookup)"),
        ("What are the requirements for excavation safety?", "Standard Path (Hybrid Search)"),
        ("Hello, how are you?", "Out of Scope Path")
    ]
    
    results = []
    print("=" * 60)
    print("GOVGIG LATENCY BENCHMARK")
    print("=" * 60)
    
    for query, label in queries:
        res = await run_benchmark(query, label)
        results.append(res)
        # Small delay between tests
        await asyncio.sleep(1)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"{'Path':<30} | {'TTFT':<10} | {'Total':<10}")
    print("-" * 60)
    for res in results:
        ttft_str = f"{res['ttft']:.3f}s" if res['ttft'] else "N/A"
        print(f"{res['label']:<30} | {ttft_str:<10} | {res['total']:.3f}s")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
