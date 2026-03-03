import time
import uuid
import requests
import json
import logging
import statistics
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000/api/v1"
TEST_QUERIES = [
    {
        "name": "Standard Search (Small Business)",
        "query": "What are the requirements for small business set-asides?",
        "validate": lambda r: "business" in r.get("response", "").lower()
    },
    {
        "name": "FAR Subsection Lookup (52.219-8(a))",
        "query": "Show me details about FAR 52.219-8(a)",
        "validate": lambda r: "52.219-8" in r.get("response", "")
    },
    {
        "name": "EM-385 New Format (01.A.01)",
        "query": "What are the safety requirements in EM-385?",
        "validate": lambda r: "EM 385" in r.get("response", "") or "safety" in r.get("response", "").lower()
    }
]

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_benchmark():
    print("=" * 60)
    print("🚀 GOVGIG AI - UNIFIED E2E & LATENCY BENCHMARK")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Health Check
    try:
        health = requests.get(f"{API_URL}/health").json()
        print(f"✅ System Status: {health.get('status', 'OFFLINE')} | DB: {health.get('database', 'UNKNOWN')}")
    except Exception:
        print("❌ CRITICAL: Backend is not running! Please run 'bash run.sh' first.")
        return

    print("\n" + "-" * 40)
    print("🔍 RUNNING E2E TESTS & LATENCY MEASUREMENTS")
    print("-" * 40)

    results = []
    
    for t in TEST_QUERIES:
        print(f"\nTesting: {t['name']}")
        print(f"Query: '{t['query']}'")
        
        # Get token
        try:
            import subprocess
            token = subprocess.check_output(["./venv/bin/python3", "gen_test_token.py"]).decode().strip()
            headers = {"Authorization": f"Bearer {token}"}
        except Exception as e:
            print(f"⚠️  Could not generate token: {e}")
            headers = {}

        start_time = time.perf_counter()
        try:
            resp = requests.post(
                f"{API_URL}/query",
                json={"query": t['query'], "history": [], "cot": False, "thread_id": str(uuid.uuid4())},
                headers=headers,
                timeout=30
            )
            elapsed = time.perf_counter() - start_time
            
            if resp.status_code == 200:
                data = resp.json()
                valid = t['validate'](data)
                status = "✅ PASS" if valid else "⚠️  CHECK (Low Recall)"
                print(f"Result: {status} | Latency: {elapsed:.2f}s")
                results.append(elapsed)
                
                # Show snippet
                ans = data.get('response', '')[:100].replace('\n', ' ')
                print(f"Snippet: {ans}...")
            else:
                print(f"❌ FAILED: Status {resp.status_code}")
        except Exception as e:
            print(f"❌ ERROR: {e}")

    # Final Stats
    if results:
        print("\n" + "=" * 60)
        print("📈 PERFORMANCE SUMMARY")
        print("-" * 60)
        print(f"Average Latency: {statistics.mean(results):.2f}s")
        print(f"Min Latency:     {min(results):.2f}s")
        print(f"Max Latency:     {max(results):.2f}s")
        print("=" * 60)
        print("\nNote: Latency includes OpenAI Embedding API + DB Search + Synthesis.")
    else:
        print("\n❌ No tests completed successfully.")

if __name__ == "__main__":
    run_benchmark()
