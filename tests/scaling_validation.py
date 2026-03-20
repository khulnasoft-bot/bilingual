"""
Scaling Validation Script for Bilingual Distributed RAG.
Simulates burst traffic to verify Ray scaling and Metrics propagation.
"""

import asyncio
import httpx
import time
import statistics
from tqdm import asyncio as tqdm_asyncio

BASE_URL = "http://localhost:8000"
API_KEY = "kothagpt-dev-2026"
CONCURRENT_USERS = 50
TOTAL_REQUESTS = 200

async def send_rag_query(client: httpx.AsyncClient, query_id: int):
    start = time.time()
    payload = {
        "query": f"Query number {query_id}: What is the importance of bilingual NLPs?",
        "top_k": 3
    }
    headers = {"X-Bilingual-Key": API_KEY}
    
    try:
        response = await client.post(f"{BASE_URL}/rag/query", json=payload, headers=headers, timeout=30.0)
        return {
            "status": response.status_code,
            "latency": (time.time() - start) * 1000,
            "request_id": response.headers.get("X-Request-ID")
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def run_load_test():
    async with httpx.AsyncClient() as client:
        print(f"🚀 Starting Load Test: {CONCURRENT_USERS} concurrent users, {TOTAL_REQUESTS} total requests.")
        
        tasks = [send_rag_query(client, i) for i in range(TOTAL_REQUESTS)]
        
        # Using a semaphore to limit concurrency correctly
        semaphore = asyncio.Semaphore(CONCURRENT_USERS)
        
        async def sem_task(t):
            async with semaphore:
                return await t

        start_time = time.time()
        results = await tqdm_asyncio.gather(*[sem_task(t) for t in tasks])
        total_duration = time.time() - start_time
        
        # Analyze Results
        latencies = [r["latency"] for r in results if isinstance(r.get("latency"), float)]
        success_count = sum(1 for r in results if r.get("status") == 200)
        error_count = TOTAL_REQUESTS - success_count
        
        print("\n--- Load Test Results ---")
        print(f"Total Time: {total_duration:.2f}s")
        print(f"Throughput: {TOTAL_REQUESTS / total_duration:.2f} req/s")
        print(f"Success: {success_count} | Errors: {error_count}")
        if latencies:
            print(f"Average Latency: {statistics.mean(latencies):.2f}ms")
            print(f"P95 Latency: {statistics.quantiles(latencies, n=20)[18]:.2f}ms")
            print(f"P99 Latency: {max(latencies):.2f}ms")

if __name__ == "__main__":
    asyncio.run(run_load_test())
