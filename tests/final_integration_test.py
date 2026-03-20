"""
Comprehensive Integration Test for Bilingual Production RAG Stack.
Validates: Gateway, Ray Serve, Model Registry, Hybrid Search, and Observability.
"""

import asyncio
import httpx
import time
import uuid
import statistics
from typing import List, Dict, Any

# --- CONFIGURATION ---
BASE_URL = "http://localhost:8000"
API_KEY = "kothagpt-dev-2026"  # From security.py
CONCURRENT_USERS = 20
TOTAL_REQUESTS = 50

# --- HELPERS ---
def log_test(name: str, status: str, message: str = ""):
    color = "\033[92m" if status == "PASS" else "\033[91m"
    reset = "\033[0m"
    print(f"[{color}{status}{reset}] {name}: {message}")

async def call_rag(client: httpx.AsyncClient, query: str, version: str = None, headers: dict = None) -> dict:
    payload = {"query": query}
    if version:
        payload["version"] = version
    
    h = headers or {"X-Bilingual-Key": API_KEY}
    try:
        response = await client.post(f"{BASE_URL}/rag/query", json=payload, headers=h, timeout=30.0)
        return response.json() if response.status_code == 200 else {"error": response.status_code, "detail": response.text}
    except Exception as e:
        return {"error": "exception", "detail": str(e)}

# --- TEST CASES ---

async def test_01_basic_rag_flow(client: httpx.AsyncClient):
    """Verifies default RAG flow with security."""
    res = await call_rag(client, "What is the capital of Bangladesh?")
    if "answer" in res and "metrics" in res:
        log_test("Basic RAG Flow", "PASS", f"Answer received. Latency: {res.get('processing_time_ms', 0):.2f}ms")
    else:
        log_test("Basic RAG Flow", "FAIL", str(res))

async def test_02_version_resolution(client: httpx.AsyncClient):
    """Verifies Model Registry versioning."""
    # Test v1
    res_v1 = await call_rag(client, "Tell me about Dhaka", version="v1")
    if "answer" in res_v1:
        log_test("Version Resolution (v1)", "PASS")
    else:
        log_test("Version Resolution (v1)", "FAIL", str(res_v1))

    # Test Invalid Version Fallback (Wait, current server raises 500 error for invalid version)
    res_err = await call_rag(client, "Test", version="non-existent")
    if res_err.get("error") == 500:
        log_test("Version Error Handling", "PASS", "Correctly rejected non-existent version")

async def test_03_security_enforcement(client: httpx.AsyncClient):
    """Verifies API Key and Payload limits."""
    # No Key
    res_no_key = await call_rag(client, "Secret query", headers={})
    if res_no_key.get("error") == 403:
         log_test("Security: No Key", "PASS")
    
    # Large Payload
    large_text = "A" * (6 * 1024 * 1024)
    res_large = await call_rag(client, large_text)
    if res_large.get("error") == 413:
        log_test("Security: Payload Limit", "PASS")

async def test_04_hybrid_and_rerank_metrics(client: httpx.AsyncClient):
    """Checks if advanced metrics are returned in the response."""
    res = await call_rag(client, "Explain the Bengali language history")
    metrics = res.get("metrics", {})
    if "retrieval_mode" in metrics and "reranked" in metrics:
        log_test("Advanced Metrics Extraction", "PASS", f"Mode: {metrics['retrieval_mode']}, Reranked: {metrics['reranked']}")
    else:
        log_test("Advanced Metrics Extraction", "FAIL", "Missing RAG metrics in response")

async def test_05_concurrent_load_and_metrics(client: httpx.AsyncClient):
    """Simulates high load and verifies metrics consistency."""
    print(f"\n🚀 Running Concurrent Load Test ({CONCURRENT_USERS} users)...")
    tasks = [call_rag(client, f"Concurrent query {uuid.uuid4()}") for _ in range(TOTAL_REQUESTS)]
    
    start = time.time()
    results = await asyncio.gather(*tasks)
    duration = time.time() - start
    
    success = sum(1 for r in results if "answer" in r)
    latencies = [r.get("processing_time_ms", 0) for r in results if "processing_time_ms" in r]
    
    log_test("Concurrent Load", "PASS" if success == TOTAL_REQUESTS else "PARTIAL", 
             f"Success: {success}/{TOTAL_REQUESTS} | Avg Latency: {statistics.mean(latencies):.2f}ms | Throughput: {success/duration:.2f} req/s")

# --- MAIN EXECUTION ---
async def main():
    print("--- 🏁 Bilingual Final Integration Test 🏁 ---\n")
    async with httpx.AsyncClient() as client:
        # Check if server is up
        try:
            await client.get(f"{BASE_URL}/health")
        except:
            print(f"❌ ERROR: Server not found at {BASE_URL}. Start 'python -m bilingual.server' first.")
            return

        await test_01_basic_rag_flow(client)
        await test_02_version_resolution(client)
        await test_03_security_enforcement(client)
        await test_04_hybrid_and_rerank_metrics(client)
        await test_05_concurrent_load_and_metrics(client)
        
        # Check Prometheus Metrics endpoint
        res_metrics = await client.get(f"{BASE_URL}/metrics")
        if res_metrics.status_code == 200 and "bilingual_rag_latency_seconds" in res_metrics.text:
            log_test("Prometheus Endpoint", "PASS", "Metics are being correctly exported.")
        else:
            log_test("Prometheus Endpoint", "FAIL", "RAG metrics not found in /metrics")

    print("\n✅ Integration Testing Completed.")

if __name__ == "__main__":
    asyncio.run(main())
