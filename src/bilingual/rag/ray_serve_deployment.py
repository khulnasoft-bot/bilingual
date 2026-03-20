"""
Ray Serve Deployment for Distributed Bilingual RAG.
Splits CPU-bound Retrieval and GPU-bound Generation.
"""

import logging
from typing import Dict, Any
from ray import serve
from starlette.requests import Request

from bilingual.models.manager import model_manager
from bilingual.rag.orchestrator import RAGOrchestrator
from bilingual.rag.vector_store.faiss_index import BilingualVectorStore

logger = logging.getLogger("ray.serve")

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 2})
class RAGService:
    """
    Ray Deployment for Retrieval-Augmented Generation.
    Centralizes orchestration while scaling across nodes.
    """
    def __init__(self, index_path: str):
        # 1. Initialize Vector Store (CPU-bound)
        # Assuming dimension 384 for MiniLM
        self.vs = BilingualVectorStore(dimension=384)
        if index_path:
            self.vs.load(index_path)
            
        # 2. Initialize Orchestrator
        self.orchestrator = RAGOrchestrator(vector_store=self.vs)
        logger.info("RAGService deployed successfully on Ray.")

    async def __call__(self, http_request: Request) -> Dict[str, Any]:
        data = await http_request.json()
        query = data.get("query")
        
        if not query:
            return {"error": "Query is required"}
            
        # Perform RAG (Orchestrator handles ModelManager singletons internally)
        try:
            result = self.orchestrator.generate_with_context(query)
            return result
        except Exception as e:
            logger.error(f"Distributed RAG Error: {e}")
            return {"error": str(e)}

# Deployment Entry Point
def deploy_rag_system(index_path: str):
    """
    Deploys the RAG system to a running Ray cluster.
    """
    serve.run(RAGService.bind(index_path=index_path), name="bilingual_rag")
    print("✅ Bilingual RAG System is live on Ray Serve.")
