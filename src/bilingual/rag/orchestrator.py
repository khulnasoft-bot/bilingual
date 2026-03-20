"""
RAG Orchestrator: Coordinates Retrieval and Generation.
Integrates Embedding Service, Vector Store, and Model Management.
"""

import torch
import numpy as np
import logging
from typing import List, Dict, Any, Optional

from bilingual.models.manager import model_manager
from bilingual.rag.vector_store.faiss_index import BilingualVectorStore
from bilingual.exceptions import InferenceError

logger = logging.getLogger("bilingual.rag.orchestrator")

from prometheus_client import Counter, Histogram, Summary

# RAG-Specific Metrics (P1.b + P3.1.5)
RAG_LATENCY = Histogram(
    "bilingual_rag_latency_seconds", 
    "Latency per RAG stage", 
    ["stage"] # embed, retrieve, generate
)
RETRIEVAL_HIT_RATE = Counter("bilingual_rag_retrieval_hits_total", "Successful retrievals above threshold")
CONTEXT_TOKENS = Summary("bilingual_rag_context_tokens", "Total tokens in context per request")

from bilingual.rag.retrieval.hybrid import HybridRetriever
from bilingual.rag.retrieval.reranker import CrossEncoderReranker

class RAGOrchestrator:
    """
    Advanced Orchestrator: Supports Hybrid Search and Cross-Encoder Re-ranking.
    """
    def __init__(
        self, 
        vector_store: BilingualVectorStore,
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        generation_model_name: str = "t5-small",
        min_similarity_threshold: float = 0.45,
        max_context_tokens: int = 1024,
        use_hybrid: bool = False,
        use_reranker: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.vector_store = vector_store
        self.embed_model_name = embedding_model_name
        self.gen_model_name = generation_model_name
        self.min_threshold = min_similarity_threshold
        self.max_context_limit = max_context_tokens
        
        # Advanced Modules (P3.5)
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker
        self.hybrid_retriever = HybridRetriever(vector_store) if use_hybrid else None
        self.reranker = CrossEncoderReranker(model_name=reranker_model) if use_reranker else None
        
        model_manager.load_model(self.embed_model_name, model_type="auto")
        model_manager.load_model(self.gen_model_name, model_type="causal")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieval with optional Hybrid Search and Re-ranking.
        """
        adaptive_k = top_k + 5 if self.use_reranker else top_k
        query_embedding = self._get_embedding(query)
        
        with RAG_LATENCY.labels(stage="retrieve").time():
            if self.use_hybrid and self.hybrid_retriever:
                results = self.hybrid_retriever.search(query, query_embedding, top_k=adaptive_k)
            else:
                results = self.vector_store.search(query_embedding, top_k=adaptive_k)
                
            filtered_results = [r for r in results if r.get("score", 1.0) >= self.min_threshold]

        # Re-ranking Stage (P3.5)
        if self.use_reranker and self.reranker and filtered_results:
            with RAG_LATENCY.labels(stage="rerank").time():
                filtered_results = self.reranker.rerank(query, filtered_results, top_n=top_k)
                
        if filtered_results:
            RETRIEVAL_HIT_RATE.inc()
                
        return filtered_results

    def generate_with_context(self, query: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Budget-aware Generation with advanced retrieval feedback.
        """
        # 1. Advanced Retrieve
        contexts = self.retrieve(query)
        
        # 2. Context Budget Controller
        final_context_list = []
        accumulated_tokens = 0
        gen_tokenizer = model_manager.get_tokenizer(self.gen_model_name)
        
        for ctx in contexts:
            tokens = len(gen_tokenizer.encode(ctx["text"]))
            if accumulated_tokens + tokens > self.max_context_limit:
                break
            final_context_list.append(ctx["text"])
            accumulated_tokens += tokens
            
        CONTEXT_TOKENS.observe(accumulated_tokens)
        context_string = "\n".join(final_context_list)
        
        # 3. Guarded Generation
        with RAG_LATENCY.labels(stage="generate").time():
            try:
                prompt = f"Context: {context_string}\n\nQuestion: {query}\n\nAnswer:"
                model = model_manager.load_model(self.gen_model_name, version=version)
                
                inputs = gen_tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, max_new_tokens=150)
                answer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                return {
                    "query": query,
                    "answer": answer,
                    "metrics": {
                        "tokens_used": accumulated_tokens,
                        "hit_count": len(final_context_list),
                        "retrieval_mode": "hybrid" if self.use_hybrid else "semantic",
                        "reranked": self.use_reranker,
                        "best_score": contexts[0].get("rerank_score") if self.use_reranker and contexts else contexts[0].get("score") if contexts else 0
                    }
                }
            except Exception as e:
                logger.error(f"Generation failure: {e}")
                raise InferenceError("RAG context was valid but generation failed.")
