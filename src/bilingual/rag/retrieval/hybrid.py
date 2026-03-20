"""
Hybrid Retriever: Combines Keyword-based (BM25) and Semantic (FAISS) search.
Optimized for Bilingual (BN/EN) retrieval precision.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from bilingual.rag.vector_store.faiss_index import BilingualVectorStore

logger = logging.getLogger("bilingual.rag.retrieval.hybrid")

class HybridRetriever:
    """
    Implements a weighted hybrid search using BM25 and FAISS.
    """
    def __init__(
        self, 
        vector_store: BilingualVectorStore,
        alpha: float = 0.7, # Weight for FAISS
        beta: float = 0.3   # Weight for BM25
    ):
        self.vs = vector_store
        self.alpha = alpha
        self.beta = beta
        self.bm25 = None
        self.corpus_chunks = []

    def fit_bm25(self, chunks: List[str]):
        """Initializes BM25 index on the provided corpus chunks."""
        try:
            from rank_bm25 import BM25Okapi
            # Simple whitespace/punctuation cleanup for tokenization
            tokenized_corpus = [doc.lower().split() for doc in chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.corpus_chunks = chunks
            logger.info(f"BM25 index fitted on {len(chunks)} chunks.")
        except ImportError:
            logger.error("rank_bm25 not installed. Install with: pip install rank-bm25")
            self.bm25 = None

    def search(self, query: str, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Performs hybrid search combining FAISS and BM25 scores."""
        # 1. Semantic Search (FAISS)
        faiss_results = self.vs.search(query_embedding, top_k=top_k * 2)
        
        # 2. Keyword Search (BM25)
        bm25_results = []
        if self.bm25:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_n_indices = np.argsort(bm25_scores)[-top_k*2:][::-1]
            for idx in top_n_indices:
                if bm25_scores[idx] > 0:
                    bm25_results.append({
                        "text": self.corpus_chunks[idx],
                        "score": float(bm25_scores[idx]),
                        "id": idx
                    })

        # 3. Score Normalization & Fusion
        combined_results = self._reciprocal_rank_fusion(faiss_results, bm25_results, top_k)
        return combined_results

    def _reciprocal_rank_fusion(self, faiss_res: List[Dict], bm25_res: List[Dict], k: int) -> List[Dict]:
        """Combines results using Reciprocal Rank Fusion (RRF) or Weighted Sum."""
        # For simplicity and robust production use, we'll use a mapped scoring approach
        fusion_map = {}
        
        # Map FAISS scores
        for res in faiss_res:
            fusion_map[res["text"]] = {"score": res["score"] * self.alpha, "meta": res}
            
        # Map BM25 scores (normalized by max score)
        if bm25_res:
            max_bm25 = max([r["score"] for r in bm25_res]) if bm25_res else 1
            for res in bm25_res:
                norm_score = (res["score"] / max_bm25) * self.beta
                if res["text"] in fusion_map:
                    fusion_map[res["text"]]["score"] += norm_score
                else:
                    fusion_map[res["text"]] = {"score": norm_score, "meta": res}

        # Sort and return top_k
        sorted_res = sorted(fusion_map.items(), key=lambda x: x[1]["score"], reverse=True)
        final = []
        for text, data in sorted_res[:k]:
            meta = data["meta"]
            meta["hybrid_score"] = data["score"]
            final.append(meta)
            
        return final
