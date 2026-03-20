"""
Re-ranker: Refines retrieval results using Cross-Encoder models.
Provides high-precision fine-grained sorting of candidate chunks.
"""

import logging
import torch
from typing import List, Dict, Any, Tuple
from bilingual.models.manager import model_manager

logger = logging.getLogger("bilingual.rag.retrieval.reranker")

class CrossEncoderReranker:
    """
    Re-ranks candidate chunks using a Cross-Encoder for maximum relevance.
    """
    def __init__(
        self, 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        """Lazy loads the reranker model."""
        if self._model is None:
            # Using model_manager to ensure singleton/lifecycle management
            self._model = model_manager.load_model(self.model_name, model_type="auto")
            self._tokenizer = model_manager.get_tokenizer(self.model_name)
            logger.info(f"Cross-Encoder {self.model_name} loaded for re-ranking.")

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        """Re-ranks candidates based on Cross-Encoder scores."""
        if not candidates:
            return []
            
        self._ensure_loaded()
        
        # Prepare pairs for cross-encoder (Query, Chunk)
        pairs = [[query, res["text"]] for res in candidates]
        
        inputs = self._tokenizer(
            pairs, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=512
        ).to(self._model.device)
        
        with torch.no_grad():
            # Cross-encoder typically returns logits for binary classification or regression
            outputs = self._model(**inputs)
            scores = outputs.logits.view(-1).cpu().numpy()
            
        # Add scores to candidates
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)
            
        # Re-sort based on rerank scores
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]
