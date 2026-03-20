"""
FAISS-based Vector Store for the Bilingual RAG Layer.
Supports metadata persistence and embedding versioning.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("bilingual.rag.vector_store")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.error("FAISS not installed. Please install with: pip install faiss-cpu or faiss-gpu")

class BilingualVectorStore:
    """
    Stateful FAISS index manager with metadata support.
    """
    def __init__(self, dimension: int, index_type: str = "FlatIP"):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu or faiss-gpu is required.")
            
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        # In-memory metadata storage (For MVP)
        self.metadata: List[Dict[str, Any]] = []
        self.embedding_version = "v1"

    def add_documents(self, embeddings: np.ndarray, docs_metadata: List[Dict[str, Any]]):
        """Adds embeddings and their corresponding metadata."""
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}")
            
        # Normalize for Inner Product (Cosine Similarity)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata.extend(docs_metadata)
        logger.info(f"Added {len(docs_metadata)} documents to FAISS index.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for top-k nearest neighbors."""
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1: # FAISS returns -1 for no match
                meta = self.metadata[idx].copy()
                meta["score"] = float(dist)
                results.append(meta)
        return results

    def save(self, path_prefix: str):
        """Persists index and metadata to disk."""
        faiss.write_index(self.index, f"{path_prefix}.index")
        with open(f"{path_prefix}.meta", "w", encoding="utf-8") as f:
            json.dump({
                "version": self.embedding_version,
                "metadata": self.metadata
            }, f, ensure_ascii=False)
        logger.info(f"Index persisted to {path_prefix}")

    def load(self, path_prefix: str):
        """Loads index and metadata from disk."""
        self.index = faiss.read_index(f"{path_prefix}.index")
        with open(f"{path_prefix}.meta", "r", encoding="utf-8") as f:
            data = json.load(f)
            self._metadata = data["metadata"]
            self.embedding_version = data["version"]
        logger.info(f"Loaded index from {path_prefix}")
