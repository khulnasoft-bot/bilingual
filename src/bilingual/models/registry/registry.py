"""
Central Model Registry for versioning and model discovery.
"""

import logging
from typing import Dict, List, Optional
from bilingual.models.registry.model_entry import ModelEntry

logger = logging.getLogger("bilingual.models.registry")

class ModelRegistry:
    """
    Manages discovery and version resolution of models.
    """
    def __init__(self):
        self._models: Dict[str, Dict[str, ModelEntry]] = {}  # name -> version -> entry
        self._default_versions: Dict[str, str] = {}  # name -> version

    def register(self, entry: ModelEntry, is_default: bool = False):
        """Register a new model version."""
        if entry.name not in self._models:
            self._models[entry.name] = {}
        
        self._models[entry.name][entry.version] = entry
        
        if is_default or entry.name not in self._default_versions:
            self._default_versions[entry.name] = entry.version
            
        logger.info(f"Registered model {entry.name} version {entry.version}")

    def resolve(self, name: str, version: Optional[str] = None) -> ModelEntry:
        """Resolve a model name and version to its entry."""
        if name not in self._models:
            raise ValueError(f"Model {name} not found in registry.")
            
        v = version or self._default_versions.get(name)
        if not v or v not in self._models[name]:
            available = list(self._models[name].keys())
            raise ValueError(f"Version {version} not found for model {name}. Available: {available}")
            
        return self._models[name][v]

    def list_models(self) -> List[ModelEntry]:
        """List all registered models."""
        all_entries = []
        for versions in self._models.values():
            all_entries.extend(versions.values())
        return all_entries

# Global Instance
model_registry = ModelRegistry()

# Pre-register some defaults for MVP
model_registry.register(ModelEntry(
    name="bilingual-small",
    version="v1",
    path="t5-small",
    model_type="causal"
), is_default=True)

model_registry.register(ModelEntry(
    name="bilingual-embed",
    version="v1",
    path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_type="embedding"
), is_default=True)
