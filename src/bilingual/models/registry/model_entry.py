"""
Model Entry metadata for the Bilingual Model Registry.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional

@dataclass
class ModelEntry:
    """Represents a single model version in the registry."""
    name: str
    version: str
    path: str  # Local path or HuggingFace ID
    model_type: str = "causal"  # causal, masked, embedding
    tokenizer_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "active"  # active, deprecated, experimental
    trust_remote_code: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "path": self.path,
            "model_type": self.model_type,
            "tokenizer_path": self.tokenizer_path,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "status": self.status
        }
