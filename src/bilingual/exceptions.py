"""
Custom exceptions for the Bilingual NLP Toolkit.
Defined to prevent silent failures and provide structured error reporting.
"""

class BilingualError(Exception):
    """Base exception for all bilingual toolkit errors."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}

class ModelError(BilingualError):
    """Base exception for model-related errors."""
    pass

class ModelLoadError(ModelError):
    """Raised when a model fails to load (e.g., OOM, missing files)."""
    pass

class InferenceError(ModelError):
    """Raised when model inference fails."""
    pass

class TokenizerError(BilingualError):
    """Raised when tokenization fails or tokenizer is incompatible."""
    pass

class ConfigurationError(BilingualError):
    """Raised when system configuration is invalid."""
    pass

class EvaluationError(BilingualError):
    """Raised when evaluation metrics cannot be computed."""
    pass

class ResourceUnavailableError(BilingualError):
    """Raised when a required resource (GPU, File, API) is missing."""
    pass

class ValidationError(BilingualError):
    """Raised when input validation fails (e.g. payload too large)."""
    pass
