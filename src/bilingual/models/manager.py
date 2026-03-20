"""
Central Model Manager for the Bilingual NLP Toolkit.

Provides a singleton interface for loading, caching, and managing 
model lifecycles to prevent OOM and redundant loading.
"""

import logging
import torch
from typing import Dict, Any, Optional, Union
from threading import Lock
from pathlib import Path

from bilingual.exceptions import ModelLoadError, ConfigurationError

logger = logging.getLogger(__name__)

from bilingual.models.registry.registry import model_registry

from prometheus_client import Counter, Histogram, Gauge

# Internal Metrics for ModelManager (P1.b / A. Observability)
MODEL_CACHE_SIZE = Gauge("bilingual_model_cache_items", "Number of models currently in memory cache")
TOKENIZER_CACHE_SIZE = Gauge("bilingual_tokenizer_cache_items", "Number of tokenizers currently in memory cache")
BATCH_INFERENCE_SIZE = Histogram("bilingual_batch_inference_size", "Number of items in a single batch inference call", buckets=[1, 2, 4, 8, 16, 32, 64])

class ModelManager:
    """
    Singleton Manager for LLM Models and Tokenizers.
    Integrated with ModelRegistry for versioning support.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._init_manager()
        return cls._instance

    def _get_optimal_device(self) -> str:
        """Determines the best available device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        # xpu support for Intel GPUs if intel_extension_for_pytorch is installed
        try:
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                return "xpu"
        except ImportError:
            pass
        return "cpu"

    def _init_manager(self):
        """Initialize the manager internal state."""
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._pipelines: Dict[str, Any] = {}
        self.device = self._get_optimal_device()
        logger.info(f"ModelManager initialized on device: {self.device}")

    def load_model(
        self, 
        model_name: str, 
        version: Optional[str] = None,
        model_type: Optional[str] = None, 
        load_in_8bit: bool = True,
        force_reload: bool = False
    ) -> Any:
        """
        Load a model into memory with version support.
        """
        # Resolve via Registry if possible
        try:
            entry = model_registry.resolve(model_name, version)
            actual_path = entry.path
            actual_type = model_type or entry.model_type
            actual_version = entry.version
            trust_remote_code = getattr(entry, "trust_remote_code", True)
        except Exception:
            # Fallback for direct loading by path/HF ID
            actual_path = model_name
            actual_type = model_type or "auto"
            actual_version = "raw"
            trust_remote_code = True

        model_key = f"{model_name}_{actual_version}_{actual_type}_{'8bit' if load_in_8bit else 'fp32'}"
        
        with self._lock:
            if model_key in self._models and not force_reload:
                return self._models[model_key]

            logger.info(f"Loading model {actual_path} ({actual_version}) on {self.device}...")
            
            try:
                from transformers import (
                    AutoModelForCausalLM, 
                    AutoModelForMaskedLM, 
                    AutoModelForSeq2SeqLM,
                    AutoModel
                )
                
                quant_config = None
                if load_in_8bit and self.device == "cuda":
                    from transformers import BitsAndBytesConfig
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)

                loading_kwargs = {
                    "quantization_config": quant_config,
                    "device_map": "auto" if self.device == "cuda" else None,
                    "trust_remote_code": trust_remote_code
                }

                if actual_type == "causal" or "gpt" in actual_path.lower():
                    model = AutoModelForCausalLM.from_pretrained(actual_path, **loading_kwargs)
                elif actual_type == "masked" or "bert" in actual_path.lower():
                    model = AutoModelForMaskedLM.from_pretrained(actual_path, **loading_kwargs)
                elif actual_type in ["seq2seq", "translate", "summarize"] or any(x in actual_path.lower() for x in ["t5", "bart", "mt5"]):
                    model = AutoModelForSeq2SeqLM.from_pretrained(actual_path, **loading_kwargs)
                else:
                    model = AutoModel.from_pretrained(actual_path, **loading_kwargs)

                if self.device != "cuda" and hasattr(model, "to"):
                    model.to(self.device)

                self._models[model_key] = model
                MODEL_CACHE_SIZE.set(len(self._models))
                return model

            except Exception as e:
                logger.error(f"Failed to load model {actual_path}: {e}")
                raise ModelLoadError(f"Could not load model {actual_path}")

    def get_tokenizer(self, model_name: str, force_reload: bool = False) -> Any:
        """Load and cache tokenizer."""
        with self._lock:
            if model_name in self._tokenizers and not force_reload:
                return self._tokenizers[model_name]

            try:
                from transformers import AutoTokenizer
                # Try to resolve path from registry first
                try:
                    entry = model_registry.resolve(model_name)
                    path = entry.tokenizer_path or entry.path
                except:
                    path = model_name

                tokenizer = AutoTokenizer.from_pretrained(path)
                self._tokenizers[model_name] = tokenizer
                TOKENIZER_CACHE_SIZE.set(len(self._tokenizers))
                return tokenizer
            except Exception as e:
                logger.error(f"Failed to load tokenizer for {model_name}: {str(e)}")
                raise ModelLoadError(f"Tokenizer fail for {model_name}")

    def get_pipeline(
        self, 
        task: str, 
        model_name: str, 
        version: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Get a Hugging Face pipeline, cached for efficiency.
        """
        pipeline_key = f"{task}_{model_name}_{version or 'default'}"
        
        with self._lock:
            if pipeline_key in self._pipelines:
                return self._pipelines[pipeline_key]
            
            from transformers import pipeline
            
            model = self.load_model(model_name, version=version)
            tokenizer = self.get_tokenizer(model_name)
            
            pipe = pipeline(
                task, 
                model=model, 
                tokenizer=tokenizer, 
                device=0 if self.device == "cuda" else -1,
                **kwargs
            )
            self._pipelines[pipeline_key] = pipe
            return pipe

    def _get_dynamic_batch_size(self, input_count: int) -> int:
        """
        Optimizes batch size based on input count and available hardware.
        """
        if self.device == "cpu":
            return 1 # Batching usually slower on CPU
        elif self.device == "cuda":
            if input_count > 64: return 32
            if input_count > 16: return 16
            return 8
        return 4 # Default for MPS/XPU

    def predict_batch(
        self, 
        model_name: str, 
        inputs: list, 
        task: Optional[str] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> list:
        """
        Perform vectorized batch inference with dynamic sizing.
        """
        BATCH_INFERENCE_SIZE.observe(len(inputs))
        
        target_batch_size = batch_size or self._get_dynamic_batch_size(len(inputs))
        
        if task:
            pipe = self.get_pipeline(task, model_name)
            return pipe(inputs, batch_size=target_batch_size, **kwargs)
        
        # Fallback to manual batching
        model = self.load_model(model_name)
        tokenizer = self.get_tokenizer(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        all_decoded = []
        for i in range(0, len(inputs), target_batch_size):
            batch = inputs[i:i+target_batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = model.generate(**encoded, **kwargs)
            all_decoded.extend([tokenizer.decode(o, skip_special_tokens=True) for o in outputs])
        
        return all_decoded

    def warmup(self, model_list: list):
        """
        Preload specified models to avoid latency on first request.
        Optimized for production RAG & Translation stack (P3.b / B. Warm-up).
        """
        logger.info(f"System Warm-up started for: {model_list}")
        for m in model_list:
            try:
                # Resolve type for better loading
                actual_type = "causal"
                if any(x in m.lower() for x in ["bert", "masked"]): actual_type = "masked"
                if any(x in m.lower() for x in ["t5", "mt5"]): actual_type = "seq2seq"
                
                self.load_model(m, model_type=actual_type)
                self.get_tokenizer(m)
                logger.info(f"✅ Warm-up complete for {m}")
            except Exception as e:
                logger.error(f"❌ Warmup failed for {m}: {e}")

    def clear_cache(self, model_name: Optional[str] = None):
        """Free memory by clearing cached models."""
        with self._lock:
            if model_name:
                keys_to_del = [k for k in self._models if k.startswith(model_name)]
                for k in keys_to_del:
                    del self._models[k]
                    logger.info(f"Evicted {k} from cache.")
                
                pipe_keys = [k for k in self._pipelines if model_name in k]
                for k in pipe_keys:
                    del self._pipelines[k]
            else:
                self._models.clear()
                self._tokenizers.clear()
                self._pipelines.clear()
                logger.info("Cleared all cached models, tokenizers, and pipelines.")
            
            MODEL_CACHE_SIZE.set(len(self._models))
            TOKENIZER_CACHE_SIZE.set(len(self._tokenizers))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Global access point
model_manager = ModelManager()

def get_model_manager() -> ModelManager:
    return model_manager
