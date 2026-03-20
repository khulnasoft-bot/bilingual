#!/usr/bin/env python3
"""
Transformer model integration for bilingual generation tasks.

This module provides integration with Transformer-based models:
- T5 for text generation and translation
- BART for summarization and generation
- mT5 for multilingual tasks
- Zero-shot learning capabilities
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import torch

try:
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        BartForConditionalGeneration,
        BartTokenizer,
        GenerationConfig,
        MT5ForConditionalGeneration,
        MT5Tokenizer,
        T5ForConditionalGeneration,
        T5Tokenizer,
        pipeline,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers torch")

warnings.warn(
    "TransformerModelManager is deprecated and will be removed in future versions. "
    "Please use bilingual.models.manager.ModelManager instead for consolidated loading, "
    "caching, and quantization support.",
    DeprecationWarning,
    stacklevel=2
)


class TransformerModelManager:
    """
    Manager for Transformer-based models for bilingual generation.
    """

    def __init__(self):
        """Initialize the model manager."""
        if not TRANSFORMERS_AVAILABLE:
            warnings.warn(
                "Transformers not available. Install with: pip install transformers torch"
            )

        self.models = {}
        self.tokenizers = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_name: str, model_type: str = "auto") -> None:
        """
        Load a Transformer model for generation.

        Args:
            model_name: Name or path of the model
            model_type: Type of model ('t5', 'bart', 'mt5', 'auto')
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")

        try:
            # Auto-detect model type if not specified
            if model_type == "auto":
                model_type = self._detect_model_type(model_name)

            if model_type == "t5":
                tokenizer = T5Tokenizer.from_pretrained(model_name)
                model = T5ForConditionalGeneration.from_pretrained(model_name)
            elif model_type == "bart":
                tokenizer = BartTokenizer.from_pretrained(model_name)
                model = BartForConditionalGeneration.from_pretrained(model_name)
            elif model_type == "mt5":
                tokenizer = MT5Tokenizer.from_pretrained(model_name)
                model = MT5ForConditionalGeneration.from_pretrained(model_name)
            else:
                # Use auto-loading for other models
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            model.to(self.device)
            model.eval()

            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer

            print(f"✅ Loaded {model_type.upper()} model: {model_name}")

        except Exception as e:
            print(f"❌ Failed to load model {model_name}: {e}")
            raise

    def _detect_model_type(self, model_name: str) -> str:
        """Auto-detect model type from name."""
        name_lower = model_name.lower()

        if "t5" in name_lower:
            return "t5"
        elif "bart" in name_lower:
            return "bart"
        elif "mt5" in name_lower:
            return "mt5"
        elif "mbart" in name_lower:
            return "mbart"
        else:
            return "auto"

    def generate(
        self,
        model_name: str,
        prompt: str,
        max_length: int = 50,
        min_length: int = 10,
        num_beams: int = 4,
        temperature: float = 1.0,
        repetition_penalty: float = 1.2,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 0.95,
        **kwargs,
    ) -> str:
        """
        Generate text using a loaded model.

        Args:
            model_name: Name of the loaded model
            prompt: Input prompt for generation
            max_length: Maximum length of generated text
            min_length: Minimum length of generated text
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            repetition_penalty: Penalty for repetition
            do_sample: Whether to use sampling
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]

        # Encode input
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True)
        inputs = inputs.to(self.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def translate(
        self, model_name: str, text: str, src_lang: str = "en", tgt_lang: str = "bn", **kwargs
    ) -> str:
        """
        Translate text using a translation model.

        Args:
            model_name: Name of the translation model
            text: Text to translate
            src_lang: Source language code
            tgt_lang: Target language code
            **kwargs: Additional generation parameters

        Returns:
            Translated text
        """
        # For demonstration, use a simple template-based approach
        # In practice, you'd use dedicated translation models

        if src_lang == "en" and tgt_lang == "bn":
            # Simple English to Bengali translation prompt
            prompt = f"Translate English to Bengali: {text}"
        elif src_lang == "bn" and tgt_lang == "en":
            # Simple Bengali to English translation prompt
            prompt = f"Translate Bengali to English: {text}"
        else:
            return text  # No translation available

        return self.generate(model_name, prompt, **kwargs)

    def summarize(self, model_name: str, text: str, max_length: int = 150, **kwargs) -> str:
        """
        Summarize text using a summarization model.

        Args:
            model_name: Name of the summarization model
            text: Text to summarize
            max_length: Maximum length of summary
            **kwargs: Additional generation parameters

        Returns:
            Summary text
        """
        prompt = f"Summarize: {text}"
        return self.generate(model_name, prompt, max_length=max_length, **kwargs)

    def zero_shot_classify(
        self, model_name: str, text: str, labels: List[str], **kwargs
    ) -> Dict[str, float]:
        """
        Perform zero-shot classification.

        Args:
            model_name: Name of the zero-shot model
            text: Text to classify
            labels: List of possible labels
            **kwargs: Additional parameters

        Returns:
            Dictionary of label probabilities
        """
        if not TRANSFORMERS_AVAILABLE:
            # Simple fallback when transformers not available
            return {label: 1.0 / len(labels) for label in labels}

        try:
            # Use pipeline for zero-shot classification
            classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
            )

            result = classifier(text, labels, **kwargs)
            return dict(zip(result["labels"], result["scores"]))

        except Exception as e:
            print(f"Zero-shot classification failed: {e}")
            return {label: 1.0 / len(labels) for label in labels}

    def multilingual_generate(
        self, model_name: str, prompt: str, target_language: str = "english", **kwargs
    ) -> str:
        """
        Generate text in a specific target language.

        Args:
            model_name: Name of the multilingual model
            prompt: Input prompt
            target_language: Target language for generation
            **kwargs: Additional generation parameters

        Returns:
            Generated text in target language
        """
        # Create language-specific prompt
        lang_prompts = {
            "english": f"Generate in English: {prompt}",
            "bengali": f"Generate in Bengali: {prompt}",
            "spanish": f"Generate in Spanish: {prompt}",
            "french": f"Generate in French: {prompt}",
        }

        prompt_with_lang = lang_prompts.get(target_language.lower(), prompt)
        return self.generate(model_name, prompt_with_lang, **kwargs)

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a loaded model."""
        if model_name not in self.models:
            return {"error": "Model not loaded"}

        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]

        return {
            "model_name": model_name,
            "model_type": type(model).__name__,
            "vocab_size": getattr(tokenizer, "vocab_size", "unknown"),
            "device": str(self.device),
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

    def list_loaded_models(self) -> List[str]:
        """List all loaded models."""
        return list(self.models.keys())


# Global model manager instance
_model_manager = None


def get_model_manager() -> TransformerModelManager:
    """Get or create the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = TransformerModelManager()
    return _model_manager


def load_model(model_name: str, model_type: str = "auto") -> None:
    """Convenience function to load a model."""
    return get_model_manager().load_model(model_name, model_type)


def generate_text(model_name: str, prompt: str, **kwargs) -> str:
    """Convenience function to generate text."""
    return get_model_manager().generate(model_name, prompt, **kwargs)


def translate_text(
    model_name: str, text: str, src_lang: str = "en", tgt_lang: str = "bn", **kwargs
) -> str:
    """Convenience function to translate text."""
    return get_model_manager().translate(model_name, text, src_lang, tgt_lang, **kwargs)


def summarize_text(model_name: str, text: str, **kwargs) -> str:
    """Convenience function to summarize text."""
    return get_model_manager().summarize(model_name, text, **kwargs)


def zero_shot_classify(model_name: str, text: str, labels: List[str], **kwargs) -> Dict[str, float]:
    """Convenience function for zero-shot classification."""
    return get_model_manager().zero_shot_classify(model_name, text, labels, **kwargs)


def multilingual_generate(
    model_name: str, prompt: str, target_language: str = "english", **kwargs
) -> str:
    """Convenience function for multilingual generation."""
    return get_model_manager().multilingual_generate(model_name, prompt, target_language, **kwargs)
