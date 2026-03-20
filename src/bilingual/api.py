"""
High-level API for bilingual package.

Provides simple functions for common NLP tasks in Bangla and English.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from cachetools import TTLCache
from bilingual.normalize import detect_language
from bilingual.normalize import normalize_text as _normalize_text
from bilingual.tokenizer import BilingualTokenizer
from bilingual.tokenizer import load_tokenizer as _load_tokenizer

# Global cache for loaded models and tokenizers
_TOKENIZER_CACHE: TTLCache = TTLCache(maxsize=128, ttl=3600)
_MODEL_CACHE: TTLCache = TTLCache(maxsize=32, ttl=3600)


def load_tokenizer(
    model_name: str = "bilingual-tokenizer", force_reload: bool = False
) -> BilingualTokenizer:
    """
    Load a tokenizer (with caching).

    Args:
        model_name: Name or path of the tokenizer model
        force_reload: Force reload even if cached

    Returns:
        BilingualTokenizer instance
    """
    if not model_name:
        raise ValueError("model_name must be a non-empty string")

    if model_name not in _TOKENIZER_CACHE or force_reload:
        _TOKENIZER_CACHE[model_name] = _load_tokenizer(model_name)
    return _TOKENIZER_CACHE[model_name]


def load_model(model_name: str, force_reload: bool = False, **kwargs) -> Any:
    """
    Load a language model (with caching).

    Args:
        model_name: Name or path of the model
        force_reload: Force reload even if cached
        **kwargs: Additional arguments for model loading

    Returns:
        Loaded model instance
    """
    if not model_name:
        raise ValueError("model_name must be a non-empty string")

    if model_name not in _MODEL_CACHE or force_reload:
        # Import here to avoid circular dependencies
        from bilingual.models.loader import load_model_from_name

        _MODEL_CACHE[model_name] = load_model_from_name(model_name, **kwargs)
    return _MODEL_CACHE[model_name]


def normalize_text(text: str, lang: Optional[str] = None, **kwargs) -> str:
    """
    Normalize text for Bangla or English.

    Args:
        text: Input text
        lang: Language code ('bn' or 'en'), auto-detected if None
        **kwargs: Additional normalization options

    Returns:
        Normalized text

    Examples:
        >>> normalize_text("আমি   স্কুলে যাচ্ছি।", lang="bn")
        'আমি স্কুলে যাচ্ছি.'
    """
    if lang is not None and lang not in {"bn", "en"}:
        warnings.warn(
            f"Unsupported language code '{lang}' in normalize_text; falling back to auto-detection."
        )
        lang = None

    return _normalize_text(text, lang=lang, **kwargs)


def tokenize(
    text: str,
    tokenizer: Optional[Union[str, BilingualTokenizer]] = None,
    return_ids: bool = False,
) -> Union[List[str], List[int]]:
    """
    Tokenize text.

    Args:
        text: Input text
        tokenizer: Tokenizer name/path or instance (default: "bilingual-tokenizer")
        return_ids: If True, return token IDs instead of strings

    Returns:
        List of tokens or token IDs

    Examples:
        >>> tokenize("আমি বই পড়ি।")
        ['▁আমি', '▁বই', '▁পড়ি', '.']
    """
    if tokenizer is None:
        tokenizer = "bilingual-tokenizer"

    if isinstance(tokenizer, str):
        tokenizer = load_tokenizer(tokenizer)

    if not isinstance(tokenizer, BilingualTokenizer):
        raise TypeError("tokenizer must be a BilingualTokenizer instance or a valid model name")

    result = tokenizer.encode(text, as_ids=return_ids)
    # Return appropriate type based on return_ids
    return result  # type: ignore[return-value]


def generate(
    prompt: Union[str, List[str]],
    model_name: str = "bilingual-small-lm",
    version: Optional[str] = None,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    **kwargs,
) -> Union[str, List[str]]:
    """
    Generate text continuation from a prompt.

    Args:
        prompt: Input prompt text or a list of prompts
        model_name: Name of the generation model
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        **kwargs: Additional generation parameters

    Returns:
        Generated text or a list of generated texts

    Examples:
        >>> generate("Once upon a time, there was a brave rabbit")
        'Once upon a time, there was a brave rabbit who lived in a forest...'
    """
    if not prompt or len(prompt) > 5000:
        from bilingual.exceptions import ValidationError
        raise ValidationError(f"Prompt length {len(prompt) if prompt else 0} exceeds limit of 5000 or is empty.")

    model = load_model(model_name, version=version)

    # Import here to avoid circular dependencies
    from bilingual.models.lm import generate_text

    return generate_text(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        **kwargs,
    )


def translate(
    text: Union[str, List[str]],
    src: str = "bn",
    tgt: str = "en",
    model_name: str = "bilingual-translate",
    **kwargs,
) -> Union[str, List[str]]:
    """
    Translate text between Bangla and English.

    Args:
        text: Input text to translate or a list of texts
        src: Source language code ('bn' or 'en')
        tgt: Target language code ('bn' or 'en')
        model_name: Name of the translation model
        **kwargs: Additional translation parameters

    Returns:
        Translated text or a list of translated texts

    Examples:
        >>> translate("আমি বই পড়তে ভালোবাসি।", src="bn", tgt="en")
        'I love to read books.'
    """
    supported_langs = {"bn", "en"}
    if src not in supported_langs:
        raise ValueError(f"Unsupported source language code: {src}")
    if tgt not in supported_langs:
        raise ValueError(f"Unsupported target language code: {tgt}")

    if src == tgt:
        warnings.warn(f"Source and target languages are the same ({src}). Returning original text.")
        return text

    # Import here to avoid circular dependencies
    from bilingual.models.translate import translate_text

    return translate_text(
        model=model_name, 
        text=text, 
        src_lang=src, 
        tgt_lang=tgt, 
        version=version,
        **kwargs
    )


def readability_check(
    text: Union[str, List[str]],
    lang: Optional[str] = None,
    model_name: str = "bilingual-readability",
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Check readability level of text.

    Args:
        text: Input text or a list of texts
        lang: Language code ('bn' or 'en'), auto-detected if None
        model_name: Name of the readability model

    Returns:
        Dictionary with readability metrics or a list of dictionaries

    Examples:
        >>> readability_check("আমি স্কুলে যাই।", lang="bn")
        {'level': 'elementary', 'age_range': '6-8', 'score': 2.5}
    """
    warnings.warn("The 'readability_check' function is using a simple heuristic and not a trained model.")
    if isinstance(text, list):
        return [readability_check(t, lang, model_name) for t in text]
    if lang is not None and lang not in {"bn", "en"}:
        warnings.warn(
            f"Unsupported language code '{lang}' in readability_check; falling back to auto-detection."
        )
        lang = None

    if lang is None:
        lang = detect_language(text)

    # Implement actual readability checking using linguistic features
    # Extract features for better readability assessment
    features = _extract_readability_features(text, lang)
    return _calculate_readability_score(features, lang)


def _extract_readability_features(text: str, lang: str) -> Dict[str, float]:
    """
    Extract features for readability scoring.

    Args:
        text: Input text
        lang: Language code

    Returns:
        Dictionary of readability features
    """
    import re

    # Split into sentences and words
    if lang == "bn":
        # For Bangla, use simple sentence splitting
        sentences = re.split(r"[।!?]", text)
        words = re.findall(r"\w+", text)
    else:
        sentences = re.split(r"[.!?]+", text)
        words = re.findall(r"\b\w+\b", text)

    sentences = [s.strip() for s in sentences if s.strip()]
    words = [w for w in words if w.strip()]

    num_sentences = len(sentences)
    num_words = len(words)

    if num_words == 0:
        return {
            "avg_words_per_sentence": 0,
            "avg_word_length": 0,
            "avg_syllables_per_word": 0,
            "complexity_ratio": 0,
        }

    # Average words per sentence
    avg_words_per_sentence = num_words / max(num_sentences, 1)

    # Average word length
    avg_word_length = sum(len(w) for w in words) / num_words

    # Simple syllable estimation (works for both languages approximately)
    def estimate_syllables(word):
        # Simple heuristic: count vowel groups
        vowels = "aeiou" if lang == "en" else "অআইঈউঊএঐওঔ"
        syllables = 0
        prev_vowel = False
        for char in word:
            is_vowel = char.lower() in vowels if lang == "en" else char in vowels
            if is_vowel and not prev_vowel:
                syllables += 1
            prev_vowel = is_vowel
        return max(syllables, 1)

    syllables = sum(estimate_syllables(w) for w in words)
    avg_syllables_per_word = syllables / num_words

    # Complexity ratio (words with >6 characters / total words)
    complex_words = sum(1 for w in words if len(w) > 6)
    complexity_ratio = complex_words / num_words

    return {
        "avg_words_per_sentence": avg_words_per_sentence,
        "avg_word_length": avg_word_length,
        "avg_syllables_per_word": avg_syllables_per_word,
        "complexity_ratio": complexity_ratio,
    }


def _calculate_readability_score(features: Dict[str, float], lang: str) -> Dict[str, Any]:
    """
    Calculate readability score based on extracted features.

    Args:
        features: Dictionary of readability features
        lang: Language code

    Returns:
        Readability assessment dictionary
    """
    # Simple scoring model (can be replaced with trained model)
    score = 0.0

    # Weight different features
    score += features["avg_words_per_sentence"] * 0.3
    score += features["avg_word_length"] * 0.2
    score += features["avg_syllables_per_word"] * 0.3
    score += features["complexity_ratio"] * 0.2

    # Normalize score to 0-10 scale
    score = min(max(score, 0), 10)

    # Determine level based on score
    if score < 3:
        level = "elementary"
        age_range = "6-8"
    elif score < 6:
        level = "intermediate"
        age_range = "9-12"
    else:
        level = "advanced"
        age_range = "13+"

    return {
        "level": level,
        "age_range": age_range,
        "score": score,
        "language": lang,
    }


def safety_check(
    text: Union[str, List[str]],
    lang: Optional[str] = None,
    model_name: str = "bilingual-safety",
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Check if text is safe and appropriate for children.

    Args:
        text: Input text or a list of texts
        lang: Language code ('bn' or 'en'), auto-detected if None
        model_name: Name of the safety model

    Returns:
        Dictionary with safety assessment or a list of dictionaries

    Examples:
        >>> safety_check("This is a nice story about rabbits.")
        {'is_safe': True, 'confidence': 0.95, 'flags': [], 'recommendation': 'approved'}
    """
    warnings.warn("The 'safety_check' function is using a simple heuristic and not a trained model.")
    if isinstance(text, list):
        return [safety_check(t, lang, model_name) for t in text]
    if lang is not None and lang not in {"bn", "en"}:
        warnings.warn(
            f"Unsupported language code '{lang}' in safety_check; falling back to auto-detection."
        )
        lang = None

    if lang is None:
        lang = detect_language(text)

    # Implement actual safety checking with keyword-based filtering
    # Enhanced safety check for child-friendly content
    unsafe_keywords = {
        "en": ["violence", "hate", "kill", "death", "blood", "weapon", "drug", "alcohol"],
        "bn": ["হিংসা", "ঘৃণা", "মারা", "মৃত্যু", "রক্ত", "অস্ত্র", "মাদক", "মদ"],
    }

    text_lower = text.lower()
    flags = []

    # Check for unsafe keywords
    keywords = unsafe_keywords.get(lang, unsafe_keywords["en"])
    for keyword in keywords:
        if keyword in text_lower:
            flags.append(f"Contains potentially unsafe content: {keyword}")

    # Basic safety assessment
    is_safe = len(flags) == 0
    confidence = 0.9 if is_safe else 0.6

    return {
        "is_safe": is_safe,
        "confidence": confidence,
        "flags": flags,
        "recommendation": "approved" if is_safe else "review_required",
        "language": lang,
    }


def classify(
    text: Union[str, List[str]],
    labels: List[str],
    model_name: str = "bilingual-classifier",
    **kwargs,
) -> Union[Dict[str, float], List[Dict[str, float]]]:
    """
    Classify text into one or more categories.

    Args:
        text: Input text or a list of texts
        labels: List of possible labels
        model_name: Name of the classification model
        **kwargs: Additional classification parameters

    Returns:
        Dictionary mapping labels to confidence scores or a list of dictionaries

    Examples:
        >>> classify("This is a story about animals.", labels=["story", "news", "dialogue"])
        {'story': 0.85, 'news': 0.05, 'dialogue': 0.10}
    """
    warnings.warn("The 'classify' function is using a simple heuristic and not a trained model.")
    if isinstance(text, list):
        return [classify(t, labels, model_name, **kwargs) for t in text]
    # Implement actual text classification using simple heuristics
    # Simple rule-based classification for common categories
    if not labels:
        raise ValueError("labels must be a non-empty list")

    text_lower = text.lower()
    scores = {}

    # Define category keywords
    category_keywords = {
        "story": ["story", "tale", "once upon", "গল্প", "কাহিনী", "একদা"],
        "news": ["news", "report", "announce", "খবর", "সংবাদ", "রিপোর্ট"],
        "dialogue": ["said", "asked", "replied", "বলল", "জিজ্ঞাসা", "উত্তর"],
        "poetry": ["poem", "verse", "rhyme", "কবিতা", "ছন্দ", "পদ্য"],
        "instruction": ["how to", "step by step", "guide", "কীভাবে", "ধাপে ধাপে", "নির্দেশনা"],
    }

    # Calculate scores based on keyword matches
    for label in labels:
        keywords = category_keywords.get(label.lower(), [])
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        # Normalize score based on number of keywords
        score = matches / max(len(keywords), 1) if keywords else 0
        scores[label] = min(score, 1.0)  # Cap at 1.0

    # Ensure at least some score is distributed if no matches
    if sum(scores.values()) == 0:
        base_score = 1.0 / len(labels)
        scores = {label: base_score for label in labels}
    else:
        # Normalize scores to sum to 1.0
        total = sum(scores.values())
        scores = {label: score / total for label, score in scores.items()}

    return scores


def batch_process(texts: List[str], operation: str, **kwargs) -> List[Any]:
    """
    Process multiple texts in batch for improved efficiency.

    Args:
        texts: List of input texts to process
        operation: Type of operation ('tokenize', 'normalize', 'generate',
            'translate', 'readability_check', 'safety_check', 'classify')
        **kwargs: Additional arguments for the operation

    Returns:
        List of results corresponding to each input text

    Examples:
        >>> texts = ["Hello world", "আমি বাংলায় কথা বলি"]
        >>> results = batch_process(texts, 'tokenize')
        >>> len(results)
        2
    """
    if not isinstance(texts, list):
        raise TypeError("texts must be a list of strings")

    if operation == "tokenize":
        return tokenize(texts, **kwargs)
    elif operation == "normalize":
        return normalize_text(texts, **kwargs)
    elif operation == "generate":
        return generate(texts, **kwargs)
    elif operation == "translate":
        return translate(texts, **kwargs)
    elif operation == "readability_check":
        return readability_check(texts, **kwargs)
    elif operation == "safety_check":
        return safety_check(texts, **kwargs)
    elif operation == "classify":
        return classify(texts, **kwargs)
    else:
        raise ValueError(f"Unsupported operation: {operation}")


def fine_tune_model(
    model_name: str,
    train_data: List[Dict[str, str]],
    output_dir: str,
    epochs: int = 3,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    **kwargs,
) -> str:
    """
    Fine-tune a language model on custom data.
    Now delegated to bilingual.training.manager.
    """
    raise NotImplementedError("Fine-tuning is not yet implemented.")


def list_available_models(base_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """List available local models and related artifacts.

    This inspects the package's ``models/`` directory and returns a
    structured summary that can be used for debugging or tooling.

    Args:
        base_dir: Optional base directory to inspect. Defaults to the
            project root inferred from this file.

    Returns:
        Dictionary describing available models, tokenizers, and
        classifier/checkpoint subdirectories.
    """
    # Resolve base models directory
    if base_dir is None:
        package_root = Path(__file__).resolve().parent.parent
        models_dir = package_root / "models"
    else:
        models_dir = Path(base_dir)

    summary: Dict[str, Any] = {
        "root": str(models_dir),
        "exists": models_dir.exists(),
        "tokenizer": {},
        "subdirs": {},
    }

    if not models_dir.exists():
        return summary

    # Tokenizer info
    tokenizer_info: Dict[str, Any] = {}
    tokenizer_dir = models_dir / "tokenizer"
    if tokenizer_dir.exists():
        tokenizer_files = sorted(p.name for p in tokenizer_dir.iterdir() if p.is_file())
        tokenizer_info["path"] = str(tokenizer_dir)
        tokenizer_info["files"] = tokenizer_files

        # Try to infer model file name
        model_files = [f for f in tokenizer_files if f.endswith(".model")]
        vocab_files = [f for f in tokenizer_files if f.endswith(".vocab")]
        if model_files:
            tokenizer_info["model_file"] = model_files[0]
        if vocab_files:
            tokenizer_info["vocab_file"] = vocab_files[0]

    # Also check for legacy top-level tokenizer vocab
    legacy_vocab = models_dir / "bilingual-tokenizer.vocab"
    if legacy_vocab.exists():
        tokenizer_info.setdefault("legacy", {})["vocab"] = str(legacy_vocab)

    summary["tokenizer"] = tokenizer_info

    # Subdirectory-based models (classifiers, LMs, etc.)
    for entry in sorted(models_dir.iterdir()):
        if not entry.is_dir() or entry.name in {"tokenizer"}:
            continue

        sub_summary: Dict[str, Any] = {
            "path": str(entry),
            "files": [],
            "has_training_args": False,
            "has_task_config": False,
        }

        for child in entry.iterdir():
            if child.is_file():
                name = child.name
                sub_summary["files"].append(name)
                if name == "training_args.json":
                    sub_summary["has_training_args"] = True
                if name == "task_config.json":
                    sub_summary["has_task_config"] = True

        summary["subdirs"][entry.name] = sub_summary

    return summary


