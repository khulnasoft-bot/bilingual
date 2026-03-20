"""
Translation model implementations.

Handles Bangla ↔ English translation.
"""

import warnings
from typing import Any, List


from typing import Any, List, Optional
from bilingual.models.manager import model_manager

def translate_text(
    model: str, 
    text: str, 
    src_lang: str = "bn", 
    tgt_lang: str = "en", 
    version: Optional[str] = None,
    **kwargs
) -> str:
    """
    Translate text using the central ModelManager pipeline.
    """
    # Use ModelManager to get a cached pipeline
    try:
        pipe = model_manager.get_pipeline(
            "translation", 
            model_name=model, 
            version=version,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            **kwargs
        )
        
        result = pipe(text)
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("translation_text", "")
        return ""
        
    except Exception as e:
        warnings.warn(f"Error during translation: {e}")
        return text


def batch_translate(
    model: str,
    texts: List[str],
    src_lang: str = "bn",
    tgt_lang: str = "en",
    version: Optional[str] = None,
    batch_size: int = 8,
    **kwargs
) -> List[str]:
    """
    Translate multiple texts using vectorized batching.
    """
    try:
        results = model_manager.predict_batch(
            model_name=model,
            inputs=texts,
            task="translation",
            version=version,
            batch_size=batch_size,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            **kwargs
        )
        
        return [r.get("translation_text", "") if isinstance(r, dict) else r for r in results]
    except Exception as e:
        warnings.warn(f"Batch translation failed, falling back to sequential: {e}")
        return [translate_text(model, t, src_lang, tgt_lang, version, **kwargs) for t in texts]
