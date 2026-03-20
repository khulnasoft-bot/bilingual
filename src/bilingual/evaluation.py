"""
Evaluation metrics and utilities for the Bilingual NLP Toolkit.

Provides metrics for generation, translation, and classification tasks.
"""

import warnings
from typing import Any, Dict, List


def compute_bleu(
    predictions: List[str],
    references: List[List[str]],
) -> float:
    """
    Compute BLEU score for translation.
    Args:
        predictions: List of predicted translations
        references: List of reference translations (can have multiple refs per prediction)

    Returns:
        BLEU score (0-100)
    """
    try:
        from sacrebleu import corpus_bleu

        # Convert references format if needed
        if isinstance(references[0], str):
            references = [[ref] for ref in references]

        # Transpose references for sacrebleu format
        refs_transposed = list(zip(*references))

        bleu = corpus_bleu(predictions, refs_transposed)
        return float(bleu.score)

    except ImportError:
        warnings.warn("sacrebleu not installed. Install with: pip install sacrebleu")
        return 0.0


def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE scores for generation.

    Args:
        predictions: List of predicted texts
        references: List of reference texts

    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

        scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            for key in scores:
                scores[key] += score[key].fmeasure

        # Average scores
        n = len(predictions)
        scores = {k: v / n for k, v in scores.items()}

        return scores

    except ImportError:
        warnings.warn("rouge-score not installed. Install with: pip install rouge-score")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def compute_perplexity(
    model: Any,
    texts: List[str],
) -> float:
    """
    Compute perplexity of texts under a language model.

    Args:
        model: Language model
        texts: List of texts to evaluate

    Returns:
        Average perplexity
    """
    # Placeholder - will be implemented with actual model
    return 0.0


def compute_accuracy(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Predicted labels
        references: True labels

    Returns:
        Accuracy (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    correct = sum(p == r for p, r in zip(predictions, references))
    return correct / len(predictions)


def compute_f1(
    predictions: List[str],
    references: List[str],
    average: str = "macro",
) -> float:
    """
    Compute F1 score for classification.

    Args:
        predictions: Predicted labels
        references: True labels
        average: Averaging method ('micro', 'macro', 'weighted')

    Returns:
        F1 score (0-1)
    """
    try:
        from sklearn.metrics import f1_score

        score = f1_score(references, predictions, average=average)
        return float(score)

    except ImportError:
        warnings.warn("scikit-learn not installed. Install with: pip install scikit-learn")
        return 0.0


def evaluate_model(
    dataset_path: str,
    model_name: str,
    metric: str = "all",
) -> Dict[str, Any]:
    """
    Evaluate a model on a dataset.

    Args:
        dataset_path: Path to evaluation dataset
        model_name: Name of model to evaluate
        metric: Metric to compute ('all', 'bleu', 'rouge', 'accuracy', etc.)

    Returns:
        Dictionary of evaluation results
    """
    from bilingual import api as bb
    from bilingual.data_utils import BilingualDataset

    # Load dataset
    dataset = BilingualDataset(file_path=dataset_path)

    # Load model (not used yet in placeholder implementation)
    _ = bb.load_model(model_name)

    results = {}

    # Compute requested metrics
    # This is a placeholder - actual implementation depends on task type
    warnings.warn("Model evaluation not fully implemented yet")

    results["dataset"] = dataset_path
    results["model"] = model_name
    results["num_samples"] = str(len(dataset))


"""
Evaluation metrics and utilities for bilingual models.

Provides metrics for generation, translation, and classification tasks.
"""

import math
import re
import logging
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional

from bilingual.exceptions import EvaluationError

logger = logging.getLogger(__name__)

# Try internal imports
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.translate.meteor_score import meteor_score
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Standardizing on fallback tokenization.")

class BilingualEvaluator:
    """
    Consolidated evaluator for bilingual NLP tasks.
    Replaces fragmented and duplicated evaluation logic.
    """
    
    def __init__(self):
        self.smoothing = SmoothingFunction().method4 if NLTK_AVAILABLE else None
        self._ensure_resources()

    def _ensure_resources(self):
        """Prepare NLTK resources silently."""
        if NLTK_AVAILABLE:
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)

    def tokenize(self, text: str) -> List[str]:
        """Unified tokenization for all metrics."""
        if not text: return []
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text.lower())
            except Exception:
                pass
        # Fallback regex tokenization
        return re.sub(r"[^\w\s]", " ", text.lower()).split()

    def compute_bleu(self, reference: str, candidate: str) -> float:
        """Compute BLEU score (0-1)."""
        ref_tokens = [self.tokenize(reference)]
        cand_tokens = self.tokenize(candidate)
        
        if not ref_tokens[0] or not cand_tokens:
            return 0.0
            
        try:
            if NLTK_AVAILABLE:
                return float(sentence_bleu(ref_tokens, cand_tokens, smoothing_function=self.smoothing))
            return self._fallback_bleu(ref_tokens[0], cand_tokens)
        except Exception as e:
            logger.error(f"BLEU computation failed: {e}")
            return 0.0

    def _fallback_bleu(self, ref_tokens: List[str], cand_tokens: List[str]) -> float:
        """Simplified BLEU for environments without NLTK."""
        overlap = len(set(ref_tokens) & set(cand_tokens))
        precision = overlap / len(cand_tokens) if cand_tokens else 0
        
        # Brevity penalty
        bp = math.exp(1 - len(ref_tokens)/len(cand_tokens)) if len(cand_tokens) < len(ref_tokens) else 1.0
        return bp * precision

    def compute_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """Compute basic ROUGE-like overlap metrics."""
        ref_tokens = self.tokenize(reference)
        cand_tokens = self.tokenize(candidate)
        
        if not ref_tokens or not cand_tokens:
            return {"rouge_1": 0.0}
            
        overlap = len(set(ref_tokens) & set(cand_tokens))
        precision = overlap / len(cand_tokens)
        recall = overlap / len(ref_tokens)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {"rouge_1": f1}

    def evaluate_batch(self, references: List[str], candidates: List[str], task: str = "translation") -> Dict[str, Any]:
        """Perform batch evaluation for a task."""
        if len(references) != len(candidates):
            raise EvaluationError("Batch mismatch: references and candidates must have same length.")
            
        results = {"bleu": [], "rouge": []}
        for ref, cand in zip(references, candidates):
            results["bleu"].append(self.compute_bleu(ref, cand))
            results["rouge"].append(self.compute_rouge(ref, cand)["rouge_1"])
            
        count = len(references)
        summary = {
            "avg_bleu": sum(results["bleu"]) / count if count > 0 else 0,
            "avg_rouge": sum(results["rouge"]) / count if count > 0 else 0,
            "sample_count": count
        }
        return summary

# Global instance
_evaluator = BilingualEvaluator()

def evaluate_translation(references: List[str], candidates: List[str]) -> Dict[str, Any]:
    return _evaluator.evaluate_batch(references, candidates, task="translation")

def evaluate_generation(references: List[str], candidates: List[str]) -> Dict[str, Any]:
    return _evaluator.evaluate_batch(references, candidates, task="generation")
