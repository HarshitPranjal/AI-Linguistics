"""Cross-lingual semantic equivalence verification for paired parallel prompts."""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class SemanticVerifier:
    """Verifies that translations and code-mixed variants retain the semantic intent of anchor prompts."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model: Optional[SentenceTransformer] = None
        self._load_failed = False

    def _get_model(self) -> Optional[SentenceTransformer]:
        if not SENTENCE_TRANSFORMERS_AVAILABLE or self._load_failed:
            return None
        if self._model is None:
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer model {self.model_name}: {e}. Using n-gram fallback.")
                self._load_failed = True
                return None
        return self._model

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Computes cosine similarity between two texts (0.0 to 1.0)."""
        model = self._get_model()
        if model is not None:
            try:
                embeddings = model.encode([text_a, text_b], convert_to_numpy=True)
                emb_a, emb_b = embeddings[0], embeddings[1]
                norm_a = np.linalg.norm(emb_a)
                norm_b = np.linalg.norm(emb_b)
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
                return max(0.0, min(1.0, (sim + 1.0) / 2.0)) # normalized 0-1
            except Exception as e:
                logger.debug(f"Error computing neural embedding similarity: {e}")

        # Fallback heuristic: token-level jaccard + length penalty
        return self._fallback_lexical_similarity(text_a, text_b)

    def _fallback_lexical_similarity(self, text_a: str, text_b: str) -> float:
        """Lightweight fallback similarity for offline / test environments."""
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a.intersection(tokens_b)
        union = tokens_a.union(tokens_b)
        jaccard = len(intersection) / len(union) if union else 0.0
        len_ratio = min(len(text_a), len(text_b)) / max(len(text_a), len(text_b))
        return 0.5 * jaccard + 0.5 * len_ratio

    def verify_bundle_consistency(self, english: str, hindi: str, hinglish: str, bhojpuri: str, threshold: float = 0.50) -> dict[str, float]:
        """Checks semantic consistency between English anchor and each target variety."""
        scores = {
            "hi_similarity": self.compute_similarity(english, hindi),
            "hinglish_similarity": self.compute_similarity(english, hinglish),
            "bhojpuri_similarity": self.compute_similarity(english, bhojpuri),
        }
        return scores

