import re
import numpy as np
from typing import List
from rapidfuzz import fuzz


TOKENIZER = re.compile(r"\b\w+\b")


def keyword_score(
    query_words: List[str],
    text: str,
    *,
    case_sensitive: bool = False,
    coverage_weight: float = 3.0,
    prefix_weight: float = 0.5,
    prefix_cap: int = 10,
) -> float:
    """
    Universal lexical scoring function for both tag search and embedding support.

    - Normalized tokenization
    - Tunable weights
    - Safe prefix bonus (capped)
    """

    if not query_words or not text:
        return 0.0

    # ---- Normalize ----
    if not case_sensitive:
        query_words = [q.lower() for q in query_words]
        text = text.lower()

    words = TOKENIZER.findall(text)

    if not words:
        return 0.0

    # ---- Core overlap ----
    query_set = set(query_words)
    word_set = set(words)

    overlap = len(query_set & word_set)
    coverage = overlap / len(query_set)

    # ---- Prefix matching (controlled) ----
    prefix_hits = sum(1 for qw in query_words for w in words if w.startswith(qw))
    prefix_hits = min(prefix_hits, prefix_cap)

    # ---- Final score ----
    return coverage * coverage_weight + prefix_hits * prefix_weight


def fuzzy_score(query, text):
    return fuzz.partial_ratio(query, text) / 100


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    vec1 = np.asarray(vec1, dtype=np.float32)
    vec2 = np.asarray(vec2, dtype=np.float32)
    dot = np.dot(vec1, vec2)
    norm1 = np.sum(vec1**2)
    norm2 = np.sum(vec2**2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (np.sqrt(norm1) * np.sqrt(norm2))
