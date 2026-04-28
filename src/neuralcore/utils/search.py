import re
from typing import List
import numpy as np
from rapidfuzz import fuzz


TOKENIZER = re.compile(r"\b\w+\b")


def keyword_score(
    query_words: list,
    text,
    *,
    case_sensitive: bool = False,
    coverage_weight: float = 3.0,
    prefix_weight: float = 0.5,
    prefix_cap: int = 10,
) -> float:
    """
    Lexical scoring that is robust to lists or non-string inputs.
    """
    if not query_words or text is None:
        return 0.0

    # Ensure text is a string
    if isinstance(text, list):
        text = " ".join(map(str, text))
    else:
        text = str(text)

    # Normalize case
    if not case_sensitive:
        query_words = [str(q).lower() for q in query_words]
        text = text.lower()

    words = TOKENIZER.findall(text)
    if not words:
        return 0.0

    # Core overlap
    query_set = set(query_words)
    word_set = set(words)
    overlap = len(query_set & word_set)
    coverage = overlap / len(query_set)

    # Prefix bonus
    prefix_hits = sum(1 for qw in query_words for w in words if w.startswith(qw))
    prefix_hits = min(prefix_hits, prefix_cap)

    return coverage * coverage_weight + prefix_hits * prefix_weight


def fuzzy_score(query, text):
    return fuzz.partial_ratio(query, text) / 100


def cosine_sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Fast, robust cosine similarity.
    Compatible with all call sites in ContextManager + KnowledgeConsolidator.
    """
    if vec1 is None or vec2 is None:
        return 0.0

    # Convert + flatten (handles (384,) and (1, 384) safely)
    vec1 = np.asarray(vec1, dtype=np.float32).ravel()
    vec2 = np.asarray(vec2, dtype=np.float32).ravel()

    if vec1.size == 0 or vec2.size == 0:
        return 0.0

    if vec1.shape != vec2.shape:
        return 0.0

    if not np.isfinite(vec1).all() or not np.isfinite(vec2).all():
        return 0.0

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    # Pure numpy dot product (fastest path)
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def cosine_sim_batch(
    query_emb: np.ndarray,
    embeddings: List[np.ndarray],
) -> np.ndarray:
    """
    Vectorized cosine similarity — much faster than looping.
    Returns array of similarities in the same order as embeddings.
    """
    if not embeddings:
        return np.array([])

    # Stack all embeddings into a matrix (N x D)
    emb_matrix = np.vstack(
        [e.ravel() for e in embeddings if e is not None and e.size > 0]
    )

    if emb_matrix.size == 0:
        return np.zeros(len(embeddings))

    query = query_emb.ravel().astype(np.float32)

    # Normalize
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(len(embeddings))

    emb_norms = np.linalg.norm(emb_matrix, axis=1)
    emb_norms[emb_norms == 0] = 1.0  # avoid div by zero

    # Batch dot product
    dots = emb_matrix @ query
    sims = dots / (query_norm * emb_norms)

    return sims.astype(np.float32)
