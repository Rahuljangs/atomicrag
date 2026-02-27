"""Vector similarity utilities (numpy-only, no external deps)."""
from __future__ import annotations

from typing import List

import numpy as np


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    va, vb = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def top_k_similar(
    query_vec: List[float],
    candidates: List[List[float]],
    k: int = 10,
) -> List[tuple[int, float]]:
    """Return indices and scores of the *k* most similar candidates.

    Args:
        query_vec: The query embedding.
        candidates: List of candidate embeddings.
        k: How many to return.

    Returns:
        List of ``(index, score)`` tuples sorted descending by score.
    """
    if not candidates:
        return []

    q = np.asarray(query_vec)
    mat = np.asarray(candidates)

    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return []

    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0] = 1.0  # avoid division by zero

    scores = mat @ q / (norms * q_norm)

    actual_k = min(k, len(scores))
    if actual_k >= len(scores):
        top_idx = np.argsort(-scores)
    else:
        top_idx = np.argpartition(-scores, actual_k)[:actual_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

    return [(int(i), float(scores[i])) for i in top_idx]
