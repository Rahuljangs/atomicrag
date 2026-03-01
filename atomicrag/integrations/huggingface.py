"""Hugging Face integration adapter for AtomicRAG.

Provides HuggingFaceEmbedding class that satisfies the BaseEmbedding
protocol using SentenceTransformer models. Requires sentence-transformers;
install with ``pip install atomicrag[huggingface]``.
"""

from __future__ import annotations

from typing import List


def _import_sentence_transformers():
    """Lazily import the sentence_transformers package."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Hugging Face embedding integration requires the sentence-transformers package. "
            "Install it with: pip install atomicrag[huggingface]"
        ) from e


class HuggingFaceEmbedding:
    """Thin wrapper around SentenceTransformer for embeddings.

    Satisfies the BaseEmbedding protocol for use with AtomicRAG pipelines.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """Initialize the Hugging Face embedding adapter.

        Args:
            model_name: Hugging Face model ID (e.g. BAAI/bge-small-en-v1.5).
        """
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        """Lazily load the SentenceTransformer model."""
        if self._model is None:
            SentenceTransformer = _import_sentence_transformers()
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Return the embedding vector for a single piece of text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a list of floats.
        """
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return embedding vectors for a batch of texts.

        Uses a single encode call for throughput.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]
