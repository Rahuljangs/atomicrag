"""LangChain integration adapters for AtomicRAG.

Provides LangChainLLMAdapter and LangChainEmbeddingAdapter classes that
wrap any LangChain BaseLLM or Embeddings object to satisfy the BaseLLM
and BaseEmbedding protocols. Requires langchain-core; install with
``pip install atomicrag[langchain]``.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseLLM


def _import_langchain_core():
    """Lazily import langchain_core."""
    try:
        import langchain_core
        return langchain_core
    except ImportError as e:
        raise ImportError(
            "LangChain integration requires the langchain-core package. "
            "Install it with: pip install atomicrag[langchain]"
        ) from e


class LangChainLLMAdapter:
    """Adapter that wraps a LangChain BaseLLM to satisfy the BaseLLM protocol.

    Use this to plug any LangChain-compatible LLM into AtomicRAG pipelines.
    """

    def __init__(self, llm: "BaseLLM"):
        """Initialize the adapter with a LangChain LLM.

        Args:
            llm: A LangChain BaseLLM instance (e.g. ChatOpenAI, ChatOllama).
        """
        _import_langchain_core()
        self._llm = llm

    def generate(self, prompt: str) -> str:
        """Send a prompt and return the model's text response.

        Delegates to the wrapped LLM's invoke method.

        Args:
            prompt: The user prompt to send.

        Returns:
            The model's generated text.
        """
        response = self._llm.invoke(prompt)
        if hasattr(response, "content"):
            return response.content
        return str(response)


class LangChainEmbeddingAdapter:
    """Adapter that wraps a LangChain Embeddings to satisfy the BaseEmbedding protocol.

    Use this to plug any LangChain-compatible embedding model into AtomicRAG pipelines.
    """

    def __init__(self, embeddings: "Embeddings"):
        """Initialize the adapter with a LangChain Embeddings instance.

        Args:
            embeddings: A LangChain Embeddings instance (e.g. OpenAIEmbeddings).
        """
        _import_langchain_core()
        self._embeddings = embeddings

    def embed_text(self, text: str) -> List[float]:
        """Return the embedding vector for a single piece of text.

        Delegates to the wrapped Embeddings' embed_query or embed_documents.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a list of floats.
        """
        return self._embeddings.embed_query(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return embedding vectors for a batch of texts.

        Uses embed_documents for throughput.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        return self._embeddings.embed_documents(texts)
