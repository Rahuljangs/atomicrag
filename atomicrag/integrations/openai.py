"""OpenAI integration adapters for AtomicRAG.

Provides OpenAILLM and OpenAIEmbedding classes that satisfy the BaseLLM
and BaseEmbedding protocols. Requires the openai package; install with
``pip install atomicrag[openai]``.
"""

from __future__ import annotations

from typing import List


def _import_openai():
    """Lazily import the openai package."""
    try:
        from openai import OpenAI

        return OpenAI
    except ImportError as e:
        raise ImportError(
            "OpenAI integration requires the openai package. "
            "Install it with: pip install atomicrag[openai]"
        ) from e


class OpenAILLM:
    """Thin wrapper around OpenAI Chat Completions API.

    Satisfies the BaseLLM protocol for use with AtomicRAG pipelines.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        """Initialize the OpenAI LLM adapter.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Model name (e.g. gpt-4o-mini, gpt-4o).
        """
        self._api_key = api_key
        self._model = model
        self._client = None

    def _get_client(self):
        """Lazily create the OpenAI client."""
        if self._client is None:
            OpenAI = _import_openai()
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def generate(self, prompt: str) -> str:
        """Send a prompt and return the model's text response.

        Args:
            prompt: The user prompt to send.

        Returns:
            The model's generated text.
        """
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


class OpenAIEmbedding:
    """Thin wrapper around OpenAI Embeddings API.

    Satisfies the BaseEmbedding protocol for use with AtomicRAG pipelines.
    """

    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small"):
        """Initialize the OpenAI embedding adapter.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Embedding model name (e.g. text-embedding-3-small).
        """
        self._api_key = api_key
        self._model = model
        self._client = None

    def _get_client(self):
        """Lazily create the OpenAI client."""
        if self._client is None:
            OpenAI = _import_openai()
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed_text(self, text: str) -> List[float]:
        """Return the embedding vector for a single piece of text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a list of floats.
        """
        client = self._get_client()
        response = client.embeddings.create(model=self._model, input=text)
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return embedding vectors for a batch of texts.

        Uses a single API call for throughput.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        client = self._get_client()
        response = client.embeddings.create(model=self._model, input=texts)
        # API returns embeddings in order; sort by index to handle any reordering
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
