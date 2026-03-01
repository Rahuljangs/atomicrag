"""Ollama integration adapters for AtomicRAG.

Provides OllamaLLM and OllamaEmbedding classes that satisfy the BaseLLM
and BaseEmbedding protocols. Requires the ollama package; install with
``pip install atomicrag[ollama]``. Assumes Ollama is running locally
or at the configured host.
"""

from __future__ import annotations

from typing import List


def _import_ollama():
    """Lazily import the ollama package."""
    try:
        import ollama

        return ollama
    except ImportError as e:
        raise ImportError(
            "Ollama integration requires the ollama package. "
            "Install it with: pip install atomicrag[ollama]"
        ) from e


class OllamaLLM:
    """Thin wrapper around Ollama Chat API.

    Satisfies the BaseLLM protocol for use with AtomicRAG pipelines.
    """

    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3"):
        """Initialize the Ollama LLM adapter.

        Args:
            host: Ollama server URL (default http://localhost:11434).
            model: Model name (e.g. llama3, mistral, gemma).
        """
        self._host = host
        self._model = model
        self._client = None

    def _get_client(self):
        """Lazily create the Ollama client."""
        if self._client is None:
            ollama = _import_ollama()
            self._client = ollama.Client(host=self._host)
        return self._client

    def generate(self, prompt: str) -> str:
        """Send a prompt and return the model's text response.

        Args:
            prompt: The user prompt to send.

        Returns:
            The model's generated text.
        """
        client = self._get_client()
        response = client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content or ""


class OllamaEmbedding:
    """Thin wrapper around Ollama Embed API.

    Satisfies the BaseEmbedding protocol for use with AtomicRAG pipelines.
    """

    def __init__(self, host: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        """Initialize the Ollama embedding adapter.

        Args:
            host: Ollama server URL (default http://localhost:11434).
            model: Embedding model name (e.g. nomic-embed-text).
        """
        self._host = host
        self._model = model
        self._client = None

    def _get_client(self):
        """Lazily create the Ollama client."""
        if self._client is None:
            ollama = _import_ollama()
            self._client = ollama.Client(host=self._host)
        return self._client

    def embed_text(self, text: str) -> List[float]:
        """Return the embedding vector for a single piece of text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a list of floats.
        """
        client = self._get_client()
        response = client.embed(model=self._model, input=text)
        return response.embeddings[0]

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
        response = client.embed(model=self._model, input=texts)
        return response.embeddings
