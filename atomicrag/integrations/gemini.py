"""Google Gemini integration adapters for AtomicRAG.

Provides GeminiLLM and GeminiEmbedding classes that satisfy the BaseLLM
and BaseEmbedding protocols. Requires the google-generativeai package;
install with ``pip install atomicrag[gemini]``.
"""

from __future__ import annotations

from typing import List


def _import_google_genai():
    """Lazily import the google.generativeai package."""
    try:
        import google.generativeai as genai
        return genai
    except ImportError as e:
        raise ImportError(
            "Gemini integration requires the google-generativeai package. "
            "Install it with: pip install atomicrag[gemini]"
        ) from e


class GeminiLLM:
    """Thin wrapper around Google Gemini Generate Content API.

    Satisfies the BaseLLM protocol for use with AtomicRAG pipelines.
    """

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.0-flash"):
        """Initialize the Gemini LLM adapter.

        Args:
            api_key: Google AI API key. If None, uses GOOGLE_API_KEY env var.
            model: Model name (e.g. gemini-2.0-flash, gemini-1.5-pro).
        """
        self._api_key = api_key
        self._model = model
        self._model_instance = None

    def _get_model(self):
        """Lazily create the GenerativeModel instance."""
        if self._model_instance is None:
            genai = _import_google_genai()
            if self._api_key:
                genai.configure(api_key=self._api_key)
            self._model_instance = genai.GenerativeModel(self._model)
        return self._model_instance

    def generate(self, prompt: str) -> str:
        """Send a prompt and return the model's text response.

        Args:
            prompt: The user prompt to send.

        Returns:
            The model's generated text.
        """
        model = self._get_model()
        response = model.generate_content(prompt)
        return response.text or ""


class GeminiEmbedding:
    """Thin wrapper around Google Gemini Embedding API.

    Satisfies the BaseEmbedding protocol for use with AtomicRAG pipelines.
    """

    def __init__(self, api_key: str | None = None, model: str = "models/text-embedding-004"):
        """Initialize the Gemini embedding adapter.

        Args:
            api_key: Google AI API key. If None, uses GOOGLE_API_KEY env var.
            model: Embedding model name (e.g. models/text-embedding-004).
        """
        self._api_key = api_key
        self._model = model
        self._configured = False

    def _ensure_configured(self):
        """Lazily configure the API."""
        if not self._configured:
            genai = _import_google_genai()
            if self._api_key:
                genai.configure(api_key=self._api_key)
            self._configured = True

    def embed_text(self, text: str) -> List[float]:
        """Return the embedding vector for a single piece of text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a list of floats.
        """
        genai = _import_google_genai()
        self._ensure_configured()
        result = genai.embed_content(model=self._model, content=text)
        return result["embedding"]

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
        genai = _import_google_genai()
        self._ensure_configured()
        result = genai.embed_content(model=self._model, content=texts)
        embeddings = result["embedding"]
        # For batch, API may return list of lists or single list; normalize
        if embeddings and isinstance(embeddings[0], (int, float)):
            return [embeddings]
        return embeddings
