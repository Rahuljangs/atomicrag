"""Google Gemini integration adapters for AtomicRAG.

Supports both the new ``google-genai`` SDK and the legacy
``google-generativeai`` SDK. Installs with ``pip install atomicrag[gemini]``.
"""
from __future__ import annotations

from typing import List


def _get_genai():
    """Import whichever Gemini SDK is available (new or legacy)."""
    # Try new SDK first (google-genai)
    try:
        from google import genai
        return genai, "new"
    except ImportError:
        pass

    # Fall back to legacy SDK (google-generativeai)
    try:
        import google.generativeai as genai
        return genai, "legacy"
    except ImportError:
        raise ImportError(
            "Gemini integration requires either 'google-genai' or 'google-generativeai'. "
            "Install with: pip install google-genai  OR  pip install google-generativeai"
        )


class GeminiLLM:
    """Thin wrapper around Google Gemini Generate Content API.

    Satisfies the BaseLLM protocol for use with AtomicRAG pipelines.
    Automatically detects and uses whichever SDK is installed.
    """

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash"):
        self._api_key = api_key
        self._model_name = model
        self._client = None
        self._sdk_type = None

    def _init_client(self):
        if self._client is not None:
            return

        genai, sdk_type = _get_genai()
        self._sdk_type = sdk_type

        if sdk_type == "new":
            self._client = genai.Client(api_key=self._api_key)
        else:
            if self._api_key:
                genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self._model_name)

    def generate(self, prompt: str) -> str:
        self._init_client()

        if self._sdk_type == "new":
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
            )
            return response.text or ""
        else:
            response = self._client.generate_content(prompt)
            return response.text or ""


class GeminiEmbedding:
    """Thin wrapper around Google Gemini Embedding API.

    Satisfies the BaseEmbedding protocol for use with AtomicRAG pipelines.
    Automatically detects and uses whichever SDK is installed.
    """

    def __init__(self, api_key: str | None = None, model: str = "models/gemini-embedding-001"):
        self._api_key = api_key
        self._model_name = model
        self._client = None
        self._sdk_type = None

    def _init_client(self):
        if self._client is not None:
            return

        genai, sdk_type = _get_genai()
        self._sdk_type = sdk_type

        if sdk_type == "new":
            self._client = genai.Client(api_key=self._api_key)
        else:
            if self._api_key:
                genai.configure(api_key=self._api_key)
            self._client = genai  # module-level for legacy embed_content

    def embed_text(self, text: str) -> List[float]:
        self._init_client()

        if self._sdk_type == "new":
            result = self._client.models.embed_content(
                model=self._model_name,
                contents=text,
            )
            return list(result.embeddings[0].values)
        else:
            result = self._client.embed_content(model=self._model_name, content=text)
            return result["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        self._init_client()

        if self._sdk_type == "new":
            result = self._client.models.embed_content(
                model=self._model_name,
                contents=texts,
            )
            return [list(e.values) for e in result.embeddings]
        else:
            result = self._client.embed_content(model=self._model_name, content=texts)
            embeddings = result["embedding"]
            if embeddings and isinstance(embeddings[0], (int, float)):
                return [embeddings]
            return embeddings
