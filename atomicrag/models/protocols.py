"""Model-agnostic Protocol interfaces for LLMs and Embedding models.

Any class that implements the methods described here can be used
with AtomicRAG — no inheritance or registration required.

Example — bring your own LLM::

    class MyLLM:
        def generate(self, prompt: str) -> str:
            return my_api_call(prompt)

    pipeline = IndexPipeline(llm=MyLLM(), ...)

Example — bring your own Embedding::

    class MyEmbed:
        def embed_text(self, text: str) -> list[float]:
            return encoder.encode(text).tolist()
        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return encoder.encode(texts).tolist()

    pipeline = IndexPipeline(embedding=MyEmbed(), ...)
"""
from __future__ import annotations

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class BaseLLM(Protocol):
    """Protocol that any LLM wrapper must satisfy.

    Only one method is required: ``generate``.

    Optionally implement ``agenerate`` for async pipelines.
    The library will fall back to running ``generate`` in a thread
    if ``agenerate`` is not present.
    """

    def generate(self, prompt: str) -> str:
        """Send a prompt and return the model's text response."""
        ...


@runtime_checkable
class BaseEmbedding(Protocol):
    """Protocol that any embedding model must satisfy.

    Two methods are required:

    * ``embed_text``  — embed a single string
    * ``embed_batch`` — embed a list of strings (for throughput)
    """

    def embed_text(self, text: str) -> List[float]:
        """Return the embedding vector for a single piece of text."""
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return embedding vectors for a batch of texts.

        Implementations should optimise this for throughput (e.g. a single
        API call) rather than looping over ``embed_text``.
        """
        ...
