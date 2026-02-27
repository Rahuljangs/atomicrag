"""Integration adapters for popular LLM and embedding providers.

These thin wrappers satisfy the BaseLLM and BaseEmbedding protocols,
enabling use with AtomicRAG pipelines without custom glue code.
"""

from atomicrag.integrations.openai import OpenAIEmbedding, OpenAILLM
from atomicrag.integrations.gemini import GeminiEmbedding, GeminiLLM
from atomicrag.integrations.ollama import OllamaEmbedding, OllamaLLM
from atomicrag.integrations.langchain import (
    LangChainEmbeddingAdapter,
    LangChainLLMAdapter,
)
from atomicrag.integrations.huggingface import HuggingFaceEmbedding

__all__ = [
    "OpenAILLM",
    "OpenAIEmbedding",
    "GeminiLLM",
    "GeminiEmbedding",
    "OllamaLLM",
    "OllamaEmbedding",
    "LangChainLLMAdapter",
    "LangChainEmbeddingAdapter",
    "HuggingFaceEmbedding",
]
