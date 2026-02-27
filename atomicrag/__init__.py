"""AtomicRAG: Graph-based RAG using atomic knowledge units and entity graphs.

Quick start::

    from atomicrag import IndexPipeline, RetrievePipeline, AtomicRAGConfig

    graph = IndexPipeline(llm=my_llm, embedding=my_emb).run(["document text"])
    results = RetrievePipeline(graph=graph, llm=my_llm, embedding=my_emb).search("query")
"""

__version__ = "0.1.0"

from atomicrag.config import AtomicRAGConfig
from atomicrag.index.pipeline import IndexPipeline
from atomicrag.retrieve.pipeline import RetrievePipeline

__all__ = ["IndexPipeline", "RetrievePipeline", "AtomicRAGConfig"]
