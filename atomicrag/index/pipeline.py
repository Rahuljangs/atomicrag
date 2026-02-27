"""IndexPipeline — the main entry point for building a knowledge graph."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union

from atomicrag.config import AtomicRAGConfig
from atomicrag.index.chunker import TextChunker
from atomicrag.index.entity_extractor import EntityExtractor
from atomicrag.index.extractor import KnowledgeUnitExtractor
from atomicrag.index.graph_builder import GraphBuilder
from atomicrag.models.graph import KnowledgeGraph
from atomicrag.models.protocols import BaseEmbedding, BaseLLM

logger = logging.getLogger(__name__)


class IndexPipeline:
    """Build a ``KnowledgeGraph`` from raw documents.

    This is the primary offline pipeline.  It chains:

    1. **TextChunker** — split documents into chunks
    2. **KnowledgeUnitExtractor** — extract atomic facts via LLM
    3. **EntityExtractor** — extract and deduplicate entities
    4. **GraphBuilder** — embed everything and assemble the graph

    Example::

        from atomicrag import IndexPipeline

        graph = IndexPipeline(llm=my_llm, embedding=my_emb).run([
            "First document text ...",
            "Second document text ...",
        ])
        graph.to_json("knowledge_graph.json")
    """

    def __init__(
        self,
        llm: BaseLLM,
        embedding: BaseEmbedding,
        config: Optional[AtomicRAGConfig] = None,
    ):
        self.config = config or AtomicRAGConfig()
        self.llm = llm
        self.embedding = embedding

        self.chunker = TextChunker(config=self.config)
        self.ku_extractor = KnowledgeUnitExtractor(llm=llm, config=self.config)
        self.entity_extractor = EntityExtractor(config=self.config, llm=llm)
        self.graph_builder = GraphBuilder(embedding=embedding, config=self.config)

    def run(
        self,
        documents: Union[List[str], List[dict]],
    ) -> KnowledgeGraph:
        """Process documents and return a ``KnowledgeGraph``.

        Args:
            documents: Either a list of plain strings, or a list of dicts
                with keys ``text`` and optional ``doc_id`` / ``metadata``.

        Returns:
            A fully populated ``KnowledgeGraph``.
        """
        verbose = self.config.verbose
        on_progress = self.config.on_progress
        on_chunk = self.config.on_chunk_processed

        # ---- Step 1: Chunking ----
        if verbose:
            print(f"[1/4] Chunking {len(documents)} documents...")

        all_chunks = []
        for i, doc in enumerate(documents):
            if isinstance(doc, dict):
                text = doc.get("text", "")
                doc_id = doc.get("doc_id", f"doc-{i}")
            else:
                text = str(doc)
                doc_id = f"doc-{i}"

            chunks = self.chunker.chunk(text, doc_id=doc_id)
            all_chunks.extend(chunks)

        if verbose:
            print(f"  -> {len(all_chunks)} chunks")
        if on_progress:
            on_progress(len(all_chunks), len(all_chunks), "Chunking")

        # ---- Step 2: KU Extraction ----
        concurrency = self.config.ku_concurrency
        total = len(all_chunks)
        if verbose:
            mode = "parallel" if concurrency > 1 else "sequential"
            print(f"[2/4] Extracting knowledge units from {total} chunks ({mode}, workers={concurrency})...")

        all_kus = []

        if concurrency <= 1:
            for i, chunk in enumerate(all_chunks):
                kus, _ = self.ku_extractor.extract(chunk)
                all_kus.extend(kus)

                if on_chunk:
                    on_chunk(i + 1, total)
                if on_progress:
                    on_progress(i + 1, total, "KU Extraction")
                if verbose and (i + 1) % 10 == 0:
                    print(f"  -> {i + 1}/{total} chunks, {len(all_kus)} KUs so far")
        else:
            completed = 0
            futures_map = {}
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                for chunk in all_chunks:
                    future = executor.submit(self.ku_extractor.extract, chunk)
                    futures_map[future] = chunk

                for future in as_completed(futures_map):
                    completed += 1
                    try:
                        kus, _ = future.result()
                        all_kus.extend(kus)
                    except Exception as e:
                        chunk = futures_map[future]
                        logger.warning(f"KU extraction failed for chunk {chunk.id[:8]}: {e}")

                    if on_chunk:
                        on_chunk(completed, total)
                    if on_progress:
                        on_progress(completed, total, "KU Extraction")
                    if verbose and completed % 10 == 0:
                        print(f"  -> {completed}/{total} chunks, {len(all_kus)} KUs so far")

        if verbose:
            print(f"  -> {len(all_kus)} knowledge units extracted")

        # ---- Step 3: Entity Extraction ----
        if verbose:
            print(f"[3/4] Extracting entities from {len(all_kus)} knowledge units...")

        entities, ku_entity_map = self.entity_extractor.extract(all_kus)

        if verbose:
            print(f"  -> {len(entities)} unique entities")
        if on_progress:
            on_progress(len(entities), len(entities), "Entity Extraction")

        # ---- Step 4: Graph Building ----
        if verbose:
            print(f"[4/4] Building graph (embedding + assembly)...")

        graph = self.graph_builder.build(all_chunks, all_kus, entities, ku_entity_map)

        if verbose:
            s = graph.stats()
            print(f"  -> Done! {s}")

        return graph
