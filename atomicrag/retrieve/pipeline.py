"""RetrievePipeline — query a KnowledgeGraph using Q-Iter."""

from __future__ import annotations

import logging
from typing import Optional

from atomicrag.config import AtomicRAGConfig
from atomicrag.models.graph import KnowledgeGraph
from atomicrag.models.protocols import BaseEmbedding, BaseLLM
from atomicrag.models.results import RetrievalResult
from atomicrag.retrieve.anchoring import Anchoring
from atomicrag.retrieve.ranking import ResultRanker
from atomicrag.retrieve.traversal import GraphTraversal

logger = logging.getLogger(__name__)


class RetrievePipeline:
    """Query a ``KnowledgeGraph`` using the Q-Iter algorithm.

    Chains three steps:

    1. **Anchoring** — find seed entities and KUs from the query
    2. **Traversal** — iteratively expand through the graph
    3. **Ranking** — score, group, and return top results

    Example::

        retriever = RetrievePipeline(graph=graph, llm=my_llm, embedding=my_emb)
        results = retriever.search("What features does RHEL 9 have?")
        for item in results.items:
            print(item.score, item.content)
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        llm: BaseLLM,
        embedding: BaseEmbedding,
        config: Optional[AtomicRAGConfig] = None,
    ):
        self.config = config or AtomicRAGConfig()
        self.graph = graph
        self.llm = llm
        self.embedding = embedding

        self.anchoring = Anchoring(graph=graph, llm=llm, embedding=embedding, config=self.config)
        self.traversal = GraphTraversal(graph=graph, config=self.config)
        self.ranker = ResultRanker(graph=graph, config=self.config)

    def search(self, query: str) -> RetrievalResult:
        """Search the knowledge graph for information relevant to *query*.

        Args:
            query: The user's natural-language question.

        Returns:
            A ``RetrievalResult`` with ranked items.
        """
        verbose = self.config.verbose

        # Step 1: Anchoring
        if verbose:
            print(f"[1/3] Anchoring query: '{query[:60]}...'")

        entity_ids, ku_ids, query_entities, query_embedding = self.anchoring.anchor(query)

        if verbose:
            print(
                f"  -> {len(entity_ids)} entities, {len(ku_ids)} seed KUs, "
                f"query entities: {query_entities}"
            )

        if not entity_ids and not ku_ids:
            if verbose:
                print("  -> No anchors found, returning empty result")
            return RetrievalResult(query=query, entities_extracted=query_entities)

        # Step 2: Graph Traversal
        if verbose:
            print(f"[2/3] Traversing graph (depth={self.config.traversal_depth})...")

        all_ku_ids = self.traversal.traverse(entity_ids, ku_ids, query_embedding)

        if verbose:
            print(f"  -> {len(all_ku_ids)} KUs collected")

        # Step 3: Ranking
        if verbose:
            print(f"[3/3] Ranking and returning top-{self.config.result_top_n}...")

        result = self.ranker.rank(all_ku_ids, query, query_embedding, query_entities)

        if verbose:
            print(f"  -> {len(result.items)} items returned")
            for i, item in enumerate(result.items, 1):
                print(f"     {i}. score={item.score:.3f}: {item.content[:60]}...")

        return result
