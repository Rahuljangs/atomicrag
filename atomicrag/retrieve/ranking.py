"""Step 3 of Q-Iter: Rank retrieved KUs and map back to chunks."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import numpy as np

from atomicrag.config import AtomicRAGConfig
from atomicrag.models.graph import KnowledgeGraph
from atomicrag.models.results import RetrievalItem, RetrievalResult
from atomicrag.utils.similarity import cosine_similarity

logger = logging.getLogger(__name__)


class ResultRanker:
    """Score retrieved KUs, group by source chunk, and return top results."""

    def __init__(
        self,
        graph: KnowledgeGraph,
        config: Optional[AtomicRAGConfig] = None,
    ):
        cfg = config or AtomicRAGConfig()
        self.graph = graph
        self.top_n = cfg.result_top_n
        self.min_score = cfg.min_score_threshold
        self.group_by_chunk = cfg.group_by_chunk
        self.aggregation = cfg.score_aggregation

    def rank(
        self,
        ku_ids: Set[str],
        query: str,
        query_embedding: List[float],
        query_entities: List[str],
    ) -> RetrievalResult:
        """Rank the collected KUs and produce a ``RetrievalResult``.

        Args:
            ku_ids: Set of KU IDs collected during traversal.
            query: The original query string.
            query_embedding: The query vector.
            query_entities: Entities extracted from the query.

        Returns:
            A ``RetrievalResult`` with ranked items.
        """
        if not ku_ids:
            return RetrievalResult(query=query, entities_extracted=query_entities)

        np.array(query_embedding)

        # Score each KU
        scored_kus: List[dict] = []
        for ku_id in ku_ids:
            ku = self.graph.get_ku(ku_id)
            if not ku or not ku.embedding:
                continue

            score = cosine_similarity(query_embedding, ku.embedding)
            if score < self.min_score:
                continue

            entity_names = []
            for eid in self.graph.entities_for_ku(ku_id):
                entity = self.graph.get_entity(eid)
                if entity:
                    entity_names.append(entity.name)

            scored_kus.append(
                {
                    "ku_id": ku_id,
                    "content": ku.content,
                    "chunk_id": ku.chunk_id,
                    "score": score,
                    "entity_names": entity_names,
                }
            )

        if not scored_kus:
            return RetrievalResult(query=query, entities_extracted=query_entities)

        # Group by chunk or return individual KUs
        if self.group_by_chunk:
            items = self._group_by_chunk(scored_kus)
        else:
            scored_kus.sort(key=lambda x: x["score"], reverse=True)
            items = [
                RetrievalItem(
                    content=ku["content"],
                    score=ku["score"],
                    source_chunk_id=ku["chunk_id"],
                    knowledge_unit_ids=[ku["ku_id"]],
                    entity_names=ku["entity_names"],
                )
                for ku in scored_kus[: self.top_n]
            ]

        return RetrievalResult(
            items=items[: self.top_n],
            query=query,
            entities_extracted=query_entities,
            graph_stats={
                "kus_retrieved": len(ku_ids),
                "kus_scored": len(scored_kus),
                "items_returned": len(items[: self.top_n]),
            },
        )

    def _group_by_chunk(self, scored_kus: List[dict]) -> List[RetrievalItem]:
        """Group KU scores by source chunk and aggregate."""
        chunk_groups: Dict[str, List[dict]] = {}
        for ku in scored_kus:
            chunk_groups.setdefault(ku["chunk_id"], []).append(ku)

        items = []
        for chunk_id, kus in chunk_groups.items():
            scores = [ku["score"] for ku in kus]

            if self.aggregation == "mean":
                agg_score = float(np.mean(scores))
            elif self.aggregation == "max":
                agg_score = float(np.max(scores))
            elif self.aggregation == "sum":
                agg_score = float(np.sum(scores))
            else:
                agg_score = float(np.mean(scores))

            # Merge content from all KUs in this chunk
            all_entity_names = set()
            all_ku_ids = []
            contents = []
            for ku in kus:
                contents.append(ku["content"])
                all_ku_ids.append(ku["ku_id"])
                all_entity_names.update(ku["entity_names"])

            # Try to get original chunk content, fall back to concatenated KUs
            chunk = self.graph.get_chunk(chunk_id)
            content = chunk.content if chunk else "\n".join(contents)

            items.append(
                RetrievalItem(
                    content=content,
                    score=agg_score,
                    source_chunk_id=chunk_id,
                    knowledge_unit_ids=all_ku_ids,
                    entity_names=list(all_entity_names),
                )
            )

        items.sort(key=lambda x: x.score, reverse=True)
        return items
