"""Assemble a KnowledgeGraph from extracted components."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from atomicrag.config import AtomicRAGConfig
from atomicrag.models.graph import Chunk, Entity, KnowledgeGraph, KnowledgeUnit
from atomicrag.models.protocols import BaseEmbedding

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Combine chunks, knowledge units, and entities into a ``KnowledgeGraph``.

    Handles embedding generation and edge construction.

    Example::

        builder = GraphBuilder(embedding=my_embedding)
        graph = builder.build(chunks, knowledge_units, entities, ku_entity_map)
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        config: Optional[AtomicRAGConfig] = None,
    ):
        cfg = config or AtomicRAGConfig()
        self.embedding = embedding
        self.batch_size = cfg.embedding_batch_size
        self.min_occurrences = cfg.min_entity_occurrences
        self.verbose = cfg.verbose
        self.on_progress = cfg.on_progress

    def build(
        self,
        chunks: List[Chunk],
        knowledge_units: List[KnowledgeUnit],
        entities: List[Entity],
        ku_entity_map: Dict[str, List[str]],
    ) -> KnowledgeGraph:
        """Build the complete knowledge graph.

        Args:
            chunks: Source text chunks.
            knowledge_units: Extracted KUs (embeddings will be generated).
            entities: Extracted entities (embeddings will be generated).
            ku_entity_map: Mapping of ``ku_id -> [entity_id, ...]``.

        Returns:
            A fully populated ``KnowledgeGraph``.
        """
        # Filter rare entities
        if self.min_occurrences > 1:
            entities, ku_entity_map = self._filter_rare_entities(
                entities, ku_entity_map
            )

        # Generate embeddings for KUs
        if self.verbose:
            print(f"  Embedding {len(knowledge_units)} knowledge units...")
        self._embed_items(
            knowledge_units,
            key=lambda ku: ku.content,
            setter=lambda ku, emb: setattr(ku, "embedding", emb),
            stage="Embedding KUs",
        )

        # Generate embeddings for entities
        if self.verbose:
            print(f"  Embedding {len(entities)} entities...")
        self._embed_items(
            entities,
            key=lambda e: e.name,
            setter=lambda e, emb: setattr(e, "embedding", emb),
            stage="Embedding entities",
        )

        # Wire up KU -> entity id lists
        for ku in knowledge_units:
            ku.entity_ids = ku_entity_map.get(ku.id, [])

        # Build edges
        edges: List[Tuple[str, str]] = []
        for ku_id, entity_ids in ku_entity_map.items():
            for entity_id in entity_ids:
                edges.append((ku_id, entity_id))

        graph = KnowledgeGraph(
            chunks=chunks,
            knowledge_units=knowledge_units,
            entities=entities,
            edges=edges,
        )

        if self.verbose:
            s = graph.stats()
            print(f"  Graph built: {s}")

        return graph

    def _embed_items(self, items, key, setter, stage: str = "") -> None:
        """Batch-embed a list of items."""
        texts = [key(item) for item in items]
        total = len(texts)

        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self.embedding.embed_batch(batch)

            for j, emb in enumerate(embeddings):
                setter(items[i + j], emb)

            if self.on_progress:
                self.on_progress(min(i + len(batch), total), total, stage)

    def _filter_rare_entities(
        self,
        entities: List[Entity],
        ku_entity_map: Dict[str, List[str]],
    ) -> Tuple[List[Entity], Dict[str, List[str]]]:
        """Remove entities that appear fewer than min_occurrences times."""
        entity_count: Dict[str, int] = {}
        for entity_ids in ku_entity_map.values():
            for eid in entity_ids:
                entity_count[eid] = entity_count.get(eid, 0) + 1

        keep_ids = {
            eid for eid, count in entity_count.items()
            if count >= self.min_occurrences
        }

        filtered_entities = [e for e in entities if e.id in keep_ids]
        filtered_map = {
            ku_id: [eid for eid in eids if eid in keep_ids]
            for ku_id, eids in ku_entity_map.items()
        }

        removed = len(entities) - len(filtered_entities)
        if removed > 0:
            logger.info(f"Filtered out {removed} rare entities (min={self.min_occurrences})")

        return filtered_entities, filtered_map
