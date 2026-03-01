"""Step 2 of Q-Iter: Iterative graph traversal with beam search."""

from __future__ import annotations

import logging
from typing import List, Optional, Set

import numpy as np

from atomicrag.config import AtomicRAGConfig
from atomicrag.models.graph import KnowledgeGraph
from atomicrag.utils.similarity import top_k_similar

logger = logging.getLogger(__name__)


class GraphTraversal:
    """Iteratively expand from seed KUs through the entity graph.

    At each depth level:
    1. From current KUs -> find connected entities
    2. From those entities -> find connected KUs (new ones only)
    3. Score new KUs against the (updated) query vector
    4. Prune to beam_size best paths

    Implements "Query Updating" from the paper: subtract retrieved KU
    embeddings from the query vector to encourage diversity.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        config: Optional[AtomicRAGConfig] = None,
    ):
        cfg = config or AtomicRAGConfig()
        self.graph = graph
        self.depth = cfg.traversal_depth
        self.beam_size = cfg.beam_size
        self.query_update_weight = cfg.query_update_weight
        self.max_kus_per_depth = cfg.max_kus_per_depth

    def traverse(
        self,
        seed_entity_ids: Set[str],
        seed_ku_ids: Set[str],
        query_embedding: List[float],
    ) -> Set[str]:
        """Traverse the graph starting from seed nodes.

        Args:
            seed_entity_ids: Initial entity anchors.
            seed_ku_ids: Initial KU anchors (from entity + semantic).
            query_embedding: The query vector (will be updated iteratively).

        Returns:
            Set of all collected KU IDs after traversal.
        """
        all_kus = set(seed_ku_ids)
        current_entities = set(seed_entity_ids)
        current_kus = set(seed_ku_ids)

        query_vec = np.array(query_embedding, dtype=np.float64)

        for d in range(self.depth):
            logger.debug(
                f"Depth {d + 1}/{self.depth}: "
                f"{len(current_entities)} entities, {len(current_kus)} KUs"
            )

            # Step a: KU -> Entity expansion
            new_entities: Set[str] = set()
            for ku_id in current_kus:
                for eid in self.graph.entities_for_ku(ku_id):
                    new_entities.add(eid)
            current_entities = new_entities

            # Step b: Entity -> KU expansion
            candidate_kus: Set[str] = set()
            for eid in current_entities:
                for ku_id in self.graph.kus_for_entity(eid):
                    if ku_id not in all_kus:
                        candidate_kus.add(ku_id)

            if not candidate_kus:
                logger.debug(f"  No new KUs at depth {d + 1}, stopping")
                break

            # Step c: Score candidates by similarity to (updated) query
            candidate_list = list(candidate_kus)
            candidate_embeddings = []
            for ku_id in candidate_list:
                ku = self.graph.get_ku(ku_id)
                if ku and ku.embedding:
                    candidate_embeddings.append(ku.embedding)
                else:
                    candidate_embeddings.append([0.0] * len(query_vec))

            top_results = top_k_similar(
                query_vec.tolist(),
                candidate_embeddings,
                k=min(self.beam_size, len(candidate_list)),
            )

            # Step d: Keep top beam_size KUs
            new_kus = set()
            for idx, score in top_results:
                new_kus.add(candidate_list[idx])
                if len(new_kus) >= self.max_kus_per_depth:
                    break

            all_kus.update(new_kus)
            current_kus = new_kus

            # Step e: Query Updating — subtract retrieved to encourage diversity
            if self.query_update_weight > 0 and new_kus:
                retrieved_vecs = []
                for ku_id in new_kus:
                    ku = self.graph.get_ku(ku_id)
                    if ku and ku.embedding:
                        retrieved_vecs.append(np.array(ku.embedding))
                if retrieved_vecs:
                    avg_retrieved = np.mean(retrieved_vecs, axis=0)
                    query_vec = query_vec - self.query_update_weight * avg_retrieved
                    norm = np.linalg.norm(query_vec)
                    if norm > 0:
                        query_vec = query_vec / norm

            logger.debug(f"  Depth {d + 1}: added {len(new_kus)} KUs, total {len(all_kus)}")

        return all_kus
