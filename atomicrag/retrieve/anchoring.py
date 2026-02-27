"""Step 1 of Q-Iter: Entity Anchoring and Semantic Anchoring."""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Set, Tuple

from atomicrag.config import AtomicRAGConfig
from atomicrag.models.graph import KnowledgeGraph
from atomicrag.models.protocols import BaseEmbedding, BaseLLM
from atomicrag.utils.prompts import DEFAULT_QUERY_ENTITY_PROMPT, get_prompt
from atomicrag.utils.similarity import cosine_similarity, top_k_similar

logger = logging.getLogger(__name__)


class Anchoring:
    """Find seed nodes in the graph that are relevant to a query.

    Combines two strategies:
    1. **Entity anchoring** — extract entities from the query, match to graph entities.
    2. **Semantic anchoring** — embed the query, find top-K similar KUs by cosine.

    Returns a set of seed KU ids and entity ids.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        llm: BaseLLM,
        embedding: BaseEmbedding,
        config: Optional[AtomicRAGConfig] = None,
    ):
        cfg = config or AtomicRAGConfig()
        self.graph = graph
        self.llm = llm
        self.embedding = embedding

        self.prompt_template = get_prompt(
            cfg.query_entity_prompt,
            "ATOMICRAG_QUERY_ENTITY_PROMPT",
            DEFAULT_QUERY_ENTITY_PROMPT,
        )
        self.top_k = cfg.anchor_top_k
        self.match_threshold = cfg.entity_match_threshold

    def anchor(self, query: str) -> Tuple[Set[str], Set[str], List[str], List[float]]:
        """Find seed nodes for the given query.

        Returns:
            Tuple of:
            - entity_ids: Set of matched graph entity IDs
            - ku_ids: Set of seed KU IDs (from both entity + semantic anchoring)
            - query_entities: List of entity names extracted from query
            - query_embedding: The query embedding vector
        """
        # 1. Extract entities from query
        query_entities = self._extract_query_entities(query)
        logger.info(f"Query entities: {query_entities}")

        # 2. Entity anchoring — match to graph
        entity_ids = set()
        entity_ku_ids = set()
        for name in query_entities:
            entity = self.graph.get_entity_by_name(name)
            if entity:
                entity_ids.add(entity.id)
                for ku_id in self.graph.kus_for_entity(entity.id):
                    entity_ku_ids.add(ku_id)

        logger.info(
            f"Entity anchoring: {len(entity_ids)} entities, {len(entity_ku_ids)} KUs"
        )

        # 3. Semantic anchoring — embed query, find top-K KUs
        query_embedding = self.embedding.embed_text(query)
        ku_embeddings = [ku.embedding for ku in self.graph.knowledge_units]

        if ku_embeddings and ku_embeddings[0]:
            top_results = top_k_similar(query_embedding, ku_embeddings, k=self.top_k)
            semantic_ku_ids = {
                self.graph.knowledge_units[idx].id for idx, _ in top_results
            }
        else:
            semantic_ku_ids = set()

        logger.info(f"Semantic anchoring: {len(semantic_ku_ids)} KUs")

        # Merge both sets
        all_ku_ids = entity_ku_ids | semantic_ku_ids

        return entity_ids, all_ku_ids, query_entities, query_embedding

    def _extract_query_entities(self, query: str) -> List[str]:
        """Use the LLM to extract entity names from the query."""
        prompt = self.prompt_template.format(query=query)
        try:
            response = self.llm.generate(prompt)
            text = response.strip()

            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)

            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                entities = data.get("entities", [])
                return [str(e) for e in entities if e]
        except Exception as e:
            logger.warning(f"Query entity extraction failed: {e}")

        return []
