"""Extract and deduplicate entities from knowledge units."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from atomicrag.config import AtomicRAGConfig
from atomicrag.models.graph import Entity, KnowledgeUnit
from atomicrag.models.protocols import BaseLLM

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extract entities from Knowledge Units and produce deduplicated Entity nodes.

    Two methods are supported:

    - ``llm``: Already extracted by the KU extraction step (entities are in
      ``ku.metadata["raw_entities"]``). This just deduplicates and normalises.
    - ``spacy``: Uses spaCy NER on the KU content (requires ``pip install atomicrag[spacy]``).

    Example::

        extractor = EntityExtractor(method="llm")
        entities, ku_entity_map = extractor.extract(knowledge_units)
    """

    def __init__(
        self,
        config: Optional[AtomicRAGConfig] = None,
        *,
        method: Optional[str] = None,
        merge_similar: Optional[bool] = None,
        entity_types: Optional[List[str]] = None,
        llm: Optional[BaseLLM] = None,
    ):
        cfg = config or AtomicRAGConfig()
        self.method = method or cfg.entity_extraction_method
        self.merge_similar = (
            merge_similar if merge_similar is not None else cfg.entity_merge_similar
        )
        self.entity_types = entity_types or cfg.entity_types
        self.llm = llm

    def extract(
        self, knowledge_units: List[KnowledgeUnit]
    ) -> Tuple[List[Entity], Dict[str, List[str]]]:
        """Extract entities from a list of KUs.

        Returns:
            Tuple of:
            - Deduplicated list of Entity objects
            - Dict mapping ``ku_id`` -> list of ``entity_id``s
        """
        if self.method == "llm":
            return self._extract_from_metadata(knowledge_units)
        elif self.method == "spacy":
            return self._extract_with_spacy(knowledge_units)
        else:
            raise ValueError(f"Unknown entity extraction method: {self.method}")

    def _extract_from_metadata(
        self, knowledge_units: List[KnowledgeUnit]
    ) -> Tuple[List[Entity], Dict[str, List[str]]]:
        """Use the entity names already extracted during KU extraction."""
        name_to_entity: Dict[str, Entity] = {}
        ku_entity_map: Dict[str, List[str]] = {}

        for ku in knowledge_units:
            raw_entities = ku.metadata.get("raw_entities", [])
            entity_ids = []

            for name in raw_entities:
                if not name or not name.strip():
                    continue

                norm = name.strip()
                key = norm.lower() if self.merge_similar else norm

                if key not in name_to_entity:
                    entity = Entity(name=norm, entity_type="UNKNOWN")
                    name_to_entity[key] = entity
                entity_ids.append(name_to_entity[key].id)

            ku_entity_map[ku.id] = entity_ids

        entities = list(name_to_entity.values())
        return entities, ku_entity_map

    def _extract_with_spacy(
        self, knowledge_units: List[KnowledgeUnit]
    ) -> Tuple[List[Entity], Dict[str, List[str]]]:
        """Use spaCy NER to extract entities from KU content."""
        try:
            import spacy
        except ImportError:
            raise ImportError(
                "spaCy is required for entity extraction method 'spacy'. "
                "Install with: pip install atomicrag[spacy]"
            )

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError(
                "spaCy model 'en_core_web_sm' not found. "
                "Install with: python -m spacy download en_core_web_sm"
            )

        name_to_entity: Dict[str, Entity] = {}
        ku_entity_map: Dict[str, List[str]] = {}

        for ku in knowledge_units:
            doc = nlp(ku.content)
            entity_ids = []

            for ent in doc.ents:
                if self.entity_types and ent.label_ not in self.entity_types:
                    continue

                norm = ent.text.strip()
                key = norm.lower() if self.merge_similar else norm

                if key not in name_to_entity:
                    entity = Entity(name=norm, entity_type=ent.label_)
                    name_to_entity[key] = entity
                entity_ids.append(name_to_entity[key].id)

            ku_entity_map[ku.id] = entity_ids

        entities = list(name_to_entity.values())
        return entities, ku_entity_map
