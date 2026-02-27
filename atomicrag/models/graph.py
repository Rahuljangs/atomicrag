"""Data models for the AtomicRAG knowledge graph.

All models are plain dataclasses with built-in JSON/dict serialisation.
They carry **no database dependency** — users decide where to store them.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


def _new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class Chunk:
    """A segment of the original document."""

    id: str = field(default_factory=_new_id)
    content: str = ""
    doc_id: str = ""
    index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeUnit:
    """An atomic, self-contained statement of fact extracted from a chunk."""

    id: str = field(default_factory=_new_id)
    content: str = ""
    embedding: List[float] = field(default_factory=list)
    chunk_id: str = ""
    entity_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    """A named concept (product, person, technology, etc.) extracted from KUs."""

    id: str = field(default_factory=_new_id)
    name: str = ""
    entity_type: str = ""
    embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    """The complete multi-partite graph produced by ``IndexPipeline``.

    Three layers:  Chunk  -->  KnowledgeUnit  -->  Entity

    Edges are stored as ``(knowledge_unit_id, entity_id)`` tuples.
    """

    chunks: List[Chunk] = field(default_factory=list)
    knowledge_units: List[KnowledgeUnit] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Quick look-ups (built lazily)
    # ------------------------------------------------------------------ #

    def _build_indexes(self) -> None:
        self._ku_by_id = {ku.id: ku for ku in self.knowledge_units}
        self._entity_by_id = {e.id: e for e in self.entities}
        self._entity_by_name = {e.name.lower(): e for e in self.entities}
        self._chunk_by_id = {c.id: c for c in self.chunks}

        self._entity_to_kus: Dict[str, List[str]] = {}
        self._ku_to_entities: Dict[str, List[str]] = {}
        for ku_id, entity_id in self.edges:
            self._entity_to_kus.setdefault(entity_id, []).append(ku_id)
            self._ku_to_entities.setdefault(ku_id, []).append(entity_id)

    def get_ku(self, ku_id: str) -> Optional[KnowledgeUnit]:
        if not hasattr(self, "_ku_by_id"):
            self._build_indexes()
        return self._ku_by_id.get(ku_id)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        if not hasattr(self, "_entity_by_id"):
            self._build_indexes()
        return self._entity_by_id.get(entity_id)

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        if not hasattr(self, "_entity_by_name"):
            self._build_indexes()
        return self._entity_by_name.get(name.lower())

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        if not hasattr(self, "_chunk_by_id"):
            self._build_indexes()
        return self._chunk_by_id.get(chunk_id)

    def kus_for_entity(self, entity_id: str) -> List[str]:
        """Return KU ids connected to a given entity."""
        if not hasattr(self, "_entity_to_kus"):
            self._build_indexes()
        return self._entity_to_kus.get(entity_id, [])

    def entities_for_ku(self, ku_id: str) -> List[str]:
        """Return entity ids connected to a given KU."""
        if not hasattr(self, "_ku_to_entities"):
            self._build_indexes()
        return self._ku_to_entities.get(ku_id, [])

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict[str, int]:
        return {
            "chunks": len(self.chunks),
            "knowledge_units": len(self.knowledge_units),
            "entities": len(self.entities),
            "edges": len(self.edges),
        }

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks": [asdict(c) for c in self.chunks],
            "knowledge_units": [asdict(ku) for ku in self.knowledge_units],
            "entities": [asdict(e) for e in self.entities],
            "edges": self.edges,
        }

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        return cls(
            chunks=[Chunk(**c) for c in data.get("chunks", [])],
            knowledge_units=[KnowledgeUnit(**ku) for ku in data.get("knowledge_units", [])],
            entities=[Entity(**e) for e in data.get("entities", [])],
            edges=[tuple(edge) for edge in data.get("edges", [])],
        )

    @classmethod
    def from_json(cls, path: str) -> "KnowledgeGraph":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
