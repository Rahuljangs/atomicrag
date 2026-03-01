"""Data models for retrieval results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class RetrievalItem:
    """A single retrieved result from the knowledge graph."""

    content: str = ""
    score: float = 0.0
    source_chunk_id: str = ""
    knowledge_unit_ids: List[str] = field(default_factory=list)
    entity_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Complete output from ``RetrievePipeline.search()``."""

    items: List[RetrievalItem] = field(default_factory=list)
    query: str = ""
    entities_extracted: List[str] = field(default_factory=list)
    graph_stats: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "entities_extracted": self.entities_extracted,
            "graph_stats": self.graph_stats,
            "items": [asdict(item) for item in self.items],
        }

    def to_json(self, path: str | None = None) -> str:
        """Serialise to JSON string. Optionally write to *path*."""
        text = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalResult":
        return cls(
            query=data.get("query", ""),
            entities_extracted=data.get("entities_extracted", []),
            graph_stats=data.get("graph_stats", {}),
            items=[RetrievalItem(**item) for item in data.get("items", [])],
        )
