"""JSON file storage adapter for KnowledgeGraph."""
from __future__ import annotations

from pathlib import Path

from atomicrag.models.graph import KnowledgeGraph


class JSONStorage:
    """Save and load a ``KnowledgeGraph`` to/from a JSON file.

    Example::

        storage = JSONStorage("my_graph.json")
        storage.save(graph)
        loaded = storage.load()
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def save(self, graph: KnowledgeGraph) -> None:
        """Write the graph to the JSON file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        graph.to_json(str(self.path))

    def load(self) -> KnowledgeGraph:
        """Read the graph from the JSON file."""
        return KnowledgeGraph.from_json(str(self.path))

    def exists(self) -> bool:
        """Check whether the file exists."""
        return self.path.exists()
