"""Central configuration for all AtomicRAG pipelines and components."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AtomicRAGConfig:
    """Every tunable parameter in AtomicRAG, with sensible defaults.

    Pass this to ``IndexPipeline`` or ``RetrievePipeline`` to override
    any behaviour.  Only set the fields you care about; everything else
    uses battle-tested defaults from the Clue-RAG paper.

    Example::

        config = AtomicRAGConfig(chunk_size=500, traversal_depth=3)
        pipeline = IndexPipeline(llm=my_llm, embedding=my_emb, config=config)
    """

    # ------------------------------------------------------------------ #
    # Chunking
    # ------------------------------------------------------------------ #
    chunk_size: int = 1000
    """Maximum characters per chunk."""
    chunk_overlap: int = 200
    """Character overlap between consecutive chunks."""
    chunk_strategy: str = "recursive"
    """Splitting strategy: ``recursive``, ``sentence``, ``fixed``, or ``custom``."""
    chunk_separators: Optional[List[str]] = None
    """Custom separator list for the ``recursive`` strategy (default: newlines / spaces)."""

    # ------------------------------------------------------------------ #
    # Knowledge-Unit Extraction
    # ------------------------------------------------------------------ #
    ku_extraction_prompt: Optional[str] = None
    """Custom prompt template for extracting atomic facts.
    Must contain ``{text_chunk}`` placeholder.  ``None`` = built-in default."""
    ku_max_units_per_chunk: int = 50
    """Safety cap on KUs returned per chunk."""
    ku_batch_size: int = 10
    """Chunks to feed per LLM call batch."""
    ku_concurrency: int = 1
    """Parallel LLM calls for KU extraction (1 = sequential)."""

    # ------------------------------------------------------------------ #
    # Entity Extraction
    # ------------------------------------------------------------------ #
    entity_extraction_prompt: Optional[str] = None
    """Custom prompt for entity extraction.
    Must contain ``{text}`` placeholder.  ``None`` = built-in default."""
    entity_extraction_method: str = "llm"
    """Extraction backend: ``llm`` (uses your LLM) or ``spacy``."""
    entity_merge_similar: bool = True
    """Merge entities that share the same normalised name."""
    entity_types: Optional[List[str]] = None
    """Restrict to specific entity types (``None`` = keep all)."""

    # ------------------------------------------------------------------ #
    # Embedding
    # ------------------------------------------------------------------ #
    embedding_batch_size: int = 100
    """Texts to embed per batch call."""
    embedding_concurrency: int = 1
    """Parallel embedding batch calls (1 = sequential). Each thread processes one batch."""
    embedding_dimensions: Optional[int] = None
    """Expected embedding dimensions (``None`` = auto-detect from first call)."""

    # ------------------------------------------------------------------ #
    # Graph Building
    # ------------------------------------------------------------------ #
    deduplicate_entities: bool = True
    """Merge duplicate entity nodes by normalised name."""
    min_entity_occurrences: int = 1
    """Discard entities that appear fewer times than this threshold."""

    # ------------------------------------------------------------------ #
    # Retrieval – Entity Anchoring
    # ------------------------------------------------------------------ #
    query_entity_prompt: Optional[str] = None
    """Custom prompt to extract entities from the user query.
    Must contain ``{query}`` placeholder.  ``None`` = built-in default."""
    anchor_top_k: int = 10
    """Top-K knowledge units for initial semantic anchoring."""
    entity_match_threshold: float = 0.8
    """Minimum cosine similarity for fuzzy entity name matching."""

    # ------------------------------------------------------------------ #
    # Retrieval – Q-Iter Traversal
    # ------------------------------------------------------------------ #
    traversal_depth: int = 2
    """Number of graph hops (paper default: 2)."""
    beam_size: int = 10
    """Beam-search width per depth level (paper default: 10)."""
    query_update_weight: float = 1.0
    """Weight for subtracting retrieved embeddings from the query vector."""
    max_kus_per_depth: int = 50
    """Cap on knowledge units collected per depth level."""

    # ------------------------------------------------------------------ #
    # Retrieval – Ranking
    # ------------------------------------------------------------------ #
    result_top_n: int = 6
    """Final number of items to return."""
    min_score_threshold: float = 0.0
    """Discard results below this cosine-similarity score."""
    group_by_chunk: bool = True
    """Aggregate KU scores per source chunk before ranking."""
    score_aggregation: str = "mean"
    """Aggregation function: ``mean``, ``max``, or ``sum``."""

    # ------------------------------------------------------------------ #
    # Progress / Callbacks
    # ------------------------------------------------------------------ #
    verbose: bool = False
    """Print progress messages to stdout."""
    on_chunk_processed: Optional[Callable] = field(default=None, repr=False)
    """Called after each chunk is processed: ``fn(chunk_index, total_chunks)``."""
    on_batch_complete: Optional[Callable] = field(default=None, repr=False)
    """Called after each LLM/embedding batch: ``fn(batch_index, total_batches)``."""
    on_progress: Optional[Callable] = field(default=None, repr=False)
    """General progress callback: ``fn(current, total, stage_name)``."""

    # ------------------------------------------------------------------ #
    # Serialisation helpers
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Export config as a plain dictionary (callbacks excluded)."""
        d = asdict(self)
        for key in ("on_chunk_processed", "on_batch_complete", "on_progress"):
            d.pop(key, None)
        return d

    def to_json(self, path: str) -> None:
        """Write config to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtomicRAGConfig":
        """Create config from a dictionary, ignoring unknown keys."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str) -> "AtomicRAGConfig":
        """Load config from a JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_yaml(cls, path: str) -> "AtomicRAGConfig":
        """Load config from a YAML file (requires PyYAML)."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required: pip install pyyaml")
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_env(cls, prefix: str = "ATOMICRAG_") -> "AtomicRAGConfig":
        """Build config from environment variables.

        Each field maps to ``{prefix}{FIELD_NAME_UPPER}``.
        Example: ``ATOMICRAG_CHUNK_SIZE=500``
        """
        overrides: Dict[str, Any] = {}
        for f in cls.__dataclass_fields__.values():
            env_key = f"{prefix}{f.name.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is None:
                continue

            ftype = f.type
            if ftype in ("int", int):
                overrides[f.name] = int(env_val)
            elif ftype in ("float", float):
                overrides[f.name] = float(env_val)
            elif ftype in ("bool", bool):
                overrides[f.name] = env_val.lower() in ("true", "1", "yes")
            elif ftype in ("str", str):
                overrides[f.name] = env_val

        return cls(**overrides)
