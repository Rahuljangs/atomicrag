"""Text chunking strategies for splitting documents before extraction."""
from __future__ import annotations

import re
from typing import List, Optional

from atomicrag.config import AtomicRAGConfig
from atomicrag.models.graph import Chunk


_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class TextChunker:
    """Split raw text into overlapping chunks.

    Supports multiple strategies:
    - ``recursive``: Recursively split on a hierarchy of separators (default).
    - ``sentence``: Split on sentence boundaries.
    - ``fixed``: Fixed-size character windows.

    All parameters come from ``AtomicRAGConfig`` but can be overridden
    at construction time.

    Example::

        chunker = TextChunker(config=AtomicRAGConfig(chunk_size=500))
        chunks = chunker.chunk("long document text ...", doc_id="doc-1")
    """

    def __init__(
        self,
        config: Optional[AtomicRAGConfig] = None,
        *,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: Optional[str] = None,
        separators: Optional[List[str]] = None,
    ):
        cfg = config or AtomicRAGConfig()
        self.chunk_size = chunk_size or cfg.chunk_size
        self.chunk_overlap = chunk_overlap or cfg.chunk_overlap
        self.strategy = strategy or cfg.chunk_strategy
        self.separators = separators or cfg.chunk_separators or _DEFAULT_SEPARATORS

    def chunk(self, text: str, doc_id: str = "") -> List[Chunk]:
        """Split *text* and return a list of ``Chunk`` objects."""
        if self.strategy == "recursive":
            pieces = self._recursive_split(text, self.separators)
        elif self.strategy == "sentence":
            pieces = self._sentence_split(text)
        elif self.strategy == "fixed":
            pieces = self._fixed_split(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

        return [
            Chunk(content=p, doc_id=doc_id, index=i)
            for i, p in enumerate(pieces)
            if p.strip()
        ]

    # ------------------------------------------------------------------ #
    # Strategies
    # ------------------------------------------------------------------ #

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using a hierarchy of separators."""
        if not text:
            return []

        sep = separators[0]
        remaining_seps = separators[1:] if len(separators) > 1 else []

        if sep == "":
            splits = list(text)
        else:
            splits = text.split(sep)

        chunks: List[str] = []
        current = ""

        for piece in splits:
            candidate = f"{current}{sep}{piece}" if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(piece) > self.chunk_size and remaining_seps:
                    chunks.extend(self._recursive_split(piece, remaining_seps))
                    current = ""
                else:
                    current = piece

        if current:
            chunks.append(current)

        return self._merge_with_overlap(chunks)

    def _sentence_split(self, text: str) -> List[str]:
        """Split on sentence boundaries, then merge into chunks."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks: List[str] = []
        current = ""

        for sent in sentences:
            candidate = f"{current} {sent}".strip() if current else sent
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sent

        if current:
            chunks.append(current)

        return self._merge_with_overlap(chunks)

    def _fixed_split(self, text: str) -> List[str]:
        """Simple fixed-size windows with overlap."""
        step = max(1, self.chunk_size - self.chunk_overlap)
        return [text[i:i + self.chunk_size] for i in range(0, len(text), step)]

    # ------------------------------------------------------------------ #
    # Overlap
    # ------------------------------------------------------------------ #

    def _merge_with_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks."""
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            overlap_text = chunks[i - 1][-self.chunk_overlap:]
            result.append(overlap_text + chunks[i])

        return result
