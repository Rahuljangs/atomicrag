"""Extract atomic Knowledge Units from text chunks using an LLM."""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple

from atomicrag.config import AtomicRAGConfig
from atomicrag.models.graph import Chunk, KnowledgeUnit
from atomicrag.models.protocols import BaseLLM
from atomicrag.utils.prompts import DEFAULT_KU_EXTRACTION_PROMPT, get_prompt

logger = logging.getLogger(__name__)


class KnowledgeUnitExtractor:
    """Use an LLM to decompose chunks into atomic knowledge units.

    Each KU is a self-contained statement of fact with associated
    entity names (entity linking is handled by ``EntityExtractor``).

    Example::

        extractor = KnowledgeUnitExtractor(llm=my_llm)
        kus, raw = extractor.extract(chunk)
    """

    def __init__(
        self,
        llm: BaseLLM,
        config: Optional[AtomicRAGConfig] = None,
        *,
        prompt: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_units_per_chunk: Optional[int] = None,
    ):
        cfg = config or AtomicRAGConfig()
        self.llm = llm
        self.prompt_template = get_prompt(
            prompt or cfg.ku_extraction_prompt,
            "ATOMICRAG_KU_PROMPT",
            DEFAULT_KU_EXTRACTION_PROMPT,
        )
        self.batch_size = batch_size or cfg.ku_batch_size
        self.max_units = max_units_per_chunk or cfg.ku_max_units_per_chunk

    def extract(self, chunk: Chunk) -> Tuple[List[KnowledgeUnit], List[Dict]]:
        """Extract knowledge units from a single chunk.

        Returns:
            Tuple of (list of KnowledgeUnit, list of raw dicts from LLM)
            The raw dicts contain ``content`` and ``entities`` keys.
        """
        prompt = self.prompt_template.format(text_chunk=chunk.content)

        try:
            response = self.llm.generate(prompt)
            parsed = self._parse_response(response)
        except Exception as e:
            logger.warning(f"KU extraction failed for chunk {chunk.id[:8]}: {e}")
            return [], []

        # Cap the number of units
        parsed = parsed[: self.max_units]

        kus = []
        for item in parsed:
            ku = KnowledgeUnit(
                content=item["content"],
                chunk_id=chunk.id,
                entity_ids=[],  # filled later by EntityExtractor
                metadata={"raw_entities": item.get("entities", [])},
            )
            kus.append(ku)

        return kus, parsed

    def extract_batch(self, chunks: List[Chunk]) -> Tuple[List[KnowledgeUnit], List[Dict]]:
        """Extract KUs from multiple chunks sequentially.

        Returns:
            Tuple of (all KUs, all raw dicts)
        """
        all_kus: List[KnowledgeUnit] = []
        all_raw: List[Dict] = []

        for chunk in chunks:
            kus, raw = self.extract(chunk)
            all_kus.extend(kus)
            all_raw.extend(raw)

        return all_kus, all_raw

    @staticmethod
    def _parse_response(response: str) -> List[Dict]:
        """Parse the LLM JSON response into a list of KU dicts."""
        text = response.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    logger.warning("Could not parse LLM response as JSON")
                    return []
            else:
                return []

        units = data.get("knowledge_units", [])
        if not isinstance(units, list):
            return []

        valid = []
        for u in units:
            if isinstance(u, dict) and "content" in u:
                valid.append(
                    {
                        "content": str(u["content"]),
                        "entities": [str(e) for e in u.get("entities", [])],
                    }
                )

        return valid
