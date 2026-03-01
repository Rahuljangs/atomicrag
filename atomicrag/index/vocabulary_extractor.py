"""Vocabulary-based extraction: build a global entity vocabulary, then
sentence-split chunks and match entities — almost zero LLM cost.

Pipeline:
    1. NLP scan all chunks → candidate terms (NER + n-grams + TF-IDF)
    2. Deduplicate and normalise candidates
    3. Batch-send candidates to LLM for filtering + categorisation
    4. Sentence-split every chunk into KUs (no LLM)
    5. Fast string-match entities to KUs to create edges

Designed for large corpora (100K+ chunks) where per-chunk LLM calls
are too expensive.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

from atomicrag.config import AtomicRAGConfig
from atomicrag.models.graph import Chunk, Entity, KnowledgeUnit
from atomicrag.models.protocols import BaseLLM

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Default prompt for LLM entity filtering
# ------------------------------------------------------------------ #
_ENTITY_FILTER_PROMPT = """\
You are a knowledge engineer. Below is a list of candidate terms extracted \
from a technical document corpus using NLP.

Your job:
1. REMOVE terms that are generic stopwords, filler, or not meaningful \
   (e.g. "also", "using", "new", "example", "section", "page").
2. KEEP terms that are specific entities: products, technologies, people, \
   organisations, features, standards, versions, or domain concepts.
3. For each kept term, assign a TYPE from: \
   PRODUCT, TECHNOLOGY, FEATURE, PERSON, ORG, VERSION, STANDARD, CONCEPT.

Output JSON:
{{"entities": [{{"name": "term", "type": "TYPE"}}, ...]}}

Candidate terms:
{terms}

JSON:
"""


class VocabularyExtractor:
    """Extract KUs and entities from chunks using the vocabulary approach.

    This is the cost-efficient alternative to per-chunk LLM extraction.
    It uses NLP for heavy lifting and LLM only for a small entity-filtering
    step.

    Example::

        extractor = VocabularyExtractor(llm=my_llm)
        kus, entities, ku_entity_map = extractor.extract_all(chunks)
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        config: Optional[AtomicRAGConfig] = None,
        *,
        llm_batch_size: int = 500,
        min_term_freq: int = 2,
        max_terms_per_llm_call: int = 500,
        ngram_range: Tuple[int, int] = (1, 3),
    ):
        """
        Args:
            llm: Optional LLM for entity filtering. If None, all NER/n-gram
                 candidates are kept (fully LLM-free mode).
            config: AtomicRAGConfig for shared settings.
            llm_batch_size: Max terms to send per LLM call.
            min_term_freq: Minimum corpus frequency to consider a term.
            max_terms_per_llm_call: Hard cap per LLM call to avoid token limits.
            ngram_range: (min_n, max_n) for n-gram extraction.
        """
        self.llm = llm
        cfg = config or AtomicRAGConfig()
        self.verbose = cfg.verbose
        self.concurrency = cfg.ku_concurrency
        self.llm_batch_size = llm_batch_size
        self.min_term_freq = min_term_freq
        self.max_terms_per_llm_call = max_terms_per_llm_call
        self.ngram_range = ngram_range

    # ================================================================ #
    # PUBLIC API
    # ================================================================ #

    def extract_all(
        self, chunks: List[Chunk]
    ) -> Tuple[List[KnowledgeUnit], List[Entity], Dict[str, List[str]]]:
        """Run the full vocabulary extraction pipeline.

        Returns:
            Tuple of:
            - knowledge_units: List of KUs (one per sentence)
            - entities: Deduplicated entity list
            - ku_entity_map: Mapping ku_id -> [entity_id, ...]
        """
        if self.verbose:
            print(f"[VocabExtractor] Starting on {len(chunks)} chunks")

        # Step 1: Collect candidate terms from all chunks
        if self.verbose:
            print("[VocabExtractor] Step 1/4: Extracting candidate terms (NER + n-grams)...")
        candidates, term_freq = self._collect_candidates(chunks)
        if self.verbose:
            print(f"  -> {len(candidates)} unique candidate terms (min_freq={self.min_term_freq})")

        # Step 2: LLM filtering (or skip if no LLM)
        if self.llm is not None:
            if self.verbose:
                print(f"[VocabExtractor] Step 2/4: LLM filtering {len(candidates)} terms...")
            entities = self._llm_filter_entities(candidates)
            if self.verbose:
                print(f"  -> {len(entities)} entities after LLM filtering")
        else:
            if self.verbose:
                print(
                    "[VocabExtractor] Step 2/4: No LLM provided, keeping all candidates as entities"
                )
            entities = [Entity(name=term, entity_type="UNKNOWN") for term in candidates]

        # Build lookup structures
        entity_by_name: Dict[str, Entity] = {}
        for e in entities:
            entity_by_name[e.name.lower()] = e

        # Step 3: Sentence-split chunks into KUs
        if self.verbose:
            print(f"[VocabExtractor] Step 3/4: Sentence-splitting {len(chunks)} chunks...")
        kus = self._sentence_split_chunks(chunks)
        if self.verbose:
            print(f"  -> {len(kus)} knowledge units (sentences)")

        # Step 4: Match entities to KUs
        if self.verbose:
            print("[VocabExtractor] Step 4/4: Matching entities to KUs...")
        ku_entity_map = self._match_entities_to_kus(kus, entity_by_name)

        # Count edges
        total_edges = sum(len(eids) for eids in ku_entity_map.values())
        if self.verbose:
            print(f"  -> {total_edges} entity-KU links created")

        return kus, entities, ku_entity_map

    # ================================================================ #
    # STEP 1: Candidate Collection
    # ================================================================ #

    def _collect_candidates(self, chunks: List[Chunk]) -> Tuple[List[str], Counter]:
        """Extract candidate terms from all chunks using NLP techniques."""
        term_counter: Counter = Counter()

        # Use spaCy if available, otherwise fall back to regex n-grams
        try:
            self._collect_with_spacy(chunks, term_counter)
        except (ImportError, OSError):
            logger.info("spaCy not available, falling back to regex-based extraction")
            self._collect_with_regex(chunks, term_counter)

        # Filter by minimum frequency
        filtered = [
            term
            for term, count in term_counter.items()
            if count >= self.min_term_freq and len(term) > 1
        ]

        # Sort by frequency (most common first)
        filtered.sort(key=lambda t: term_counter[t], reverse=True)

        return filtered, term_counter

    def _collect_with_spacy(self, chunks: List[Chunk], counter: Counter) -> None:
        """Use spaCy NER + noun chunks for candidate extraction."""
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError(
                "spaCy model 'en_core_web_sm' not found. "
                "Install with: python -m spacy download en_core_web_sm"
            )

        # Disable unnecessary pipes for speed
        disabled = [p for p in nlp.pipe_names if p not in ("ner", "parser", "tagger")]

        # Process in batches for memory efficiency
        batch_size = 500
        texts = [c.content for c in chunks]

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for doc in nlp.pipe(batch, disable=disabled, batch_size=64):
                # NER entities
                for ent in doc.ents:
                    name = ent.text.strip()
                    if len(name) > 1 and not name.isdigit():
                        counter[name] += 1

                # Noun chunks (catches multi-word terms NER misses)
                for nc in doc.noun_chunks:
                    name = nc.text.strip()
                    # Remove leading determiners/articles
                    name = re.sub(
                        r"^(the|a|an|this|that|these|those)\s+", "", name, flags=re.IGNORECASE
                    )
                    if len(name) > 2 and not name.isdigit():
                        counter[name] += 1

    def _collect_with_regex(self, chunks: List[Chunk], counter: Counter) -> None:
        """Fallback: extract n-grams and capitalized phrases via regex."""
        # Common English stopwords
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "shall",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "and",
            "but",
            "or",
            "if",
            "while",
            "about",
            "up",
            "out",
            "off",
            "over",
            "just",
            "also",
            "new",
            "use",
            "used",
            "using",
            "based",
            "include",
            "including",
            "includes",
            "it",
            "its",
            "they",
            "them",
            "their",
            "this",
            "that",
            "these",
            "those",
            "which",
            "what",
            "who",
            "whom",
            "i",
            "we",
            "you",
            "he",
            "she",
            "me",
            "us",
        }

        for chunk in chunks:
            text = chunk.content

            # Extract capitalized phrases (likely proper nouns / product names)
            caps = re.findall(r"\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b", text)
            for phrase in caps:
                if phrase.lower() not in stopwords and len(phrase) > 1:
                    counter[phrase] += 1

            # Extract n-grams
            words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9.-]*\b", text)
            words = [w for w in words if w.lower() not in stopwords and len(w) > 1]

            min_n, max_n = self.ngram_range
            for n in range(min_n, max_n + 1):
                for j in range(len(words) - n + 1):
                    gram = " ".join(words[j : j + n])
                    if len(gram) > 2:
                        counter[gram] += 1

    # ================================================================ #
    # STEP 2: LLM Filtering
    # ================================================================ #

    def _llm_filter_entities(self, candidates: List[str]) -> List[Entity]:
        """Send candidate terms to LLM in batches for filtering."""
        entities: List[Entity] = []
        seen_names: Set[str] = set()

        # Divide candidates into batches
        total_batches = math.ceil(len(candidates) / self.max_terms_per_llm_call)
        if self.verbose:
            print(f"  Sending {len(candidates)} terms in {total_batches} LLM calls")

        def _process_batch(batch_terms: List[str]) -> List[dict]:
            terms_str = ", ".join(batch_terms)
            prompt = _ENTITY_FILTER_PROMPT.format(terms=terms_str)
            try:
                response = self.llm.generate(prompt)
                return self._parse_entity_response(response)
            except Exception as e:
                logger.warning(f"LLM entity filter batch failed: {e}")
                return []

        batches = [
            candidates[i : i + self.max_terms_per_llm_call]
            for i in range(0, len(candidates), self.max_terms_per_llm_call)
        ]

        if self.concurrency > 1 and len(batches) > 1:
            # Parallel LLM calls
            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = {executor.submit(_process_batch, b): b for b in batches}
                for i, future in enumerate(as_completed(futures)):
                    results = future.result()
                    for item in results:
                        name_lower = item["name"].lower()
                        if name_lower not in seen_names:
                            seen_names.add(name_lower)
                            entities.append(
                                Entity(
                                    name=item["name"],
                                    entity_type=item.get("type", "UNKNOWN"),
                                )
                            )
                    if self.verbose:
                        print(f"  Batch {i + 1}/{len(batches)}: +{len(results)} entities")
        else:
            # Sequential
            for i, batch in enumerate(batches):
                results = _process_batch(batch)
                for item in results:
                    name_lower = item["name"].lower()
                    if name_lower not in seen_names:
                        seen_names.add(name_lower)
                        entities.append(
                            Entity(
                                name=item["name"],
                                entity_type=item.get("type", "UNKNOWN"),
                            )
                        )
                if self.verbose:
                    print(
                        f"  Batch {i + 1}/{len(batches)}: +{len(results)} entities, total={len(entities)}"
                    )

        return entities

    @staticmethod
    def _parse_entity_response(response: str) -> List[dict]:
        """Parse LLM JSON response into entity dicts."""
        import json

        text = response.strip()

        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    return []
            else:
                return []

        items = data.get("entities", [])
        valid = []
        for item in items:
            if isinstance(item, dict) and "name" in item:
                valid.append(
                    {
                        "name": str(item["name"]).strip(),
                        "type": str(item.get("type", "UNKNOWN")).strip(),
                    }
                )
        return valid

    # ================================================================ #
    # STEP 3: Sentence Splitting
    # ================================================================ #

    def _sentence_split_chunks(self, chunks: List[Chunk]) -> List[KnowledgeUnit]:
        """Split each chunk into sentences, one KU per sentence."""
        kus: List[KnowledgeUnit] = []

        try:
            import spacy

            nlp = spacy.load("en_core_web_sm")

            # Process in batches
            batch_size = 1000
            texts = [c.content for c in chunks]
            chunk_ids = [c.id for c in chunks]

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_ids = chunk_ids[i : i + batch_size]

                for doc, cid in zip(nlp.pipe(batch_texts, batch_size=64), batch_ids):
                    for sent in doc.sents:
                        text = sent.text.strip()
                        if len(text) > 10:  # Skip very short fragments
                            kus.append(
                                KnowledgeUnit(
                                    content=text,
                                    chunk_id=cid,
                                    metadata={"extraction_method": "sentence_split"},
                                )
                            )
        except (ImportError, OSError):
            # Fallback: regex sentence splitting
            logger.info("spaCy not available, using regex sentence splitting")
            for chunk in chunks:
                sentences = re.split(r"(?<=[.!?])\s+", chunk.content)
                for sent in sentences:
                    text = sent.strip()
                    if len(text) > 10:
                        kus.append(
                            KnowledgeUnit(
                                content=text,
                                chunk_id=chunk.id,
                                metadata={"extraction_method": "regex_split"},
                            )
                        )

        return kus

    # ================================================================ #
    # STEP 4: Entity-KU Matching
    # ================================================================ #

    def _match_entities_to_kus(
        self,
        kus: List[KnowledgeUnit],
        entity_by_name: Dict[str, Entity],
    ) -> Dict[str, List[str]]:
        """Match entities to KUs via case-insensitive string search.

        Uses an optimized approach:
        - Build a regex pattern from all entity names (compiled once)
        - Scan each KU against the pattern
        - Handles overlapping matches and substring issues
        """
        ku_entity_map: Dict[str, List[str]] = {}

        if not entity_by_name:
            return {ku.id: [] for ku in kus}

        # Sort entity names by length (longest first) to prioritize longer matches
        # e.g., "Red Hat OpenShift" should match before "Red Hat"
        sorted_names = sorted(entity_by_name.keys(), key=len, reverse=True)

        # Build compiled regex with word boundaries
        # Escape special regex chars in entity names
        patterns = [re.escape(name) for name in sorted_names]

        # Chunk the patterns to avoid regex size limits (>10K entities)
        pattern_chunk_size = 2000
        compiled_patterns = []
        for i in range(0, len(patterns), pattern_chunk_size):
            chunk = patterns[i : i + pattern_chunk_size]
            combined = r"\b(?:" + "|".join(chunk) + r")\b"
            compiled_patterns.append(re.compile(combined, re.IGNORECASE))

        # Match entities to each KU
        for ku in kus:
            matched_ids: List[str] = []
            seen_entity_ids: Set[str] = set()
            ku.content.lower()

            for pattern in compiled_patterns:
                for match in pattern.finditer(ku.content):
                    matched_name = match.group().lower()
                    entity = entity_by_name.get(matched_name)
                    if entity and entity.id not in seen_entity_ids:
                        seen_entity_ids.add(entity.id)
                        matched_ids.append(entity.id)

            ku.entity_ids = matched_ids
            ku.metadata["matched_entities"] = len(matched_ids)
            ku_entity_map[ku.id] = matched_ids

        return ku_entity_map
