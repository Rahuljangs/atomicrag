"""Default prompt templates used throughout AtomicRAG.

Every prompt uses ``str.format()``-style placeholders.  Users can
override any prompt via ``AtomicRAGConfig`` or by passing a custom
string directly to the component.

Available placeholders per prompt:
  - KU extraction:     ``{text_chunk}``
  - Entity extraction:  ``{text}``
  - Query entities:     ``{query}``
"""

from __future__ import annotations

import os
from typing import Optional


# ------------------------------------------------------------------ #
# Knowledge-Unit Extraction
# ------------------------------------------------------------------ #

DEFAULT_KU_EXTRACTION_PROMPT = """\
You are an expert knowledge engineer. Your task is to extract atomic \
"Knowledge Units" from the given text chunk.

A Knowledge Unit is a simplified, self-contained statement of fact.
Follow these rules:
1. **Sentence Simplification:** Break down complex sentences into atomic facts.
2. **Disambiguation:** Replace pronouns (it, he, they) with the specific \
names/entities from the context.
3. **De-contextualization:** Each unit must make sense on its own without \
the surrounding text.
4. **Entity Identification:** List the key entities mentioned in each unit.

Output must be a JSON object with a single key "knowledge_units" containing \
a list of objects, where each object has:
- "content": The atomic fact text.
- "entities": A list of specific entities (nouns) mentioned in the fact.

Example Input:
"Jesus Aranguren played in nearly 400 games for Athletic Bilbao."

Example Output:
{{
    "knowledge_units": [
        {{
            "content": "Jesus Aranguren had a professional career with Athletic Bilbao.",
            "entities": ["Jesus Aranguren", "Athletic Bilbao"]
        }},
        {{
            "content": "Jesus Aranguren played in nearly 400 official games for Athletic Bilbao.",
            "entities": ["Jesus Aranguren", "Athletic Bilbao"]
        }}
    ]
}}

Text Chunk:
{text_chunk}

JSON Output:
"""


# ------------------------------------------------------------------ #
# Entity Extraction (standalone, when not embedded in KU extraction)
# ------------------------------------------------------------------ #

DEFAULT_ENTITY_EXTRACTION_PROMPT = """\
Extract all named entities from the following text.
Entities can be: people, organisations, products, technologies, \
locations, versions, features, or important concepts.

Return a JSON object:
{{
    "entities": [
        {{"name": "entity name", "type": "PRODUCT|PERSON|ORG|TECHNOLOGY|FEATURE|LOCATION|OTHER"}}
    ]
}}

Text:
{text}

JSON Output:
"""


# ------------------------------------------------------------------ #
# Query Entity Extraction
# ------------------------------------------------------------------ #

DEFAULT_QUERY_ENTITY_PROMPT = """\
Extract the key entities and concepts from this query.
Entities can be: products, features, technologies, organisations, \
versions, or important nouns.

Output as JSON:
{{"entities": ["entity1", "entity2", ...]}}

Query: {query}

JSON:
"""


# ------------------------------------------------------------------ #
# Helper
# ------------------------------------------------------------------ #


def get_prompt(
    config_value: Optional[str],
    env_var: str,
    default: str,
) -> str:
    """Resolve a prompt with priority: config > env var > default.

    Args:
        config_value: Value from ``AtomicRAGConfig`` (highest priority).
        env_var: Environment-variable name to check.
        default: Built-in default (lowest priority).
    """
    if config_value is not None:
        return config_value
    env = os.environ.get(env_var)
    if env is not None:
        return env
    return default
