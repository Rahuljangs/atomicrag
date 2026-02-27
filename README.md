# AtomicRAG

> *"Vector search finds what looks similar. AtomicRAG finds what's actually connected."*

**Stop retrieving chunks. Start retrieving facts.**

AtomicRAG is a graph-based Retrieval-Augmented Generation (RAG) library that breaks documents into atomic knowledge units, links them through an entity graph, and retrieves answers via iterative graph traversal. It significantly outperforms traditional vector-only RAG on complex, multi-hop queries.

Inspired by the [Clue-RAG](https://arxiv.org/html/2507.08445v3) research paper. Reimagined as a production-ready, model-agnostic Python library.

---

## Why AtomicRAG?

Traditional RAG retrieves **chunks** -- large blocks of text matched by cosine similarity. This works for simple queries but fails when:

- The answer requires connecting information across multiple documents
- Two chunks look similar but are about different things (disambiguation)
- The query requires multi-hop reasoning ("Who founded the company that acquired X?")

AtomicRAG solves this by:

1. **Decomposing chunks into atomic facts** (Knowledge Units) -- self-contained statements
2. **Linking facts through an entity graph** -- products, people, features, versions
3. **Traversing the graph** to find connected information, not just similar text

---

## Features

- **Graph-based RAG** -- multi-partite graph (Chunks -> Knowledge Units -> Entities)
- **Model-agnostic** -- bring any LLM and embedding model (OpenAI, Gemini, Ollama, HuggingFace, or your own)
- **Zero required dependencies** -- core needs only `numpy`. Everything else is optional.
- **Every parameter configurable** -- chunk size, prompts, traversal depth, beam size, scoring, callbacks
- **DB-agnostic output** -- export to JSON, dict, PostgreSQL, or any storage you want
- **Protocol-based** -- no forced inheritance. If your class has a `generate()` method, it works.
- **Save and reload** -- build the graph once, query it forever without re-indexing

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
- [Bring Your Own Model](#bring-your-own-model)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Built-in Integrations](#built-in-integrations)
- [Storage Adapters](#storage-adapters)
- [Output Format](#output-format)
- [Architecture](#architecture)
- [Configuration Reference](#configuration-reference)
- [License](#license)
- [Contributing](#contributing)

---

## Installation

### Prerequisites

- Python 3.10 or higher
- `pip` package manager

### Install

```bash
# Core library (only requires numpy)
pip install atomicrag

# With a specific LLM/embedding provider:
pip install atomicrag[openai]        # OpenAI (GPT-4, text-embedding-3)
pip install atomicrag[gemini]        # Google Gemini (gemini-2.5-flash, gemini-embedding-001)
pip install atomicrag[ollama]        # Ollama (local models like llama3)
pip install atomicrag[huggingface]   # HuggingFace (sentence-transformers)
pip install atomicrag[langchain]     # LangChain adapter (wrap any LangChain model)

# With database storage:
pip install atomicrag[pgvector]      # PostgreSQL + pgvector

# Everything:
pip install atomicrag[all]
```

### Install from source (development)

```bash
git clone https://github.com/Rahuljangs/atomicrag.git
cd atomicrag
pip install -e ".[dev]"
```

---

## Quick Start

The fastest way to get running -- 5 lines of code:

```python
from atomicrag import IndexPipeline, RetrievePipeline
from atomicrag.integrations.openai import OpenAILLM, OpenAIEmbedding

# 1. Build the knowledge graph from your documents
graph = IndexPipeline(
    llm=OpenAILLM(api_key="sk-..."),
    embedding=OpenAIEmbedding(api_key="sk-..."),
).run(["Your document text here...", "Another document..."])

# 2. Query it
results = RetrievePipeline(
    graph=graph,
    llm=OpenAILLM(api_key="sk-..."),
    embedding=OpenAIEmbedding(api_key="sk-..."),
).search("What are the key features?")

# 3. Use the results
for item in results.items:
    print(f"{item.score:.3f}: {item.content}")
```

---

## Step-by-Step Guide

### Step 1: Choose your models

Pick an LLM and an embedding model. AtomicRAG supports any provider:

```python
# Option A: OpenAI
from atomicrag.integrations.openai import OpenAILLM, OpenAIEmbedding
llm = OpenAILLM(api_key="sk-...", model="gpt-4o-mini")
embedding = OpenAIEmbedding(api_key="sk-...", model="text-embedding-3-small")

# Option B: Google Gemini
from atomicrag.integrations.gemini import GeminiLLM, GeminiEmbedding
llm = GeminiLLM(api_key="AIza...", model="gemini-2.5-flash")
embedding = GeminiEmbedding(api_key="AIza...", model="models/gemini-embedding-001")

# Option C: Local Ollama (no API key needed)
from atomicrag.integrations.ollama import OllamaLLM, OllamaEmbedding
llm = OllamaLLM(model="llama3")
embedding = OllamaEmbedding(model="nomic-embed-text")

# Option D: Mix and match (e.g., Ollama LLM + HuggingFace embeddings)
from atomicrag.integrations.ollama import OllamaLLM
from atomicrag.integrations.huggingface import HuggingFaceEmbedding
llm = OllamaLLM(model="llama3")
embedding = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
```

### Step 2: Prepare your documents

Documents can be plain strings or dicts with metadata:

```python
# Simple: list of strings
documents = [
    "First document text...",
    "Second document text...",
]

# With metadata: list of dicts
documents = [
    {"text": "First document text...", "doc_id": "doc-001"},
    {"text": "Second document text...", "doc_id": "doc-002"},
]
```

### Step 3: Build the knowledge graph

```python
from atomicrag import IndexPipeline, AtomicRAGConfig

# Optional: customize the config
config = AtomicRAGConfig(
    chunk_size=800,        # Characters per chunk
    verbose=True,          # Show progress
)

# Run the pipeline
graph = IndexPipeline(llm=llm, embedding=embedding, config=config).run(documents)

# See what was built
print(graph.stats())
# {'chunks': 15, 'knowledge_units': 142, 'entities': 87, 'edges': 312}
```

### Step 4: Save the graph (optional but recommended)

```python
# Save to JSON -- you never need to re-index
graph.to_json("my_knowledge_graph.json")

# Load it back later
from atomicrag.models.graph import KnowledgeGraph
graph = KnowledgeGraph.from_json("my_knowledge_graph.json")
```

### Step 5: Query the graph

```python
from atomicrag import RetrievePipeline

retriever = RetrievePipeline(graph=graph, llm=llm, embedding=embedding, config=config)

results = retriever.search("What are the security certifications?")

for item in results.items:
    print(f"Score: {item.score:.3f}")
    print(f"Content: {item.content[:200]}")
    print(f"Entities: {item.entity_names}")
    print()
```

### Step 6: Use the results in your application

```python
# Get as JSON (for APIs)
json_str = results.to_json()

# Get as dict (for further processing)
data = results.to_dict()

# Access individual fields
results.query                # The original query
results.entities_extracted   # Entities found in the query
results.graph_stats          # How many nodes were traversed
results.items[0].content     # Top result text
results.items[0].score       # Relevance score (0-1)
results.items[0].entity_names  # Entities in the retrieval path
```

---

## Bring Your Own Model

AtomicRAG uses Python Protocol classes -- no inheritance required. Any object with the right methods works automatically.

### Custom LLM

Your class just needs a `generate(prompt: str) -> str` method:

```python
import requests

class MyCustomLLM:
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def generate(self, prompt: str) -> str:
        response = requests.post(self.endpoint, json={"prompt": prompt})
        return response.json()["text"]

# Use it directly
graph = IndexPipeline(llm=MyCustomLLM("http://my-api/v1/generate"), embedding=...).run(docs)
```

### Custom Embedding

Your class needs `embed_text(text: str) -> list[float]` and `embed_batch(texts: list[str]) -> list[list[float]]`:

```python
from sentence_transformers import SentenceTransformer

class MyEmbedding:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_text(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()
```

### Wrap an existing LangChain model

```python
from atomicrag.integrations.langchain import LangChainLLMAdapter, LangChainEmbeddingAdapter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = LangChainLLMAdapter(ChatOpenAI(model="gpt-4o-mini"))
embedding = LangChainEmbeddingAdapter(OpenAIEmbeddings())
```

---

## How It Works

```
Documents --> Chunks --> [LLM] --> Knowledge Units --> [LLM/NER] --> Entities
                                        |                             |
                                  Embed (vectors)               Embed (vectors)
                                        |                             |
                                  +------------- Graph ---------------+
                                  |  KU <--> Entity <--> KU <--> ... |
                                  +-----------------------------------+
                                              |
               Query --> Entity Anchoring --> Graph Traversal --> Ranking --> Results
```

### Index Pipeline (offline, run once)

| Step | Component | What it does |
|------|-----------|-------------|
| 1 | **TextChunker** | Splits documents into overlapping chunks (configurable size, overlap, strategy) |
| 2 | **KnowledgeUnitExtractor** | Uses your LLM to extract atomic facts from each chunk |
| 3 | **EntityExtractor** | Identifies entities (products, people, features) from each fact |
| 4 | **GraphBuilder** | Embeds all KUs and entities, assembles the multi-partite graph |

### Retrieve Pipeline (online, per query)

| Step | Component | What it does |
|------|-----------|-------------|
| 1 | **Anchoring** | Extracts entities from query, finds matching graph nodes + top-K similar KUs |
| 2 | **GraphTraversal** | Q-Iter algorithm: iteratively expands through entity-KU connections |
| 3 | **ResultRanker** | Scores KUs by similarity, groups by source chunk, returns top-N |

### The Q-Iter Algorithm

Based on the Clue-RAG paper, Q-Iter improves retrieval through:

1. **Entity Anchoring** -- find where the query connects to the graph
2. **Iterative Expansion** -- hop through Entity -> KU -> Entity connections (configurable depth)
3. **Query Updating** -- subtract retrieved embeddings from the query vector to find diverse (not redundant) information
4. **Beam Search Pruning** -- keep only the top-M most relevant paths at each depth

---

## Configuration

Every parameter has a sensible default. Override only what you need:

```python
from atomicrag import AtomicRAGConfig

config = AtomicRAGConfig(
    # -- Chunking --
    chunk_size=500,                    # Max characters per chunk
    chunk_overlap=100,                 # Overlap between chunks
    chunk_strategy="sentence",         # "recursive", "sentence", or "fixed"

    # -- Knowledge Unit Extraction --
    ku_extraction_prompt="Your custom prompt here: {text_chunk}",
    ku_max_units_per_chunk=30,         # Safety cap
    ku_batch_size=5,                   # Chunks per LLM batch

    # -- Entity Extraction --
    entity_extraction_method="llm",    # "llm" or "spacy"
    entity_merge_similar=True,         # Deduplicate "RHEL 9" and "rhel 9"

    # -- Embedding --
    embedding_batch_size=50,           # Texts per embedding API call

    # -- Retrieval --
    anchor_top_k=10,                   # Semantic anchor count
    traversal_depth=3,                 # Graph hops (1=fast, 3=thorough)
    beam_size=15,                      # Beam search width
    query_update_weight=1.0,           # Diversity factor (0=off, 1=full)
    result_top_n=10,                   # Final results to return
    min_score_threshold=0.3,           # Drop low-relevance results
    score_aggregation="max",           # "mean", "max", or "sum"

    # -- Progress --
    verbose=True,
    on_progress=lambda cur, total, stage: print(f"{stage}: {cur}/{total}"),
)
```

### Load config from file

```python
# From JSON
config = AtomicRAGConfig.from_json("config.json")

# From YAML (requires pyyaml)
config = AtomicRAGConfig.from_yaml("config.yaml")

# From environment variables (prefix: ATOMICRAG_)
# e.g., ATOMICRAG_CHUNK_SIZE=500, ATOMICRAG_VERBOSE=true
config = AtomicRAGConfig.from_env()

# From dict
config = AtomicRAGConfig.from_dict({"chunk_size": 500, "traversal_depth": 3})
```

### Custom prompts

Override any LLM prompt used internally:

```python
config = AtomicRAGConfig(
    # Custom KU extraction prompt (must contain {text_chunk})
    ku_extraction_prompt="""
    Break this text into individual facts. Each fact should be self-contained.
    Output JSON: {{"knowledge_units": [{{"content": "...", "entities": ["..."]}}]}}
    Text: {text_chunk}
    """,

    # Custom entity extraction prompt (must contain {text})
    entity_extraction_prompt="Find all entities in: {text}\nJSON: ...",

    # Custom query entity prompt (must contain {query})
    query_entity_prompt="Extract key concepts from: {query}\nJSON: ...",
)
```

You can also set prompts via environment variables:
- `ATOMICRAG_KU_PROMPT` -- overrides KU extraction prompt
- `ATOMICRAG_ENTITY_PROMPT` -- overrides entity extraction prompt
- `ATOMICRAG_QUERY_ENTITY_PROMPT` -- overrides query entity prompt

---

## Built-in Integrations

| Provider | LLM Class | Embedding Class | Install |
|----------|-----------|-----------------|---------|
| OpenAI | `OpenAILLM(api_key, model)` | `OpenAIEmbedding(api_key, model)` | `pip install atomicrag[openai]` |
| Google Gemini | `GeminiLLM(api_key, model)` | `GeminiEmbedding(api_key, model)` | `pip install atomicrag[gemini]` |
| Ollama | `OllamaLLM(host, model)` | `OllamaEmbedding(host, model)` | `pip install atomicrag[ollama]` |
| HuggingFace | -- | `HuggingFaceEmbedding(model_name)` | `pip install atomicrag[huggingface]` |
| LangChain | `LangChainLLMAdapter(lc_llm)` | `LangChainEmbeddingAdapter(lc_emb)` | `pip install atomicrag[langchain]` |

All integrations import lazily -- you only need the SDK for the provider you use.

---

## Storage Adapters

### JSON (built-in, no extra dependencies)

```python
from atomicrag.storage.json_storage import JSONStorage

storage = JSONStorage("my_graph.json")
storage.save(graph)                  # Save
graph = storage.load()               # Load
print(storage.exists())              # Check if file exists
```

### PostgreSQL + pgvector

```python
from atomicrag.storage.pgvector_storage import PGVectorStorage

storage = PGVectorStorage("postgresql://user:pass@localhost:5432/mydb", schema="atomicrag")
storage.create_tables()              # Create tables (idempotent)
storage.save(graph)                  # Save graph to DB
graph = storage.load()               # Load graph from DB
```

### Direct serialization (no adapter needed)

```python
# To/from JSON file
graph.to_json("graph.json")
graph = KnowledgeGraph.from_json("graph.json")

# To/from Python dict
data = graph.to_dict()
graph = KnowledgeGraph.from_dict(data)
```

---

## Output Format

All outputs are plain Python dataclasses with no vendor lock-in.

### RetrievalResult

```python
results = retriever.search("my query")

results.query                # "my query"
results.entities_extracted   # ["Entity1", "Entity2"] -- extracted from query
results.graph_stats          # {"kus_retrieved": 47, "kus_scored": 42, "items_returned": 6}
results.items                # List[RetrievalItem]
```

### RetrievalItem

```python
item = results.items[0]

item.content                 # The retrieved text (original chunk or aggregated KUs)
item.score                   # Relevance score (0.0 to 1.0)
item.source_chunk_id         # UUID of the source chunk
item.knowledge_unit_ids      # List of KU UUIDs that contributed to this result
item.entity_names            # Entity names in the retrieval path
item.metadata                # Any additional metadata
```

### Export

```python
results.to_json()            # Returns JSON string
results.to_json("out.json")  # Writes to file
results.to_dict()            # Returns plain dict
```

---

## Architecture

```
atomicrag/
├── __init__.py              # Public API: IndexPipeline, RetrievePipeline, AtomicRAGConfig
├── config.py                # AtomicRAGConfig -- every tunable parameter
├── models/
│   ├── protocols.py         # BaseLLM, BaseEmbedding (Protocol classes)
│   ├── graph.py             # Chunk, KnowledgeUnit, Entity, KnowledgeGraph
│   └── results.py           # RetrievalItem, RetrievalResult
├── index/
│   ├── pipeline.py          # IndexPipeline orchestrator
│   ├── chunker.py           # TextChunker (recursive, sentence, fixed)
│   ├── extractor.py         # KnowledgeUnitExtractor (LLM-based)
│   ├── entity_extractor.py  # EntityExtractor (LLM or spaCy)
│   └── graph_builder.py     # GraphBuilder (embed + assemble)
├── retrieve/
│   ├── pipeline.py          # RetrievePipeline orchestrator
│   ├── anchoring.py         # Entity + Semantic anchoring
│   ├── traversal.py         # Q-Iter graph traversal
│   └── ranking.py           # ResultRanker (score + group + sort)
├── integrations/            # OpenAI, Gemini, Ollama, LangChain, HuggingFace
├── storage/                 # JSON file, PostgreSQL adapters
└── utils/
    ├── prompts.py           # Default prompts (all overridable)
    └── similarity.py        # Cosine similarity, top-K search
```

---

## Configuration Reference

### Chunking

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | int | 1000 | Maximum characters per chunk |
| `chunk_overlap` | int | 200 | Character overlap between consecutive chunks |
| `chunk_strategy` | str | "recursive" | Splitting strategy: "recursive", "sentence", or "fixed" |
| `chunk_separators` | list | None | Custom separator list for recursive splitting |

### Knowledge Unit Extraction

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ku_extraction_prompt` | str | None | Custom prompt template (must contain `{text_chunk}`) |
| `ku_max_units_per_chunk` | int | 50 | Maximum KUs to extract per chunk |
| `ku_batch_size` | int | 10 | Chunks to process per LLM batch |
| `ku_concurrency` | int | 1 | Parallel LLM calls (1 = sequential) |

### Entity Extraction

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entity_extraction_prompt` | str | None | Custom prompt (must contain `{text}`) |
| `entity_extraction_method` | str | "llm" | Method: "llm" or "spacy" |
| `entity_merge_similar` | bool | True | Merge entities with same normalized name |
| `entity_types` | list | None | Restrict to specific entity types |

### Embedding

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_batch_size` | int | 100 | Texts to embed per API batch call |
| `embedding_dimensions` | int | None | Expected dimensions (auto-detected if None) |

### Graph Building

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deduplicate_entities` | bool | True | Merge duplicate entity nodes |
| `min_entity_occurrences` | int | 1 | Discard entities appearing fewer times |

### Retrieval: Anchoring

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query_entity_prompt` | str | None | Custom prompt for query entity extraction |
| `anchor_top_k` | int | 10 | Top-K KUs for semantic anchoring |
| `entity_match_threshold` | float | 0.8 | Min similarity for fuzzy entity matching |

### Retrieval: Q-Iter Traversal

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `traversal_depth` | int | 2 | Number of graph hops (1=fast, 2=balanced, 3=thorough) |
| `beam_size` | int | 10 | Beam search width per depth level |
| `query_update_weight` | float | 1.0 | Weight for query updating (0=disabled, 1=full) |
| `max_kus_per_depth` | int | 50 | Cap on KUs collected per depth level |

### Retrieval: Ranking

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result_top_n` | int | 6 | Number of final results to return |
| `min_score_threshold` | float | 0.0 | Minimum cosine similarity score to include |
| `group_by_chunk` | bool | True | Aggregate KU scores per source chunk |
| `score_aggregation` | str | "mean" | Aggregation: "mean", "max", or "sum" |

### Progress / Callbacks

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | bool | False | Print progress messages to stdout |
| `on_chunk_processed` | callable | None | `fn(chunk_index, total_chunks)` |
| `on_batch_complete` | callable | None | `fn(batch_index, total_batches)` |
| `on_progress` | callable | None | `fn(current, total, stage_name)` |

---

## License

Apache 2.0 -- use freely in commercial and open-source projects.

## Citation

If you use AtomicRAG in your research or projects, please cite:

```bibtex
@software{atomicrag,
  title={AtomicRAG: Graph-based RAG using Atomic Knowledge Units},
  author={Rahul Jangir},
  year={2026},
  url={https://github.com/Rahuljangs/atomicrag}
}
```

## Contributing

Contributions are welcome! Please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
