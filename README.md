# AtomicRAG

**Stop retrieving chunks. Start retrieving facts.**

AtomicRAG breaks documents into atomic knowledge units, links them via an entity graph, and traverses it to answer complex queries with precision. Built on Postgres. No graph DB required.

Based on the [Clue-RAG](https://arxiv.org/html/2507.08445v3) research paper — reimagined as a production-ready, model-agnostic Python library.

---

## Features

- **Graph-based RAG** — multi-partite graph (Chunks → Knowledge Units → Entities)
- **Model-agnostic** — bring any LLM and embedding model (OpenAI, Gemini, Ollama, HuggingFace, or your own)
- **Zero required dependencies** — core needs only `numpy`. Everything else is optional.
- **Every parameter configurable** — chunk size, prompts, traversal depth, beam size, scoring, callbacks
- **DB-agnostic output** — export to JSON, dict, Parquet, PostgreSQL, or any storage you want
- **Protocol-based** — no forced inheritance. If your class has a `generate()` method, it works.

## Installation

```bash
# Core (numpy only)
pip install atomicrag

# With OpenAI
pip install atomicrag[openai]

# With Google Gemini
pip install atomicrag[gemini]

# With local Ollama
pip install atomicrag[ollama]

# With LangChain adapters
pip install atomicrag[langchain]

# Everything
pip install atomicrag[all]
```

## Quick Start (5 lines)

```python
from atomicrag import IndexPipeline, RetrievePipeline
from atomicrag.integrations.openai import OpenAILLM, OpenAIEmbedding

# Build the knowledge graph
graph = IndexPipeline(
    llm=OpenAILLM(api_key="sk-..."),
    embedding=OpenAIEmbedding(api_key="sk-..."),
).run(["Your document text here...", "Another document..."])

# Query it
results = RetrievePipeline(
    graph=graph,
    llm=OpenAILLM(api_key="sk-..."),
    embedding=OpenAIEmbedding(api_key="sk-..."),
).search("What are the key features?")

for item in results.items:
    print(f"{item.score:.3f}: {item.content}")
```

## Bring Your Own Model

No SDK required. Just implement the protocol:

```python
class MyLLM:
    def generate(self, prompt: str) -> str:
        return my_api_call(prompt)

class MyEmbedding:
    def embed_text(self, text: str) -> list[float]:
        return my_encoder.encode(text)
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]

graph = IndexPipeline(llm=MyLLM(), embedding=MyEmbedding()).run(documents)
```

## How It Works

```
Documents → Chunks → [LLM] → Knowledge Units → [LLM] → Entities
                                    ↓                       ↓
                              Embed (vectors)         Embed (vectors)
                                    ↓                       ↓
                              ┌─────────── Graph ───────────┐
                              │  KU ←→ Entity ←→ KU ←→ ... │
                              └─────────────────────────────┘
                                           ↓
                    Query → Entity Anchoring → Graph Traversal → Ranking → Results
```

**Index Pipeline (offline):**
1. Split documents into chunks
2. Extract atomic facts (Knowledge Units) from each chunk using an LLM
3. Extract entities from each KU
4. Embed everything and build the graph

**Retrieve Pipeline (online):**
1. Extract entities from the query
2. Anchor to matching entities and similar KUs in the graph
3. Iteratively traverse the graph (Q-Iter algorithm)
4. Rank results by similarity and return top-N

## Configuration

Every parameter has a sensible default. Override only what you need:

```python
from atomicrag import AtomicRAGConfig, IndexPipeline

config = AtomicRAGConfig(
    # Chunking
    chunk_size=500,
    chunk_overlap=100,
    chunk_strategy="sentence",

    # KU Extraction
    ku_extraction_prompt="Your custom prompt: {text_chunk}",
    ku_max_units_per_chunk=30,
    ku_batch_size=5,

    # Entity Extraction
    entity_extraction_method="llm",  # or "spacy"
    entity_merge_similar=True,

    # Retrieval
    traversal_depth=3,      # more hops = more context
    beam_size=15,            # wider beam = more candidates
    result_top_n=10,
    score_aggregation="max", # "mean", "max", or "sum"

    # Progress tracking
    verbose=True,
    on_progress=lambda cur, total, stage: print(f"{stage}: {cur}/{total}"),
)

graph = IndexPipeline(llm=my_llm, embedding=my_emb, config=config).run(docs)
```

### Load config from file

```python
# From JSON
config = AtomicRAGConfig.from_json("config.json")

# From YAML (requires pyyaml)
config = AtomicRAGConfig.from_yaml("config.yaml")

# From environment variables
# Set ATOMICRAG_CHUNK_SIZE=500, ATOMICRAG_VERBOSE=true, etc.
config = AtomicRAGConfig.from_env()
```

## Save & Load Graphs

```python
# Save to JSON
graph.to_json("knowledge_graph.json")

# Load later (no re-indexing needed!)
from atomicrag.models.graph import KnowledgeGraph
graph = KnowledgeGraph.from_json("knowledge_graph.json")

# Or use storage adapters
from atomicrag.storage.json_storage import JSONStorage
storage = JSONStorage("my_graph.json")
storage.save(graph)
graph = storage.load()

# PostgreSQL storage
from atomicrag.storage.pgvector_storage import PGVectorStorage
storage = PGVectorStorage("postgresql://user:pass@localhost/db")
storage.create_tables()
storage.save(graph)
```

## Built-in Integrations

| Provider | LLM | Embedding | Install |
|----------|-----|-----------|---------|
| OpenAI | `OpenAILLM` | `OpenAIEmbedding` | `pip install atomicrag[openai]` |
| Google Gemini | `GeminiLLM` | `GeminiEmbedding` | `pip install atomicrag[gemini]` |
| Ollama (local) | `OllamaLLM` | `OllamaEmbedding` | `pip install atomicrag[ollama]` |
| HuggingFace | — | `HuggingFaceEmbedding` | `pip install atomicrag[huggingface]` |
| LangChain | `LangChainLLMAdapter` | `LangChainEmbeddingAdapter` | `pip install atomicrag[langchain]` |

## Output Format

All outputs are plain Python dataclasses. No vendor lock-in.

```python
# RetrievalResult
results.query               # "What are the key features?"
results.entities_extracted   # ["RHEL 9", "Kernel"]
results.graph_stats          # {"kus_retrieved": 42, "kus_scored": 38, ...}
results.items                # List[RetrievalItem]

# RetrievalItem
item.content                 # The retrieved text
item.score                   # Relevance score (0-1)
item.source_chunk_id         # Which original chunk
item.knowledge_unit_ids      # Which KUs contributed
item.entity_names            # Entities in the retrieval path

# Export
results.to_json()            # JSON string
results.to_dict()            # Plain dict
```

## Architecture

```
atomicrag/
├── __init__.py          # Public API
├── config.py            # AtomicRAGConfig
├── models/
│   ├── protocols.py     # BaseLLM, BaseEmbedding (Protocol classes)
│   ├── graph.py         # Chunk, KnowledgeUnit, Entity, KnowledgeGraph
│   └── results.py       # RetrievalItem, RetrievalResult
├── index/
│   ├── pipeline.py      # IndexPipeline (orchestrator)
│   ├── chunker.py       # TextChunker
│   ├── extractor.py     # KnowledgeUnitExtractor
│   ├── entity_extractor.py  # EntityExtractor
│   └── graph_builder.py # GraphBuilder
├── retrieve/
│   ├── pipeline.py      # RetrievePipeline (orchestrator)
│   ├── anchoring.py     # Entity + Semantic anchoring
│   ├── traversal.py     # Q-Iter graph traversal
│   └── ranking.py       # ResultRanker
├── integrations/        # OpenAI, Gemini, Ollama, LangChain, HuggingFace
├── storage/             # JSON, PostgreSQL adapters
└── utils/               # Prompts, similarity functions
```

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 1000 | Max characters per chunk |
| `chunk_overlap` | 200 | Overlap between chunks |
| `chunk_strategy` | "recursive" | "recursive", "sentence", "fixed" |
| `ku_extraction_prompt` | None | Custom prompt (uses `{text_chunk}`) |
| `ku_max_units_per_chunk` | 50 | Cap on KUs per chunk |
| `ku_batch_size` | 10 | Chunks per LLM batch |
| `entity_extraction_method` | "llm" | "llm" or "spacy" |
| `entity_merge_similar` | True | Deduplicate by name |
| `embedding_batch_size` | 100 | Texts per embedding batch |
| `anchor_top_k` | 10 | Semantic anchor KU count |
| `traversal_depth` | 2 | Graph hops (1-3) |
| `beam_size` | 10 | Beam search width |
| `query_update_weight` | 1.0 | Diversity factor |
| `result_top_n` | 6 | Final results to return |
| `min_score_threshold` | 0.0 | Minimum score filter |
| `group_by_chunk` | True | Aggregate by source chunk |
| `score_aggregation` | "mean" | "mean", "max", "sum" |
| `verbose` | False | Print progress |

## License

Apache 2.0 — use it freely in commercial and open-source projects.

## Citation

If you use AtomicRAG in your research, please cite:

```bibtex
@software{atomicrag,
  title={AtomicRAG: Graph-based RAG using Atomic Knowledge Units},
  author={Rohit Jangir},
  year={2026},
  url={https://github.com/rjangir/atomicrag}
}
```

## Contributing

Contributions welcome! Please open an issue first to discuss what you'd like to change.
