# Benchmark Results

Evaluated on [GraphRAG-Bench](https://graphrag-bench.github.io/) (Medical subset) — the standard benchmark for Graph RAG systems. 2,062 questions across 4 difficulty levels, derived from NCCN clinical guidelines.

## AtomicRAG Scores (ACC)

Evaluated using the **exact same metric pipeline** as the GraphRAG-Bench leaderboard: `compute_answer_correctness` from their official codebase.

**ACC** = 0.75 \* Factuality(F1) + 0.25 \* SemanticSimilarity(cosine)

### Vocabulary Method + Gemini 2.5 Pro

| Task | Questions | ACC |
|------|:---------:|:---:|
| Fact Retrieval | 1,098 | **61.45** |
| Complex Reasoning | 509 | **55.85** |
| Contextual Summarize | 289 | **62.35** |
| Creative Generation | 166 | **45.87** |
| **Average** | **2,062** | **56.38** |

- Graph indexing: ~120-140 LLM calls total (vocabulary method)
- Answer generation: Gemini 2.5 Pro
- LLM Judge: Gemini 2.5 Flash
- Embeddings: Gemini Embedding 001

---

## Comparison with GraphRAG-Bench Medical Leaderboard

All leaderboard scores from the [official paper](https://arxiv.org/abs/2506.05690) (Table 2), evaluated with GPT-4o-mini as judge.

### Answer Accuracy (ACC) — All Task Types

| Model | Fact Retrieval | Complex Reasoning | Ctx Summarize | Creative Gen | Avg ACC |
|-------|:---:|:---:|:---:|:---:|:---:|
| HippoRAG2 | 66.28 | 61.98 | 63.08 | 68.05 | 64.85 |
| Fast-GraphRAG | 60.93 | 61.73 | 67.88 | 65.93 | 64.12 |
| LightRAG | 63.32 | 61.32 | 63.14 | 67.91 | 63.92 |
| RAG (w/ rerank) | 64.73 | 58.64 | 65.75 | 60.61 | 62.43 |
| RAG (w/o rerank) | 63.72 | 57.61 | 63.72 | 58.94 | 61.00 |
| HippoRAG | 56.14 | 55.87 | 59.86 | 64.43 | 59.08 |
| RAPTOR | 54.07 | 53.20 | 58.73 | 62.38 | 57.10 |
| **AtomicRAG (vocab)** | **61.45** | **55.85** | **62.35** | **45.87** | **56.38** |
| MS-GraphRAG | 38.63 | 47.04 | 41.87 | 53.11 | 45.16 |

> **Note on judge model**: The leaderboard uses GPT-4o-mini as the LLM judge, while our evaluation uses Gemini 2.5 Flash. Different judges produce different absolute scores. Cross-judge comparison should be interpreted directionally, not as exact rankings.

---

## The Real Story: LLM Call Efficiency

AtomicRAG's vocabulary method is radically more efficient at graph construction than any competing GraphRAG system.

### LLM Calls to Build the Knowledge Graph

| System | LLM Calls (Indexing) | Method |
|--------|:---:|------|
| **AtomicRAG (vocabulary)** | **~120-140 total** | NLP-based vocab extraction + batch LLM filtering. No per-chunk calls. |
| AtomicRAG (llm) | ~1 per chunk | Per-chunk LLM extraction (highest quality) |
| MS-GraphRAG | ~2-3x per chunk | Entity extraction + community summarization + multi-level summaries |
| HippoRAG / HippoRAG2 | ~1 per chunk | Per-chunk entity & triple extraction |
| LightRAG | ~1-2x per chunk | Entity/relationship extraction + keyword generation per chunk |
| Fast-GraphRAG | ~1 per chunk | Per-chunk LLM extraction |
| RAPTOR | ~1 per chunk + tree | Per-chunk summarization + hierarchical clustering |

### Enterprise Scale Cost Comparison

At 100,000 documents (~500K chunks):

| System | LLM Calls | Est. Cost (GPT-4o-mini) | Est. Cost (Gemini Flash) |
|--------|----------:|------------------------:|-------------------------:|
| **AtomicRAG (vocab)** | **~500-1,000** | **~$0.50-1.00** | **~$0.10-0.20** |
| MS-GraphRAG | ~1,000,000+ | ~$1,000-1,500 | ~$200-300 |
| HippoRAG2 | ~500,000 | ~$500 | ~$100 |
| LightRAG | ~500,000+ | ~$500-1,000 | ~$100-200 |
| Fast-GraphRAG | ~500,000 | ~$500 | ~$100 |

The vocabulary method scales with **vocabulary size, not document count**. 100x more documents does not mean 100x more LLM calls. This makes AtomicRAG the only practical GraphRAG option for large-scale enterprise deployments where cost is a constraint.

---

## Retrieval Quality (Evidence Recall)

For reference, Evidence Recall scores from the paper (Table 3, Medical dataset):

| Model | Fact Retrieval | Complex Reasoning | Ctx Summarize | Creative Gen |
|-------|:---:|:---:|:---:|:---:|
| RAG (w/ rerank) | 87.83 | 86.49 | 85.87 | 45.23 |
| HippoRAG | 87.25 | 83.80 | 83.46 | 81.66 |
| RAPTOR | 85.40 | 89.70 | 88.86 | 72.70 |
| LightRAG | 80.32 | 82.91 | 85.71 | 81.34 |
| HippoRAG2 | 78.70 | 77.00 | 77.40 | 61.12 |
| Fast-GraphRAG | 66.82 | 74.93 | 77.27 | 62.99 |
| MS-GraphRAG | 38.06 | 61.32 | 59.66 | 66.59 |

---

## Files

```
benchmark/
  results/
    eval_scores_vocab_gemini_2.5_pro.json   # ACC scores — vocabulary method
    eval_scores_llm_gemini_2.5_pro.json     # ACC scores — LLM method
```

## Evaluation Setup

| Parameter | Value |
|-----------|-------|
| Benchmark | GraphRAG-Bench Medical (NCCN guidelines) |
| Questions | 2,062 |
| Metric | ACC = 0.75 * F1 + 0.25 * CosineSim |
| LLM Judge | Gemini 2.5 Flash |
| Embeddings | Gemini Embedding 001 |
| Answer LLM | Gemini 2.5 Pro |
| Extraction | Vocabulary method |
| Config | chunk_size=1000, overlap=200, depth=3, beam=15, top_n=8 |

## Reproduce

```bash
# Generate answers (graph auto-builds on first run)
cd benchmark
python run_atomicrag.py --subset medical --workers 2

# Evaluate
python eval_atomicrag.py --data_file results/atomicrag/predictions_medical.json --workers 5
```
