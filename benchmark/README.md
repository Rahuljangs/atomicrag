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

### Measured Scaling Results on Public Corpora

We ran the vocabulary extraction pipeline (spaCy NER + noun chunks → frequency filter → batch-divide for LLM) on three public datasets at varying sizes. **No actual LLM calls were made** — these are exact counts of how many batch calls the pipeline would issue.

Config: `chunk_size=1000`, `overlap=200`, `min_freq=2`, `max_terms_per_call=500`.

| Corpus | Documents | Chunks | Vocab LLM Calls | Per-Chunk Calls | Reduction |
|--------|----------:|-------:|----------------:|----------------:|----------:|
| **PubMedQA** (biomedical) | 1,000 | 2,043 | **7** | 2,043 | **292x** |
| **MS MARCO** (web passages) | 10,000 | 10,001 | **12** | 10,001 | **833x** |
| **MS MARCO** | 50,000 | 50,008 | **49** | 50,008 | **1,020x** |
| **MS MARCO** | 100,000 | 100,020 | **88** | 100,020 | **1,137x** |
| **WikiText-103** (Wikipedia) | 10,000 | 67,796 | **226** | 67,796 | **300x** |
| **WikiText-103** | 29,023 | 198,145 | **523** | 198,145 | **379x** |

**Key insight:** 100x more documents (1K → 100K MS MARCO) = only ~12x more LLM calls (7 → 88). Vocabulary method cost scales with vocabulary size, not document count.

### Comparison with Other Systems

| System | LLM Calls (Indexing) | Method |
|--------|:---:|------|
| **AtomicRAG (vocabulary)** | **7–88 total** (measured) | NLP vocab extraction + batch LLM filtering. No per-chunk calls. |
| AtomicRAG (llm) | ~1 per chunk | Per-chunk LLM extraction (highest quality) |
| MS-GraphRAG | ~2-3x per chunk | Entity extraction + community summarization |
| HippoRAG / HippoRAG2 | ~1 per chunk | Per-chunk entity & triple extraction |
| LightRAG | ~1-2x per chunk | Entity/relationship extraction + keyword generation |
| Fast-GraphRAG | ~1 per chunk | Per-chunk LLM extraction |
| RAPTOR | ~1 per chunk + tree | Per-chunk summarization + hierarchical clustering |

---

## Retrieval Quality — AtomicRAG (Vocabulary Method)

Evaluated on all 2,062 questions. LLM Judge: Gemini 2.5 Pro. Metrics from the GraphRAG-Bench evaluation codebase.

| Task | Questions | Context Relevance | Context Recall | Faithfulness |
|------|:---------:|:-----------------:|:--------------:|:------------:|
| Fact Retrieval | 1,098 | **85.54** | **77.37** | **91.29** |
| Complex Reasoning | 509 | **80.60** | **62.77** | **85.49** |
| Contextual Summarize | 289 | **82.53** | **46.71** | **89.40** |
| Creative Generation | 166 | **78.61** | **47.56** | **74.30** |
| **Average** | **2,062** | **81.82** | **58.60** | **85.12** |

- **Context Relevance**: How relevant is the retrieved context to the question (0–1).
- **Context Recall**: What fraction of gold-answer statements are attributable to the context.
- **Faithfulness**: What fraction of generated-answer statements are supported by the context.
- Some items received partial scores (1–2 metrics as NaN due to unparseable LLM judge output). NaN values are excluded from averages. See [results/README.md](results/README.md) for full transparency breakdown.

### Evidence Recall from Paper (Other Systems)

For reference, Evidence Recall scores from the GraphRAG-Bench paper (Table 3, Medical dataset):

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
  run_atomicrag.py                             # Generate answers (builds graph on first run)
  eval_atomicrag.py                            # Evaluate answer accuracy (ACC)
  eval_retrieval.py                            # Evaluate retrieval quality (CR, recall, faithfulness)
  scaling_analysis.py                          # Scaling analysis — LLM call counts (dry-run, no LLM)
  results/
    README.md                                  # Detailed results with methodology & transparency
    eval_scores_vocab_gemini_2.5_pro.json      # ACC scores — vocabulary method
    eval_scores_llm_gemini_2.5_pro.json        # ACC scores — LLM method
    eval_retrieval_scores.json                 # Retrieval metrics (CR, recall, faithfulness)
    scaling_analysis.json                      # LLM call counts on public corpora
```

## Evaluation Setup

| Parameter | Value |
|-----------|-------|
| Benchmark | GraphRAG-Bench Medical (NCCN guidelines) |
| Questions | 2,062 |
| ACC Metric | 0.75 * F1 + 0.25 * CosineSim |
| ACC LLM Judge | Gemini 2.5 Flash |
| Retrieval Metrics Judge | Gemini 2.5 Pro (10 parallel workers) |
| Embeddings | Gemini Embedding 001 |
| Answer LLM | Gemini 2.5 Pro |
| Extraction | Vocabulary method |
| Config | chunk_size=1000, overlap=200, depth=3, beam=15, top_n=8 |

## Reproduce

```bash
cd benchmark

# 1. Generate answers (graph auto-builds on first run)
python run_atomicrag.py --subset medical --workers 2

# 2. Evaluate answer accuracy (ACC)
python eval_atomicrag.py --data_file results/predictions_medical.json --workers 5

# 3. Evaluate retrieval quality (context relevance, context recall, faithfulness)
python eval_retrieval.py \
  --data_file results/checkpoint_medical_Medical_gemini_2.5_pro_vocab.json \
  --output_file results/eval_retrieval_scores.json \
  --workers 10 --delay 1.0

# 4. Scaling analysis on public corpora (no LLM calls, dry-run)
python scaling_analysis.py --output_file results/scaling_analysis.json
```
