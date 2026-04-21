"""
Scaling Analysis: Vocabulary Extraction LLM Call Count on Public Corpora.

Demonstrates that AtomicRAG's vocabulary method has near-constant LLM
indexing cost regardless of corpus size. Runs the NLP pipeline (spaCy NER
+ noun chunks + frequency filtering) on 2-3 public corpora of varying
sizes, then computes the exact number of LLM calls that would be needed
for the batch entity-filtering step.

NO actual LLM calls are made — this is a dry-run simulation.

Datasets used (downloaded automatically via HuggingFace datasets):
  1. PubMedQA     — biomedical abstracts (~1K documents)
  2. MS MARCO     — web passages (sampled at 10K, 50K, 100K)
  3. WikiText-103 — Wikipedia-derived articles (sampled at 10K, 50K, 100K)

Output: JSON with per-corpus stats + comparison table printed to stdout.

Usage:
    python scaling_analysis.py
    python scaling_analysis.py --output_file results/atomicrag/scaling_analysis.json
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# ─── Configuration matching AtomicRAG defaults ──────────────────────

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_TERM_FREQ = 2
MAX_TERMS_PER_LLM_CALL = 500


# ─── Text chunking (recursive, matching atomicrag) ──────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


# ─── Candidate extraction (mirrors VocabularyExtractor) ─────────────

def collect_candidates_spacy(texts: List[str], min_freq: int = MIN_TERM_FREQ
                             ) -> Tuple[List[str], Counter, dict]:
    """Run spaCy NER + noun chunks on texts, return filtered candidates."""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    disabled = [p for p in nlp.pipe_names if p not in ("ner", "parser", "tagger")]

    counter: Counter = Counter()
    batch_size = 500

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for doc in nlp.pipe(batch, disable=disabled, batch_size=64):
            for ent in doc.ents:
                name = ent.text.strip()
                if len(name) > 1 and not name.isdigit():
                    counter[name] += 1
            for nc in doc.noun_chunks:
                name = nc.text.strip()
                name = re.sub(r"^(the|a|an|this|that|these|those)\s+", "",
                              name, flags=re.IGNORECASE)
                if len(name) > 2 and not name.isdigit():
                    counter[name] += 1

    filtered = [t for t, c in counter.items() if c >= min_freq and len(t) > 1]
    filtered.sort(key=lambda t: counter[t], reverse=True)

    stats = {
        "total_unique_candidates": len(counter),
        "candidates_after_freq_filter": len(filtered),
        "top_20_terms": [(t, counter[t]) for t in filtered[:20]],
    }
    return filtered, counter, stats


def compute_llm_calls(num_candidates: int,
                      max_per_call: int = MAX_TERMS_PER_LLM_CALL) -> int:
    """Number of LLM calls for the batch entity-filtering step."""
    if num_candidates == 0:
        return 0
    return math.ceil(num_candidates / max_per_call)


# ─── Sentence splitting (mirrors VocabularyExtractor) ────────────────

def count_sentences_spacy(texts: List[str]) -> int:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    total = 0
    batch_size = 1000
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for doc in nlp.pipe(batch, batch_size=64):
            total += sum(1 for s in doc.sents if len(s.text.strip()) > 10)
    return total


# ─── Dataset loaders ────────────────────────────────────────────────

def load_pubmedqa(max_docs: int = None) -> Tuple[List[str], str]:
    """PubMedQA: biomedical research abstracts."""
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    texts = []
    for row in ds:
        ctx = row.get("context", {})
        if isinstance(ctx, dict):
            context_list = ctx.get("contexts", [])
            if context_list:
                texts.append(" ".join(context_list))
        elif isinstance(ctx, str) and ctx.strip():
            texts.append(ctx)
        if max_docs and len(texts) >= max_docs:
            break
    return texts[:max_docs] if max_docs else texts, "PubMedQA (biomedical abstracts)"


def load_msmarco(max_docs: int = 10000) -> Tuple[List[str], str]:
    """MS MARCO: web search passages."""
    from datasets import load_dataset
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="train",
                      streaming=True)
    texts = []
    seen = set()
    for row in ds:
        passages = row.get("passages", {})
        passage_texts = passages.get("passage_text", [])
        for p in passage_texts:
            if p and p not in seen:
                seen.add(p)
                texts.append(p)
                if len(texts) >= max_docs:
                    break
        if len(texts) >= max_docs:
            break
    return texts, f"MS MARCO passages ({len(texts):,} docs)"


def load_wikitext(max_docs: int = 10000) -> Tuple[List[str], str]:
    """WikiText-103: Wikipedia-derived long-form text."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
    texts = []
    current_doc = []
    for row in ds:
        line = row.get("text", "").strip()
        if line.startswith("= ") and line.endswith(" =") and not line.startswith("= ="):
            if current_doc:
                doc = " ".join(current_doc)
                if len(doc) > 100:
                    texts.append(doc[:5000])
                    if len(texts) >= max_docs:
                        break
                current_doc = []
        elif line:
            current_doc.append(line)
    if current_doc and len(texts) < max_docs:
        doc = " ".join(current_doc)
        if len(doc) > 100:
            texts.append(doc[:5000])
    return texts, f"WikiText-103 ({len(texts):,} articles)"


# ─── Analyze a single corpus ────────────────────────────────────────

def analyze_corpus(texts: List[str], corpus_name: str) -> dict:
    print(f"\n{'─'*70}", flush=True)
    print(f"  Analyzing: {corpus_name}", flush=True)
    print(f"  Documents: {len(texts):,}", flush=True)

    t0 = time.time()

    total_chars = sum(len(t) for t in texts)
    print(f"  Total chars: {total_chars:,}", flush=True)

    # Chunk
    all_chunks = []
    for text in texts:
        all_chunks.extend(chunk_text(text))
    num_chunks = len(all_chunks)
    print(f"  Chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}): {num_chunks:,}",
          flush=True)

    # Candidate extraction
    print(f"  Running spaCy NER + noun chunks...", flush=True)
    candidates, counter, stats = collect_candidates_spacy(
        all_chunks, min_freq=MIN_TERM_FREQ
    )
    num_candidates = len(candidates)

    # LLM call count
    llm_calls_vocab = compute_llm_calls(num_candidates)
    llm_calls_per_chunk = num_chunks  # what other systems would need

    # Sentence count (KUs)
    print(f"  Counting sentences (KUs)...", flush=True)
    num_kus = count_sentences_spacy(all_chunks)

    elapsed = time.time() - t0

    print(f"  Unique candidates (raw): {stats['total_unique_candidates']:,}", flush=True)
    print(f"  Candidates after freq filter: {num_candidates:,}", flush=True)
    print(f"  LLM calls (vocabulary method): {llm_calls_vocab}", flush=True)
    print(f"  LLM calls (per-chunk method): {llm_calls_per_chunk:,}", flush=True)
    print(f"  Knowledge Units (sentences): {num_kus:,}", flush=True)
    print(f"  Analysis time: {elapsed:.1f}s", flush=True)

    return {
        "corpus_name": corpus_name,
        "num_documents": len(texts),
        "total_characters": total_chars,
        "num_chunks": num_chunks,
        "unique_candidates_raw": stats["total_unique_candidates"],
        "candidates_after_freq_filter": num_candidates,
        "llm_calls_vocab_method": llm_calls_vocab,
        "llm_calls_per_chunk_method": llm_calls_per_chunk,
        "knowledge_units": num_kus,
        "cost_reduction_factor": (
            round(llm_calls_per_chunk / llm_calls_vocab, 1)
            if llm_calls_vocab > 0 else float("inf")
        ),
        "top_20_terms": stats["top_20_terms"],
        "analysis_time_seconds": round(elapsed, 1),
    }


# ─── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scaling analysis: vocabulary extraction LLM call counts on public corpora",
    )
    parser.add_argument("--output_file",
                        default="results/atomicrag/scaling_analysis.json")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("  AtomicRAG Vocabulary Method — Scaling Analysis", flush=True)
    print("  (No LLM calls are made — dry-run simulation)", flush=True)
    print("=" * 70, flush=True)
    print(f"\n  Config: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, "
          f"min_freq={MIN_TERM_FREQ}, max_terms_per_call={MAX_TERMS_PER_LLM_CALL}",
          flush=True)

    results = []

    # 1. PubMedQA — small biomedical corpus
    try:
        texts, name = load_pubmedqa()
        results.append(analyze_corpus(texts, name))
    except Exception as e:
        print(f"\n  [SKIP] PubMedQA: {e}", flush=True)

    # 2. MS MARCO — varying sizes
    for size in [10_000, 50_000, 100_000]:
        try:
            texts, name = load_msmarco(max_docs=size)
            results.append(analyze_corpus(texts, name))
        except Exception as e:
            print(f"\n  [SKIP] MS MARCO ({size:,}): {e}", flush=True)

    # 3. WikiText-103 — varying sizes
    for size in [10_000, 50_000, 100_000]:
        try:
            texts, name = load_wikitext(max_docs=size)
            results.append(analyze_corpus(texts, name))
        except Exception as e:
            print(f"\n  [SKIP] WikiText ({size:,}): {e}", flush=True)

    # ─── Summary table ───────────────────────────────────────────────
    print(f"\n\n{'='*100}", flush=True)
    print("  SCALING ANALYSIS SUMMARY", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"  {'Corpus':<40s}  {'Docs':>8s}  {'Chunks':>8s}  "
          f"{'Vocab LLM':>10s}  {'Per-Chunk':>10s}  {'Reduction':>10s}",
          flush=True)
    print(f"  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}",
          flush=True)

    for r in results:
        print(
            f"  {r['corpus_name']:<40s}  "
            f"{r['num_documents']:>8,}  "
            f"{r['num_chunks']:>8,}  "
            f"{r['llm_calls_vocab_method']:>10,}  "
            f"{r['llm_calls_per_chunk_method']:>10,}  "
            f"{r['cost_reduction_factor']:>9.0f}x",
            flush=True,
        )

    print(f"\n  Key insight: Vocabulary method LLM calls scale with VOCABULARY SIZE,",
          flush=True)
    print(f"  not document count. 10x more documents ≠ 10x more LLM calls.", flush=True)

    # Save
    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"config": {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "min_term_freq": MIN_TERM_FREQ,
                "max_terms_per_llm_call": MAX_TERMS_PER_LLM_CALL,
            }, "results": results}, f, indent=2)
        print(f"\n  Results saved to {args.output_file}", flush=True)


if __name__ == "__main__":
    main()
