"""
GraphRAG-Bench evaluation runner for AtomicRAG.

Uses Groq (Llama 3.3 70B) for LLM generation and Gemini for embeddings.
Saves results incrementally and supports resume from where it left off.

Usage:
    python run_atomicrag.py --subset medical --sample 50
    python run_atomicrag.py --subset medical                  # full run
    python run_atomicrag.py --subset medical --resume         # resume interrupted run
"""

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent / "atomicrag"
sys.path.insert(0, str(PROJECT_ROOT))

from atomicrag import IndexPipeline, RetrievePipeline, AtomicRAGConfig
from atomicrag.integrations.gemini import GeminiLLM, GeminiEmbedding
from atomicrag.models.graph import KnowledgeGraph

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_atomicrag")

DATASET_DIR = SCRIPT_DIR / "datasets" / "graphrag_bench"

SUBSET_PATHS = {
    "medical": {
        "corpus": DATASET_DIR / "Datasets" / "Corpus" / "medical.json",
        "questions": DATASET_DIR / "Datasets" / "Questions" / "medical_questions.json",
    },
    "novel": {
        "corpus": DATASET_DIR / "Datasets" / "Corpus" / "novel.json",
        "questions": DATASET_DIR / "Datasets" / "Questions" / "novel_questions.json",
    },
}

QUESTION_TYPE_MAP = {
    "Fact Retrieval": "type1",
    "Complex Reasoning": "type2",
    "Contextual Summarize": "type3",
    "Creative Generation": "type4",
}

ANSWER_SYSTEM_PROMPT = """
---Role---
You are a helpful assistant responding to user queries.

---Goal---
Generate direct and concise answers based strictly on the provided Knowledge Base.
Respond in plain text without explanations or formatting.
Maintain conversation continuity and use the same language as the query.
If the answer is unknown, respond with "I don't know".

---Conversation History---

---Knowledge Base---
{context}
"""


# ─── Groq LLM (satisfies AtomicRAG's BaseLLM protocol) ──────────────

class GroqLLM:
    """OpenAI-compatible LLM client for Groq. Implements generate(prompt) -> str."""

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0,
        max_tokens: int = 2048,
        max_retries: int = 100,
        retry_delay: float = 5.0,
    ):
        from openai import OpenAI
        self._client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def generate(self, prompt: str) -> str:
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                err = str(e)
                if "rate_limit" in err.lower() or "429" in err:
                    wait = self._retry_delay * attempt
                    log.warning("Rate limited (attempt %d/%d), waiting %.0fs...",
                                attempt, self._max_retries, wait)
                    time.sleep(wait)
                else:
                    if attempt == self._max_retries:
                        log.error("Groq call failed after %d attempts: %s", self._max_retries, e)
                        raise
                    time.sleep(self._retry_delay)
        return ""


# ─── Checkpoint / Resume ─────────────────────────────────────────────

def load_checkpoint(path: Path) -> list[dict]:
    if path.exists() and path.stat().st_size > 0:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_checkpoint(path: Path, results: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ─── Core logic ──────────────────────────────────────────────────────

def load_corpus(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_questions(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_questions_by_source(questions: list[dict]) -> dict[str, list[dict]]:
    grouped = {}
    for q in questions:
        source = q.get("source", "unknown")
        grouped.setdefault(source, []).append(q)
    return grouped


def build_or_load_graph(
    corpus_name: str,
    corpus_text: str,
    workspace: Path,
    llm,
    emb,
    config: AtomicRAGConfig,
) -> KnowledgeGraph:
    graph_path = workspace / f"{corpus_name}_graph.json"

    if graph_path.exists():
        log.info("Loading cached graph: %s", graph_path.name)
        graph = KnowledgeGraph.from_json(str(graph_path))
        log.info(
            "  Chunks=%d  KUs=%d  Entities=%d",
            len(graph.chunks),
            len(graph.knowledge_units),
            len(graph.entities),
        )
        return graph

    log.info("Indexing corpus '%s' (%d chars, ~%d words)...",
             corpus_name, len(corpus_text), len(corpus_text.split()))

    t0 = time.time()
    graph = IndexPipeline(llm=llm, embedding=emb, config=config).run([corpus_text])
    elapsed = time.time() - t0

    log.info(
        "Indexed in %.0fs — Chunks=%d  KUs=%d  Entities=%d",
        elapsed, len(graph.chunks), len(graph.knowledge_units), len(graph.entities),
    )

    graph.to_json(str(graph_path))
    log.info("Graph saved → %s", graph_path.name)
    return graph


def generate_answer(llm, question: str, context: str) -> str:
    prompt = ANSWER_SYSTEM_PROMPT.format(context=context) + f"\nQuestion: {question}\nAnswer:"
    try:
        return llm.generate(prompt).strip()
    except Exception as e:
        log.warning("LLM generation failed for '%s...': %s", question[:50], e)
        return "I don't know"


def _process_single_question(
    q: dict,
    corpus_name: str,
    retriever: "RetrievePipeline",
    llm,
) -> dict:
    """Retrieve context and generate answer for one question (thread-safe)."""
    t0 = time.time()
    retrieval_result = retriever.search(q["question"])
    context_parts = [item.content for item in retrieval_result.items if item.content]
    context_str = "\n\n".join(context_parts) if context_parts else ""
    answer = generate_answer(llm, q["question"], context_str)
    elapsed = time.time() - t0
    return {
        "id": q["id"],
        "question": q["question"],
        "source": corpus_name,
        "context": context_str,
        "evidence": q.get("evidence", []),
        "question_type": q["question_type"],
        "generated_answer": answer,
        "gold_answer": q.get("answer", ""),
        "_elapsed": elapsed,
    }


def process_corpus(
    corpus_name: str,
    corpus_text: str,
    questions: list[dict],
    llm,
    emb,
    workspace: Path,
    sample: Optional[int],
    retrieve_top_n: int,
    checkpoint_path: Path,
    workers: int = 1,
) -> list[dict]:

    index_config = AtomicRAGConfig(
        chunk_size=1000,
        chunk_overlap=200,
        chunk_strategy="recursive",
        ku_extraction_method="llm",
        ku_batch_size=5,
        ku_concurrency=2,
        verbose=True,
    )

    graph = build_or_load_graph(
        corpus_name, corpus_text, workspace, llm, emb, index_config
    )

    retrieve_config = AtomicRAGConfig(
        traversal_depth=3,
        beam_size=15,
        result_top_n=retrieve_top_n,
        verbose=False,
    )
    retriever = RetrievePipeline(
        graph=graph, llm=llm, embedding=emb, config=retrieve_config
    )

    if sample and sample < len(questions):
        questions = questions[:sample]

    results = load_checkpoint(checkpoint_path)
    done_ids = {r["id"] for r in results}
    remaining = [q for q in questions if q["id"] not in done_ids]

    total = len(questions)
    done_count = len(done_ids)

    log.info(
        "Corpus '%s': %d total, %d already done, %d remaining, workers=%d",
        corpus_name, len(questions), len(done_ids), len(remaining), workers,
    )

    if workers <= 1:
        # Sequential path (original behaviour)
        for q in remaining:
            result = _process_single_question(q, corpus_name, retriever, llm)
            elapsed = result.pop("_elapsed")
            results.append(result)
            done_count += 1
            save_checkpoint(checkpoint_path, results)
            if done_count % 10 == 0 or done_count == total:
                log.info("  [%d/%d] %.1fs — Q: %s...", done_count, total, elapsed, q["question"][:60])
    else:
        # Parallel path
        lock = threading.Lock()

        def on_result(result: dict):
            nonlocal done_count
            elapsed = result.pop("_elapsed")
            with lock:
                results.append(result)
                done_count += 1
                current = done_count
                save_checkpoint(checkpoint_path, results)
            if current % 10 == 0 or current == total:
                log.info("  [%d/%d] %.1fs — Q: %s...", current, total, elapsed, result["question"][:60])

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_single_question, q, corpus_name, retriever, llm): q
                for q in remaining
            }
            for future in concurrent.futures.as_completed(futures):
                q = futures[future]
                try:
                    result = future.result()
                    on_result(result)
                except Exception as exc:
                    log.error("Question '%s...' raised %s", q["question"][:50], exc)

    return results


def reformat_for_eval(results: list[dict]) -> dict:
    grouped = {"type1": [], "type2": [], "type3": [], "type4": []}
    for r in results:
        type_key = QUESTION_TYPE_MAP.get(r["question_type"])
        if type_key:
            grouped[type_key].append(r)
    return grouped


def main():
    parser = argparse.ArgumentParser(
        description="AtomicRAG runner for GraphRAG-Bench (Groq LLM + Gemini Embedding)"
    )
    parser.add_argument(
        "--subset", required=True, choices=["medical", "novel"],
        help="Which subset to evaluate",
    )
    parser.add_argument(
        "--workspace", default="./atomicrag_workspace",
        help="Directory for cached graphs and results",
    )
    parser.add_argument(
        "--llm_provider", default="gemini", choices=["gemini", "groq"],
        help="LLM provider: gemini or groq",
    )
    parser.add_argument(
        "--model", default=None,
        help="LLM model name (default: gemini-2.5-pro for gemini, llama-3.3-70b-versatile for groq)",
    )
    parser.add_argument(
        "--groq_api_key", default=None,
        help="Groq API key (or set GROQ_API_KEY env var). Only needed if --llm_provider groq",
    )
    parser.add_argument(
        "--embedding_model", default=None,
        help="Gemini embedding model (default: from .env or models/gemini-embedding-001)",
    )
    parser.add_argument(
        "--google_api_key", default=None,
        help="Google API key for embeddings + Gemini LLM (default: from .env or GOOGLE_API_KEY)",
    )
    parser.add_argument(
        "--retrieve_top_n", type=int, default=8,
        help="Number of retrieval results to use as context",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Limit questions per corpus (for quick testing)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers for answer generation (default: 1 = sequential)",
    )

    args = parser.parse_args()

    # Load .env for Google API key (embeddings)
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    google_key = args.google_api_key or os.getenv("GOOGLE_API_KEY", "")
    groq_key = args.groq_api_key or os.getenv("GROQ_API_KEY", "")
    embed_model = args.embedding_model or os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

    if not google_key:
        log.error("No Google API key. Set GOOGLE_API_KEY or pass --google_api_key.")
        sys.exit(1)

    provider = args.llm_provider

    if provider == "groq":
        if not groq_key:
            log.error("No Groq API key. Set GROQ_API_KEY or pass --groq_api_key.")
            sys.exit(1)
        model_name = args.model or "llama-3.3-70b-versatile"
        llm = GroqLLM(api_key=groq_key, model=model_name)
    else:
        model_name = args.model or os.getenv("MODEL_NAME", "gemini-2.5-pro")
        llm = GeminiLLM(api_key=google_key, model=model_name)

    log.info("AtomicRAG GraphRAG-Bench Runner")
    log.info("  Subset     : %s", args.subset)
    log.info("  LLM        : %s / %s", provider, model_name)
    log.info("  Embedding  : Gemini / %s", embed_model)
    log.info("  Top-N      : %d", args.retrieve_top_n)
    log.info("  Workers    : %d", args.workers)
    log.info("  Sample     : %s", args.sample or "all")

    emb = GeminiEmbedding(api_key=google_key, model=embed_model)

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    results_dir = Path("./results/atomicrag")
    results_dir.mkdir(parents=True, exist_ok=True)

    paths = SUBSET_PATHS[args.subset]
    corpus_data = load_corpus(paths["corpus"])
    all_questions = load_questions(paths["questions"])
    grouped_questions = group_questions_by_source(all_questions)

    log.info("Loaded %d corpus document(s), %d total questions",
             len(corpus_data), len(all_questions))

    all_results = []

    for item in corpus_data:
        corpus_name = item["corpus_name"]
        corpus_text = item["context"]
        corpus_qs = grouped_questions.get(corpus_name, [])

        if not corpus_qs:
            log.warning("No questions found for corpus '%s', skipping.", corpus_name)
            continue

        checkpoint_path = results_dir / f"checkpoint_{args.subset}_{corpus_name}.json"

        results = process_corpus(
            corpus_name=corpus_name,
            corpus_text=corpus_text,
            questions=corpus_qs,
            llm=llm,
            emb=emb,
            workspace=workspace,
            sample=args.sample,
            retrieve_top_n=args.retrieve_top_n,
            checkpoint_path=checkpoint_path,
            workers=args.workers,
        )
        all_results.extend(results)

    # Save final flat results
    flat_path = results_dir / f"predictions_{args.subset}_flat.json"
    with open(flat_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info("Saved flat results → %s (%d items)", flat_path, len(all_results))

    # Save eval-formatted results
    eval_path = results_dir / f"predictions_{args.subset}.json"
    eval_data = reformat_for_eval(all_results)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    type_counts = {k: len(v) for k, v in eval_data.items()}
    log.info("Saved eval results → %s  %s", eval_path, type_counts)

    log.info("Done! Run evaluation with:")
    log.info("  python -m Evaluation.generation_eval --data_file %s --output_file results/atomicrag/eval_%s.json",
             eval_path, args.subset)


if __name__ == "__main__":
    main()
