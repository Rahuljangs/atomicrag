"""
GraphRAG-Bench Answer Accuracy (ACC) evaluation for AtomicRAG.

Computes ACC for all 4 question types using the EXACT same
compute_answer_correctness function from GraphRAG-Bench.

ACC = 0.75 * Factuality(F1) + 0.25 * SemanticSimilarity(cosine)

LLM Judge: Gemini 2.5 Pro
Embeddings: Gemini (gemini-embedding-001)

Usage:
    python eval_atomicrag.py --data_file results/atomicrag/checkpoint_medical_Medical.json --sample 3
    python eval_atomicrag.py --data_file results/atomicrag/checkpoint_medical_Medical.json --workers 5
    python eval_atomicrag.py --data_file results/atomicrag/checkpoint_medical_Medical.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPT_DIR / "datasets" / "graphrag_bench"
sys.path.insert(0, str(BENCH_DIR))
from Evaluation.metrics.answer_accuracy import (
    compute_answer_correctness,
    generate_statements,
    calculate_factuality,
    calculate_semantic_similarity,
)

QUESTION_TYPE_MAP = {
    "Fact Retrieval": "type1",
    "Complex Reasoning": "type2",
    "Contextual Summarize": "type3",
    "Creative Generation": "type4",
}
TYPE_LABELS = {
    "type1": "Fact Retrieval",
    "type2": "Complex Reasoning",
    "type3": "Contextual Summarize",
    "type4": "Creative Generation",
}

GEMINI_JUDGE_MODEL = "gemini-2.5-pro"
GEMINI_EMBED_MODEL = "models/gemini-embedding-001"

# ─── Gemini LLM Judge wrapper ────────────────────────────────────────

from langchain_core.messages import AIMessage
from langchain_core.embeddings import Embeddings as LCEmbeddings


class GeminiJudgeLLM:
    """LangChain-compatible wrapper for Gemini as LLM judge."""

    def __init__(self, api_key: str, model: str = GEMINI_JUDGE_MODEL,
                 temperature: float = 0.0, max_retries: int = 5):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            lines = t.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            t = "\n".join(lines).strip()
        return t

    def _call_sync(self, text: str) -> str:
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=text,
                    config={
                        "temperature": self._temperature,
                        "max_output_tokens": 4096,
                    },
                )
                raw = resp.text or ""
                return self._strip_markdown_fences(raw)
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "resource" in err_str or "quota" in err_str:
                    wait = min(10 * attempt, 60)
                    print(f"    [rate-limit] attempt {attempt}/{self._max_retries}, wait {wait}s...")
                    time.sleep(wait)
                elif attempt == self._max_retries:
                    print(f"    [FAILED] after {self._max_retries} attempts: {e}")
                    return ""
                else:
                    time.sleep(3 * attempt)
        return ""

    def invoke(self, prompt, **kwargs) -> AIMessage:
        text = prompt if isinstance(prompt, str) else str(prompt)
        return AIMessage(content=self._call_sync(text))

    async def ainvoke(self, prompt, **kwargs) -> AIMessage:
        text = prompt if isinstance(prompt, str) else str(prompt)
        content = await asyncio.get_event_loop().run_in_executor(None, self._call_sync, text)
        return AIMessage(content=content)


# ─── Gemini Embeddings ───────────────────────────────────────────────

class GeminiEmbeddings(LCEmbeddings):
    def __init__(self, api_key: str, model: str = GEMINI_EMBED_MODEL):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self._client.models.embed_content(model=self._model, contents=texts)
        return [list(e.values) for e in result.embeddings]

    def embed_query(self, text: str) -> List[float]:
        result = self._client.models.embed_content(model=self._model, contents=text)
        return list(result.embeddings[0].values)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)


# ─── Sequential ACC evaluation ───────────────────────────────────────

async def compute_acc_single(question, answer, ground_truth, llm, embeddings) -> dict:
    answer_stmts = await generate_statements(llm, answer, None)
    await asyncio.sleep(1)

    gt_stmts = await generate_statements(llm, ground_truth, None)
    await asyncio.sleep(1)

    if not answer_stmts and not gt_stmts:
        return {
            "acc": float("nan"),
            "factuality": float("nan"),
            "similarity": float("nan"),
            "answer_stmts": 0,
            "gt_stmts": 0,
            "status": "EMPTY_BOTH",
        }

    factuality = await calculate_factuality(
        llm, question, answer_stmts, gt_stmts, None, 1.0
    )
    await asyncio.sleep(1)

    similarity = await calculate_semantic_similarity(embeddings, answer, ground_truth)

    acc = float(np.average([factuality, similarity], weights=[0.75, 0.25]))

    return {
        "acc": acc,
        "factuality": factuality,
        "similarity": similarity,
        "answer_stmts": len(answer_stmts),
        "gt_stmts": len(gt_stmts),
        "status": "OK",
    }


# ─── Checkpoint helpers ──────────────────────────────────────────────

def _load_checkpoint(path: Path) -> Dict:
    if path.exists() and path.stat().st_size > 0:
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_checkpoint(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def checkpoint_to_typed(data: list) -> Dict[str, list]:
    typed = {}
    for item in data:
        qtype = item.get("question_type", "")
        type_key = QUESTION_TYPE_MAP.get(qtype)
        if type_key:
            typed.setdefault(type_key, []).append(item)
    return typed


# ─── Main evaluation loop ───────────────────────────────────────────

def _log_result(i, total, item, result, elapsed, acc_vals, existing):
    acc = result["acc"]
    status = result["status"]
    running_avg = np.mean(acc_vals) * 100 if acc_vals else 0.0
    n_ok = len(acc_vals)
    n_fail = len(existing) - n_ok
    fact_str = f"F1={result['factuality']:.3f}" if not np.isnan(result.get('factuality', float('nan'))) else "F1=---"
    sim_str = f"Sim={result['similarity']:.3f}" if not np.isnan(result.get('similarity', float('nan'))) else "Sim=---"
    print(
        f"  [{i+1}/{total}] {elapsed:.1f}s  "
        f"ACC={acc:.4f}  {fact_str}  {sim_str}  "
        f"stmts({result['answer_stmts']}/{result['gt_stmts']})  [{status}]  "
        f"⟨avg ACC={running_avg:.2f} n={n_ok} fail={n_fail}⟩  "
        f"Q: {item['question'][:45]}..."
    )


async def run_evaluation(file_data, llm, embeddings, sample=None, output_file=None, delay=2.0, workers=1):
    checkpoint_path = Path(output_file).with_suffix(".checkpoint.json") if output_file else None
    all_results = _load_checkpoint(checkpoint_path) if checkpoint_path else {}
    eval_start = time.time()
    total_ok = 0
    total_failed = 0

    for question_type in ["type1", "type2", "type3", "type4"]:
        items = file_data.get(question_type, [])
        if not items:
            continue
        if sample and sample < len(items):
            items = items[:sample]

        label = TYPE_LABELS[question_type]
        ck_key = f"{question_type}_acc_scores"
        existing = all_results.get(ck_key, [])
        start_idx = len(existing)

        if start_idx >= len(items):
            avg_acc = np.nanmean([s["acc"] for s in existing if not np.isnan(s.get("acc", float("nan")))])
            print(f"\n  {label} ({question_type}): already complete ({start_idx} items), ACC={avg_acc*100:.2f}")
            continue

        mode_str = "sequential" if workers <= 1 else f"parallel x{workers}"
        print(f"\n{'='*70}")
        print(f"  {label} ({question_type}) — {len(items)} samples, ACC only ({mode_str})")
        if start_idx > 0:
            print(f"  Resuming from item {start_idx + 1}")
        print(f"{'='*70}")

        acc_vals = [s["acc"] for s in existing if not np.isnan(s.get("acc", float("nan")))]

        if workers <= 1:
            # Sequential path
            for i in range(start_idx, len(items)):
                item = items[i]
                t0 = time.time()
                try:
                    result = await compute_acc_single(
                        item["question"], item["generated_answer"],
                        item["gold_answer"], llm, embeddings,
                    )
                except Exception as e:
                    print(f"  [{i+1}/{len(items)}] EXCEPTION — skipped: {e}")
                    result = {"acc": float("nan"), "factuality": float("nan"),
                              "similarity": float("nan"), "answer_stmts": 0,
                              "gt_stmts": 0, "status": f"ERROR: {e}"}

                elapsed = time.time() - t0
                existing.append(result)
                all_results[ck_key] = existing

                if not np.isnan(result["acc"]):
                    acc_vals.append(result["acc"])
                    total_ok += 1
                else:
                    total_failed += 1

                _log_result(i, len(items), item, result, elapsed, acc_vals, existing)

                if checkpoint_path and (i + 1) % 5 == 0:
                    _save_checkpoint(checkpoint_path, all_results)

                if delay > 0:
                    await asyncio.sleep(delay)
        else:
            # Parallel batch path
            remaining_indices = list(range(start_idx, len(items)))

            for batch_start in range(0, len(remaining_indices), workers):
                batch_idx = remaining_indices[batch_start : batch_start + workers]
                batch_t0 = time.time()

                async def _eval_one(idx: int) -> tuple:
                    item = items[idx]
                    t0 = time.time()
                    try:
                        result = await compute_acc_single(
                            item["question"], item["generated_answer"],
                            item["gold_answer"], llm, embeddings,
                        )
                    except Exception as e:
                        result = {"acc": float("nan"), "factuality": float("nan"),
                                  "similarity": float("nan"), "answer_stmts": 0,
                                  "gt_stmts": 0, "status": f"ERROR: {e}"}
                    return idx, result, time.time() - t0

                batch_results = await asyncio.gather(
                    *[_eval_one(idx) for idx in batch_idx]
                )

                for idx, result, elapsed in sorted(batch_results, key=lambda x: x[0]):
                    existing.append(result)
                    all_results[ck_key] = existing

                    if not np.isnan(result["acc"]):
                        acc_vals.append(result["acc"])
                        total_ok += 1
                    else:
                        total_failed += 1

                    _log_result(idx, len(items), items[idx], result, elapsed, acc_vals, existing)

                if checkpoint_path:
                    _save_checkpoint(checkpoint_path, all_results)

                batch_elapsed = time.time() - batch_t0
                print(f"    [batch {batch_start//workers + 1} done in {batch_elapsed:.1f}s]")

                if delay > 0:
                    await asyncio.sleep(delay)

        avg = float(np.mean(acc_vals)) if acc_vals else 0.0
        all_results[question_type] = {"answer_correctness": avg}
        if checkpoint_path:
            _save_checkpoint(checkpoint_path, all_results)

        print(f"\n  >> {label} ACC = {avg*100:.2f}  (n={len(acc_vals)}, failed={len(existing)-len(acc_vals)})")

    total_time = time.time() - eval_start
    print(f"\n{'='*70}")
    print(f"  FINAL ACC RESULTS  ({total_time/60:.1f} min, {total_ok} ok, {total_failed} failed)")
    print(f"{'='*70}")
    for qtype in ["type1", "type2", "type3", "type4"]:
        scores = all_results.get(qtype)
        if isinstance(scores, dict) and "answer_correctness" in scores:
            label = TYPE_LABELS[qtype]
            print(f"  {label:25s}  ACC = {scores['answer_correctness']*100:.2f}")

    if output_file:
        clean = {k: v for k, v in all_results.items()
                 if not k.endswith("_scores") and isinstance(v, dict)}
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG-Bench ACC evaluation (Gemini judge, supports parallel workers)",
    )
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_file", default="results/atomicrag/eval_scores.json")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds to wait between samples/batches (rate-limit safety)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for ACC evaluation (default: 1 = sequential)")
    args = parser.parse_args()

    load_dotenv(SCRIPT_DIR / ".env")
    google_key = os.getenv("GOOGLE_API_KEY", "")
    if not google_key:
        print("ERROR: GOOGLE_API_KEY not set in .env")
        sys.exit(1)

    llm = GeminiJudgeLLM(api_key=google_key, model=GEMINI_JUDGE_MODEL, max_retries=5)
    print(f"LLM Judge: {GEMINI_JUDGE_MODEL}")

    print(f"Embeddings: {GEMINI_EMBED_MODEL}")
    embeddings = GeminiEmbeddings(api_key=google_key, model=GEMINI_EMBED_MODEL)

    print(f"Loading {args.data_file}...")
    with open(args.data_file) as f:
        raw = json.load(f)
    file_data = checkpoint_to_typed(raw) if isinstance(raw, list) else raw

    for tkey in ["type1", "type2", "type3", "type4"]:
        n = len(file_data.get(tkey, []))
        if n:
            print(f"  {TYPE_LABELS[tkey]}: {n}")

    print(f"\n  Workers: {args.workers}")
    print(f"  Delay between {'batches' if args.workers > 1 else 'samples'}: {args.delay}s")
    print(f"  Sample limit: {args.sample or 'all'}\n")

    asyncio.run(run_evaluation(file_data, llm, embeddings, args.sample, args.output_file, args.delay, args.workers))


if __name__ == "__main__":
    main()
