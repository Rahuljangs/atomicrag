"""
Retrieval-quality evaluation for AtomicRAG on GraphRAG-Bench.

Computes three metrics using the EXACT same metric functions from GraphRAG-Bench:
  1. Context Relevance  — Is the retrieved context relevant to the question?
  2. Context Recall     — Does the retrieved context cover the ground-truth answer?
  3. Faithfulness       — Is the generated answer supported by the retrieved context?

LLM Judge: Gemini 2.5 Pro (configurable)

Features:
  - Parallel evaluation (default 10 workers)
  - Resume from checkpoint (safe to Ctrl+C and restart)
  - Per-item exponential-backoff retry (up to 6 attempts)
  - Live running averages printed after every item
  - Checkpoint saved after every batch

Usage:
    # Full run (resume-safe, 10 workers)
    python eval_retrieval.py \\
        --data_file results/atomicrag/checkpoint_medical_Medical_gemini_2.5_pro_vocab.json \\
        --workers 10 --delay 1.0

    # Quick test
    python eval_retrieval.py \\
        --data_file results/atomicrag/checkpoint_medical_Medical_gemini_2.5_pro_vocab.json \\
        --sample 5 --workers 3
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

from Evaluation.metrics.context_relevance import compute_context_relevance
from Evaluation.metrics.context_recall import compute_context_recall
from Evaluation.metrics.faithfulness import compute_faithfulness_score

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

# ─── Gemini LLM Judge wrapper with robust retry ─────────────────────

from langchain_core.messages import AIMessage


class GeminiJudgeLLM:
    """LangChain-compatible Gemini wrapper with exponential-backoff retry."""

    def __init__(self, api_key: str, model: str = GEMINI_JUDGE_MODEL,
                 temperature: float = 0.0, max_retries: int = 6):
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
                is_rate_limit = any(k in err_str for k in
                                    ["429", "resource", "quota", "rate", "limit",
                                     "overloaded", "unavailable", "503", "500"])
                if is_rate_limit or attempt < self._max_retries:
                    wait = min(2 ** attempt + 1, 120)
                    if is_rate_limit:
                        print(f"    [rate-limit] attempt {attempt}/{self._max_retries}, "
                              f"backoff {wait}s ...", flush=True)
                    else:
                        print(f"    [error] attempt {attempt}/{self._max_retries}: "
                              f"{str(e)[:80]}, retry in {wait}s ...", flush=True)
                    time.sleep(wait)
                else:
                    print(f"    [FAILED] after {self._max_retries} attempts: {e}",
                          flush=True)
                    return ""
        return ""

    def invoke(self, prompt, **kwargs) -> AIMessage:
        text = prompt if isinstance(prompt, str) else str(prompt)
        return AIMessage(content=self._call_sync(text))

    async def ainvoke(self, prompt, **kwargs) -> AIMessage:
        text = prompt if isinstance(prompt, str) else str(prompt)
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, self._call_sync, text)
        return AIMessage(content=content)


# ─── Compute all 3 retrieval metrics for one item ────────────────────

async def compute_retrieval_single(item: dict, llm) -> dict:
    question = item["question"]
    contexts = [item["context"]]
    generated_answer = item.get("generated_answer", "")
    gold_answer = item.get("gold_answer", "")

    results = {}

    try:
        cr = await compute_context_relevance(question, contexts, llm, None)
        results["context_relevance"] = float(cr) if cr is not None else float("nan")
    except Exception as e:
        results["context_relevance"] = float("nan")
        results["cr_error"] = str(e)[:120]

    try:
        recall = await compute_context_recall(question, contexts, gold_answer, llm, None)
        results["context_recall"] = float(recall) if recall is not None else float("nan")
    except Exception as e:
        results["context_recall"] = float("nan")
        results["recall_error"] = str(e)[:120]

    try:
        faith = await compute_faithfulness_score(question, generated_answer, contexts, llm, None)
        results["faithfulness"] = float(faith) if faith is not None else float("nan")
    except Exception as e:
        results["faithfulness"] = float("nan")
        results["faith_error"] = str(e)[:120]

    ok = all(not np.isnan(results.get(m, float("nan")))
             for m in ["context_relevance", "context_recall", "faithfulness"])
    results["status"] = "OK" if ok else "PARTIAL"
    return results


# ─── Checkpoint helpers ──────────────────────────────────────────────

def _load_checkpoint(path: Path) -> Dict:
    if path and path.exists() and path.stat().st_size > 0:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_checkpoint(path: Path, data: Dict):
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def checkpoint_to_typed(data: list) -> Dict[str, list]:
    typed = {}
    for item in data:
        qtype = item.get("question_type", "")
        type_key = QUESTION_TYPE_MAP.get(qtype)
        if type_key:
            typed.setdefault(type_key, []).append(item)
    return typed


# ─── Running-average helper ──────────────────────────────────────────

def _running_avg(scores: list, key: str) -> float:
    vals = [s[key] for s in scores
            if not np.isnan(s.get(key, float("nan")))]
    return np.mean(vals) * 100 if vals else 0.0


# ─── Main evaluation loop ───────────────────────────────────────────

async def run_evaluation(file_data, llm, sample=None, output_file=None,
                         delay=1.0, workers=10):
    checkpoint_path = (Path(output_file).with_suffix(".checkpoint.json")
                       if output_file else None)
    all_results = _load_checkpoint(checkpoint_path)
    eval_start = time.time()

    for question_type in ["type1", "type2", "type3", "type4"]:
        items = file_data.get(question_type, [])
        if not items:
            continue
        if sample and sample < len(items):
            items = items[:sample]

        label = TYPE_LABELS[question_type]
        ck_key = f"{question_type}_retrieval_scores"
        existing: list = all_results.get(ck_key, [])
        start_idx = len(existing)

        if start_idx >= len(items):
            cr_avg = _running_avg(existing, "context_relevance")
            rc_avg = _running_avg(existing, "context_recall")
            fa_avg = _running_avg(existing, "faithfulness")
            print(f"\n  {label} ({question_type}): already complete "
                  f"({start_idx} items)  CR={cr_avg:.1f}  Recall={rc_avg:.1f}  "
                  f"Faith={fa_avg:.1f}", flush=True)
            continue

        print(f"\n{'='*80}", flush=True)
        print(f"  {label} ({question_type}) — {len(items)} items, "
              f"parallel x{workers}", flush=True)
        if start_idx > 0:
            print(f"  Resuming from item {start_idx + 1} "
                  f"({start_idx} already done)", flush=True)
        print(f"{'='*80}", flush=True)

        remaining = list(range(start_idx, len(items)))
        batch_num = 0

        for batch_start in range(0, len(remaining), workers):
            batch_idx = remaining[batch_start: batch_start + workers]
            batch_num += 1
            batch_t0 = time.time()

            async def _eval_one(idx: int) -> tuple:
                t0 = time.time()
                try:
                    result = await compute_retrieval_single(items[idx], llm)
                except Exception as e:
                    result = {
                        "context_relevance": float("nan"),
                        "context_recall": float("nan"),
                        "faithfulness": float("nan"),
                        "status": f"ERROR: {str(e)[:100]}",
                    }
                return idx, result, time.time() - t0

            batch_results = await asyncio.gather(
                *[_eval_one(idx) for idx in batch_idx]
            )

            for idx, result, elapsed in sorted(batch_results, key=lambda x: x[0]):
                existing.append(result)
                all_results[ck_key] = existing

                cr = result.get("context_relevance", float("nan"))
                rc = result.get("context_recall", float("nan"))
                fa = result.get("faithfulness", float("nan"))
                cr_s = f"CR={cr:.3f}" if not np.isnan(cr) else "CR=---"
                rc_s = f"Rc={rc:.3f}" if not np.isnan(rc) else "Rc=---"
                fa_s = f"Fa={fa:.3f}" if not np.isnan(fa) else "Fa=---"

                avg_cr = _running_avg(existing, "context_relevance")
                avg_rc = _running_avg(existing, "context_recall")
                avg_fa = _running_avg(existing, "faithfulness")

                print(
                    f"  [{idx+1}/{len(items)}] {elapsed:.0f}s  "
                    f"{cr_s}  {rc_s}  {fa_s}  [{result['status']}]  "
                    f"avg(CR={avg_cr:.1f} Rc={avg_rc:.1f} Fa={avg_fa:.1f})  "
                    f"Q: {items[idx]['question'][:45]}...",
                    flush=True,
                )

            _save_checkpoint(checkpoint_path, all_results)

            done = len(existing)
            total = len(items)
            elapsed_total = time.time() - eval_start
            items_per_min = done / (elapsed_total / 60) if elapsed_total > 0 else 0
            eta_min = (total - done) / items_per_min if items_per_min > 0 else 0
            batch_elapsed = time.time() - batch_t0

            print(
                f"    [batch {batch_num} done in {batch_elapsed:.0f}s | "
                f"{done}/{total} complete | "
                f"{items_per_min:.1f} items/min | "
                f"ETA ~{eta_min:.0f} min]",
                flush=True,
            )

            if delay > 0:
                await asyncio.sleep(delay)

        # Aggregate for this type
        n_ok = sum(1 for r in existing if r.get("status") == "OK")
        agg = {
            "context_relevance": _running_avg(existing, "context_relevance") / 100,
            "context_recall": _running_avg(existing, "context_recall") / 100,
            "faithfulness": _running_avg(existing, "faithfulness") / 100,
            "n_evaluated": len(existing),
            "n_ok": n_ok,
        }
        all_results[question_type] = agg
        _save_checkpoint(checkpoint_path, all_results)

        print(f"\n  >> {label}:", flush=True)
        print(f"     Context Relevance = {agg['context_relevance']*100:.2f}",
              flush=True)
        print(f"     Context Recall    = {agg['context_recall']*100:.2f}",
              flush=True)
        print(f"     Faithfulness      = {agg['faithfulness']*100:.2f}",
              flush=True)
        print(f"     (n={agg['n_evaluated']}, ok={agg['n_ok']})", flush=True)

    # ─── Final summary ───────────────────────────────────────────────
    total_time = time.time() - eval_start
    print(f"\n{'='*80}", flush=True)
    print(f"  FINAL RETRIEVAL RESULTS  ({total_time/60:.1f} min)", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  {'Task':<25s}  {'Ctx Relevance':>13s}  {'Ctx Recall':>10s}  "
          f"{'Faithfulness':>12s}", flush=True)
    print(f"  {'-'*25}  {'-'*13}  {'-'*10}  {'-'*12}", flush=True)

    all_cr, all_rc, all_fa = [], [], []
    for qtype in ["type1", "type2", "type3", "type4"]:
        agg = all_results.get(qtype)
        if isinstance(agg, dict) and "context_relevance" in agg:
            lbl = TYPE_LABELS[qtype]
            cr_v = agg["context_relevance"] * 100
            rc_v = agg["context_recall"] * 100
            fa_v = agg["faithfulness"] * 100
            print(f"  {lbl:<25s}  {cr_v:>12.2f}%  {rc_v:>9.2f}%  {fa_v:>11.2f}%",
                  flush=True)
            all_cr.append(agg["context_relevance"])
            all_rc.append(agg["context_recall"])
            all_fa.append(agg["faithfulness"])

    if all_cr:
        print(f"  {'-'*25}  {'-'*13}  {'-'*10}  {'-'*12}", flush=True)
        print(f"  {'Average':<25s}  "
              f"{np.mean(all_cr)*100:>12.2f}%  "
              f"{np.mean(all_rc)*100:>9.2f}%  "
              f"{np.mean(all_fa)*100:>11.2f}%", flush=True)

    if output_file:
        clean = {}
        for k, v in all_results.items():
            if not k.endswith("_scores") and isinstance(v, dict):
                clean[k] = v
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"\nResults saved to {output_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG-Bench Retrieval Metrics "
                    "(context relevance, context recall, faithfulness)",
    )
    parser.add_argument("--data_file", required=True,
                        help="Checkpoint JSON with question/context/evidence/"
                             "generated_answer/gold_answer")
    parser.add_argument("--output_file",
                        default="results/atomicrag/eval_retrieval_scores.json")
    parser.add_argument("--sample", type=int, default=None,
                        help="Evaluate only first N items per type (for testing)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between batches (rate-limit safety)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel workers per batch (default: 10)")
    parser.add_argument("--judge_model", default=GEMINI_JUDGE_MODEL,
                        help=f"Gemini model for judge (default: {GEMINI_JUDGE_MODEL})")
    args = parser.parse_args()

    load_dotenv(SCRIPT_DIR / ".env")
    google_key = os.getenv("GOOGLE_API_KEY", "")
    if not google_key:
        print("ERROR: GOOGLE_API_KEY not set in .env", flush=True)
        sys.exit(1)

    llm = GeminiJudgeLLM(api_key=google_key, model=args.judge_model, max_retries=6)
    print(f"LLM Judge   : {args.judge_model}", flush=True)
    print(f"Workers     : {args.workers}", flush=True)
    print(f"Batch delay : {args.delay}s", flush=True)
    print(f"Sample limit: {args.sample or 'all'}", flush=True)

    print(f"\nLoading {args.data_file}...", flush=True)
    with open(args.data_file) as f:
        raw = json.loads(f.read(), strict=False)
    file_data = checkpoint_to_typed(raw) if isinstance(raw, list) else raw

    total = 0
    for tkey in ["type1", "type2", "type3", "type4"]:
        n = len(file_data.get(tkey, []))
        if n:
            print(f"  {TYPE_LABELS[tkey]}: {n}", flush=True)
            total += n
    print(f"  Total: {total}\n", flush=True)

    asyncio.run(run_evaluation(
        file_data, llm, args.sample, args.output_file, args.delay, args.workers
    ))


if __name__ == "__main__":
    main()
