"""CLI execution harness for Structured Generation Pipeline over Dev Set (Milestone M5).

Executes end-to-end RAG generation with multi-trial repeats, citation validation,
and abstain verification across Golden QA dev questions.
Logs structured transcripts to runs/transcripts/rag_dev_trials.jsonl.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any, Literal

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

from eval.schema import GoldenDataset, GoldenItem
from src.generation import GenerationResult, generate_answer
from src.retrieval import RetrievalResult, retrieve

DEFAULT_DATASET_PATH = REPO_ROOT / "eval" / "golden_set.json"
DEFAULT_TRANSCRIPTS_PATH = REPO_ROOT / "runs" / "transcripts" / "rag_dev_trials.jsonl"


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for generation evaluation."""
    parser = argparse.ArgumentParser(
        description="Run structured generation pipeline over Golden QA dev set."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to golden dataset JSON (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_TRANSCRIPTS_PATH,
        help=f"Path for transcripts JSONL (default: {DEFAULT_TRANSCRIPTS_PATH})",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "dense", "bm25"],
        default="hybrid",
        help="Retrieval mode to supply context chunks (default: hybrid)",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks per query (default: 5)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of generation trials per question (default: 3)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of questions to evaluate (for smoke testing / dry-running)",
    )
    parser.add_argument(
        "--question-id",
        type=str,
        default=None,
        help="Filter and evaluate a specific question by ID (e.g. gold-049)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output file instead of overwriting",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens for generation response (default: 8192)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["anthropic", "gemini"],
        default=None,
        help="Primary LLM provider (default: from src.config.DEFAULT_LLM_PROVIDER)",
    )
    parser.add_argument(
        "--effort",
        type=str,
        choices=["xhigh", "high", "medium", "low"],
        default="xhigh",
        help="Reasoning effort for extended thinking models (default: xhigh)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=5,
        help="Number of concurrent worker threads for generation (default: 5)",
    )
    return parser.parse_args()


def compute_summary_statistics(trial_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate generation metrics across all executed trial records."""
    total_trials = len(trial_records)
    if total_trials == 0:
        return {
            "total_trials": 0,
            "parse_success_rate": 0.0,
            "valid_output_rate": 0.0,
            "abstain_accuracy": 0.0,
            "hallucinated_citation_rate": 0.0,
            "trials_with_hallucinations_rate": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "mean_latency_ms": 0.0,
            "fallback_count": 0,
            "fallback_rate": 0.0,
        }

    parsed_count = sum(1 for r in trial_records if r.get("output_parsed"))
    valid_count = sum(1 for r in trial_records if r.get("is_valid"))
    fallback_count = sum(1 for r in trial_records if r.get("fell_back"))

    # Unanswerable questions abstain metrics
    unanswerable_trials = [r for r in trial_records if not r.get("answerable")]
    unanswerable_count = len(unanswerable_trials)
    correct_abstains = sum(
        1 for r in unanswerable_trials if r.get("abstain") is True and len(r.get("citations", [])) == 0
    )
    abstain_accuracy = (
        (correct_abstains / unanswerable_count) if unanswerable_count > 0 else 1.0
    )

    # Answerable questions false abstain metrics
    answerable_trials = [r for r in trial_records if r.get("answerable")]
    answerable_count = len(answerable_trials)
    false_abstains = sum(1 for r in answerable_trials if r.get("abstain") is True)
    false_abstain_rate = (
        (false_abstains / answerable_count) if answerable_count > 0 else 0.0
    )

    # Citation hallucination metrics
    total_citations_cited = sum(len(r.get("citations", [])) for r in trial_records)
    total_hallucinated_citations = sum(
        len(r.get("hallucinated_citations", [])) for r in trial_records
    )
    hallucinated_citation_rate = (
        (total_hallucinated_citations / total_citations_cited)
        if total_citations_cited > 0
        else 0.0
    )
    trials_with_hallucinations = sum(
        1 for r in trial_records if len(r.get("hallucinated_citations", [])) > 0
    )

    # Latency statistics
    latencies = [float(r.get("latency_ms", 0.0)) for r in trial_records]
    p50_latency = float(np.percentile(latencies, 50))
    p95_latency = float(np.percentile(latencies, 95))
    mean_latency = float(np.mean(latencies))

    return {
        "total_trials": total_trials,
        "parsed_count": parsed_count,
        "parse_success_rate": parsed_count / total_trials,
        "valid_count": valid_count,
        "valid_output_rate": valid_count / total_trials,
        "unanswerable_trials": unanswerable_count,
        "correct_abstains": correct_abstains,
        "abstain_accuracy": abstain_accuracy,
        "answerable_trials": answerable_count,
        "false_abstains": false_abstains,
        "false_abstain_rate": false_abstain_rate,
        "total_citations_cited": total_citations_cited,
        "total_hallucinated_citations": total_hallucinated_citations,
        "hallucinated_citation_rate": hallucinated_citation_rate,
        "trials_with_hallucinations": trials_with_hallucinations,
        "trials_with_hallucinations_rate": trials_with_hallucinations / total_trials,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "mean_latency_ms": mean_latency,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / total_trials,
    }


def display_summary_report(
    summary: dict[str, Any],
    num_questions: int,
    trials_per_question: int,
    mode: str,
    top_k: int,
    output_path: Path,
) -> None:
    """Print formatted evaluation summary to standard output."""
    sep = "=" * 80
    subsep = "-" * 80
    print("\n" + sep)
    print("           STRUCTURED GENERATION PIPELINE EVALUATION SUMMARY (M5)")
    print(sep)
    print(f"Evaluated Questions:      {num_questions}")
    print(f"Trials per Question:      {trials_per_question}")
    print(f"Total Generation Trials:  {summary['total_trials']}")
    print(f"Retrieval Configuration:  mode='{mode}', top_k={top_k}")
    print(f"Transcripts Output:       {output_path}")
    print(subsep)
    print(
        f"Parse Success Rate:       {summary['parse_success_rate']:.1%} "
        f"({summary['parsed_count']}/{summary['total_trials']})"
    )
    print(
        f"Strict Valid Output Rate: {summary['valid_output_rate']:.1%} "
        f"({summary['valid_count']}/{summary['total_trials']})"
    )
    print(
        f"Abstain Accuracy:         {summary['abstain_accuracy']:.1%} "
        f"({summary['correct_abstains']}/{summary['unanswerable_trials']} unanswerable trials)"
    )
    print(
        f"False Abstain Rate:       {summary['false_abstain_rate']:.1%} "
        f"({summary['false_abstains']}/{summary['answerable_trials']} answerable trials)"
    )
    print(
        f"Hallucinated Citations:   {summary['hallucinated_citation_rate']:.1%} "
        f"({summary['total_hallucinated_citations']}/{summary['total_citations_cited']} total citations)"
    )
    print(
        f"Trials with Hallucination: {summary['trials_with_hallucinations_rate']:.1%} "
        f"({summary['trials_with_hallucinations']}/{summary['total_trials']})"
    )
    print(subsep)
    print(f"P50 Generation Latency:   {summary['p50_latency_ms']:.1f} ms")
    print(f"P95 Generation Latency:   {summary['p95_latency_ms']:.1f} ms")
    print(f"Mean Generation Latency:  {summary['mean_latency_ms']:.1f} ms")
    print(
        f"Fallback Invocations:     {summary['fallback_count']} "
        f"({summary['fallback_rate']:.1%})"
    )
    print(sep + "\n")


def run_generation_evaluation(
    dataset_path: Path,
    output_path: Path,
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
    top_k: int = 5,
    trials: int = 3,
    sample: int | None = None,
    question_id: str | None = None,
    append: bool = False,
    max_tokens: int = 8192,
    provider: str | None = None,
    effort: str | None = None,
    concurrency: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute generation evaluation loop over dev set questions."""
    dataset = GoldenDataset.from_file(dataset_path)
    dev_items = dataset.filter_by_split("dev")

    if question_id:
        dev_items = [it for it in dev_items if it.id == question_id]
        if not dev_items:
            raise ValueError(f"Question ID '{question_id}' not found in dev set")

    if not dev_items:
        raise ValueError(f"No dev items found in golden dataset at {dataset_path}")

    if sample is not None and sample > 0:
        dev_items = dev_items[:sample]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not append and output_path.exists():
        output_path.unlink()

    print(
        f"Starting generation evaluation: {len(dev_items)} items x {trials} trials "
        f"= {len(dev_items) * trials} generations (retrieval mode: {mode}, k: {top_k}, "
        f"effort: {effort or 'xhigh'}, concurrency: {concurrency})..."
    )

    file_lock = threading.Lock()
    all_trial_records: list[dict[str, Any]] = []

    def evaluate_item(item_idx: int, item: GoldenItem) -> list[dict[str, Any]]:
        # 1. Retrieve candidate chunks once per question
        retrieved_chunks = retrieve(
            query=item.question,
            mode=mode,
            k=top_k,
            trace=False,
        )

        item_records: list[dict[str, Any]] = []
        for trial_idx in range(1, trials + 1):
            gen_result: GenerationResult = generate_answer(
                query=item.question,
                retrieved_chunks=retrieved_chunks,
                max_tokens=max_tokens,
                provider=provider,
                reasoning_effort=effort,
            )

            output_obj = gen_result.output
            record: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "question_id": item.id,
                "question": item.question,
                "type": item.type,
                "answerable": item.answerable,
                "gold_urls": item.gold_urls,
                "trial": trial_idx,
                "retrieval_mode": mode,
                "retrieval_k": top_k,
                "retrieved_chunk_ids": [r.chunk_id for r in retrieved_chunks],
                "output_parsed": output_obj is not None,
                "answer": output_obj.answer if output_obj else None,
                "citations": output_obj.citations if output_obj else [],
                "abstain": output_obj.abstain if output_obj else None,
                "valid_citations": gen_result.valid_citations,
                "hallucinated_citations": gen_result.hallucinated_citations,
                "is_valid": gen_result.is_valid,
                "model": gen_result.model,
                "provider": gen_result.provider,
                "effort": effort or "xhigh",
                "fell_back": gen_result.fell_back,
                "latency_ms": round(gen_result.latency_ms, 2),
                "raw_text": gen_result.raw_text,
                "error_message": gen_result.error_message,
            }
            item_records.append(record)

        with file_lock:
            with open(output_path, "a", encoding="utf-8") as f_out:
                for rec in item_records:
                    f_out.write(json.dumps(rec) + "\n")
                f_out.flush()

            status_summary = ", ".join(
                f"T{r['trial']}:{'OK' if r['is_valid'] else 'ERR'}({r['latency_ms']:.0f}ms)"
                for r in item_records
            )
            print(f"[{item_idx}/{len(dev_items)}] ({item.id}) [{item.type}] {status_summary}")

        return item_records

    if concurrency > 1 and len(dev_items) > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {
                executor.submit(evaluate_item, idx, item): idx
                for idx, item in enumerate(dev_items, start=1)
            }
            for future in as_completed(future_to_idx):
                item_records = future.result()
                all_trial_records.extend(item_records)
    else:
        for idx, item in enumerate(dev_items, start=1):
            item_records = evaluate_item(idx, item)
            all_trial_records.extend(item_records)

    # Sort records by question_id and trial for stable downstream ordering
    all_trial_records.sort(key=lambda r: (r["question_id"], r["trial"]))

    summary = compute_summary_statistics(all_trial_records)
    display_summary_report(
        summary=summary,
        num_questions=len(dev_items),
        trials_per_question=trials,
        mode=mode,
        top_k=top_k,
        output_path=output_path,
    )
    return all_trial_records, summary


def main() -> None:
    args = parse_arguments()
    run_generation_evaluation(
        dataset_path=args.dataset,
        output_path=args.output_path,
        mode=args.mode,
        top_k=args.top_k,
        trials=args.trials,
        sample=args.sample,
        question_id=args.question_id,
        append=args.append,
        max_tokens=args.max_tokens,
        provider=args.provider,
        effort=args.effort,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
