"""Agent Evaluation Suite for Milestone M10.

Evaluates the ReAct agent across single-hop, multi-hop, and unanswerable documentation tasks.
Applies trajectory constraint graders (search called, multi-hop depth, step budget, loop guard,
and citation fabrication) and outcome graders (abstain accuracy, citation accuracy, and semantic correctness).
Executes multi-trial evaluations and computes pass@1, pass@3, and pass^3 metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pydantic import BaseModel, Field

from src.agent import AgentResult, arun_agent, run_agent
from src.config import (
    AGENT_MODEL,
    MAX_STEPS,
    REPO_ROOT,
    RUNS_DIR,
)

logger = logging.getLogger(__name__)

DEFAULT_TASKS_PATH = REPO_ROOT / "eval" / "agent_tasks.json"
DEFAULT_REPORT_PATH = REPO_ROOT / "reports" / "agent_eval.md"
DEFAULT_EVAL_TRANSCRIPTS_DIR = RUNS_DIR / "transcripts" / "agent_eval"


class TaskItem(BaseModel):
    """Specification of an agent evaluation task."""

    id: str
    question: str
    type: str  # "single_hop", "multi_hop", "unanswerable"
    answerable: bool
    gold_urls: list[str] = Field(default_factory=list)
    expected_facts: list[str | list[str]] = Field(default_factory=list)


class TrialGrade(BaseModel):
    """Grading results for an individual agent trial execution."""

    trial_num: int
    task_id: str
    constraint_search_called: bool
    constraint_multihop_depth: bool
    constraint_step_budget: bool
    constraint_no_duplicate_queries: bool
    constraint_no_fabricated_citations: bool
    abstain_accuracy: bool
    citation_accuracy: bool
    semantic_correctness: bool
    constraints_passed: bool
    outcomes_passed: bool
    success: bool
    failure_reasons: list[str] = Field(default_factory=list)
    steps: int
    total_tool_calls: int
    latency_ms: float


class TaskEvaluation(BaseModel):
    """Multi-trial evaluation aggregation for a single task."""

    task: TaskItem
    trials: list[TrialGrade]
    success_count: int
    pass_at_k: bool  # pass@k (>= 1 success)
    pass_all: bool   # pass^k (all k successes)
    mean_pass_rate: float


class SuiteEvaluation(BaseModel):
    """Aggregated evaluation benchmark report across all tasks and trials."""

    model: str
    total_tasks: int
    trials_per_task: int
    total_trials: int
    pass_at_1: float
    pass_at_k: float  # pass@3
    pass_all: float   # pass^3
    task_evaluations: list[TaskEvaluation]
    type_breakdown: dict[str, dict[str, float]]
    constraint_pass_rates: dict[str, float]
    outcome_pass_rates: dict[str, float]
    total_duration_sec: float


def load_tasks(path: Path = DEFAULT_TASKS_PATH) -> list[TaskItem]:
    """Load evaluation tasks from agent_tasks.json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("tasks", data) if isinstance(data, dict) else data
    return [TaskItem(**item) for item in items]


def get_chunk_url_map() -> dict[str, str]:
    """Load mapping from chunk_id to canonical URL from index lookup."""
    try:
        from src.retrieval import get_bm25_and_lookup
        _, lookup = get_bm25_and_lookup()
        return {cid: chunk.url for cid, chunk in lookup._by_id.items() if hasattr(chunk, "url")}
    except Exception as exc:
        logger.warning(f"Could not load chunk lookup index: {exc}")
        return {}


# --- Trajectory Constraint Graders ---


def grade_constraint_search_called(task: TaskItem, result: AgentResult) -> tuple[bool, Optional[str]]:
    """Check constraint: >= 1 search_docs call on answerable tasks."""
    if task.answerable:
        search_calls = [
            c for c in result.tool_call_history if c.get("tool_name") == "search_docs"
        ]
        passed = len(search_calls) >= 1
        return (
            passed,
            None if passed else "Constraint failed: No search_docs tool call made on answerable task",
        )
    return True, None


def grade_constraint_multihop_depth(task: TaskItem, result: AgentResult) -> tuple[bool, Optional[str]]:
    """Check constraint: >= 2 distinct searches on multi-hop tasks."""
    if task.type == "multi_hop":
        unique_searches = set(
            str((c.get("tool_input") or {}).get("query", "")).strip().lower()
            for c in result.tool_call_history
            if c.get("tool_name") == "search_docs"
        )
        passed = len(unique_searches) >= 2
        return (
            passed,
            None
            if passed
            else f"Constraint failed: Multi-hop task required >= 2 distinct searches (got {len(unique_searches)})",
        )
    return True, None


def grade_constraint_step_budget(task: TaskItem, result: AgentResult) -> tuple[bool, Optional[str]]:
    """Check constraint: total steps and tool calls within MAX_STEPS (6)."""
    passed = result.steps <= MAX_STEPS and result.total_tool_calls <= MAX_STEPS
    return (
        passed,
        None
        if passed
        else f"Constraint failed: Exceeded step budget (steps={result.steps}, tool_calls={result.total_tool_calls}, limit={MAX_STEPS})",
    )


def grade_constraint_no_duplicate_queries(task: TaskItem, result: AgentResult) -> tuple[bool, Optional[str]]:
    """Check constraint: no duplicate queries executed."""
    duplicates = [c for c in result.tool_call_history if c.get("is_duplicate")]
    passed = len(duplicates) == 0
    return (
        passed,
        None
        if passed
        else f"Constraint failed: Found {len(duplicates)} duplicate queries in trajectory",
    )


def grade_constraint_no_fabricated_citations(task: TaskItem, result: AgentResult) -> tuple[bool, Optional[str]]:
    """Check constraint: no citations on unanswerable tasks, zero hallucinated citations on answerable."""
    if not task.answerable:
        passed = (
            len(result.output.citations) == 0
            and len(result.valid_citations) == 0
            and len(result.hallucinated_citations) == 0
        )
        return (
            passed,
            None
            if passed
            else f"Constraint failed: Fabricated {len(result.output.citations)} citations on unanswerable task",
        )

    passed = len(result.hallucinated_citations) == 0
    return (
        passed,
        None
        if passed
        else f"Constraint failed: Hallucinated citations detected: {result.hallucinated_citations}",
    )


# --- Outcome Graders ---


def grade_abstain_accuracy(task: TaskItem, result: AgentResult) -> tuple[bool, Optional[str]]:
    """Check outcome: abstain is True if and only if task is unanswerable."""
    expected_abstain = not task.answerable
    passed = result.output.abstain == expected_abstain
    if passed:
        return True, None
    if expected_abstain and not result.output.abstain:
        return False, "Outcome failed: Expected agent to abstain on unanswerable task, but agent answered"
    return False, "Outcome failed: Agent abstained on an answerable task"


def grade_citation_accuracy(
    task: TaskItem,
    result: AgentResult,
    chunk_url_map: dict[str, str],
) -> tuple[bool, Optional[str]]:
    """Check outcome: cited URLs overlap with gold_urls on answerable tasks."""
    if not task.answerable:
        passed = len(result.output.citations) == 0
        return (
            passed,
            None if passed else "Outcome failed: Unanswerable task must not cite any URLs",
        )

    if not result.output.citations:
        return False, "Outcome failed: No citations provided for answerable task"

    cited_urls: set[str] = set()
    for cid in result.output.citations:
        clean_cid = str(cid).strip()
        url = chunk_url_map.get(clean_cid)
        if url:
            cited_urls.add(url.strip())
        else:
            for g in task.gold_urls:
                slug = g.rstrip("/").split("/")[-1]
                if len(slug) >= 4 and slug in clean_cid:
                    cited_urls.add(g)

    gold_set = set(u.strip() for u in task.gold_urls)
    overlap = cited_urls.intersection(gold_set)
    passed = len(overlap) > 0
    return (
        passed,
        None
        if passed
        else f"Outcome failed: Cited URLs {list(cited_urls)} do not overlap with gold URLs {list(gold_set)}",
    )


def grade_semantic_correctness(task: TaskItem, result: AgentResult) -> tuple[bool, Optional[str]]:
    """Check outcome: key expected facts containment or clear unanswerable indication."""
    answer_text = result.output.answer.lower()

    if not task.answerable:
        abstain_cues = [
            "not support",
            "does not provide",
            "no native",
            "cannot",
            "do not have enough",
            "unsupported",
            "abstain",
            "not available",
            "no documentation",
        ]
        passed = result.output.abstain or any(cue in answer_text for cue in abstain_cues)
        return (
            passed,
            None
            if passed
            else "Outcome failed: Unanswerable answer did not express lack of support or documentation",
        )

    if result.output.abstain:
        return False, "Outcome failed: Agent abstained on answerable task"

    if not task.expected_facts:
        return True, None

    matches = []
    for fact in task.expected_facts:
        if isinstance(fact, list):
            if any(str(syn).lower() in answer_text for syn in fact):
                matches.append(str(fact[0]))
        else:
            if str(fact).lower() in answer_text:
                matches.append(str(fact))

    required = 1 if len(task.expected_facts) <= 2 else max(1, len(task.expected_facts) // 2)
    passed = len(matches) >= required
    return (
        passed,
        None
        if passed
        else f"Outcome failed: Matched {len(matches)}/{len(task.expected_facts)} expected facts (needed {required}). Missing: {[f for f in task.expected_facts if (f[0] if isinstance(f, list) else f) not in matches]}",
    )


def grade_trial(
    task: TaskItem,
    result: AgentResult,
    trial_num: int,
    chunk_url_map: dict[str, str],
) -> TrialGrade:
    """Evaluate a single trial execution against all trajectory constraints and outcome criteria."""
    failure_reasons: list[str] = []

    # 1. Trajectory Constraints
    c_search, r_search = grade_constraint_search_called(task, result)
    if not c_search and r_search:
        failure_reasons.append(r_search)

    c_depth, r_depth = grade_constraint_multihop_depth(task, result)
    if not c_depth and r_depth:
        failure_reasons.append(r_depth)

    c_budget, r_budget = grade_constraint_step_budget(task, result)
    if not c_budget and r_budget:
        failure_reasons.append(r_budget)

    c_nodup, r_nodup = grade_constraint_no_duplicate_queries(task, result)
    if not c_nodup and r_nodup:
        failure_reasons.append(r_nodup)

    c_nofab, r_nofab = grade_constraint_no_fabricated_citations(task, result)
    if not c_nofab and r_nofab:
        failure_reasons.append(r_nofab)

    constraints_passed = c_search and c_depth and c_budget and c_nodup and c_nofab

    # 2. Outcome Criteria
    o_abstain, r_abstain = grade_abstain_accuracy(task, result)
    if not o_abstain and r_abstain:
        failure_reasons.append(r_abstain)

    o_cite, r_cite = grade_citation_accuracy(task, result, chunk_url_map)
    if not o_cite and r_cite:
        failure_reasons.append(r_cite)

    o_sem, r_sem = grade_semantic_correctness(task, result)
    if not o_sem and r_sem:
        failure_reasons.append(r_sem)

    outcomes_passed = o_abstain and o_cite and o_sem
    overall_success = constraints_passed and outcomes_passed

    return TrialGrade(
        trial_num=trial_num,
        task_id=task.id,
        constraint_search_called=c_search,
        constraint_multihop_depth=c_depth,
        constraint_step_budget=c_budget,
        constraint_no_duplicate_queries=c_nodup,
        constraint_no_fabricated_citations=c_nofab,
        abstain_accuracy=o_abstain,
        citation_accuracy=o_cite,
        semantic_correctness=o_sem,
        constraints_passed=constraints_passed,
        outcomes_passed=outcomes_passed,
        success=overall_success,
        failure_reasons=failure_reasons,
        steps=result.steps,
        total_tool_calls=result.total_tool_calls,
        latency_ms=result.latency_ms,
    )


async def evaluate_task(
    task: TaskItem,
    num_trials: int,
    model: str,
    chunk_url_map: dict[str, str],
    transcripts_dir: Path,
    agent_runner: Optional[Callable[..., Any]] = None,
) -> TaskEvaluation:
    """Run num_trials evaluations for a given task and aggregate results."""
    trials: list[TrialGrade] = []

    for t_idx in range(1, num_trials + 1):
        task_trial_id = f"{task.id}_t{t_idx}"
        transcript_path = transcripts_dir / f"{task_trial_id}.jsonl"

        if agent_runner is not None:
            # Used for tests or custom runners
            res = agent_runner(task=task.question, task_id=task_trial_id)
            if asyncio.iscoroutine(res):
                res = await res
        else:
            res = await arun_agent(
                task=task.question,
                model=model,
                max_steps=MAX_STEPS,
                transcripts_path=transcript_path,
                task_id=task_trial_id,
            )

        grade = grade_trial(task, res, t_idx, chunk_url_map)
        trials.append(grade)

    successes = sum(1 for tr in trials if tr.success)
    pass_at_k = successes >= 1
    pass_all = successes == num_trials
    mean_rate = successes / float(num_trials) if num_trials > 0 else 0.0

    return TaskEvaluation(
        task=task,
        trials=trials,
        success_count=successes,
        pass_at_k=pass_at_k,
        pass_all=pass_all,
        mean_pass_rate=round(mean_rate, 4),
    )


async def evaluate_suite(
    tasks: list[TaskItem],
    model: str = AGENT_MODEL,
    trials_per_task: int = 3,
    transcripts_dir: Path = DEFAULT_EVAL_TRANSCRIPTS_DIR,
    agent_runner: Optional[Callable[..., Any]] = None,
) -> SuiteEvaluation:
    """Execute complete benchmark suite across all tasks and trials."""
    t_start = time.perf_counter()
    chunk_url_map = get_chunk_url_map()
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    task_evaluations: list[TaskEvaluation] = []
    total_trials = len(tasks) * trials_per_task
    all_trials: list[TrialGrade] = []

    for task in tasks:
        logger.info(f"Evaluating {task.id} ({task.type}) - {trials_per_task} trials...")
        task_eval = await evaluate_task(
            task=task,
            num_trials=trials_per_task,
            model=model,
            chunk_url_map=chunk_url_map,
            transcripts_dir=transcripts_dir,
            agent_runner=agent_runner,
        )
        task_evaluations.append(task_eval)
        all_trials.extend(task_eval.trials)

    # Calculate overall metrics
    total_successes = sum(te.success_count for te in task_evaluations)
    pass_at_1 = total_successes / float(total_trials) if total_trials > 0 else 0.0
    pass_at_k = sum(1 for te in task_evaluations if te.pass_at_k) / float(len(tasks)) if tasks else 0.0
    pass_all = sum(1 for te in task_evaluations if te.pass_all) / float(len(tasks)) if tasks else 0.0

    # Type breakdown
    type_breakdown: dict[str, dict[str, float]] = {}
    for task_type in ["single_hop", "multi_hop", "unanswerable"]:
        matching_evals = [te for te in task_evaluations if te.task.type == task_type]
        if matching_evals:
            n_t = len(matching_evals)
            matching_trials = [tr for te in matching_evals for tr in te.trials]
            m_success = sum(1 for tr in matching_trials if tr.success)
            p1 = m_success / float(len(matching_trials))
            pk = sum(1 for te in matching_evals if te.pass_at_k) / float(n_t)
            pall = sum(1 for te in matching_evals if te.pass_all) / float(n_t)
            type_breakdown[task_type] = {
                "count": float(n_t),
                "pass_at_1": round(p1, 4),
                "pass_at_k": round(pk, 4),
                "pass_all": round(pall, 4),
            }

    # Grader pass rates
    total_tr_count = len(all_trials)
    constraint_pass_rates = {
        "search_called": round(sum(1 for tr in all_trials if tr.constraint_search_called) / total_tr_count, 4),
        "multihop_depth": round(sum(1 for tr in all_trials if tr.constraint_multihop_depth) / total_tr_count, 4),
        "step_budget": round(sum(1 for tr in all_trials if tr.constraint_step_budget) / total_tr_count, 4),
        "no_duplicate_queries": round(sum(1 for tr in all_trials if tr.constraint_no_duplicate_queries) / total_tr_count, 4),
        "no_fabricated_citations": round(sum(1 for tr in all_trials if tr.constraint_no_fabricated_citations) / total_tr_count, 4),
    }

    outcome_pass_rates = {
        "abstain_accuracy": round(sum(1 for tr in all_trials if tr.abstain_accuracy) / total_tr_count, 4),
        "citation_accuracy": round(sum(1 for tr in all_trials if tr.citation_accuracy) / total_tr_count, 4),
        "semantic_correctness": round(sum(1 for tr in all_trials if tr.semantic_correctness) / total_tr_count, 4),
    }

    duration = time.perf_counter() - t_start

    return SuiteEvaluation(
        model=model,
        total_tasks=len(tasks),
        trials_per_task=trials_per_task,
        total_trials=total_trials,
        pass_at_1=round(pass_at_1, 4),
        pass_at_k=round(pass_at_k, 4),
        pass_all=round(pass_all, 4),
        task_evaluations=task_evaluations,
        type_breakdown=type_breakdown,
        constraint_pass_rates=constraint_pass_rates,
        outcome_pass_rates=outcome_pass_rates,
        total_duration_sec=round(duration, 2),
    )


def generate_markdown_report(suite: SuiteEvaluation) -> str:
    """Generate comprehensive evaluation report in Markdown format."""
    lines: list[str] = [
        "# Agent Evaluation Benchmark Report (Milestone M10)",
        "",
        "## Executive Summary",
        "",
        f"- **Model**: `{suite.model}`",
        f"- **Total Tasks**: {suite.total_tasks} (8 single-hop, 2 multi-hop, 2 unanswerable)",
        f"- **Trials per Task**: {suite.trials_per_task} (Total Trials: {suite.total_trials})",
        f"- **Overall Pass@1**: {suite.pass_at_1 * 100:.1f}%",
        f"- **Overall Pass@3 ($pass@3$)**: {suite.pass_at_k * 100:.1f}%",
        f"- **Overall Pass^3 ($pass^3$)**: {suite.pass_all * 100:.1f}%",
        f"- **Total Benchmark Duration**: {suite.total_duration_sec:.1f}s",
        "",
        "### Key High-Level Metrics",
        "",
        "| Metric | Definition | Value |",
        "| :--- | :--- | :--- |",
        f"| **Pass@1** | Mean trial success rate across all {suite.total_trials} trials | {suite.pass_at_1 * 100:.1f}% |",
        f"| **Pass@3 ($pass@3$)** | Task success rate where at least 1 of 3 trials succeeds (capability/coverage) | {suite.pass_at_k * 100:.1f}% |",
        f"| **Pass^3 ($pass^3$)** | Task success rate where all 3 of 3 trials succeed (consistency/reliability) | {suite.pass_all * 100:.1f}% |",
        "",
        "---",
        "",
        "## Task Type Breakdown",
        "",
        "| Task Type | Task Count | Pass@1 | Pass@3 ($pass@3$) | Pass^3 ($pass^3$) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for t_type, stats in suite.type_breakdown.items():
        count = int(stats["count"])
        p1 = stats["pass_at_1"] * 100
        pk = stats["pass_at_k"] * 100
        pall = stats["pass_all"] * 100
        lines.append(f"| `{t_type}` | {count} | {p1:.1f}% | {pk:.1f}% | {pall:.1f}% |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Grader Performance Breakdown",
            "",
            "### Trajectory Constraint Graders (Deterministic, $0 Cost)",
            "",
            "Trajectory constraints verify the agent's behavior and protocol discipline without imposing brittle tool sequence requirements:",
            "",
            "| Constraint | Description | Pass Rate |",
            "| :--- | :--- | :---: |",
            f"| `constraint_search_called` | >= 1 search call on answerable queries | {suite.constraint_pass_rates['search_called'] * 100:.1f}% |",
            f"| `constraint_multihop_depth` | >= 2 distinct searches on multi-hop tasks | {suite.constraint_pass_rates['multihop_depth'] * 100:.1f}% |",
            f"| `constraint_step_budget` | Trajectory finished within MAX_STEPS (6) | {suite.constraint_pass_rates['step_budget'] * 100:.1f}% |",
            f"| `constraint_no_duplicate_queries` | Loop guard prevented redundant queries | {suite.constraint_pass_rates['no_duplicate_queries'] * 100:.1f}% |",
            f"| `constraint_no_fabricated_citations` | Zero hallucinated or fabricated citations | {suite.constraint_pass_rates['no_fabricated_citations'] * 100:.1f}% |",
            "",
            "### Outcome Graders",
            "",
            "Outcome graders verify that the final answer is truthful, well-grounded, and correctly abstains when information is absent:",
            "",
            "| Grader | Description | Pass Rate |",
            "| :--- | :--- | :---: |",
            f"| `abstain_accuracy` | Abstains if and only if question is unanswerable | {suite.outcome_pass_rates['abstain_accuracy'] * 100:.1f}% |",
            f"| `citation_accuracy` | Cited passages resolve to verified gold URLs | {suite.outcome_pass_rates['citation_accuracy'] * 100:.1f}% |",
            f"| `semantic_correctness` | Answer accurately reflects key documented facts | {suite.outcome_pass_rates['semantic_correctness'] * 100:.1f}% |",
            "",
            "---",
            "",
            "## Detailed Per-Task Trial Results",
            "",
            "| ID | Type | Answerable | Successes | Pass@3 | Pass^3 | Failures / Notes |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
        ]
    )

    for te in suite.task_evaluations:
        t = te.task
        ans_str = "Yes" if t.answerable else "No"
        pk_str = "PASS" if te.pass_at_k else "FAIL"
        pall_str = "PASS" if te.pass_all else "FAIL"

        reasons = []
        for tr in te.trials:
            if not tr.success and tr.failure_reasons:
                reasons.extend(tr.failure_reasons)
        unique_reasons = list(dict.fromkeys(reasons))
        reason_display = "; ".join(unique_reasons[:2]) if unique_reasons else "All trials passed cleanly"

        lines.append(
            f"| `{t.id}` | `{t.type}` | {ans_str} | {te.success_count}/{suite.trials_per_task} | {pk_str} | {pall_str} | {reason_display} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Analysis: Why Pass@3 and Pass^3 Differ",
            "",
            "A critical finding from Anthropic's 'Demystifying evals' framework is the fundamental operational distinction between $pass@k$ and $pass^k$:",
            "",
            "1. **Capability vs Reliability Frontier**:",
            "   - **$pass@3$ (Capability/Coverage)** measures whether the agent is capable of finding the right tool calls and answers given multiple independent attempts. A high $pass@3$ indicates strong reasoning capacity and broad coverage.",
            "   - **$pass^3$ (Reliability/Consistency)** measures whether the agent produces correct, grounded results 100% of the time without failing any trajectory constraints or hallucinating citations.",
            "",
            "2. **Stochastic Trajectory Branching**:",
            "   - Even at low sampling temperature, LLM tool selection exhibits non-deterministic branching. On complex multi-hop tasks, trial 1 might query an optimal keyword on step 1, whereas trial 2 might issue a slightly broader query requiring follow-up retrieval.",
            "   - When multi-hop depth requires combining two distinct documents, occasional single-search shortcuts cause variance between individual trials.",
            "",
            "3. **Poka-Yoke Guards Preserve Safety**:",
            "   - The duplicate query loop guard successfully prevents infinite loops when queries repeat, allowing the agent to recover or conclude safely.",
            "   - Step budget clamping prevents context overflow, converting runaway searches into safe fallback conclusions.",
        ]
    )

    return "\n".join(lines)


def run_cli() -> None:
    """CLI execution entrypoint for running the agent evaluation suite."""
    parser = argparse.ArgumentParser(
        description="Run Agent Evaluation Benchmark Suite (Milestone M10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=AGENT_MODEL,
        help=f"Anthropic model name (default: {AGENT_MODEL})",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials per task (default: 3)",
    )
    parser.add_argument(
        "--tasks-path",
        type=str,
        default=str(DEFAULT_TASKS_PATH),
        help=f"Path to agent tasks JSON (default: {DEFAULT_TASKS_PATH})",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=str(DEFAULT_REPORT_PATH),
        help=f"Path to output markdown report (default: {DEFAULT_REPORT_PATH})",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=str,
        default=str(DEFAULT_EVAL_TRANSCRIPTS_DIR),
        help=f"Directory for saving trial transcripts (default: {DEFAULT_EVAL_TRANSCRIPTS_DIR})",
    )
    args = parser.parse_args()

    tasks_path = Path(args.tasks_path)
    output_report_path = Path(args.output_report)
    transcripts_dir = Path(args.transcripts_dir)

    tasks = load_tasks(tasks_path)
    print(f"Loaded {len(tasks)} tasks from {tasks_path}")
    print(f"Running {args.trials} trials per task using model '{args.model}'...")

    suite_eval = asyncio.run(
        evaluate_suite(
            tasks=tasks,
            model=args.model,
            trials_per_task=args.trials,
            transcripts_dir=transcripts_dir,
        )
    )

    report_md = generate_markdown_report(suite_eval)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nEvaluation completed in {suite_eval.total_duration_sec:.1f}s!")
    print(f"Overall Pass@1: {suite_eval.pass_at_1 * 100:.1f}%")
    print(f"Overall Pass@3: {suite_eval.pass_at_k * 100:.1f}%")
    print(f"Overall Pass^3: {suite_eval.pass_all * 100:.1f}%")
    print(f"Report written to: {output_report_path}")


if __name__ == "__main__":
    run_cli()
