"""Agent Trajectory Failure Analysis Pipeline and Diagnostic Report Generator.

Deterministically re-evaluates saved agent trajectory transcripts under
runs/transcripts/agent_eval/ using eval/eval_agent.py graders (zero live LLM calls),
audits standalone loop trials in runs/transcripts/agent_trials.jsonl, compares
re-evaluated metrics against historical reports/agent_eval.md, classifies every
failing or flaky run under both the 5-category (+ abstention) failure taxonomy and
the ReAct loop failure taxonomy, names the top 2 root causes, and writes
reports/agent_failure_analysis.md.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from eval.eval_agent import (
    DEFAULT_TASKS_PATH,
    TaskItem,
    get_chunk_url_map,
    grade_trial,
    load_tasks,
)
from src.agent import AgentResult
from src.config import MAX_STEPS, REPO_ROOT, RUNS_DIR
from src.generation import GenerationOutput

PrimaryFailureCategory = Literal[
    "bad query",
    "stopped early",
    "loop or duplicate queries",
    "fabricated citation",
    "grader bug",
    "over-helpfulness / failed abstain",
]

ReactTaxonomyCategory = Literal[
    "duplicate queries",
    "premature stop",
    "excessive tool calls",
    "context rot",
    "failed abstention / over-helpfulness",
    "grader bug",
]

PRIMARY_FAILURE_CATEGORIES: tuple[PrimaryFailureCategory, ...] = (
    "bad query",
    "stopped early",
    "loop or duplicate queries",
    "fabricated citation",
    "grader bug",
    "over-helpfulness / failed abstain",
)

REACT_TAXONOMY_CATEGORIES: tuple[ReactTaxonomyCategory, ...] = (
    "duplicate queries",
    "premature stop",
    "excessive tool calls",
    "context rot",
    "failed abstention / over-helpfulness",
    "grader bug",
)

DEFAULT_TRANSCRIPTS_DIR = RUNS_DIR / "transcripts" / "agent_eval"
DEFAULT_AGENT_TRIALS_PATH = RUNS_DIR / "transcripts" / "agent_trials.jsonl"
DEFAULT_HISTORICAL_REPORT_PATH = REPO_ROOT / "reports" / "agent_eval.md"
DEFAULT_OUTPUT_REPORT = REPO_ROOT / "reports" / "agent_failure_analysis.md"


class TranscriptAuditAnnotation(BaseModel):
    """Evidence-verified root cause annotation for an inspected agent trajectory."""

    run_id: str
    source: Literal["eval", "standalone"]
    primary_category: PrimaryFailureCategory
    react_category: ReactTaxonomyCategory
    secondary_signals: list[str] = Field(default_factory=list)
    trajectory_evidence: str
    root_cause_summary: str


class AnalyzedAgentRun(BaseModel):
    """Combined trajectory metrics, re-evaluation grade, and verified failure diagnosis."""

    run_id: str
    source: Literal["eval", "standalone"]
    task_id: str
    trial_num: Optional[int] = None
    question: str
    task_type: str
    answerable: Optional[bool] = None
    status: str
    steps: int
    total_tool_calls: int
    duplicate_query_count: int
    tokens_in: int
    tokens_out: int
    latency_ms: float
    queries: list[str] = Field(default_factory=list)
    retrieved_chunk_count: int
    valid_citations: list[str] = Field(default_factory=list)
    hallucinated_citations: list[str] = Field(default_factory=list)
    abstain: bool
    re_eval_success: Optional[bool] = None
    failure_reasons: list[str] = Field(default_factory=list)
    primary_category: PrimaryFailureCategory
    react_category: ReactTaxonomyCategory
    secondary_signals: list[str] = Field(default_factory=list)
    trajectory_evidence: str
    root_cause_summary: str


class TaskEvalSummary(BaseModel):
    """Re-evaluated per-task summary across its saved trials."""

    task_id: str
    task_type: str
    answerable: bool
    trials_count: int
    success_count: int
    pass_at_k: bool
    pass_all: bool
    failure_reasons: list[str] = Field(default_factory=list)


class TrajectoryEfficiencyStats(BaseModel):
    """Aggregate trajectory step, tool-call, duplicate-guard, and token statistics."""

    total_trials: int
    total_tool_calls: int
    mean_tool_calls: float
    duplicate_queries_total: int
    step_counts: dict[int, int] = Field(default_factory=dict)
    tool_call_counts: dict[int, int] = Field(default_factory=dict)
    mean_tokens_in: float
    max_tokens_in: int
    max_tokens_in_trial: str
    mean_tokens_in_1_to_2_calls: float
    mean_tokens_in_3_plus_calls: float
    total_in_loop_duration_sec: float


class HistoricalReportComparison(BaseModel):
    """Comparison between historical reports/agent_eval.md and current re-evaluation."""

    historical_present: bool
    historical_pass_at_1: Optional[float] = None
    historical_pass_at_3: Optional[float] = None
    historical_pass_all: Optional[float] = None
    historical_semantic_correctness: Optional[float] = None
    historical_task05_successes: Optional[str] = None
    historical_narrative_mismatch: Optional[str] = None
    current_pass_at_1: float
    current_pass_at_3: float
    current_pass_all: float
    current_semantic_correctness: float
    current_task05_successes: Optional[str] = None
    divergence_summary: str


class RootCauseSummary(BaseModel):
    """Named top root cause backed by exact cohort counts and evidence."""

    rank: int
    title: str
    current_failure_count: int
    current_failure_total: int
    cohort_count: int
    cohort_total: int
    affected_runs: list[str]
    mechanism: str


class AgentFailureAnalysisReport(BaseModel):
    """Complete deterministic agent trajectory failure analysis summary."""

    tasks_path: str
    transcripts_dir: str
    agent_trials_path: str
    historical_report_path: str
    model: str
    total_tasks: int
    trials_per_task: int
    total_trials: int
    passing_trials: int
    failing_trials: int
    pass_at_1: float
    pass_at_3: float
    pass_all: float
    type_breakdown: dict[str, dict[str, float]]
    constraint_pass_rates: dict[str, float]
    outcome_pass_rates: dict[str, float]
    task_summaries: list[TaskEvalSummary]
    efficiency_stats: TrajectoryEfficiencyStats
    standalone_summary: dict[str, Any]
    historical_comparison: HistoricalReportComparison
    current_failing_ids: list[str]
    historical_failing_ids: list[str]
    current_primary_counts: dict[str, int]
    current_react_counts: dict[str, int]
    historical_primary_counts: dict[str, int]
    historical_react_counts: dict[str, int]
    cohort_primary_counts: dict[str, int]
    cohort_react_counts: dict[str, int]
    top_root_causes: list[RootCauseSummary]
    analyzed_runs: list[AnalyzedAgentRun]


# Evidence-verified audit annotations covering:
# - All 7 currently failing eval trials (task-03_t2, task-10_t1, task-11_t1, task-11_t3, task-12_t1, task-12_t2, task-12_t3)
# - All 3 historically failing task-05 trials (resolved by synonym-list grader update)
# - 4 flaky/high-step/incomplete trajectories (task-06_t2, task-08_t1, task-10_t2, 2c815ecfbe79)
AGENT_AUDIT_ANNOTATIONS: tuple[TranscriptAuditAnnotation, ...] = (
    TranscriptAuditAnnotation(
        run_id="task-03_t2",
        source="eval",
        primary_category="stopped early",
        react_category="premature stop",
        secondary_signals=["bad query"],
        trajectory_evidence=(
            "Executed 2 parallel search_docs calls in step 1 (total_tool_calls=2 of MAX_STEPS=6, tokens_in=8,796). "
            "Call 1 ('MCP tool delete multiple dashboards at once bulk delete maximum') appended 'maximum', "
            "which pushed product-analytics/surfaces/mcp.mdx#5 out of top-5 (unlike task-03_t1/task-03_t3 where Call 1 omitted "
            "'maximum' and retrieved mcp.mdx#5). Call 2 ('PostHog MCP dashboards bulk delete tool') also missed "
            "mcp.mdx#5. Despite 4 remaining tool-budget steps, the agent abstained immediately at step 2."
        ),
        root_cause_summary=(
            "Query-term dilution in Call 1 dropped mcp.mdx#5 from top-5, and the agent stopped prematurely "
            "at 2 tool calls (4 unused budget steps) with a false abstention instead of reformulating."
        ),
    ),
    TranscriptAuditAnnotation(
        run_id="task-10_t1",
        source="eval",
        primary_category="fabricated citation",
        react_category="context rot",
        secondary_signals=["excessive tool calls"],
        trajectory_evidence=(
            "Executed 3 steps / 3 tool calls accumulating 15 unique chunks and 17,133 input tokens (2.32x "
            "task-10_t3's 7,377 tokens). Call 3 retrieved 'experiments/testing-and-launching.mdx#0' at display "
            "heading '### [3] Testing and launching an experiment'. In final_output.citations, the model emitted "
            "'experiments/testing-and-launching.mdx#3', conflating rank index [3] with chunk suffix #0."
        ),
        root_cause_summary=(
            "Under 17,133 input tokens and 15 accumulated chunks across 3 steps, the agent fabricated chunk "
            "suffix #3 for retrieved chunk experiments/testing-and-launching.mdx#0."
        ),
    ),
    TranscriptAuditAnnotation(
        run_id="task-11_t1",
        source="eval",
        primary_category="over-helpfulness / failed abstain",
        react_category="failed abstention / over-helpfulness",
        secondary_signals=["fabricated citation (unanswerable policy)"],
        trajectory_evidence=(
            "Unanswerable CAN bus telematics query (steps=2, calls=2, tokens_in=7,748). Retrieved general "
            "ingestion pipeline docs (how-posthog-works/ingestion-pipeline.mdx#0, metrics/architecture.mdx#1, "
            "npm-registry.md#1). Output began 'No. The documentation I searched contains no mention...' and "
            "proposed a custom bridge/OTLP workaround with abstain=false and 3 citations."
        ),
        root_cause_summary=(
            "Model treated a capability-absence yes/no question as answerable ('No' + custom bridge advice), "
            "setting abstain=false with 3 citations and failing unanswerable policy checks (including substring "
            "abstain_cues in semantic_correctness)."
        ),
    ),
    TranscriptAuditAnnotation(
        run_id="task-11_t3",
        source="eval",
        primary_category="over-helpfulness / failed abstain",
        react_category="failed abstention / over-helpfulness",
        secondary_signals=["fabricated citation (unanswerable policy)"],
        trajectory_evidence=(
            "Unanswerable CAN bus telematics query (steps=2, calls=2, tokens_in=6,510). Retrieved general "
            "ingestion architecture and SaaS connector chunks. Answered 'No. The PostHog documentation I searched "
            "does not describe any native telematics protocol adapter...' and advised writing custom SDK/OTLP "
            "software, setting abstain=false with 4 citations."
        ),
        root_cause_summary=(
            "Over-helpfulness on negative capability question caused abstain=false and 4 cited general-ingestion "
            "chunks (contrast with task-11_t2, which missed ingestion-pipeline.mdx#0 and abstained cleanly)."
        ),
    ),
    TranscriptAuditAnnotation(
        run_id="task-12_t1",
        source="eval",
        primary_category="over-helpfulness / failed abstain",
        react_category="failed abstention / over-helpfulness",
        secondary_signals=["excessive tool calls", "fabricated citation (unanswerable policy)"],
        trajectory_evidence=(
            "Unanswerable COBOL/z/OS mainframe query (steps=3, calls=3, tokens_in=16,005, 15 retrieved chunks). "
            "After Call 1/2 confirmed COBOL is absent from supported SDKs (health-checks/sdk-health.mdx#4), "
            "the agent issued a 3rd search ('send events via HTTP capture API without an SDK') to propose a "
            "custom HTTP integration, outputting abstain=false and 4 citations."
        ),
        root_cause_summary=(
            "Model spent an extra 3rd tool call searching HTTP fallback endpoints to offer a workaround for an "
            "unsupported mainframe stack, failing abstain_accuracy and unanswerable citation checks."
        ),
    ),
    TranscriptAuditAnnotation(
        run_id="task-12_t2",
        source="eval",
        primary_category="over-helpfulness / failed abstain",
        react_category="failed abstention / over-helpfulness",
        secondary_signals=["fabricated citation (unanswerable policy)"],
        trajectory_evidence=(
            "Unanswerable COBOL/z/OS mainframe query (steps=2, calls=2, tokens_in=6,935). Retrieved capture API "
            "(api/capture.mdx#5) and supported SDK list (health-checks/sdk-health.mdx#4). Answered 'No. The "
            "documentation does not describe a native COBOL SDK...' and suggested HTTP POSTs to /i/v0/e/, setting "
            "abstain=false with 3 citations."
        ),
        root_cause_summary=(
            "Answered a negative-capability question with 'No' plus a custom capture API workaround and 3 "
            "citations instead of setting abstain=true and citations=[]."
        ),
    ),
    TranscriptAuditAnnotation(
        run_id="task-12_t3",
        source="eval",
        primary_category="over-helpfulness / failed abstain",
        react_category="failed abstention / over-helpfulness",
        secondary_signals=["excessive tool calls", "fabricated citation (unanswerable policy)"],
        trajectory_evidence=(
            "Unanswerable COBOL/z/OS mainframe query (steps=3, calls=3, tokens_in=16,427, 15 chunks). Issued "
            "a 3rd search ('PostHog capture API send events over HTTP without SDK') after confirming COBOL is "
            "missing from sdk-health.mdx#4, then answered 'No.' with custom HTTP advice, abstain=false, and 2 citations."
        ),
        root_cause_summary=(
            "Repeated the same over-helpfulness pattern as task-12_t1/task-12_t2: extra fallback search for /i/v0/e plus "
            "abstain=false and 2 citations, leaving unanswerable Pass^3 at 0.0%."
        ),
    ),
    TranscriptAuditAnnotation(
        run_id="task-05_t1",
        source="eval",
        primary_category="grader bug",
        react_category="grader bug",
        secondary_signals=[],
        trajectory_evidence=(
            "Single-hop key rotation task (steps=2, calls=2, tokens_in=7,164, 3 valid citations, 0 hallucinated). "
            "Answer accurately cited feature-flags/project-wide-settings.mdx#7/#8/#9 and stated 'backup key' and "
            "'delete the backup key'. Failed historical reports/agent_eval.md (0/3) only because expected_facts "
            "required exact strings ['secondary key', 'delete old key']; passes under current synonym lists."
        ),
        root_cause_summary=(
            "Historical grader bug: naive exact-substring matching penalized accurate 'backup key' terminology "
            "before eval/agent_tasks.json was updated to synonym lists."
        ),
    ),
    TranscriptAuditAnnotation(
        run_id="task-05_t2",
        source="eval",
        primary_category="grader bug",
        react_category="grader bug",
        secondary_signals=[],
        trajectory_evidence=(
            "Single-hop key rotation task (steps=2, calls=2, tokens_in=7,161, 4 valid citations, 0 hallucinated). "
            "Provided both legacy rotation ('backup key', 'delete the backup key') and secret-key migration steps. "
            "Falsely failed in historical reports/agent_eval.md due to exact-string expected_facts mismatch; passes now."
        ),
        root_cause_summary=(
            "Historical grader bug: identical substring mismatch against 'backup key' / 'delete the backup key' "
            "caused a false-negative failure in the recorded benchmark report."
        ),
    ),
    TranscriptAuditAnnotation(
        run_id="task-05_t3",
        source="eval",
        primary_category="grader bug",
        react_category="grader bug",
        secondary_signals=[],
        trajectory_evidence=(
            "Single-hop key rotation task (steps=2, calls=2, tokens_in=7,164, 4 valid citations, 0 hallucinated). "
            "Explained primary/backup key transition and 'delete the backup key'. Falsely failed in historical "
            "reports/agent_eval.md under rigid substring matching; passes under current synonym-list grading."
        ),
        root_cause_summary=(
            "Historical grader bug: completed the 3/3 false-negative failure on task-05 in reports/agent_eval.md "
            "despite accurate, fully grounded answers."
        ),
    ),
    TranscriptAuditAnnotation(
        run_id="task-06_t2",
        source="eval",
        primary_category="bad query",
        react_category="excessive tool calls",
        secondary_signals=["context growth"],
        trajectory_evidence=(
            "Single-hop exposure event task that passed final grading, but required 3 steps / 3 calls and "
            "15,185 input tokens (2.22x task-06_t1's 6,840 tokens). Call 1 and Call 2 already retrieved "
            "experiments/exposures.mdx#10/#14/#16, yet the agent issued a 3rd search for '$feature_flag_called' "
            "definition details, duplicating troubleshooting.mdx#10 and running-experiments-without-feature-flags.mdx#3."
        ),
        root_cause_summary=(
            "Flaky step-count escalation (2 -> 3 calls) on a single-hop task more than doubled input tokens "
            "(6,840 -> 15,185) due to broad initial query framing."
        ),
    ),
    TranscriptAuditAnnotation(
        run_id="task-08_t1",
        source="eval",
        primary_category="bad query",
        react_category="excessive tool calls",
        secondary_signals=["context rot risk"],
        trajectory_evidence=(
            "Highest input-token trajectory in all 36 eval trials (tokens_in=20,044, steps=3, calls=3, 12 unique "
            "chunks across 15 retrieved results; 2.30x task-08_t2's 8,725 tokens). Call 2 ('session replay manage "
            "volume limit recordings sample rate') missed how-to-control-which-sessions-you-record.mdx completely "
            "(pulling replay-vision, dashboards, billing), forcing a 3rd query."
        ),
        root_cause_summary=(
            "Off-target Call 2 query wasted a retrieval turn and inflated context to 20,044 input tokens across "
            "12 unique chunks (15 retrieved results) before recovering on Call 3."
        ),
    ),
    TranscriptAuditAnnotation(
        run_id="task-10_t2",
        source="eval",
        primary_category="bad query",
        react_category="excessive tool calls",
        secondary_signals=["context growth"],
        trajectory_evidence=(
            "Multi-hop task that passed final grading, but escalated to 3 steps / 3 calls and 15,194 input tokens "
            "(2.06x task-10_t3's 7,377 tokens). Call 3 ('force a specific feature flag variant for testing "
            "experiment') retrieved 4 off-topic SDK chunks (godot, roblox, vscode-extension, ruby) alongside "
            "experiments/troubleshooting.mdx#6."
        ),
        root_cause_summary=(
            "Under-specified Call 3 query ('force ... variant') pulled 4 irrelevant SDK docs and doubled context "
            "size relative to task-10_t3."
        ),
    ),
    TranscriptAuditAnnotation(
        run_id="2c815ecfbe79",
        source="standalone",
        primary_category="bad query",
        react_category="excessive tool calls",
        secondary_signals=["incomplete retrieval", "context growth"],
        trajectory_evidence=(
            "Standalone trial 2c815ecfbe79 (from runs/transcripts/agent_trials.jsonl; question: "
            "'how to capture LLM traces and generations in python') "
            "ran 3 steps / 4 tool calls, 18,231 input tokens, and 19 retrieved chunks. Call 3 ('AI Observability "
            "installation Python SDK wrapper OpenAI posthog-python setup') missed ai-observability/installation.mdx "
            "entirely, forcing the final answer to admit it lacked wrapper installation steps."
        ),
        root_cause_summary=(
            "Keyword-overloaded Call 3 query missed the canonical installation guide even after 4 tool calls and "
            "18,231 input tokens, leaving part of the user's question unfulfilled."
        ),
    ),
)


def sanitize_text(text: str) -> str:
    """Normalize em and en dashes to standard ASCII hyphens."""
    return text.replace("\u2014", "-").replace("\u2013", "-")


def load_transcript_records(path: Path) -> list[dict[str, Any]]:
    """Load one or more JSON records from a JSONL or pretty-printed JSON transcript file."""
    if not path.exists():
        raise FileNotFoundError(f"Transcript file not found: {path}")
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    decoder = json.JSONDecoder()
    idx = 0
    length = len(raw)
    records: list[dict[str, Any]] = []
    while idx < length:
        while idx < length and raw[idx].isspace():
            idx += 1
        if idx >= length:
            break
        obj, next_idx = decoder.raw_decode(raw, idx)
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    records.append(item)
        elif isinstance(obj, dict):
            records.append(obj)
        idx = next_idx
    return records


def load_eval_transcripts(
    transcripts_dir: Path = DEFAULT_TRANSCRIPTS_DIR,
) -> list[dict[str, Any]]:
    """Load all benchmark trial records from runs/transcripts/agent_eval/*.jsonl."""
    if not transcripts_dir.exists():
        raise FileNotFoundError(f"Eval transcripts directory not found: {transcripts_dir}")
    files = sorted(transcripts_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl transcript files found in: {transcripts_dir}")

    records: list[dict[str, Any]] = []
    for file_path in files:
        records.extend(load_transcript_records(file_path))
    return records


def parse_task_and_trial(task_trial_id: str) -> tuple[str, int]:
    """Split a trial identifier like 'task-03_t2' into ('task-03', 2)."""
    match = re.match(r"^(.*)_t(\d+)$", task_trial_id.strip())
    if not match:
        raise ValueError(f"Invalid trial identifier format: '{task_trial_id}'")
    return match.group(1), int(match.group(2))


def detect_trajectory_signals(rec: dict[str, Any]) -> dict[str, Any]:
    """Inspect raw tool_call_history and messages for loop/context/citation anomalies."""
    history = list(rec.get("tool_call_history", []))
    queries = [str((c.get("tool_input") or {}).get("query", "")) for c in history]
    normalized_queries = [q.strip().lower() for q in queries if q.strip()]
    flagged_dups = sum(1 for c in history if c.get("is_duplicate"))
    inferred_dups = max(0, len(normalized_queries) - len(set(normalized_queries)))
    duplicate_count = max(flagged_dups, inferred_dups)

    messages = list(rec.get("messages", []))
    message_chars = [len(json.dumps(m)) for m in messages]
    return {
        "queries": queries,
        "duplicate_query_count": duplicate_count,
        "retrieved_chunk_count": len(rec.get("all_retrieved_chunk_ids", [])),
        "message_count": len(messages),
        "total_message_chars": sum(message_chars),
    }


def record_to_agent_result(rec: dict[str, Any]) -> AgentResult:
    """Reconstruct an AgentResult model directly from a saved transcript dictionary."""
    final_output_raw = rec.get("final_output") or {}
    output = GenerationOutput(
        answer=str(final_output_raw.get("answer", "")),
        citations=[str(c) for c in final_output_raw.get("citations", [])],
        abstain=bool(final_output_raw.get("abstain", False)),
    )
    return AgentResult(
        task_id=str(rec.get("task_id", "")),
        question=str(rec.get("question", "")),
        output=output,
        valid_citations=[str(c) for c in rec.get("valid_citations", [])],
        hallucinated_citations=[str(c) for c in rec.get("hallucinated_citations", [])],
        steps=int(rec.get("steps", 0)),
        total_tool_calls=int(rec.get("total_tool_calls", 0)),
        tool_call_history=list(rec.get("tool_call_history", [])),
        all_retrieved_chunk_ids=[str(c) for c in rec.get("all_retrieved_chunk_ids", [])],
        latency_ms=float(rec.get("latency_ms", 0.0)),
        tokens_in=int(rec.get("tokens_in", 0)),
        tokens_out=int(rec.get("tokens_out", 0)),
        status=str(rec.get("status", "unknown")),
        transcript_path=rec.get("transcript_path"),
    )


def parse_historical_agent_eval_report(
    path: Path = DEFAULT_HISTORICAL_REPORT_PATH,
) -> Optional[dict[str, Any]]:
    """Extract recorded benchmark metrics and task-05/task-11/task-12 notes from agent_eval.md."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")

    def _extract_pct(pattern: str) -> Optional[float]:
        match = re.search(pattern, text)
        return round(float(match.group(1)) / 100.0, 4) if match else None

    task05_match = re.search(r"^\|\s*`task-05`\s*\|[^|]+\|[^|]+\|\s*([^|]+?)\s*\|", text, re.MULTILINE)
    narrative_match = re.search(
        r"on `task-11` \(([^)]+)\) and `task-12` \(([^)]+)\)",
        text,
    )
    narrative_mismatch = None
    if narrative_match:
        narrative_mismatch = (
            "Earlier summary metrics and narrative in reports/agent_eval.md reflected an earlier task-set snapshot "
            f"(describing task-11 as '{narrative_match.group(1)}' and task-12 as '{narrative_match.group(2)}'), "
            "whereas current eval/agent_tasks.json defines task-11 as CAN bus automotive telematics and "
            "task-12 as COBOL SDK / IBM z/OS mainframe (alongside synonym-list expansion on task-05)."
        )

    return {
        "pass_at_1": _extract_pct(r"Overall Pass@1\*\*:\s*([0-9.]+)%"),
        "pass_at_3": _extract_pct(r"Overall Pass@3[^:]*:\s*([0-9.]+)%"),
        "pass_all": _extract_pct(r"Overall Pass\^3[^:]*:\s*([0-9.]+)%"),
        "semantic_correctness": _extract_pct(r"`semantic_correctness`\s*\|\s*[^|]*\|\s*([0-9.]+)%"),
        "task05_successes": task05_match.group(1).strip() if task05_match else None,
        "narrative_mismatch": narrative_mismatch,
    }


def _safe_rate(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return round(num / float(den), 4)


def _compute_efficiency_stats(eval_records: list[dict[str, Any]]) -> TrajectoryEfficiencyStats:
    total = len(eval_records)
    if total == 0:
        return TrajectoryEfficiencyStats(
            total_trials=0,
            total_tool_calls=0,
            mean_tool_calls=0.0,
            duplicate_queries_total=0,
            step_counts={},
            tool_call_counts={},
            mean_tokens_in=0.0,
            max_tokens_in=0,
            max_tokens_in_trial="none",
            mean_tokens_in_1_to_2_calls=0.0,
            mean_tokens_in_3_plus_calls=0.0,
            total_in_loop_duration_sec=0.0,
        )

    step_counter = Counter(int(r.get("steps", 0)) for r in eval_records)
    call_counter = Counter(int(r.get("total_tool_calls", 0)) for r in eval_records)
    total_calls = sum(int(r.get("total_tool_calls", 0)) for r in eval_records)
    dup_total = sum(detect_trajectory_signals(r)["duplicate_query_count"] for r in eval_records)
    tokens_list = [int(r.get("tokens_in", 0)) for r in eval_records]
    max_rec = max(eval_records, key=lambda r: int(r.get("tokens_in", 0)))

    low_call_toks = [int(r.get("tokens_in", 0)) for r in eval_records if int(r.get("total_tool_calls", 0)) <= 2]
    high_call_toks = [int(r.get("tokens_in", 0)) for r in eval_records if int(r.get("total_tool_calls", 0)) >= 3]

    return TrajectoryEfficiencyStats(
        total_trials=total,
        total_tool_calls=total_calls,
        mean_tool_calls=round(total_calls / float(total), 2),
        duplicate_queries_total=dup_total,
        step_counts=dict(sorted(step_counter.items())),
        tool_call_counts=dict(sorted(call_counter.items())),
        mean_tokens_in=round(sum(tokens_list) / float(total), 1),
        max_tokens_in=int(max_rec.get("tokens_in", 0)),
        max_tokens_in_trial=str(max_rec.get("task_id", "unknown")),
        mean_tokens_in_1_to_2_calls=round(sum(low_call_toks) / float(len(low_call_toks)), 1) if low_call_toks else 0.0,
        mean_tokens_in_3_plus_calls=round(sum(high_call_toks) / float(len(high_call_toks)), 1) if high_call_toks else 0.0,
        total_in_loop_duration_sec=round(sum(float(r.get("latency_ms", 0.0)) for r in eval_records) / 1000.0, 2),
    )


def _summarize_standalone_trials(standalone_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary telemetry across standalone agent_trials.jsonl runs."""
    total = len(standalone_records)
    if total == 0:
        return {
            "total_runs": 0,
            "duplicate_queries": 0,
            "max_tool_calls": 0,
            "max_tokens_in": 0,
            "max_tokens_in_run_id": "none",
            "max_tokens_in_tool_calls": 0,
            "run_ids": [],
        }
    max_rec = max(standalone_records, key=lambda r: int(r.get("tokens_in", 0)))
    return {
        "total_runs": total,
        "duplicate_queries": sum(detect_trajectory_signals(r)["duplicate_query_count"] for r in standalone_records),
        "max_tool_calls": max(int(r.get("total_tool_calls", 0)) for r in standalone_records),
        "max_tokens_in": int(max_rec.get("tokens_in", 0)),
        "max_tokens_in_run_id": str(max_rec.get("task_id", "unknown")),
        "max_tokens_in_tool_calls": int(max_rec.get("total_tool_calls", 0)),
        "run_ids": [str(r.get("task_id", "")) for r in standalone_records],
    }


def _count_category_map(
    runs: list[AnalyzedAgentRun],
    categories: tuple[str, ...],
    attr: str,
) -> dict[str, int]:
    """Count occurrences of each taxonomy category across analyzed runs."""
    counts = Counter(getattr(r, attr) for r in runs)
    return {cat: int(counts.get(cat, 0)) for cat in categories}


def _relative_display_path(path: Path) -> str:
    """Format a path relative to the repository root when possible."""
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _grade_eval_records(
    eval_records: list[dict[str, Any]],
    tasks_by_id: dict[str, TaskItem],
    resolved_url_map: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, list[Any]]]:
    """Deterministically grade every saved evaluation trial record."""
    eval_by_id: dict[str, dict[str, Any]] = {}
    grades_by_id: dict[str, Any] = {}
    grades_by_task: dict[str, list[Any]] = {tid: [] for tid in tasks_by_id}

    for rec in eval_records:
        run_id = str(rec.get("task_id", ""))
        if run_id in eval_by_id:
            raise ValueError(f"Duplicate eval transcript record for run_id '{run_id}'")
        task_id, trial_num = parse_task_and_trial(run_id)
        if task_id not in tasks_by_id:
            raise KeyError(f"Transcript '{run_id}' references unknown task '{task_id}'")
        agent_res = record_to_agent_result(rec)
        grade = grade_trial(tasks_by_id[task_id], agent_res, trial_num, resolved_url_map)
        eval_by_id[run_id] = rec
        grades_by_id[run_id] = grade
        grades_by_task[task_id].append(grade)

    return eval_by_id, grades_by_id, grades_by_task


def _dedupe_failure_reasons(reasons: list[str]) -> list[str]:
    """Deduplicate failure reasons while collapsing trivial article differences."""
    seen: dict[str, str] = {}
    for reason in reasons:
        clean = sanitize_text(reason)
        norm_key = re.sub(r"\s+", " ", re.sub(r"\b(?:a|an|the)\b", " ", clean.lower())).strip()
        if norm_key not in seen:
            seen[norm_key] = clean
    return list(seen.values())


def _build_task_summaries(
    tasks: list[TaskItem],
    grades_by_task: dict[str, list[Any]],
) -> list[TaskEvalSummary]:
    """Aggregate per-trial grades into per-task Pass@k and Pass^k summaries."""
    summaries: list[TaskEvalSummary] = []
    for task in tasks:
        t_grades = sorted(grades_by_task[task.id], key=lambda g: g.trial_num)
        succ = sum(1 for g in t_grades if g.success)
        reasons = _dedupe_failure_reasons([r for g in t_grades if not g.success for r in g.failure_reasons])
        summaries.append(
            TaskEvalSummary(
                task_id=task.id,
                task_type=task.type,
                answerable=task.answerable,
                trials_count=len(t_grades),
                success_count=succ,
                pass_at_k=succ >= 1,
                pass_all=len(t_grades) > 0 and succ == len(t_grades),
                failure_reasons=reasons,
            )
        )
    return summaries


def _build_type_breakdown(
    task_summaries: list[TaskEvalSummary],
    grades_by_task: dict[str, list[Any]],
) -> dict[str, dict[str, float]]:
    """Compute Pass@1, Pass@k, and Pass^k broken down by task type."""
    breakdown: dict[str, dict[str, float]] = {}
    for t_type in ("single_hop", "multi_hop", "unanswerable"):
        matching_tasks = [ts for ts in task_summaries if ts.task_type == t_type]
        if not matching_tasks:
            continue
        m_trials = [g for ts in matching_tasks for g in grades_by_task[ts.task_id]]
        m_succ = sum(1 for g in m_trials if g.success)
        breakdown[t_type] = {
            "count": float(len(matching_tasks)),
            "trials": float(len(m_trials)),
            "successes": float(m_succ),
            "pass_at_1": _safe_rate(m_succ, len(m_trials)),
            "pass_at_k": _safe_rate(sum(1 for ts in matching_tasks if ts.pass_at_k), len(matching_tasks)),
            "pass_all": _safe_rate(sum(1 for ts in matching_tasks if ts.pass_all), len(matching_tasks)),
        }
    return breakdown


def _build_grader_rates(all_grades: list[Any]) -> tuple[dict[str, float], dict[str, float]]:
    """Compute per-constraint and per-outcome pass rates across all graded trials."""
    total_trials = len(all_grades)
    constraint_pass_rates = {
        "search_called": _safe_rate(sum(1 for g in all_grades if g.constraint_search_called), total_trials),
        "multihop_depth": _safe_rate(sum(1 for g in all_grades if g.constraint_multihop_depth), total_trials),
        "step_budget": _safe_rate(sum(1 for g in all_grades if g.constraint_step_budget), total_trials),
        "no_duplicate_queries": _safe_rate(sum(1 for g in all_grades if g.constraint_no_duplicate_queries), total_trials),
        "no_fabricated_citations": _safe_rate(sum(1 for g in all_grades if g.constraint_no_fabricated_citations), total_trials),
    }
    outcome_pass_rates = {
        "abstain_accuracy": _safe_rate(sum(1 for g in all_grades if g.abstain_accuracy), total_trials),
        "citation_accuracy": _safe_rate(sum(1 for g in all_grades if g.citation_accuracy), total_trials),
        "semantic_correctness": _safe_rate(sum(1 for g in all_grades if g.semantic_correctness), total_trials),
    }
    return constraint_pass_rates, outcome_pass_rates


def _build_analyzed_run(
    ann: TranscriptAuditAnnotation,
    rec: dict[str, Any],
    task: Optional[TaskItem] = None,
    trial_num: Optional[int] = None,
    grade: Optional[Any] = None,
) -> AnalyzedAgentRun:
    """Build a unified AnalyzedAgentRun from a transcript record and audit annotation."""
    signals = detect_trajectory_signals(rec)
    final_out = rec.get("final_output") or {}
    is_eval = ann.source == "eval"
    if is_eval and (task is None or grade is None):
        raise ValueError(f"Eval run '{ann.run_id}' requires task and grade")

    return AnalyzedAgentRun(
        run_id=ann.run_id,
        source=ann.source,
        task_id=task.id if is_eval and task else ann.run_id,
        trial_num=trial_num if is_eval else None,
        question=sanitize_text(task.question if is_eval and task else str(rec.get("question", ""))),
        task_type=task.type if is_eval and task else "standalone",
        answerable=task.answerable if is_eval and task else None,
        status=str(rec.get("status", "")),
        steps=int(rec.get("steps", 0)),
        total_tool_calls=int(rec.get("total_tool_calls", 0)),
        duplicate_query_count=signals["duplicate_query_count"],
        tokens_in=int(rec.get("tokens_in", 0)),
        tokens_out=int(rec.get("tokens_out", 0)),
        latency_ms=float(rec.get("latency_ms", 0.0)),
        queries=[sanitize_text(q) for q in signals["queries"]],
        retrieved_chunk_count=signals["retrieved_chunk_count"],
        valid_citations=[str(c) for c in rec.get("valid_citations", [])],
        hallucinated_citations=[str(c) for c in rec.get("hallucinated_citations", [])],
        abstain=bool(final_out.get("abstain", False)),
        re_eval_success=grade.success if is_eval and grade is not None else None,
        failure_reasons=(
            _dedupe_failure_reasons(grade.failure_reasons)
            if is_eval and grade is not None
            else []
        ),
        primary_category=ann.primary_category,
        react_category=ann.react_category,
        secondary_signals=[sanitize_text(s) for s in ann.secondary_signals],
        trajectory_evidence=sanitize_text(ann.trajectory_evidence),
        root_cause_summary=sanitize_text(ann.root_cause_summary),
    )


def _build_analyzed_runs(
    annotations: tuple[TranscriptAuditAnnotation, ...],
    eval_by_id: dict[str, dict[str, Any]],
    grades_by_id: dict[str, Any],
    tasks_by_id: dict[str, TaskItem],
    standalone_by_id: dict[str, dict[str, Any]],
    transcripts_dir: Path,
    agent_trials_path: Path,
) -> list[AnalyzedAgentRun]:
    """Validate audit annotations against loaded transcripts and construct AnalyzedAgentRun items."""
    seen_runs: set[str] = set()
    analyzed_runs: list[AnalyzedAgentRun] = []

    for ann in annotations:
        if ann.run_id in seen_runs:
            raise ValueError(f"Duplicate audit annotation for run_id '{ann.run_id}'")
        seen_runs.add(ann.run_id)

        if ann.source == "eval":
            if ann.run_id not in eval_by_id:
                raise KeyError(f"Annotated eval run '{ann.run_id}' not found in {transcripts_dir}")
            task_id, trial_num = parse_task_and_trial(ann.run_id)
            analyzed_runs.append(
                _build_analyzed_run(
                    ann=ann,
                    rec=eval_by_id[ann.run_id],
                    task=tasks_by_id[task_id],
                    trial_num=trial_num,
                    grade=grades_by_id[ann.run_id],
                )
            )
        else:
            if ann.run_id not in standalone_by_id:
                raise KeyError(f"Annotated standalone run '{ann.run_id}' not found in {agent_trials_path}")
            analyzed_runs.append(_build_analyzed_run(ann=ann, rec=standalone_by_id[ann.run_id]))

    return analyzed_runs


def _fmt_optional_pct(value: Optional[float]) -> str:
    """Format an optional rate in [0, 1] as a percentage string."""
    return f"{value * 100:.1f}%" if value is not None else "n/a"


def _build_historical_comparison(
    historical_report_path: Path,
    task_summaries: list[TaskEvalSummary],
    pass_at_1: float,
    pass_at_3: float,
    pass_all: float,
    semantic_correctness: float,
) -> HistoricalReportComparison:
    """Compare current re-evaluated metrics against historical reports/agent_eval.md."""
    hist_parsed = parse_historical_agent_eval_report(historical_report_path)
    task05_summary = next((ts for ts in task_summaries if ts.task_id == "task-05"), None)
    cur_task05_str = f"{task05_summary.success_count}/{task05_summary.trials_count}" if task05_summary else None

    if hist_parsed:
        divergence_summary = (
            f"Historical reports/agent_eval.md reported Pass@1={_fmt_optional_pct(hist_parsed['pass_at_1'])}, "
            f"Pass@3={_fmt_optional_pct(hist_parsed['pass_at_3'])}, Pass^3={_fmt_optional_pct(hist_parsed['pass_all'])} "
            f"(with task-05 at {hist_parsed['task05_successes'] or 'n/a'}), whereas deterministic re-evaluation against "
            f"current eval/agent_tasks.json yields Pass@1={pass_at_1 * 100:.1f}%, Pass@3={pass_at_3 * 100:.1f}%, "
            f"Pass^3={pass_all * 100:.1f}% (task-05 at {cur_task05_str or 'n/a'}) after synonym-list matching was added "
            "for task-05 ('backup key', 'delete the backup key')."
        )
    else:
        divergence_summary = "Historical reports/agent_eval.md not present for comparison."

    return HistoricalReportComparison(
        historical_present=hist_parsed is not None,
        historical_pass_at_1=hist_parsed["pass_at_1"] if hist_parsed else None,
        historical_pass_at_3=hist_parsed["pass_at_3"] if hist_parsed else None,
        historical_pass_all=hist_parsed["pass_all"] if hist_parsed else None,
        historical_semantic_correctness=hist_parsed["semantic_correctness"] if hist_parsed else None,
        historical_task05_successes=hist_parsed["task05_successes"] if hist_parsed else None,
        historical_narrative_mismatch=hist_parsed["narrative_mismatch"] if hist_parsed else None,
        current_pass_at_1=pass_at_1,
        current_pass_at_3=pass_at_3,
        current_pass_all=pass_all,
        current_semantic_correctness=semantic_correctness,
        current_task05_successes=cur_task05_str,
        divergence_summary=sanitize_text(divergence_summary),
    )


def _build_top_root_causes(
    analyzed_runs: list[AnalyzedAgentRun],
    current_failing_runs: list[AnalyzedAgentRun],
    historical_failing_runs: list[AnalyzedAgentRun],
    type_breakdown: dict[str, dict[str, float]],
    efficiency_stats: TrajectoryEfficiencyStats,
) -> list[RootCauseSummary]:
    """Construct ranked root cause summaries backed by computed cohort telemetry."""
    overhelp_current = [
        r.run_id for r in current_failing_runs if r.primary_category == "over-helpfulness / failed abstain"
    ]
    overhelp_cohort = [
        r.run_id for r in analyzed_runs if r.primary_category == "over-helpfulness / failed abstain"
    ]
    traj_variance_current = [
        r.run_id for r in current_failing_runs
        if r.primary_category in ("stopped early", "fabricated citation", "bad query")
    ]
    traj_variance_cohort = [
        r for r in analyzed_runs
        if r.primary_category in ("stopped early", "fabricated citation", "bad query")
    ]
    flaky_runs = [r for r in traj_variance_cohort if r.re_eval_success is not False]
    flaky_tok_min = min((r.tokens_in for r in flaky_runs), default=0)
    flaky_tok_max = max((r.tokens_in for r in flaky_runs), default=0)
    grader_bug_cnt = sum(1 for r in historical_failing_runs if r.primary_category == "grader bug")

    unans_stats = type_breakdown.get("unanswerable", {})
    unans_trials = int(unans_stats.get("trials", 0))
    unans_succ = int(unans_stats.get("successes", 0))
    unans_fail = max(0, unans_trials - unans_succ)
    unans_fail_pct = _fmt_pct(unans_fail, unans_trials)
    unans_pall_pct = f"{unans_stats.get('pass_all', 0.0) * 100:.1f}%"

    flaky_clause = (
        f" plus {len(flaky_runs)} flaky multi-call runs whose input tokens double to "
        f"{flaky_tok_min:,}-{flaky_tok_max:,}"
        if flaky_runs
        else ""
    )
    grader_clause = (
        f" Separately, historical task-05 ({grader_bug_cnt}/{len(historical_failing_runs)} past failures) "
        "exposed brittle substring grading."
        if grader_bug_cnt > 0
        else ""
    )

    return [
        RootCauseSummary(
            rank=1,
            title="Unanswerable Over-Helpfulness & Negative-Question Abstention Mismatch",
            current_failure_count=len(overhelp_current),
            current_failure_total=len(current_failing_runs),
            cohort_count=len(overhelp_cohort),
            cohort_total=len(analyzed_runs),
            affected_runs=overhelp_cohort,
            mechanism=(
                "When asked negative-capability questions (task-11 CAN bus, task-12 COBOL/z/OS), the agent "
                "retrieves general ingestion or supported-SDK documentation, answers 'No' while proposing "
                "custom HTTP/bridge workarounds, and outputs abstain=false with citations. Because "
                f"eval_agent.py requires abstain=true and citations=[] on unanswerable tasks, {unans_fail} of "
                f"{unans_trials} unanswerable trials fail ({unans_fail_pct} failure rate, {unans_pall_pct} Pass^3)."
            ),
        ),
        RootCauseSummary(
            rank=2,
            title="Query Sensitivity & Multi-Step Context Growth (Trajectory Instability)",
            current_failure_count=len(traj_variance_current),
            current_failure_total=len(current_failing_runs),
            cohort_count=len(traj_variance_cohort),
            cohort_total=len(analyzed_runs),
            affected_runs=[r.run_id for r in traj_variance_cohort],
            mechanism=(
                "Across answerable tasks, single-token query shifts and multi-step retrieval accumulation drive "
                f"answerable failures ({', '.join(traj_variance_current) or 'none'}){flaky_clause} "
                f"(mean {efficiency_stats.mean_tokens_in_3_plus_calls:,.1f} for 3+ calls vs "
                f"{efficiency_stats.mean_tokens_in_1_to_2_calls:,.1f} for 1-2 calls).{grader_clause}"
            ),
        ),
    ]


def analyze_agent_failures(
    tasks_path: Path = DEFAULT_TASKS_PATH,
    transcripts_dir: Path = DEFAULT_TRANSCRIPTS_DIR,
    agent_trials_path: Path = DEFAULT_AGENT_TRIALS_PATH,
    historical_report_path: Path = DEFAULT_HISTORICAL_REPORT_PATH,
    chunk_url_map: Optional[dict[str, str]] = None,
    annotations: tuple[TranscriptAuditAnnotation, ...] = AGENT_AUDIT_ANNOTATIONS,
) -> AgentFailureAnalysisReport:
    """Deterministically re-grade saved agent transcripts and analyze all failing/flaky runs."""
    tasks: list[TaskItem] = load_tasks(tasks_path)
    tasks_by_id: dict[str, TaskItem] = {t.id: t for t in tasks}
    eval_records = load_eval_transcripts(transcripts_dir)
    standalone_records = load_transcript_records(agent_trials_path) if agent_trials_path.exists() else []
    resolved_url_map = chunk_url_map if chunk_url_map is not None else get_chunk_url_map()

    eval_by_id, grades_by_id, grades_by_task = _grade_eval_records(
        eval_records=eval_records,
        tasks_by_id=tasks_by_id,
        resolved_url_map=resolved_url_map,
    )
    all_grades = list(grades_by_id.values())
    total_trials = len(all_grades)
    passing_trials = sum(1 for g in all_grades if g.success)
    failing_trials = total_trials - passing_trials
    current_failing_ids = sorted(rid for rid, g in grades_by_id.items() if not g.success)

    task_summaries = _build_task_summaries(tasks, grades_by_task)
    total_tasks = len(tasks)
    trials_per_task = max((ts.trials_count for ts in task_summaries), default=0)
    pass_at_1 = _safe_rate(passing_trials, total_trials)
    pass_at_3 = _safe_rate(sum(1 for ts in task_summaries if ts.pass_at_k), total_tasks)
    pass_all = _safe_rate(sum(1 for ts in task_summaries if ts.pass_all), total_tasks)

    type_breakdown = _build_type_breakdown(task_summaries, grades_by_task)
    constraint_pass_rates, outcome_pass_rates = _build_grader_rates(all_grades)

    standalone_by_id = {str(r.get("task_id", "")): r for r in standalone_records}
    analyzed_runs = _build_analyzed_runs(
        annotations=annotations,
        eval_by_id=eval_by_id,
        grades_by_id=grades_by_id,
        tasks_by_id=tasks_by_id,
        standalone_by_id=standalone_by_id,
        transcripts_dir=transcripts_dir,
        agent_trials_path=agent_trials_path,
    )

    annotated_eval_ids = {r.run_id for r in analyzed_runs if r.source == "eval"}
    missing_failing = sorted(set(current_failing_ids) - annotated_eval_ids)
    if missing_failing:
        raise ValueError(f"Missing audit annotations for failing eval trials: {missing_failing}")

    current_failing_runs = [r for r in analyzed_runs if r.source == "eval" and r.re_eval_success is False]
    historical_failing_runs = [
        r for r in analyzed_runs
        if r.source == "eval" and (r.re_eval_success is False or r.primary_category == "grader bug")
    ]
    historical_failing_ids = sorted(r.run_id for r in historical_failing_runs)

    historical_comparison = _build_historical_comparison(
        historical_report_path=historical_report_path,
        task_summaries=task_summaries,
        pass_at_1=pass_at_1,
        pass_at_3=pass_at_3,
        pass_all=pass_all,
        semantic_correctness=outcome_pass_rates["semantic_correctness"],
    )

    current_primary_counts = _count_category_map(current_failing_runs, PRIMARY_FAILURE_CATEGORIES, "primary_category")
    current_react_counts = _count_category_map(current_failing_runs, REACT_TAXONOMY_CATEGORIES, "react_category")
    historical_primary_counts = _count_category_map(historical_failing_runs, PRIMARY_FAILURE_CATEGORIES, "primary_category")
    historical_react_counts = _count_category_map(historical_failing_runs, REACT_TAXONOMY_CATEGORIES, "react_category")
    cohort_primary_counts = _count_category_map(analyzed_runs, PRIMARY_FAILURE_CATEGORIES, "primary_category")
    cohort_react_counts = _count_category_map(analyzed_runs, REACT_TAXONOMY_CATEGORIES, "react_category")

    efficiency_stats = _compute_efficiency_stats(eval_records)
    standalone_summary = _summarize_standalone_trials(standalone_records)
    top_root_causes = _build_top_root_causes(
        analyzed_runs=analyzed_runs,
        current_failing_runs=current_failing_runs,
        historical_failing_runs=historical_failing_runs,
        type_breakdown=type_breakdown,
        efficiency_stats=efficiency_stats,
    )

    models = {str(r.get("model", "")) for r in eval_records if r.get("model")}
    model_name = ", ".join(sorted(models)) if models else "unknown"

    return AgentFailureAnalysisReport(
        tasks_path=_relative_display_path(tasks_path),
        transcripts_dir=_relative_display_path(transcripts_dir),
        agent_trials_path=_relative_display_path(agent_trials_path),
        historical_report_path=_relative_display_path(historical_report_path),
        model=model_name,
        total_tasks=total_tasks,
        trials_per_task=trials_per_task,
        total_trials=total_trials,
        passing_trials=passing_trials,
        failing_trials=failing_trials,
        pass_at_1=pass_at_1,
        pass_at_3=pass_at_3,
        pass_all=pass_all,
        type_breakdown=type_breakdown,
        constraint_pass_rates=constraint_pass_rates,
        outcome_pass_rates=outcome_pass_rates,
        task_summaries=task_summaries,
        efficiency_stats=efficiency_stats,
        standalone_summary=standalone_summary,
        historical_comparison=historical_comparison,
        current_failing_ids=current_failing_ids,
        historical_failing_ids=historical_failing_ids,
        current_primary_counts=current_primary_counts,
        current_react_counts=current_react_counts,
        historical_primary_counts=historical_primary_counts,
        historical_react_counts=historical_react_counts,
        cohort_primary_counts=cohort_primary_counts,
        cohort_react_counts=cohort_react_counts,
        top_root_causes=top_root_causes,
        analyzed_runs=analyzed_runs,
    )


def _fmt_pct(num: int, den: int) -> str:
    if den <= 0:
        return "0.0%"
    return f"{(100.0 * num / den):.1f}%"


def _render_executive_summary(report: AgentFailureAnalysisReport) -> list[str]:
    eff = report.efficiency_stats
    hc = report.historical_comparison
    sa = report.standalone_summary
    lines: list[str] = [
        "# Agent Trajectory Failure Analysis & Diagnostic Report",
        "",
        f"- **Model:** `{report.model}`",
        f"- **Tasks & Transcripts:** `{report.tasks_path}` (`{report.total_tasks}` tasks x `{report.trials_per_task}` trials = `{report.total_trials}` eval trials in `{report.transcripts_dir}`)",
        f"- **Standalone Loop Transcripts:** `{report.agent_trials_path}` (`{sa['total_runs']}` runs)",
        f"- **Inspected Failing & Flaky Cohort:** `{len(report.analyzed_runs)}` runs (`{report.failing_trials}` current eval failures, `{len(report.historical_failing_ids) - report.failing_trials}` resolved historical grader-bug failures, `{len(report.analyzed_runs) - len(report.historical_failing_ids)}` flaky/multi-step runs)",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        (
            f"Deterministic offline re-evaluation of all `{report.total_trials}` saved agent trajectories under "
            f"`eval/eval_agent.py` yields `{report.passing_trials}/{report.total_trials}` passing trials "
            f"(**Pass@1 = {report.pass_at_1 * 100:.1f}%**, **Pass@3 = {report.pass_at_3 * 100:.1f}%**, "
            f"**Pass^3 = {report.pass_all * 100:.1f}%**), with `{report.failing_trials}` failing trials "
            f"(`{', '.join(report.current_failing_ids)}`)."
        ),
        "",
        "### Re-Evaluated Benchmark vs Historical `reports/agent_eval.md`",
        "",
        "| Metric / Check | Historical `reports/agent_eval.md` | Current Re-Evaluation (`eval_agent.py`) | Delta / Verified Cause |",
        "| :--- | :---: | :---: | :--- |",
    ]
    if hc.historical_present and hc.historical_pass_at_1 is not None:
        h_p1_cnt = int(round(hc.historical_pass_at_1 * report.total_trials))
        lines.append(
            f"| **Pass@1** | {hc.historical_pass_at_1 * 100:.1f}% ({h_p1_cnt}/{report.total_trials}) | "
            f"{hc.current_pass_at_1 * 100:.1f}% ({report.passing_trials}/{report.total_trials}) | "
            f"{(hc.current_pass_at_1 - hc.historical_pass_at_1) * 100:+.1f} pp (`task-05` synonym fix) |"
        )
        if hc.historical_pass_at_3 is not None:
            h_p3_cnt = int(round(hc.historical_pass_at_3 * report.total_tasks))
            c_p3_cnt = sum(1 for t in report.task_summaries if t.pass_at_k)
            lines.append(
                f"| **Pass@3** | {hc.historical_pass_at_3 * 100:.1f}% ({h_p3_cnt}/{report.total_tasks}) | "
                f"{hc.current_pass_at_3 * 100:.1f}% ({c_p3_cnt}/{report.total_tasks}) | "
                f"{(hc.current_pass_at_3 - hc.historical_pass_at_3) * 100:+.1f} pp (`task-05` 0/3 -> {hc.current_task05_successes or '3/3'}) |"
            )
        if hc.historical_pass_all is not None:
            h_pall_cnt = int(round(hc.historical_pass_all * report.total_tasks))
            c_pall_cnt = sum(1 for t in report.task_summaries if t.pass_all)
            lines.append(
                f"| **Pass^3** | {hc.historical_pass_all * 100:.1f}% ({h_pall_cnt}/{report.total_tasks}) | "
                f"{hc.current_pass_all * 100:.1f}% ({c_pall_cnt}/{report.total_tasks}) | "
                f"{(hc.current_pass_all - hc.historical_pass_all) * 100:+.1f} pp (`task-05` reliability) |"
            )
        if hc.historical_semantic_correctness is not None:
            h_sem_cnt = int(round(hc.historical_semantic_correctness * report.total_trials))
            c_sem_cnt = int(round(hc.current_semantic_correctness * report.total_trials))
            lines.append(
                f"| **`semantic_correctness`** | {hc.historical_semantic_correctness * 100:.1f}% ({h_sem_cnt}/{report.total_trials}) | "
                f"{hc.current_semantic_correctness * 100:.1f}% ({c_sem_cnt}/{report.total_trials}) | "
                f"{(hc.current_semantic_correctness - hc.historical_semantic_correctness) * 100:+.1f} pp |"
            )
        if hc.historical_task05_successes is not None:
            lines.append(
                f"| **`task-05` Trial Score** | `{hc.historical_task05_successes}` | "
                f"`{hc.current_task05_successes}` | `agent_tasks.json` updated from exact strings to synonym lists |"
            )
    if hc.historical_narrative_mismatch:
        lines.extend(["", f"- **Historical Report Baseline:** {hc.historical_narrative_mismatch}"])

    lines.extend(
        [
            "",
            "### Task Type & Grader Breakdown (Current Re-Evaluation)",
            "",
            "| Task Type | Tasks | Trials | Pass@1 | Pass@3 | Pass^3 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for t_type, stats in report.type_breakdown.items():
        lines.append(
            f"| `{t_type}` | {int(stats['count'])} | {int(stats['trials'])} | "
            f"{stats['pass_at_1'] * 100:.1f}% | {stats['pass_at_k'] * 100:.1f}% | {stats['pass_all'] * 100:.1f}% |"
        )

    lines.extend(
        [
            "",
            "| Grader | Pass Rate | Failing Count |",
            "| :--- | :---: | :---: |",
        ]
    )
    all_rates = [
        ("`constraint_search_called`", report.constraint_pass_rates["search_called"]),
        ("`constraint_multihop_depth`", report.constraint_pass_rates["multihop_depth"]),
        ("`constraint_step_budget`", report.constraint_pass_rates["step_budget"]),
        ("`constraint_no_duplicate_queries`", report.constraint_pass_rates["no_duplicate_queries"]),
        ("`constraint_no_fabricated_citations`", report.constraint_pass_rates["no_fabricated_citations"]),
        ("`abstain_accuracy`", report.outcome_pass_rates["abstain_accuracy"]),
        ("`citation_accuracy`", report.outcome_pass_rates["citation_accuracy"]),
        ("`semantic_correctness`", report.outcome_pass_rates["semantic_correctness"]),
    ]
    for g_name, rate in all_rates:
        fail_cnt = report.total_trials - int(round(rate * report.total_trials))
        lines.append(f"| {g_name} | {rate * 100:.1f}% | {fail_cnt} / {report.total_trials} |")

    lines.extend(
        [
            "",
            (
                f"- **Trajectory & Loop-Guard Telemetry (n={eff.total_trials} eval trials):** "
                f"`{eff.duplicate_queries_total}` intercepted or normalized duplicate queries; "
                f"`{eff.total_tool_calls}` total `search_docs` calls (mean `{eff.mean_tool_calls:.2f}`/trial; "
                f"step distribution `{eff.step_counts}`, call distribution `{eff.tool_call_counts}`, budget `{MAX_STEPS}`); "
                f"mean `tokens_in` = `{eff.mean_tokens_in:,.1f}` (`{eff.mean_tokens_in_1_to_2_calls:,.1f}` for 1-2 calls vs "
                f"`{eff.mean_tokens_in_3_plus_calls:,.1f}` for 3+ calls, max `{eff.max_tokens_in:,}` in `{eff.max_tokens_in_trial}`); "
                f"in-loop latency sum `{eff.total_in_loop_duration_sec:.2f}s`."
            ),
            (
                f"- **Standalone Trials (`agent_trials.jsonl`, n={sa['total_runs']}):** "
                + (
                    f"`{sa['duplicate_queries']}` duplicate queries; max tool calls `{sa['max_tool_calls']}`, "
                    f"max `tokens_in` `{sa['max_tokens_in']:,}` "
                    f"(standalone trial `{sa.get('max_tokens_in_run_id', 'unknown')}` from "
                    f"`{report.agent_trials_path}` across {sa.get('max_tokens_in_tool_calls', 0)} calls)."
                    if sa["total_runs"] > 0
                    else "No standalone runs recorded."
                )
            ),
        ]
    )
    return lines


def _render_taxonomy_tables(report: AgentFailureAnalysisReport) -> list[str]:
    cur_n = len(report.current_failing_ids)
    hist_n = len(report.historical_failing_ids)
    coh_n = len(report.analyzed_runs)
    overhelp_cnt = report.current_primary_counts.get("over-helpfulness / failed abstain", 0)

    primary_desc = {
        "bad query": "Under-specified or keyword-overloaded query missed target doc or triggered extra search turns.",
        "stopped early": "Terminated below MAX_STEPS with false abstention instead of reformulating query.",
        "loop or duplicate queries": "Repeated identical search query intercepted by loop guard or redundant loop.",
        "fabricated citation": "Cited a chunk ID absent from all_retrieved_chunk_ids (plus unanswerable policy overlap).",
        "grader bug": "Deterministic substring matcher falsely penalized valid synonym phrasing.",
        "over-helpfulness / failed abstain": "Answered unanswerable question ('No' + workaround) with abstain=false and citations.",
    }
    react_desc = {
        "duplicate queries": "Identical query repeated within a single trajectory.",
        "premature stop": "Halted search loop early with unused step budget despite missing evidence.",
        "excessive tool calls": "Issued redundant or off-target 3rd/4th search calls inflating token context.",
        "context rot": "High multi-step token/chunk load degraded grounding (e.g., #0 cited as #3).",
        "failed abstention / over-helpfulness": "Proposed workaround with citations instead of abstaining on unsupported capabilities.",
        "grader bug": "Evaluation oracle mismatch caused false-negative grading on valid trajectory.",
    }

    lines: list[str] = [
        "",
        "---",
        "",
        "## 2. Failure Category Counts (5-Category + Abstention Taxonomy)",
        "",
        (
            f"Note: Under `eval/eval_agent.py`, the `{overhelp_cnt}` unanswerable over-helpfulness trials also fail "
            "`constraint_no_fabricated_citations` (`Fabricated N citations on unanswerable task`). Below, they are "
            "separated into `over-helpfulness / failed abstain` so distinct mechanisms remain visible."
        ),
        "",
        f"| Failure Category | Current Failures (n={cur_n}) | Current Share | Historical Failures (n={hist_n}) | Inspected Cohort (n={coh_n}) | Mechanism |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ]
    for cat in PRIMARY_FAILURE_CATEGORIES:
        c_cnt = report.current_primary_counts[cat]
        h_cnt = report.historical_primary_counts[cat]
        a_cnt = report.cohort_primary_counts[cat]
        lines.append(
            f"| `{cat}` | {c_cnt} | {_fmt_pct(c_cnt, cur_n)} | {h_cnt} | {a_cnt} | {primary_desc[cat]} |"
        )
    lines.extend(
        [
            f"| **Total** | **{cur_n}** | **{_fmt_pct(cur_n, cur_n)}** | **{hist_n}** | **{coh_n}** | - |",
            "",
            "---",
            "",
            "## 3. ReAct Loop Failure Taxonomy",
            "",
            f"| ReAct Failure Mode | Current Failures (n={cur_n}) | Current Share | Historical Failures (n={hist_n}) | Inspected Cohort (n={coh_n}) | Diagnostic Signature |",
            "| :--- | :---: | :---: | :---: | :---: | :--- |",
        ]
    )
    for cat in REACT_TAXONOMY_CATEGORIES:
        c_cnt = report.current_react_counts[cat]
        h_cnt = report.historical_react_counts[cat]
        a_cnt = report.cohort_react_counts[cat]
        lines.append(
            f"| `{cat}` | {c_cnt} | {_fmt_pct(c_cnt, cur_n)} | {h_cnt} | {a_cnt} | {react_desc[cat]} |"
        )
    lines.append(
        f"| **Total** | **{cur_n}** | **{_fmt_pct(cur_n, cur_n)}** | **{hist_n}** | **{coh_n}** | - |"
    )
    return lines


def _format_run_ref(run_id: str, standalone_ids: set[str], trials_path: str) -> str:
    """Format a run ID, labeling standalone trials with their transcript source."""
    if run_id in standalone_ids:
        return f"standalone trial `{run_id}` (from `{trials_path}`)"
    return f"`{run_id}`"


def _render_root_causes_and_cases(report: AgentFailureAnalysisReport) -> list[str]:
    standalone_ids = {r.run_id for r in report.analyzed_runs if r.source == "standalone"}
    lines: list[str] = [
        "",
        "---",
        "",
        "## 4. Top 2 Root Causes",
        "",
    ]
    for rc in report.top_root_causes:
        runs_fmt = ", ".join(
            _format_run_ref(r, standalone_ids, report.agent_trials_path) for r in rc.affected_runs
        )
        lines.extend(
            [
                f"### Root Cause {rc.rank}: {rc.title}",
                f"- **Failure Share:** `{rc.current_failure_count}/{rc.current_failure_total}` current eval failures (`{_fmt_pct(rc.current_failure_count, rc.current_failure_total)}`), `{rc.cohort_count}/{rc.cohort_total}` inspected runs (`{_fmt_pct(rc.cohort_count, rc.cohort_total)}`)",
                f"- **Affected Runs:** {runs_fmt}",
                f"- **Verified Mechanism:** {rc.mechanism}",
                "",
            ]
        )

    cur_fails = [r for r in report.analyzed_runs if r.source == "eval" and r.re_eval_success is False]
    hist_only = [r for r in report.analyzed_runs if r.source == "eval" and r.primary_category == "grader bug"]
    flaky_runs = [r for r in report.analyzed_runs if r not in cur_fails and r not in hist_only]

    sections = [
        (f"5.1 Current Re-Evaluated Failing Trials (n={len(cur_fails)})", cur_fails),
        (f"5.2 Historical Grader-Bug Failures Resolved by Synonym Expansion (n={len(hist_only)})", hist_only),
        (f"5.3 Flaky Multi-Step & Standalone Trajectory Runs (n={len(flaky_runs)})", flaky_runs),
    ]

    lines.extend(["---", "", f"## 5. Detailed Per-Run Breakdown (n={len(report.analyzed_runs)} Inspected Runs)", ""])
    for title, runs in sections:
        lines.extend([f"### {title}", ""])
        for run in runs:
            outcome_label = "PASS" if run.re_eval_success is True else ("FAIL" if run.re_eval_success is False else "STANDALONE")
            sec_str = ", ".join(f"`{s}`" for s in run.secondary_signals) if run.secondary_signals else "None"
            lines.append(
                f"#### `[{run.task_type}] {run.run_id}` ({outcome_label}) - `{run.primary_category}` | ReAct: `{run.react_category}`"
            )
            if run.source == "standalone":
                lines.append(
                    f"- **Transcript Source:** standalone trial `{run.run_id}` (from `{report.agent_trials_path}`)"
                )
            lines.append(f"- **Question:** {run.question}")
            lines.append(
                f"- **Trajectory Metrics:** Status=`{run.status}` | Steps=`{run.steps}` | Tool Calls=`{run.total_tool_calls}` | "
                f"Duplicate Queries=`{run.duplicate_query_count}` | Tokens In=`{run.tokens_in:,}` | Tokens Out=`{run.tokens_out:,}` | "
                f"Latency=`{run.latency_ms:.1f}ms` | Chunks=`{run.retrieved_chunk_count}`"
            )
            lines.append(
                f"- **Citations & Abstain:** Abstain=`{run.abstain}` | Valid=`{len(run.valid_citations)}` "
                f"({', '.join(f'`{c}`' for c in run.valid_citations) or 'None'}) | "
                f"Hallucinated=`{len(run.hallucinated_citations)}` "
                f"({', '.join(f'`{c}`' for c in run.hallucinated_citations) or 'None'})"
            )
            lines.append("- **Search Queries:** " + (" -> ".join(f"`{q}`" for q in run.queries) or "None"))
            lines.append(f"- **Secondary Signals:** {sec_str}")
            if run.failure_reasons:
                dedup_reasons = _dedupe_failure_reasons(run.failure_reasons)
                lines.append("- **Grader Failure Reasons:** " + "; ".join(f"`{r}`" for r in dedup_reasons))
            lines.append(f"- **Transcript Evidence:** {run.trajectory_evidence}")
            lines.append(f"- **Root Cause:** {run.root_cause_summary}")
            lines.append("")
    return lines


def _render_remediation_hypotheses(report: AgentFailureAnalysisReport) -> list[str]:
    overhelp_runs = [
        r.run_id for r in report.analyzed_runs
        if r.source == "eval" and r.re_eval_success is False and r.primary_category == "over-helpfulness / failed abstain"
    ]
    overhelp_cnt = report.current_primary_counts.get("over-helpfulness / failed abstain", 0)
    proj_pass = report.passing_trials + overhelp_cnt
    proj_pct = _fmt_pct(proj_pass, report.total_trials)
    cur_pct = _fmt_pct(report.passing_trials, report.total_trials)
    unans_stats = report.type_breakdown.get("unanswerable", {})
    unans_pall = f"{unans_stats.get('pass_all', 0.0) * 100:.1f}%"

    ans_fail_runs = [
        r for r in report.analyzed_runs
        if r.source == "eval" and r.re_eval_success is False and r.primary_category in ("stopped early", "fabricated citation", "bad query")
    ]
    flaky_runs = [
        r for r in report.analyzed_runs
        if r.re_eval_success is not False and r.primary_category == "bad query"
    ]
    standalone_ids = {r.run_id for r in report.analyzed_runs if r.source == "standalone"}
    overhelp_fmt = ", ".join(f"`{rid}`" for rid in overhelp_runs) or "None"
    ans_fail_fmt = ", ".join(f"`{r.run_id}` (`{r.primary_category}`)" for r in ans_fail_runs) or "None"
    flaky_fmt = ", ".join(
        _format_run_ref(r.run_id, standalone_ids, report.agent_trials_path) for r in flaky_runs
    )
    flaky_clause = f", plus `{len(flaky_runs)}` high-token multi-call runs ({flaky_fmt})" if flaky_runs else ""

    grader_bug_cnt = report.historical_primary_counts.get("grader bug", 0)
    hist_total = len(report.historical_failing_ids)
    hist_task05_clause = (
        f"Historical `task-05` false negatives (`{grader_bug_cnt}/{hist_total}`), "
        if grader_bug_cnt > 0
        else ""
    )
    eff = report.efficiency_stats

    return [
        "---",
        "",
        "## 6. Prioritized Remediation Hypotheses for Agent Release Gates (Top 3)",
        "",
        "### Hypothesis 1 (Priority 1 - Highest Release Impact): Negative-Question Abstention Calibration & Deterministic Unanswerable Guard",
        f"- **Target Failures:** {overhelp_fmt} (`{overhelp_cnt}/{report.failing_trials}` current failures).",
        (
            "- **Root Mechanism:** `src/agent.py` `SYSTEM_PROMPT` tells the agent to abstain when documentation "
            "lacks enough information, but provides no negative-capability examples. When users ask whether PostHog "
            "natively supports an out-of-domain feature (CAN bus, COBOL), the model answers 'No' with workaround "
            "suggestions and valid citations for general APIs (`abstain=false`)."
        ),
        (
            "- **Proposed Fix:** (1) Add explicit few-shot negative-capability calibration to `SYSTEM_PROMPT` instructing "
            "that when a queried native integration/SDK/adapter does not exist in PostHog docs, the agent MUST set "
            "`abstain: true` and `citations: []` without proposing unsupported workarounds. (2) Add a deterministic "
            "post-retrieval relevance gate / citation verifier when max cross-encoder score across all tool calls falls "
            "below threshold or query entities (e.g., `COBOL`, `CAN bus`) have zero matches in retrieved chunks."
        ),
        (
            f"- **Expected Impact:** Lifts unanswerable `Pass^3` from `{unans_pall}` toward `100.0%`, projected to raise overall "
            f"suite `Pass@1` from `{cur_pct}` (`{report.passing_trials}/{report.total_trials}`) to `{proj_pct}` (`{proj_pass}/{report.total_trials}`)."
        ),
        "",
        "### Hypothesis 2 (Priority 2 - Answerable Reliability & Grounding): Verbatim Chunk-ID Verification & Budget-Aware Reformulation",
        f"- **Target Failures:** {ans_fail_fmt}{flaky_clause}.",
        (
            f"- **Root Mechanism:** Result headers (`### [N] Title` followed by `- **Chunk ID**: ...`) coupled with "
            f"elevated multi-step input token load (mean `{eff.mean_tokens_in_3_plus_calls:,.1f}` for 3+ calls vs "
            f"`{eff.mean_tokens_in_1_to_2_calls:,.1f}` for 1-2 calls) induce rank-index conflation (`#3` instead of `#0`), "
            f"while single-round query misses trigger premature abstention below `MAX_STEPS` (`{MAX_STEPS}`)."
        ),
        (
            "- **Proposed Fix:** (1) On `hallucinated_citations` in `src/agent.py`, run a single deterministic repair/retry "
            "prompt listing exact `all_retrieved_chunk_ids` (or strip rank numbers `[N]` / format as `<chunk id=\"...\">`). "
            "(2) Add a budget-aware reflection guard before permitting `abstain=true` when `total_tool_calls <= 2`, "
            "requiring at least one keyword-broadened reformulation before concluding documentation is absent."
        ),
        (
            "- **Expected Impact:** Eliminates answerable citation fabrication and premature false abstention, "
            "lifting `single_hop` and `multi_hop` `Pass^3` toward `100.0%`."
        ),
        "",
        "### Hypothesis 3 (Priority 3 - Evaluation Hygiene & Gate Synchronization): Semantic Outcome Grading & Automated Report Sync",
        f"- **Target Failures:** {hist_task05_clause}brittle `abstain_cues` in `eval/eval_agent.py`, and historical `reports/agent_eval.md` snapshot drift.",
        (
            "- **Root Mechanism:** `grade_semantic_correctness` relies on naive substring matching (`expected_facts` and "
            "9 hardcoded `abstain_cues` missing `'does not mention'` / `'does not describe'` / `'does not list'`), and "
            "`reports/agent_eval.md` reflects an earlier task-set snapshot prior to synonym expansion and unanswerable-task updates in `eval/agent_tasks.json`."
        ),
        (
            "- **Proposed Fix:** (1) Expand `abstain_cues` or route borderline semantic checks through a calibrated LLM "
            "judge while preserving deterministic trajectory constraint graders. (2) Enforce a CI/regression check verifying "
            "`reports/agent_eval.md` matches deterministic re-evaluation of saved transcripts whenever `eval/agent_tasks.json` "
            "or `eval/eval_agent.py` changes."
        ),
        (
            "- **Expected Impact:** Prevents false-negative grader drift and ensures automated CI/eval summaries stay "
            "synchronized with the committed task suite."
        ),
        "",
    ]


def render_markdown_report(report: AgentFailureAnalysisReport) -> str:
    """Render the complete agent trajectory failure analysis report in Markdown."""
    lines: list[str] = []
    lines.extend(_render_executive_summary(report))
    lines.extend(_render_taxonomy_tables(report))
    lines.extend(_render_root_causes_and_cases(report))
    lines.extend(_render_remediation_hypotheses(report))
    return sanitize_text("\n".join(lines))


def write_agent_failure_report(
    report: AgentFailureAnalysisReport,
    output_path: Path = DEFAULT_OUTPUT_REPORT,
) -> Path:
    """Render and write the agent failure analysis markdown report to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown_report(report)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)
        if not markdown.endswith("\n"):
            f.write("\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze agent trajectory transcripts and failures.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS_PATH, help="Path to eval/agent_tasks.json")
    parser.add_argument("--transcripts-dir", type=Path, default=DEFAULT_TRANSCRIPTS_DIR, help="Path to agent_eval transcripts")
    parser.add_argument("--agent-trials", type=Path, default=DEFAULT_AGENT_TRIALS_PATH, help="Path to agent_trials.jsonl")
    parser.add_argument("--historical-report", type=Path, default=DEFAULT_HISTORICAL_REPORT_PATH, help="Path to reports/agent_eval.md")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_REPORT, help="Output path for markdown diagnostic report")
    args = parser.parse_args()

    report = analyze_agent_failures(
        tasks_path=args.tasks,
        transcripts_dir=args.transcripts_dir,
        agent_trials_path=args.agent_trials,
        historical_report_path=args.historical_report,
    )
    out_path = write_agent_failure_report(report, output_path=args.output)
    print(f"Wrote agent failure analysis report to {out_path}")
    print(f"Pass@1={report.pass_at_1 * 100:.1f}% | Pass@3={report.pass_at_3 * 100:.1f}% | Pass^3={report.pass_all * 100:.1f}%")
    print(f"Current primary counts (n={len(report.current_failing_ids)}): {report.current_primary_counts}")
    print(f"Cohort primary counts (n={len(report.analyzed_runs)}): {report.cohort_primary_counts}")
    print(f"Cohort ReAct counts (n={len(report.analyzed_runs)}): {report.cohort_react_counts}")


if __name__ == "__main__":
    main()
