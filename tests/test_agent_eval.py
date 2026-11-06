"""Unit and offline integration tests for Milestone M10: Agent Evaluation Suite.

All tests are completely offline ($0.00 spend) and verify:
1. Trajectory constraint graders (search called, multi-hop depth, step budget, loop guard, fabrication).
2. Outcome graders (abstain accuracy, citation accuracy, semantic correctness).
3. Multi-trial evaluation runner, pass@k / pass^k calculations, and markdown report generation.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from eval.eval_agent import (
    DEFAULT_TASKS_PATH,
    SuiteEvaluation,
    TaskEvaluation,
    TaskItem,
    TrialGrade,
    evaluate_suite,
    evaluate_task,
    generate_markdown_report,
    get_chunk_url_map,
    grade_abstain_accuracy,
    grade_citation_accuracy,
    grade_constraint_multihop_depth,
    grade_constraint_no_duplicate_queries,
    grade_constraint_no_fabricated_citations,
    grade_constraint_search_called,
    grade_constraint_step_budget,
    grade_semantic_correctness,
    grade_trial,
    load_tasks,
)
from src.agent import AgentResult
from src.config import MAX_STEPS, REPO_ROOT
from src.generation import GenerationOutput

TEST_TMP_DIR = REPO_ROOT / "tests" / "tmp"


@pytest.fixture(autouse=True)
def clean_test_tmp_dir():
    """Ensure project-local tmp directory exists and is cleaned up (adheres to D-043)."""
    TEST_TMP_DIR.mkdir(parents=True, exist_ok=True)
    yield TEST_TMP_DIR
    if TEST_TMP_DIR.exists():
        shutil.rmtree(TEST_TMP_DIR, ignore_errors=True)


def create_mock_agent_result(
    task_id: str = "test-task-1",
    question: str = "Test task question",
    answer: str = "Test answer.",
    citations: list[str] | None = None,
    abstain: bool = False,
    valid_citations: list[str] | None = None,
    hallucinated_citations: list[str] | None = None,
    tool_call_history: list[dict[str, Any]] | None = None,
    steps: int = 1,
    total_tool_calls: int = 1,
    latency_ms: float = 150.0,
    tokens_in: int = 100,
    tokens_out: int = 50,
    status: str = "success",
) -> AgentResult:
    """Helper to construct deterministic AgentResult instances for grader tests."""
    cites = citations or []
    val_cites = valid_citations if valid_citations is not None else list(cites)
    halluc_cites = hallucinated_citations or []
    output = GenerationOutput(
        answer=answer,
        citations=cites,
        abstain=abstain,
    )
    return AgentResult(
        task_id=task_id,
        question=question,
        output=output,
        steps=steps,
        total_tool_calls=total_tool_calls,
        tool_call_history=tool_call_history or [],
        all_retrieved_chunk_ids=val_cites,
        valid_citations=val_cites,
        hallucinated_citations=halluc_cites,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        status=status,
        transcript_path=None,
    )


# --- Trajectory Constraint Grader Tests ---


def test_grade_constraint_search_called():
    answerable_task = TaskItem(
        id="task-1",
        question="How to configure retention?",
        type="single_hop",
        answerable=True,
    )
    unanswerable_task = TaskItem(
        id="task-2",
        question="Does PostHog run on FreeBSD bare metal?",
        type="unanswerable",
        answerable=False,
    )

    # 1. Answerable with search call -> True
    res_with_search = create_mock_agent_result(
        tool_call_history=[{"tool_name": "search_docs", "tool_input": {"query": "retention"}}]
    )
    passed, reason = grade_constraint_search_called(answerable_task, res_with_search)
    assert passed is True
    assert reason is None

    # 2. Answerable without search call -> False
    res_no_search = create_mock_agent_result(tool_call_history=[])
    passed, reason = grade_constraint_search_called(answerable_task, res_no_search)
    assert passed is False
    assert "No search_docs tool call made" in reason

    # 3. Unanswerable without search call -> True (search is optional on unanswerable)
    passed, reason = grade_constraint_search_called(unanswerable_task, res_no_search)
    assert passed is True
    assert reason is None


def test_grade_constraint_multihop_depth():
    mh_task = TaskItem(
        id="task-mh",
        question="Combine feature flags with replay",
        type="multi_hop",
        answerable=True,
    )
    sh_task = TaskItem(
        id="task-sh",
        question="Single topic",
        type="single_hop",
        answerable=True,
    )

    # 1. Multi-hop with 2 distinct queries -> True
    res_2_distinct = create_mock_agent_result(
        tool_call_history=[
            {"tool_name": "search_docs", "tool_input": {"query": "feature flags rollout"}},
            {"tool_name": "search_docs", "tool_input": {"query": "session replay privacy"}},
        ]
    )
    passed, reason = grade_constraint_multihop_depth(mh_task, res_2_distinct)
    assert passed is True
    assert reason is None

    # 2. Multi-hop with only 1 query -> False
    res_1_search = create_mock_agent_result(
        tool_call_history=[
            {"tool_name": "search_docs", "tool_input": {"query": "feature flags rollout"}},
        ]
    )
    passed, reason = grade_constraint_multihop_depth(mh_task, res_1_search)
    assert passed is False
    assert "required >= 2 distinct searches" in reason

    # 3. Multi-hop with 2 identical queries -> False (set has length 1)
    res_identical = create_mock_agent_result(
        tool_call_history=[
            {"tool_name": "search_docs", "tool_input": {"query": "same query"}},
            {"tool_name": "search_docs", "tool_input": {"query": "  same query  "}},
        ]
    )
    passed, reason = grade_constraint_multihop_depth(mh_task, res_identical)
    assert passed is False
    assert "got 1" in reason

    # 4. Single-hop task with 1 query -> True (not multi-hop)
    passed, reason = grade_constraint_multihop_depth(sh_task, res_1_search)
    assert passed is True
    assert reason is None


def test_grade_constraint_step_budget():
    task = TaskItem(id="task-1", question="Q", type="single_hop", answerable=True)

    # Within budget (<= MAX_STEPS = 6) -> True
    res_ok = create_mock_agent_result(steps=3, total_tool_calls=3)
    passed, reason = grade_constraint_step_budget(task, res_ok)
    assert passed is True
    assert reason is None

    # Steps exceeded -> False
    res_steps_over = create_mock_agent_result(steps=MAX_STEPS + 1, total_tool_calls=2)
    passed, reason = grade_constraint_step_budget(task, res_steps_over)
    assert passed is False
    assert "Exceeded step budget" in reason

    # Tool calls exceeded -> False
    res_tools_over = create_mock_agent_result(steps=3, total_tool_calls=MAX_STEPS + 1)
    passed, reason = grade_constraint_step_budget(task, res_tools_over)
    assert passed is False
    assert "Exceeded step budget" in reason


def test_grade_constraint_no_duplicate_queries():
    task = TaskItem(id="task-1", question="Q", type="single_hop", answerable=True)

    # Clean history -> True
    res_clean = create_mock_agent_result(
        tool_call_history=[
            {"tool_name": "search_docs", "tool_input": {"query": "q1"}, "is_duplicate": False},
            {"tool_name": "search_docs", "tool_input": {"query": "q2"}, "is_duplicate": False},
        ]
    )
    passed, reason = grade_constraint_no_duplicate_queries(task, res_clean)
    assert passed is True
    assert reason is None

    # History containing duplicate flag -> False
    res_dup = create_mock_agent_result(
        tool_call_history=[
            {"tool_name": "search_docs", "tool_input": {"query": "q1"}, "is_duplicate": False},
            {"tool_name": "search_docs", "tool_input": {"query": "q1"}, "is_duplicate": True},
        ]
    )
    passed, reason = grade_constraint_no_duplicate_queries(task, res_dup)
    assert passed is False
    assert "Found 1 duplicate queries" in reason


def test_grade_constraint_no_fabricated_citations():
    answerable_task = TaskItem(id="task-1", question="Q", type="single_hop", answerable=True)
    unanswerable_task = TaskItem(id="task-2", question="Q", type="unanswerable", answerable=False)

    # 1. Answerable with 0 hallucinated citations -> True
    res_valid_ans = create_mock_agent_result(
        citations=["chunk-1"],
        valid_citations=["chunk-1"],
        hallucinated_citations=[],
    )
    passed, reason = grade_constraint_no_fabricated_citations(answerable_task, res_valid_ans)
    assert passed is True
    assert reason is None

    # 2. Answerable with hallucinated citation -> False
    res_halluc_ans = create_mock_agent_result(
        citations=["chunk-1", "fake-99"],
        valid_citations=["chunk-1"],
        hallucinated_citations=["fake-99"],
    )
    passed, reason = grade_constraint_no_fabricated_citations(answerable_task, res_halluc_ans)
    assert passed is False
    assert "Hallucinated citations detected" in reason

    # 3. Unanswerable with 0 citations -> True
    res_empty_unans = create_mock_agent_result(citations=[], valid_citations=[], hallucinated_citations=[])
    passed, reason = grade_constraint_no_fabricated_citations(unanswerable_task, res_empty_unans)
    assert passed is True
    assert reason is None

    # 4. Unanswerable with citations -> False
    res_cites_unans = create_mock_agent_result(citations=["chunk-1"], valid_citations=["chunk-1"])
    passed, reason = grade_constraint_no_fabricated_citations(unanswerable_task, res_cites_unans)
    assert passed is False
    assert "Fabricated 1 citations on unanswerable task" in reason


# --- Outcome Grader Tests ---


def test_grade_abstain_accuracy():
    answerable_task = TaskItem(id="task-1", question="Q", type="single_hop", answerable=True)
    unanswerable_task = TaskItem(id="task-2", question="Q", type="unanswerable", answerable=False)

    # 1. Answerable and abstain=False -> True
    res_answered = create_mock_agent_result(abstain=False)
    passed, reason = grade_abstain_accuracy(answerable_task, res_answered)
    assert passed is True
    assert reason is None

    # 2. Answerable and abstain=True -> False
    res_abstained = create_mock_agent_result(abstain=True)
    passed, reason = grade_abstain_accuracy(answerable_task, res_abstained)
    assert passed is False
    assert "Agent abstained on an answerable task" in reason

    # 3. Unanswerable and abstain=True -> True
    passed, reason = grade_abstain_accuracy(unanswerable_task, res_abstained)
    assert passed is True
    assert reason is None

    # 4. Unanswerable and abstain=False -> False
    passed, reason = grade_abstain_accuracy(unanswerable_task, res_answered)
    assert passed is False
    assert "Expected agent to abstain on unanswerable task" in reason


def test_grade_citation_accuracy():
    chunk_map = {
        "chunk-gold": "https://posthog.com/docs/data/retention",
        "chunk-other": "https://posthog.com/docs/other/topic",
    }
    answerable_task = TaskItem(
        id="task-1",
        question="Q",
        type="single_hop",
        answerable=True,
        gold_urls=["https://posthog.com/docs/data/retention"],
    )
    unanswerable_task = TaskItem(
        id="task-2",
        question="Q",
        type="unanswerable",
        answerable=False,
        gold_urls=[],
    )

    # 1. Answerable cites gold URL -> True
    res_gold = create_mock_agent_result(citations=["chunk-gold"])
    passed, reason = grade_citation_accuracy(answerable_task, res_gold, chunk_map)
    assert passed is True
    assert reason is None

    # 2. Answerable cites unrelated URL -> False
    res_other = create_mock_agent_result(citations=["chunk-other"])
    passed, reason = grade_citation_accuracy(answerable_task, res_other, chunk_map)
    assert passed is False
    assert "do not overlap" in reason

    # 3. Answerable has no citations -> False
    res_no_cite = create_mock_agent_result(citations=[])
    passed, reason = grade_citation_accuracy(answerable_task, res_no_cite, chunk_map)
    assert passed is False
    assert "No citations provided" in reason

    # 4. Unanswerable has no citations -> True
    passed, reason = grade_citation_accuracy(unanswerable_task, res_no_cite, chunk_map)
    assert passed is True
    assert reason is None

    # 5. Unanswerable has citations -> False
    passed, reason = grade_citation_accuracy(unanswerable_task, res_gold, chunk_map)
    assert passed is False
    assert "Unanswerable task must not cite any URLs" in reason


def test_grade_semantic_correctness():
    answerable_task = TaskItem(
        id="task-1",
        question="Q",
        type="single_hop",
        answerable=True,
        expected_facts=["7 years", "retention policy"],
    )
    unanswerable_task = TaskItem(
        id="task-2",
        question="Q",
        type="unanswerable",
        answerable=False,
    )

    # 1. Answerable with expected facts -> True
    res_facts = create_mock_agent_result(answer="PostHog stores events for 7 years under default retention policy.")
    passed, reason = grade_semantic_correctness(answerable_task, res_facts)
    assert passed is True
    assert reason is None

    # 2. Answerable missing facts -> False
    res_wrong = create_mock_agent_result(answer="PostHog stores events indefinitely.")
    passed, reason = grade_semantic_correctness(answerable_task, res_wrong)
    assert passed is False
    assert "Matched 0/2 expected facts" in reason

    # 3. Answerable but abstained -> False
    res_abstained = create_mock_agent_result(answer="I do not know.", abstain=True)
    passed, reason = grade_semantic_correctness(answerable_task, res_abstained)
    assert passed is False
    assert "Agent abstained on answerable task" in reason

    # 4. Unanswerable with abstain=True -> True
    passed, reason = grade_semantic_correctness(unanswerable_task, res_abstained)
    assert passed is True
    assert reason is None

    # 5. Unanswerable with cue phrase in answer -> True
    res_cue = create_mock_agent_result(answer="PostHog does not provide native support for FreeBSD bare metal.", abstain=False)
    passed, reason = grade_semantic_correctness(unanswerable_task, res_cue)
    assert passed is True
    assert reason is None

    # 6. Unanswerable with affirmative answer without cue -> False
    res_aff = create_mock_agent_result(answer="Install FreeBSD pkg and run posthog start.", abstain=False)
    passed, reason = grade_semantic_correctness(unanswerable_task, res_aff)
    assert passed is False
    assert "Unanswerable answer did not express lack of support" in reason


def test_grade_trial_aggregation():
    task = TaskItem(
        id="task-1",
        question="Q",
        type="single_hop",
        answerable=True,
        gold_urls=["https://posthog.com/docs/topic"],
        expected_facts=["fact A"],
    )
    chunk_map = {"c1": "https://posthog.com/docs/topic"}

    # Complete success
    res_success = create_mock_agent_result(
        answer="Here is fact A.",
        citations=["c1"],
        valid_citations=["c1"],
        tool_call_history=[{"tool_name": "search_docs", "tool_input": {"query": "topic"}}],
    )
    grade = grade_trial(task, res_success, trial_num=1, chunk_url_map=chunk_map)
    assert grade.success is True
    assert grade.constraints_passed is True
    assert grade.outcomes_passed is True
    assert len(grade.failure_reasons) == 0

    # Constraint failure (no search called)
    res_no_search = create_mock_agent_result(
        answer="Here is fact A.",
        citations=["c1"],
        valid_citations=["c1"],
        tool_call_history=[],
    )
    grade_fail_c = grade_trial(task, res_no_search, trial_num=2, chunk_url_map=chunk_map)
    assert grade_fail_c.success is False
    assert grade_fail_c.constraints_passed is False
    assert any("search_docs" in r for r in grade_fail_c.failure_reasons)

    # Outcome failure (missing expected fact)
    res_no_fact = create_mock_agent_result(
        answer="Here is wrong content.",
        citations=["c1"],
        valid_citations=["c1"],
        tool_call_history=[{"tool_name": "search_docs", "tool_input": {"query": "topic"}}],
    )
    grade_fail_o = grade_trial(task, res_no_fact, trial_num=3, chunk_url_map=chunk_map)
    assert grade_fail_o.success is False
    assert grade_fail_o.outcomes_passed is False
    assert any("expected facts" in r for r in grade_fail_o.failure_reasons)


# --- Task Loading and Offline Multi-Trial Benchmark Tests ---


def test_load_tasks_verifies_schema():
    assert DEFAULT_TASKS_PATH.exists()
    tasks = load_tasks(DEFAULT_TASKS_PATH)
    assert len(tasks) == 12

    types = [t.type for t in tasks]
    assert types.count("single_hop") == 8
    assert types.count("multi_hop") == 2
    assert types.count("unanswerable") == 2

    for t in tasks:
        assert isinstance(t.id, str) and len(t.id) > 0
        assert isinstance(t.question, str) and len(t.question) > 0
        if t.answerable:
            assert len(t.gold_urls) > 0
            assert len(t.expected_facts) > 0
        else:
            assert len(t.gold_urls) == 0


@pytest.mark.anyio
async def test_evaluate_suite_offline_mock():
    """Verify suite aggregation, pass@1, pass@k, and pass^k math using deterministic mock runner."""
    chunk_map = {
        "c_sh": "https://posthog.com/docs/sh",
        "c_mh": "https://posthog.com/docs/mh",
    }

    tasks = [
        TaskItem(
            id="t-pass-always",
            question="Always passes",
            type="single_hop",
            answerable=True,
            gold_urls=["https://posthog.com/docs/sh"],
            expected_facts=["guaranteed"],
        ),
        TaskItem(
            id="t-pass-partial",
            question="Passes 2 out of 3",
            type="multi_hop",
            answerable=True,
            gold_urls=["https://posthog.com/docs/mh"],
            expected_facts=["multi"],
        ),
        TaskItem(
            id="t-fail-always",
            question="Always fails",
            type="unanswerable",
            answerable=False,
            gold_urls=[],
            expected_facts=[],
        ),
    ]

    call_counters: dict[str, int] = {}

    def mock_runner(task: str, task_id: str) -> AgentResult:
        call_counters[task_id] = call_counters.get(task_id, 0) + 1
        trial_num = int(task_id.split("_t")[-1])

        if "Always passes" in task:
            return create_mock_agent_result(
                answer="This is guaranteed factual.",
                citations=["c_sh"],
                valid_citations=["c_sh"],
                tool_call_history=[{"tool_name": "search_docs", "tool_input": {"query": "sh query"}}],
            )
        elif "Passes 2 out of 3" in task:
            if trial_num in (1, 2):
                # Pass
                return create_mock_agent_result(
                    answer="This is multi hop information.",
                    citations=["c_mh"],
                    valid_citations=["c_mh"],
                    tool_call_history=[
                        {"tool_name": "search_docs", "tool_input": {"query": "query 1"}},
                        {"tool_name": "search_docs", "tool_input": {"query": "query 2"}},
                    ],
                )
            else:
                # Fail (only 1 query for multi-hop)
                return create_mock_agent_result(
                    answer="This is multi hop information.",
                    citations=["c_mh"],
                    valid_citations=["c_mh"],
                    tool_call_history=[
                        {"tool_name": "search_docs", "tool_input": {"query": "single query"}},
                    ],
                )
        else:
            # Unanswerable that fails (answers affirmatively and invents a citation)
            return create_mock_agent_result(
                answer="Supported on FreeBSD!",
                citations=["fake-cite"],
                hallucinated_citations=["fake-cite"],
                abstain=False,
            )

    test_transcripts = TEST_TMP_DIR / "transcripts"

    # Patch get_chunk_url_map inside eval_agent to return our test mapping
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("eval.eval_agent.get_chunk_url_map", lambda: chunk_map)
        suite_res = await evaluate_suite(
            tasks=tasks,
            model="mock-claude",
            trials_per_task=3,
            transcripts_dir=test_transcripts,
            agent_runner=mock_runner,
        )

    assert suite_res.total_tasks == 3
    assert suite_res.trials_per_task == 3
    assert suite_res.total_trials == 9

    # Task 1: 3/3 successes (pass_at_k=True, pass_all=True)
    # Task 2: 2/3 successes (pass_at_k=True, pass_all=False)
    # Task 3: 0/3 successes (pass_at_k=False, pass_all=False)
    # Total successes = 3 + 2 + 0 = 5 / 9 trials = 0.5556
    # pass@3 = 2 / 3 tasks = 0.6667
    # pass^3 = 1 / 3 tasks = 0.3333
    assert round(suite_res.pass_at_1, 4) == round(5 / 9, 4)
    assert round(suite_res.pass_at_k, 4) == round(2 / 3, 4)
    assert round(suite_res.pass_all, 4) == round(1 / 3, 4)

    # Test markdown report generation
    report_md = generate_markdown_report(suite_res)
    assert "# Agent Evaluation Benchmark Report (Milestone M10)" in report_md
    assert "Pass@1" in report_md
    assert "pass@3" in report_md
    assert "pass^3" in report_md
    assert "Trajectory Constraint Graders" in report_md
    assert "Analysis: Why Pass@3 and Pass^3 Differ" in report_md
    assert "t-pass-always" in report_md
    assert "t-pass-partial" in report_md
    assert "t-fail-always" in report_md
