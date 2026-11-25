"""Unit and integration tests for the agent trajectory failure analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import pytest

from eval.eval_agent import TaskItem
from eval.analyze_agent_failures import (
    AGENT_AUDIT_ANNOTATIONS,
    DEFAULT_AGENT_TRIALS_PATH,
    DEFAULT_HISTORICAL_REPORT_PATH,
    DEFAULT_OUTPUT_REPORT,
    DEFAULT_TASKS_PATH,
    DEFAULT_TRANSCRIPTS_DIR,
    PRIMARY_FAILURE_CATEGORIES,
    REACT_TAXONOMY_CATEGORIES,
    AgentFailureAnalysisReport,
    AnalyzedAgentRun,
    HistoricalReportComparison,
    RootCauseSummary,
    TaskEvalSummary,
    TrajectoryEfficiencyStats,
    TranscriptAuditAnnotation,
    _build_historical_comparison,
    _compute_efficiency_stats,
    _dedupe_failure_reasons,
    _fmt_pct,
    _grade_eval_records,
    _safe_rate,
    _summarize_standalone_trials,
    analyze_agent_failures,
    detect_trajectory_signals,
    load_eval_transcripts,
    load_transcript_records,
    main,
    parse_historical_agent_eval_report,
    parse_task_and_trial,
    render_markdown_report,
    sanitize_text,
    write_agent_failure_report,
)


@pytest.fixture(scope="module")
def full_agent_report() -> AgentFailureAnalysisReport:
    """Compute the deterministic agent failure analysis report once for integration tests."""
    return analyze_agent_failures(
        tasks_path=DEFAULT_TASKS_PATH,
        transcripts_dir=DEFAULT_TRANSCRIPTS_DIR,
        agent_trials_path=DEFAULT_AGENT_TRIALS_PATH,
        historical_report_path=DEFAULT_HISTORICAL_REPORT_PATH,
    )


class TestAgentTaxonomyAndAnnotations:
    """Validate taxonomy definitions and audit annotation integrity."""

    def test_taxonomy_definitions(self) -> None:
        assert PRIMARY_FAILURE_CATEGORIES == (
            "bad query",
            "stopped early",
            "loop or duplicate queries",
            "fabricated citation",
            "grader bug",
            "over-helpfulness / failed abstain",
        )
        assert REACT_TAXONOMY_CATEGORIES == (
            "duplicate queries",
            "premature stop",
            "excessive tool calls",
            "context rot",
            "failed abstention / over-helpfulness",
            "grader bug",
        )

    def test_annotations_minimum_count_and_uniqueness(self) -> None:
        assert len(AGENT_AUDIT_ANNOTATIONS) >= 10
        run_ids = [ann.run_id for ann in AGENT_AUDIT_ANNOTATIONS]
        assert len(run_ids) == len(set(run_ids)), "Duplicate run_id entries in AGENT_AUDIT_ANNOTATIONS"
        for ann in AGENT_AUDIT_ANNOTATIONS:
            assert ann.primary_category in PRIMARY_FAILURE_CATEGORIES
            assert ann.react_category in REACT_TAXONOMY_CATEGORIES
            assert len(ann.trajectory_evidence.strip()) > 20
            assert len(ann.root_cause_summary.strip()) > 20


class TestAgentAnalysisHelpers:
    """Unit tests for transcript loading, edge cases, and dynamic report rendering."""

    def test_load_transcript_records_handles_pretty_and_jsonl(self, tmp_path: Path) -> None:
        mixed_file = tmp_path / "mixed.jsonl"
        rec_a = {"task_id": "a", "steps": 2}
        rec_b = {"task_id": "b", "steps": 3}
        mixed_file.write_text(
            json.dumps(rec_a, indent=2) + "\n" + json.dumps(rec_b) + "\n",
            encoding="utf-8",
        )
        loaded = load_transcript_records(mixed_file)
        assert [r["task_id"] for r in loaded] == ["a", "b"]

        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("   \n", encoding="utf-8")
        assert load_transcript_records(empty_file) == []

    def test_missing_paths_raise_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_transcript_records(tmp_path / "missing.jsonl")
        with pytest.raises(FileNotFoundError):
            load_eval_transcripts(tmp_path / "missing_dir")
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            load_eval_transcripts(empty_dir)

    def test_zero_division_and_empty_stats_edge_cases(self) -> None:
        assert _safe_rate(0, 0) == 0.0
        assert _safe_rate(3, 0) == 0.0
        assert _fmt_pct(0, 0) == "0.0%"

        empty_eff = _compute_efficiency_stats([])
        assert empty_eff.total_trials == 0
        assert empty_eff.mean_tool_calls == 0.0
        assert empty_eff.mean_tokens_in == 0.0
        assert empty_eff.max_tokens_in_trial == "none"

        empty_sa = _summarize_standalone_trials([])
        assert empty_sa["total_runs"] == 0
        assert empty_sa["max_tool_calls"] == 0

    def test_detect_trajectory_signals_duplicate_detection(self) -> None:
        rec = {
            "tool_call_history": [
                {"tool_name": "search_docs", "tool_input": {"query": "Feature Flags"}, "is_duplicate": False},
                {"tool_name": "search_docs", "tool_input": {"query": "  feature flags "}, "is_duplicate": True},
            ],
            "all_retrieved_chunk_ids": ["a#0", "b#1"],
            "messages": [{"role": "user", "content": "hi"}],
        }
        sig = detect_trajectory_signals(rec)
        assert sig["duplicate_query_count"] == 1
        assert sig["retrieved_chunk_count"] == 2
        assert sig["message_count"] == 1

    def test_parse_task_and_trial_and_sanitize_text(self) -> None:
        assert parse_task_and_trial("task-10_t1") == ("task-10", 1)
        with pytest.raises(ValueError, match="Invalid trial identifier"):
            parse_task_and_trial("bad_format")
        assert sanitize_text("a\u2014b\u2013c") == "a-b-c"
        assert _dedupe_failure_reasons(
            [
                "Outcome failed: Agent abstained on an answerable task",
                "Outcome failed: No citations provided for answerable task",
                "Outcome failed: Agent abstained on answerable task",
            ]
        ) == [
            "Outcome failed: Agent abstained on an answerable task",
            "Outcome failed: No citations provided for answerable task",
        ]

    def test_annotation_validation_errors(self) -> None:
        dup_ann = AGENT_AUDIT_ANNOTATIONS[0]
        with pytest.raises(ValueError, match="Duplicate audit annotation"):
            analyze_agent_failures(
                chunk_url_map={},
                annotations=AGENT_AUDIT_ANNOTATIONS + (dup_ann,),
            )

        missing_subset = tuple(a for a in AGENT_AUDIT_ANNOTATIONS if a.run_id != "task-03_t2")
        with pytest.raises(ValueError, match="Missing audit annotations for failing eval trials"):
            analyze_agent_failures(
                chunk_url_map={},
                annotations=missing_subset,
            )

        unknown_eval = TranscriptAuditAnnotation(
            run_id="task-99_t1",
            source="eval",
            primary_category="bad query",
            react_category="excessive tool calls",
            trajectory_evidence="Synthetic missing eval transcript record check.",
            root_cause_summary="Synthetic missing eval transcript root cause.",
        )
        with pytest.raises(KeyError, match="not found"):
            analyze_agent_failures(
                chunk_url_map={},
                annotations=AGENT_AUDIT_ANNOTATIONS + (unknown_eval,),
            )

        unknown_standalone = TranscriptAuditAnnotation(
            run_id="missing_standalone_id",
            source="standalone",
            primary_category="bad query",
            react_category="excessive tool calls",
            trajectory_evidence="Synthetic missing standalone transcript record check.",
            root_cause_summary="Synthetic missing standalone transcript root cause.",
        )
        with pytest.raises(KeyError, match="not found"):
            analyze_agent_failures(
                chunk_url_map={},
                annotations=AGENT_AUDIT_ANNOTATIONS + (unknown_standalone,),
            )

        dummy_task = TaskItem(
            id="task-01",
            question="Q?",
            type="single_hop",
            answerable=True,
            gold_urls=[],
            expected_facts=[],
        )
        with pytest.raises(ValueError, match="Duplicate eval transcript"):
            _grade_eval_records(
                eval_records=[{"task_id": "task-01_t1"}, {"task_id": "task-01_t1"}],
                tasks_by_id={"task-01": dummy_task},
                resolved_url_map={},
            )

    def test_partial_historical_report_handles_missing_metrics(self, tmp_path: Path) -> None:
        partial_hist = tmp_path / "partial_eval.md"
        partial_hist.write_text("# Empty or partial report without metric lines\n", encoding="utf-8")
        comp = _build_historical_comparison(
            historical_report_path=partial_hist,
            task_summaries=[],
            pass_at_1=0.75,
            pass_at_3=1.0,
            pass_all=0.5,
            semantic_correctness=0.8,
        )
        assert comp.historical_present is True
        assert comp.historical_pass_at_1 is None
        assert "Pass@1=n/a" in comp.divergence_summary

    def test_render_markdown_report_is_fully_dynamic(self) -> None:
        synthetic_run = AnalyzedAgentRun(
            run_id="task-99_t1",
            source="eval",
            task_id="task-99",
            trial_num=1,
            question="Synthetic question?",
            task_type="single_hop",
            answerable=True,
            status="abstain",
            steps=2,
            total_tool_calls=1,
            duplicate_query_count=0,
            tokens_in=400,
            tokens_out=50,
            latency_ms=100.0,
            queries=["synthetic query"],
            retrieved_chunk_count=2,
            valid_citations=[],
            hallucinated_citations=[],
            abstain=True,
            re_eval_success=False,
            failure_reasons=[
                "Outcome failed: Agent abstained on an answerable task",
                "Outcome failed: Agent abstained on answerable task",
            ],
            primary_category="stopped early",
            react_category="premature stop",
            secondary_signals=[],
            trajectory_evidence="Stopped after 1 search call.",
            root_cause_summary="Premature abstention on synthetic test.",
        )
        synthetic_report = AgentFailureAnalysisReport(
            tasks_path="eval/agent_tasks.json",
            transcripts_dir="runs/transcripts/agent_eval",
            agent_trials_path="runs/transcripts/agent_trials.jsonl",
            historical_report_path="reports/agent_eval.md",
            model="synthetic-model",
            total_tasks=2,
            trials_per_task=1,
            total_trials=2,
            passing_trials=1,
            failing_trials=1,
            pass_at_1=0.5,
            pass_at_3=0.5,
            pass_all=0.5,
            type_breakdown={
                "single_hop": {"count": 2.0, "trials": 2.0, "successes": 1.0, "pass_at_1": 0.5, "pass_at_k": 0.5, "pass_all": 0.5}
            },
            constraint_pass_rates={
                "search_called": 1.0,
                "multihop_depth": 1.0,
                "step_budget": 1.0,
                "no_duplicate_queries": 1.0,
                "no_fabricated_citations": 1.0,
            },
            outcome_pass_rates={
                "abstain_accuracy": 0.5,
                "citation_accuracy": 0.5,
                "semantic_correctness": 0.5,
            },
            task_summaries=[
                TaskEvalSummary(
                    task_id="task-99",
                    task_type="single_hop",
                    answerable=True,
                    trials_count=1,
                    success_count=0,
                    pass_at_k=False,
                    pass_all=False,
                    failure_reasons=["Abstained"],
                ),
                TaskEvalSummary(
                    task_id="task-98",
                    task_type="single_hop",
                    answerable=True,
                    trials_count=1,
                    success_count=1,
                    pass_at_k=True,
                    pass_all=True,
                    failure_reasons=[],
                ),
            ],
            efficiency_stats=TrajectoryEfficiencyStats(
                total_trials=2,
                total_tool_calls=2,
                mean_tool_calls=1.0,
                duplicate_queries_total=0,
                step_counts={2: 2},
                tool_call_counts={1: 2},
                mean_tokens_in=400.0,
                max_tokens_in=400,
                max_tokens_in_trial="task-99_t1",
                mean_tokens_in_1_to_2_calls=400.0,
                mean_tokens_in_3_plus_calls=0.0,
                total_in_loop_duration_sec=0.2,
            ),
            standalone_summary={"total_runs": 0, "duplicate_queries": 0, "max_tool_calls": 0, "max_tokens_in": 0, "run_ids": []},
            historical_comparison=HistoricalReportComparison(
                historical_present=False,
                current_pass_at_1=0.5,
                current_pass_at_3=0.5,
                current_pass_all=0.5,
                current_semantic_correctness=0.5,
                divergence_summary="No historical report.",
            ),
            current_failing_ids=["task-99_t1"],
            historical_failing_ids=["task-99_t1"],
            current_primary_counts={c: (1 if c == "stopped early" else 0) for c in PRIMARY_FAILURE_CATEGORIES},
            current_react_counts={c: (1 if c == "premature stop" else 0) for c in REACT_TAXONOMY_CATEGORIES},
            historical_primary_counts={c: (1 if c == "stopped early" else 0) for c in PRIMARY_FAILURE_CATEGORIES},
            historical_react_counts={c: (1 if c == "premature stop" else 0) for c in REACT_TAXONOMY_CATEGORIES},
            cohort_primary_counts={c: (1 if c == "stopped early" else 0) for c in PRIMARY_FAILURE_CATEGORIES},
            cohort_react_counts={c: (1 if c == "premature stop" else 0) for c in REACT_TAXONOMY_CATEGORIES},
            top_root_causes=[
                RootCauseSummary(
                    rank=1,
                    title="Synthetic Premature Stop",
                    current_failure_count=1,
                    current_failure_total=1,
                    cohort_count=1,
                    cohort_total=1,
                    affected_runs=["task-99_t1"],
                    mechanism="Synthetic mechanism.",
                )
            ],
            analyzed_runs=[synthetic_run],
        )
        rendered = render_markdown_report(synthetic_report)
        assert "1/2` passing trials (**Pass@1 = 50.0%**" in rendered
        assert "Current Failures (n=1)" in rendered
        assert "Inspected Cohort (n=1)" in rendered
        assert "No standalone runs recorded." in rendered
        assert rendered.count("Outcome failed: Agent abstained on an answerable task") == 1
        assert "80.6%" not in rendered
        assert "29/36" not in rendered
        assert "2c815ecfbe79" not in rendered
        assert "task-11_t1" not in rendered
        assert "task-10_t1" not in rendered
        assert "15,185" not in rendered
        assert "3/10" not in rendered
        assert "5/7" not in rendered


class TestIntegrationAgentFailureAnalysis:
    """Integration tests against real saved transcripts, task oracle, and historical report."""

    def test_reevaluation_metrics_and_historical_divergence(
        self, full_agent_report: AgentFailureAnalysisReport
    ) -> None:
        assert full_agent_report.total_tasks == 12
        assert full_agent_report.trials_per_task == 3
        assert full_agent_report.total_trials == 36
        assert full_agent_report.passing_trials == 29
        assert full_agent_report.failing_trials == 7
        assert full_agent_report.pass_at_1 == pytest.approx(0.8056, abs=1e-4)
        assert full_agent_report.pass_at_3 == pytest.approx(0.9167, abs=1e-4)
        assert full_agent_report.pass_all == pytest.approx(0.6667, abs=1e-4)

        assert full_agent_report.current_failing_ids == [
            "task-03_t2",
            "task-10_t1",
            "task-11_t1",
            "task-11_t3",
            "task-12_t1",
            "task-12_t2",
            "task-12_t3",
        ]
        assert len(full_agent_report.historical_failing_ids) == 10

        hc = full_agent_report.historical_comparison
        assert hc.historical_present is True
        assert hc.historical_pass_at_1 == pytest.approx(0.722, abs=1e-3)
        assert hc.historical_pass_at_3 == pytest.approx(0.833, abs=1e-3)
        assert hc.historical_pass_all == pytest.approx(0.583, abs=1e-3)
        assert hc.historical_task05_successes == "0/3"
        assert hc.current_task05_successes == "3/3"
        assert hc.historical_narrative_mismatch is not None

        hist_parsed = parse_historical_agent_eval_report(DEFAULT_HISTORICAL_REPORT_PATH)
        assert hist_parsed is not None
        assert hist_parsed["semantic_correctness"] == pytest.approx(0.75, abs=1e-3)

    def test_taxonomy_counts_and_top_root_causes(
        self, full_agent_report: AgentFailureAnalysisReport
    ) -> None:
        assert len(full_agent_report.analyzed_runs) >= 10
        assert len(full_agent_report.analyzed_runs) == 14

        assert sum(full_agent_report.current_primary_counts.values()) == 7
        assert sum(full_agent_report.current_react_counts.values()) == 7
        assert sum(full_agent_report.historical_primary_counts.values()) == 10
        assert sum(full_agent_report.historical_react_counts.values()) == 10
        assert sum(full_agent_report.cohort_primary_counts.values()) == 14
        assert sum(full_agent_report.cohort_react_counts.values()) == 14

        assert full_agent_report.current_primary_counts["over-helpfulness / failed abstain"] == 5
        assert full_agent_report.current_primary_counts["stopped early"] == 1
        assert full_agent_report.current_primary_counts["fabricated citation"] == 1
        assert full_agent_report.historical_primary_counts["grader bug"] == 3
        assert full_agent_report.cohort_react_counts["excessive tool calls"] == 4
        assert full_agent_report.efficiency_stats.duplicate_queries_total == 0
        assert full_agent_report.efficiency_stats.mean_tokens_in == pytest.approx(8987.0, abs=0.1)
        assert full_agent_report.efficiency_stats.mean_tokens_in_1_to_2_calls == pytest.approx(7185.3, abs=0.1)
        assert full_agent_report.efficiency_stats.mean_tokens_in_3_plus_calls == pytest.approx(16451.0, abs=0.1)
        assert full_agent_report.efficiency_stats.max_tokens_in == 20044
        assert full_agent_report.efficiency_stats.max_tokens_in_trial == "task-08_t1"
        assert full_agent_report.standalone_summary["total_runs"] == 3
        assert full_agent_report.standalone_summary["duplicate_queries"] == 0
        assert full_agent_report.standalone_summary["max_tokens_in"] == 18231
        assert full_agent_report.standalone_summary["max_tokens_in_run_id"] == "2c815ecfbe79"

        assert [rc.rank for rc in full_agent_report.top_root_causes] == [1, 2]
        assert full_agent_report.top_root_causes[0].current_failure_count == 5

    def test_report_markdown_rendering_and_constraints(
        self, full_agent_report: AgentFailureAnalysisReport, tmp_path: Path
    ) -> None:
        out_path = write_agent_failure_report(
            full_agent_report, output_path=tmp_path / "agent_failure_analysis.md"
        )
        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")

        assert "\u2014" not in content
        assert "\u2013" not in content

        assert "## 1. Executive Summary" in content
        assert "## 2. Failure Category Counts (5-Category + Abstention Taxonomy)" in content
        assert "## 3. ReAct Loop Failure Taxonomy" in content
        assert "## 4. Top 2 Root Causes" in content
        assert "## 5. Detailed Per-Run Breakdown (n=14 Inspected Runs)" in content
        assert "## 6. Prioritized Remediation Hypotheses for Agent Release Gates (Top 3)" in content

        run_headings = re.findall(r"^#### `\[[^\]]+\] [^`]+`", content, flags=re.MULTILINE)
        assert len(run_headings) == 14

        for cat in PRIMARY_FAILURE_CATEGORIES:
            assert f"`{cat}`" in content
        for cat in REACT_TAXONOMY_CATEGORIES:
            assert f"`{cat}`" in content

        root_causes = re.findall(r"^### Root Cause \d+:", content, flags=re.MULTILINE)
        assert root_causes == ["### Root Cause 1:", "### Root Cause 2:"]

        # Public-readiness wording, expanded trial IDs, standalone source labeling, and deduplicated reasons
        assert "Stale Report Divergence" not in content
        assert "un-regenerated" not in content
        assert "release-gate dashboards" not in content
        assert "Historical Report Baseline:" in content
        assert "ensures automated CI/eval summaries stay synchronized with the committed task suite" in content
        assert "task-03_t1/task-03_t3" in content
        assert "task-12_t1/task-12_t2" in content
        assert "unlike t1/t3" not in content
        assert "as t1/t2" not in content
        assert "standalone trial `2c815ecfbe79` (from `runs/transcripts/agent_trials.jsonl`)" in content
        for line in content.splitlines():
            if line.startswith("- **Grader Failure Reasons:**"):
                reasons = [r.strip() for r in line.split(":** ", 1)[1].split(";")]
                assert len(reasons) == len(set(reasons))
                assert line.count("Agent abstained on") <= 1

    def test_committed_report_file_is_up_to_date(
        self, full_agent_report: AgentFailureAnalysisReport
    ) -> None:
        assert DEFAULT_OUTPUT_REPORT.exists(), "reports/agent_failure_analysis.md must exist"
        on_disk = DEFAULT_OUTPUT_REPORT.read_text(encoding="utf-8")
        expected = render_markdown_report(full_agent_report)
        if not expected.endswith("\n"):
            expected += "\n"
        assert on_disk == expected

    def test_cli_main_writes_report(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        out_path = tmp_path / "cli_agent_report.md"
        monkeypatch.setattr(
            sys,
            "argv",
            ["analyze_agent_failures", "--output", str(out_path)],
        )
        main()
        assert out_path.exists()
        assert out_path.read_text(encoding="utf-8") == DEFAULT_OUTPUT_REPORT.read_text(encoding="utf-8")
