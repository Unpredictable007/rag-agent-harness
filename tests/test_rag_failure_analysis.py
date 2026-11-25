"""Unit and integration tests for the RAG failure analysis pipeline and diagnostic report."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import re
import sys

import pytest

from eval.analyze_rag_failures import (
    CASCADE_LEVELS,
    DEFAULT_ABLATION_PATH,
    DEFAULT_GOLDEN_PATH,
    DEFAULT_OUTPUT_REPORT,
    DEFAULT_TRIALS_PATH,
    FAILURE_AUDIT_ANNOTATIONS,
    FAILURE_CATEGORIES,
    AnalyzedFailureCase,
    CorpusIngestionStats,
    FailureAnalysisReport,
    FailureAuditAnnotation,
    _truncate_text,
    analyze_failures,
    compute_corpus_ingestion_stats,
    identify_degraded_records,
    is_record_degraded,
    main,
    render_markdown_report,
    sanitize_text,
    summarize_dev_trials,
    write_failure_analysis_report,
)
from src.ingest import ChunkLookup, DocumentChunk


@pytest.fixture(scope="module")
def full_failure_report() -> FailureAnalysisReport:
    """Compute the full failure analysis report once for module-level integration checks."""
    return analyze_failures(
        ablation_path=DEFAULT_ABLATION_PATH,
        golden_path=DEFAULT_GOLDEN_PATH,
        trials_path=DEFAULT_TRIALS_PATH,
        include_corpus_stats=True,
    )


class TestFailureTaxonomyAndAnnotations:
    """Validate taxonomy definitions and audit annotation integrity."""

    def test_taxonomy_and_cascade_definitions(self) -> None:
        assert FAILURE_CATEGORIES == (
            "retrieval miss",
            "ranking miss",
            "chunking cut",
            "model ignored context",
            "judge error",
            "wrong golden label",
        )
        assert CASCADE_LEVELS == (
            "Ingestion",
            "Retrieval",
            "Rerank",
            "Attention",
            "Generation",
        )

    def test_audit_annotations_minimum_count_and_uniqueness(self) -> None:
        assert len(FAILURE_AUDIT_ANNOTATIONS) >= 15
        keys = [(ann.config_name, ann.item_id) for ann in FAILURE_AUDIT_ANNOTATIONS]
        assert len(keys) == len(set(keys)), "Duplicate (config_name, item_id) audit entries"

        for ann in FAILURE_AUDIT_ANNOTATIONS:
            assert ann.category in FAILURE_CATEGORIES
            assert ann.cascade_level in CASCADE_LEVELS
            assert len(ann.stage1_evidence.strip()) > 20
            assert len(ann.chunk_evidence.strip()) > 20
            assert len(ann.root_cause_summary.strip()) > 20


class TestAnalysisHelpers:
    """Unit tests for deterministic record filtering, sanitization, and trial summaries."""

    def test_identify_degraded_records_rules(self) -> None:
        synthetic = [
            {
                "item_id": "ok-ans",
                "answerable": True,
                "correctness_score": 1.0,
                "faithfulness_score": 1.0,
                "recall_at_5": 1.0,
                "abstain": False,
            },
            {
                "item_id": "ok-unans",
                "answerable": False,
                "correctness_score": 1.0,
                "faithfulness_score": 1.0,
                "recall_at_5": 0.0,
                "abstain": True,
            },
            {
                "item_id": "partial-corr",
                "answerable": True,
                "correctness_score": 0.5,
                "faithfulness_score": 1.0,
                "recall_at_5": 1.0,
                "abstain": False,
            },
            {
                "item_id": "partial-recall",
                "answerable": True,
                "correctness_score": 1.0,
                "faithfulness_score": 1.0,
                "recall_at_5": 0.5,
                "abstain": False,
            },
            {
                "item_id": "false-abstain",
                "answerable": True,
                "correctness_score": 0.0,
                "faithfulness_score": 1.0,
                "recall_at_5": 1.0,
                "abstain": True,
            },
            {
                "item_id": "unfaithful",
                "answerable": True,
                "correctness_score": 1.0,
                "faithfulness_score": 0.0,
                "recall_at_5": 1.0,
                "abstain": False,
            },
        ]
        degraded = identify_degraded_records(synthetic)
        assert [r["item_id"] for r in degraded] == [
            "partial-corr",
            "partial-recall",
            "false-abstain",
            "unfaithful",
        ]

    def test_sanitize_text_normalizes_non_ascii_dashes(self) -> None:
        raw = "Step 1\u2014check config\u2013and retry"
        cleaned = sanitize_text(raw)
        assert "\u2014" not in cleaned
        assert "\u2013" not in cleaned
        assert cleaned == "Step 1-check config-and retry"

    def test_truncate_text_snaps_to_word_boundary(self) -> None:
        assert _truncate_text("Short\u2014text   here.", limit=50) == "Short-text here."
        assert _truncate_text("hello world test", limit=14) == "hello world..."
        assert _truncate_text("hello world test", limit=13) == "hello..."
        assert _truncate_text("hello world test", limit=9) == "hello..."
        assert _truncate_text("abcdefghijkl", limit=8) == "abcde..."
        assert _truncate_text("abcdefghijkl", limit=3) == "abc"
        mid_word = _truncate_text("The candidate gives the same conclusion as expected.", limit=38)
        assert mid_word == "The candidate gives the same..."
        assert len(mid_word) <= 38

    def test_summarize_dev_trials(self) -> None:
        trials = [
            {"question_id": "q-a", "answerable": True, "abstain": True},
            {"question_id": "q-a", "answerable": True, "abstain": True},
            {"question_id": "q-b", "answerable": True, "abstain": False},
            {"question_id": "q-u", "answerable": False, "abstain": True},
        ]
        summary = summarize_dev_trials(trials)
        assert summary["total_trials"] == 4
        assert summary["answerable_trials"] == 3
        assert summary["false_abstain_trials"] == 2
        assert summary["false_abstain_items"] == ["q-a"]
        assert summary["abstain_by_item"]["q-a"] == (2, 2)

    def test_missing_input_files_raise_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            analyze_failures(
                ablation_path=tmp_path / "missing.json",
                golden_path=DEFAULT_GOLDEN_PATH,
                trials_path=DEFAULT_TRIALS_PATH,
                include_corpus_stats=False,
            )

    def test_incomplete_e3_annotations_raise_value_error(self) -> None:
        subset = tuple(ann for ann in FAILURE_AUDIT_ANNOTATIONS if ann.item_id != "gold-003")
        with pytest.raises(ValueError, match="Missing audit annotations for E3 degraded items"):
            analyze_failures(
                annotations=subset,
                include_corpus_stats=False,
            )

    def test_non_degraded_annotation_raises_value_error(self) -> None:
        passing_ann = FailureAuditAnnotation(
            config_name="E3",
            item_id="gold-001",
            category="model ignored context",
            cascade_level="Attention",
            stage1_evidence="Gold chunk ranked #1 with full recall and correctness.",
            chunk_evidence="No failure occurred on this passing record.",
            root_cause_summary="Synthetic invalid annotation for testing.",
        )
        with pytest.raises(ValueError, match="not degraded or failing"):
            analyze_failures(
                annotations=FAILURE_AUDIT_ANNOTATIONS + (passing_ann,),
                include_corpus_stats=False,
            )

    def test_is_record_degraded_edge_cases(self) -> None:
        assert is_record_degraded(
            {
                "answerable": False,
                "correctness_score": 1.0,
                "faithfulness_score": 1.0,
                "recall_at_5": 0.0,
                "abstain": True,
            }
        ) is False
        assert is_record_degraded(
            {
                "answerable": True,
                "correctness_score": 1.0,
                "faithfulness_score": 0.5,
                "recall_at_5": 1.0,
                "abstain": False,
            }
        ) is True
        assert is_record_degraded(
            {
                "answerable": True,
                "correctness_score": 1.0,
                "faithfulness_score": 1.0,
                "recall_at_5": 1.0,
                "abstain": True,
            }
        ) is True

    def test_duplicate_and_unknown_annotations_raise(self) -> None:
        dup_ann = FAILURE_AUDIT_ANNOTATIONS[0]
        with pytest.raises(ValueError, match="Duplicate audit annotation"):
            analyze_failures(
                annotations=FAILURE_AUDIT_ANNOTATIONS + (dup_ann,),
                include_corpus_stats=False,
            )

        unknown_ann = FailureAuditAnnotation(
            config_name="E3",
            item_id="gold-999",
            category="retrieval miss",
            cascade_level="Retrieval",
            stage1_evidence="Synthetic missing item evidence for validation.",
            chunk_evidence="Synthetic chunk evidence for validation.",
            root_cause_summary="Synthetic root cause for missing item check.",
        )
        with pytest.raises(KeyError, match="not found in ablation_results.json"):
            analyze_failures(
                annotations=FAILURE_AUDIT_ANNOTATIONS + (unknown_ann,),
                include_corpus_stats=False,
            )

    def test_compute_corpus_ingestion_stats_synthetic_and_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.pkl"
        assert compute_corpus_ingestion_stats(missing) is None

        chunks = [
            DocumentChunk(
                chunk_id="a.mdx#0",
                doc_path="a.mdx",
                url="https://posthog.com/docs/a",
                heading_path="Feature Flags > Rollout",
                title="Rollout",
                text="Feature flags support gradual rollout.",
                token_count=6,
            ),
            DocumentChunk(
                chunk_id="b.mdx#0",
                doc_path="b.mdx",
                url="https://posthog.com/docs/b",
                heading_path="Mintlify Connector > Available tables",
                title="Available tables",
                text='export const img = "https://example.com/a.png"\n<div>table body</div>',
                token_count=10,
            ),
        ]
        pkl_path = tmp_path / "bm25.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump((None, ChunkLookup(chunks)), f)

        stats = compute_corpus_ingestion_stats(pkl_path)
        assert stats == CorpusIngestionStats(
            total_chunks=2,
            chunks_missing_title_keywords=1,
            chunks_with_unstripped_exports=1,
            chunks_with_residual_html=1,
        )

    def test_render_markdown_report_is_fully_dynamic(self) -> None:
        sample_case = AnalyzedFailureCase(
            config_name="E3",
            item_id="gold-901",
            question="Synthetic question?",
            question_type="single-chunk",
            answerable=True,
            gold_urls=["https://posthog.com/docs/x"],
            retrieved_urls=[],
            retrieved_chunk_ids=[],
            hit_at_5=0.0,
            recall_at_5=0.0,
            mrr=0.0,
            correctness_score=0.0,
            faithfulness_score=1.0,
            abstain=True,
            generated_answer="Abstained",
            correctness_reasoning="False abstain.",
            faithfulness_reasoning="N/A",
            category="retrieval miss",
            cascade_level="Retrieval",
            stage1_evidence="Missing from top-50.",
            chunk_evidence="Missing keyword.",
            root_cause_summary="Synthetic retrieval miss.",
        )
        synthetic_report = FailureAnalysisReport(
            dataset_path="eval/golden_set.json",
            ablation_path="runs/ablation_results.json",
            trials_path="runs/transcripts/rag_dev_trials.jsonl",
            ablation_timestamp="2026-11-25T00:00:00+00:00",
            total_dev_queries=10,
            answerable_dev_queries=8,
            config_failure_counts={
                "E0": {"total_records": 10, "degraded_or_failing": 3, "correctness_lt_1": 3, "recall_lt_1": 2, "false_abstain": 1, "faithfulness_lt_1": 0},
                "E1": {"total_records": 10, "degraded_or_failing": 2, "correctness_lt_1": 2, "recall_lt_1": 1, "false_abstain": 0, "faithfulness_lt_1": 0},
                "E2": {"total_records": 10, "degraded_or_failing": 2, "correctness_lt_1": 2, "recall_lt_1": 1, "false_abstain": 1, "faithfulness_lt_1": 0},
                "E3": {"total_records": 10, "degraded_or_failing": 1, "correctness_lt_1": 1, "recall_lt_1": 1, "false_abstain": 1, "faithfulness_lt_1": 0},
            },
            dev_trials_summary={
                "total_trials": 12,
                "unique_questions": 4,
                "answerable_trials": 8,
                "false_abstain_trials": 2,
                "false_abstain_items": ["gold-901"],
                "false_abstain_item_rates": {"gold-901": "2/3"},
            },
            e3_category_counts={c: (1 if c == "retrieval miss" else 0) for c in FAILURE_CATEGORIES},
            e3_cascade_counts={l: (1 if l == "Retrieval" else 0) for l in CASCADE_LEVELS},
            cohort_category_counts={c: (1 if c == "retrieval miss" else 0) for c in FAILURE_CATEGORIES},
            cohort_cascade_counts={l: (1 if l == "Retrieval" else 0) for l in CASCADE_LEVELS},
            corpus_stats=CorpusIngestionStats(
                total_chunks=0,
                chunks_missing_title_keywords=0,
                chunks_with_unstripped_exports=0,
                chunks_with_residual_html=0,
            ),
            analyzed_cases=[sample_case],
        )
        rendered = render_markdown_report(synthetic_report)
        assert "Cross-Configuration Failure Footprint (`dev` split, n=10)" in rendered
        assert "E3 Baseline (n=1)" in rendered
        assert "Full Inspected Cohort (n=1)" in rendered
        assert "4.1 Production Baseline (`E3`) Degraded & Failing Cases (n=1)" in rendered
        assert "4.2 Hybrid RRF (`E2`) & Dev-Trial Failures (n=0)" in rendered
        assert "4.3 Cross-Ablation (`E0` / `E1`) Diagnostic Edge Cases (n=0)" in rendered
        assert "(`25.0%`)" in rendered
        assert "all 1 abstained in 2/3 trials" in rendered
        assert "n=45" not in rendered
        assert "n=22" not in rendered
        assert "11.1%" not in rendered
        assert "7/13" not in rendered
        assert "6/13" not in rendered
        assert "34.6%" not in rendered
        assert "4,895/14,148" not in rendered
        assert "0.5000" not in rendered
        assert "0.8519" not in rendered
        assert "0.8333" not in rendered
        assert "`N/A`" in rendered

        synthetic_with_metrics = synthetic_report.model_copy(
            update={
                "e3_correctness": 0.75,
                "e3_single_chunk_recall_at_5": 0.8,
                "e3_multi_chunk_recall_at_5": 0.4,
            }
        )
        rendered_metrics = render_markdown_report(synthetic_with_metrics)
        assert "0.4000" in rendered_metrics
        assert "0.8000" in rendered_metrics
        assert "0.7500" in rendered_metrics


class TestIntegrationFailureAnalysis:
    """Integration tests against real ablation, golden set, and dev trial artifacts."""

    def test_cross_config_counts_and_trials_consistency(
        self, full_failure_report: FailureAnalysisReport
    ) -> None:
        assert full_failure_report.total_dev_queries == 45
        assert full_failure_report.answerable_dev_queries == 36

        counts = full_failure_report.config_failure_counts
        assert counts["E0"]["degraded_or_failing"] == 18
        assert counts["E1"]["degraded_or_failing"] == 16
        assert counts["E2"]["degraded_or_failing"] == 17
        assert counts["E3"]["degraded_or_failing"] == 13
        assert counts["E0"]["faithfulness_lt_1"] == 1
        assert counts["E3"]["false_abstain"] == 2

        trials = full_failure_report.dev_trials_summary
        assert trials["total_trials"] == 135
        assert trials["answerable_trials"] == 108
        assert trials["false_abstain_trials"] == 12
        assert trials["false_abstain_items"] == ["gold-004", "gold-015", "gold-022", "gold-037"]

    def test_category_and_cascade_counts_sum_correctly(
        self, full_failure_report: FailureAnalysisReport
    ) -> None:
        e3_cases = [c for c in full_failure_report.analyzed_cases if c.config_name == "E3"]
        assert len(e3_cases) == 13
        assert len(full_failure_report.analyzed_cases) == 22

        assert set(full_failure_report.e3_category_counts.keys()) == set(FAILURE_CATEGORIES)
        assert set(full_failure_report.e3_cascade_counts.keys()) == set(CASCADE_LEVELS)
        assert set(full_failure_report.cohort_category_counts.keys()) == set(FAILURE_CATEGORIES)
        assert set(full_failure_report.cohort_cascade_counts.keys()) == set(CASCADE_LEVELS)

        assert sum(full_failure_report.e3_category_counts.values()) == 13
        assert sum(full_failure_report.e3_cascade_counts.values()) == 13
        assert sum(full_failure_report.cohort_category_counts.values()) == 22
        assert sum(full_failure_report.cohort_cascade_counts.values()) == 22

        # Every category and cascade level is represented in the full inspected cohort
        for cat in FAILURE_CATEGORIES:
            assert full_failure_report.cohort_category_counts[cat] >= 1
        for lvl in CASCADE_LEVELS:
            assert full_failure_report.cohort_cascade_counts[lvl] >= 1

    def test_corpus_stats_computed_from_disk(
        self, full_failure_report: FailureAnalysisReport
    ) -> None:
        cs = full_failure_report.corpus_stats
        assert cs is not None
        assert cs.total_chunks > 10000
        assert cs.chunks_missing_title_keywords > 0
        assert cs.chunks_with_unstripped_exports > 0
        assert cs.chunks_with_residual_html > 0

    def test_report_markdown_rendering_and_constraints(
        self, full_failure_report: FailureAnalysisReport, tmp_path: Path
    ) -> None:
        out_path = write_failure_analysis_report(
            full_failure_report, output_path=tmp_path / "rag_failure_analysis.md"
        )
        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")

        # Verify zero em/en dashes
        assert "\u2014" not in content
        assert "\u2013" not in content

        # Verify required sections and exact counts tables
        assert "## 1. Executive Summary" in content
        assert "## 2. Failure Category Counts (6-Category Taxonomy)" in content
        assert "## 3. Five-Level Diagnostic Cascade" in content
        assert "## 4. Detailed Case-by-Case Diagnostic Analysis" in content
        assert "## 5. Prioritized Fix Hypotheses (Top 3)" in content

        # Verify >= 15 analyzed case headings and all 6 taxonomy labels
        case_headings = re.findall(r"^#### `\[E[0-3]\] gold-\d+`", content, flags=re.MULTILINE)
        assert len(case_headings) == 22
        for cat in FAILURE_CATEGORIES:
            assert f"`{cat}`" in content
        for lvl in CASCADE_LEVELS:
            assert f"**{lvl}**" in content

        # Verify exactly 3 fix hypotheses
        hypotheses = re.findall(r"^### Hypothesis \d+", content, flags=re.MULTILINE)
        assert hypotheses == ["### Hypothesis 1", "### Hypothesis 2", "### Hypothesis 3"]

        # Verify truncated Judge Reasoning lines never cut mid-word
        for line in content.splitlines():
            if line.startswith("- **Judge Reasoning") and line.endswith("..."):
                prefix = line.split(":** ", 1)[1][:-3]
                assert prefix
                assert not prefix.endswith(" ")
                # Known mid-word cuts from naive slicing must not appear
                assert not prefix.endswith((" s", " candid", " conclu"))

    def test_committed_report_file_is_up_to_date(
        self, full_failure_report: FailureAnalysisReport
    ) -> None:
        assert DEFAULT_OUTPUT_REPORT.exists(), "reports/rag_failure_analysis.md must exist"
        on_disk = DEFAULT_OUTPUT_REPORT.read_text(encoding="utf-8")
        expected = render_markdown_report(full_failure_report)
        if not expected.endswith("\n"):
            expected += "\n"
        assert on_disk == expected

    def test_cli_main_writes_report(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        out_path = tmp_path / "cli_report.md"
        monkeypatch.setattr(
            sys,
            "argv",
            ["analyze_rag_failures", "--output", str(out_path)],
        )
        main()
        assert out_path.exists()
        assert out_path.read_text(encoding="utf-8") == DEFAULT_OUTPUT_REPORT.read_text(encoding="utf-8")
