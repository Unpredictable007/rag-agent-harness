"""Unit tests for deterministic retrieval evaluation harness (eval/eval_retrieval.py)."""

import math
from pathlib import Path
import pytest

from eval.eval_retrieval import (
    BenchmarkSummary,
    ModeEvaluationResult,
    QueryEvaluationResult,
    RetrievalMetrics,
    aggregate_metrics,
    compute_query_metrics,
    evaluate_mode,
    evaluate_query,
    filter_evaluation_items,
    generate_markdown_report,
    run_retrieval_benchmark,
)
from eval.schema import GoldenDataset, GoldenItem
from src.retrieval import RetrievalResult


class TestComputeQueryMetrics:
    """Tests for per-query Recall@k, Hit@k, and Reciprocal Rank calculation."""

    def test_single_gold_hit_at_rank_1(self):
        gold = ["https://posthog.com/docs/feature-flags"]
        retrieved = [
            "https://posthog.com/docs/feature-flags",
            "https://posthog.com/docs/experiments",
            "https://posthog.com/docs/analytics",
        ]

        recalls, hits, rr, first_hit = compute_query_metrics(gold, retrieved, k_values=[1, 3, 5])

        assert recalls[1] == 1.0
        assert recalls[3] == 1.0
        assert recalls[5] == 1.0
        assert hits[1] is True
        assert hits[3] is True
        assert hits[5] is True
        assert first_hit == 1
        assert rr == 1.0

    def test_single_gold_hit_at_rank_4(self):
        gold = ["https://posthog.com/docs/target"]
        retrieved = [
            "https://posthog.com/docs/other1",
            "https://posthog.com/docs/other2",
            "https://posthog.com/docs/other3",
            "https://posthog.com/docs/target",
            "https://posthog.com/docs/other5",
        ]

        recalls, hits, rr, first_hit = compute_query_metrics(gold, retrieved, k_values=[1, 3, 5, 10])

        assert recalls[1] == 0.0
        assert recalls[3] == 0.0
        assert recalls[5] == 1.0
        assert recalls[10] == 1.0
        assert hits[1] is False
        assert hits[3] is False
        assert hits[5] is True
        assert hits[10] is True
        assert first_hit == 4
        assert rr == 0.25

    def test_multi_gold_partial_match(self):
        gold = [
            "https://posthog.com/docs/feature-flags",
            "https://posthog.com/docs/canary",
        ]
        retrieved = [
            "https://posthog.com/docs/feature-flags",
            "https://posthog.com/docs/other",
            "https://posthog.com/docs/another",
        ]

        recalls, hits, rr, first_hit = compute_query_metrics(gold, retrieved, k_values=[1, 3, 5])

        assert recalls[1] == 0.5
        assert recalls[3] == 0.5
        assert recalls[5] == 0.5
        assert hits[1] is True
        assert hits[3] is True
        assert hits[5] is True
        assert first_hit == 1
        assert rr == 1.0

    def test_multi_gold_full_match_different_ranks(self):
        gold = [
            "https://posthog.com/docs/doc1",
            "https://posthog.com/docs/doc2",
        ]
        retrieved = [
            "https://posthog.com/docs/other",
            "https://posthog.com/docs/doc1",
            "https://posthog.com/docs/irrelevant",
            "https://posthog.com/docs/doc2",
        ]

        recalls, hits, rr, first_hit = compute_query_metrics(gold, retrieved, k_values=[1, 3, 5])

        assert recalls[1] == 0.0
        assert recalls[3] == 0.5
        assert recalls[5] == 1.0
        assert hits[1] is False
        assert hits[3] is True
        assert hits[5] is True
        assert first_hit == 2
        assert rr == 0.5

    def test_zero_match(self):
        gold = ["https://posthog.com/docs/needed"]
        retrieved = [
            "https://posthog.com/docs/foo",
            "https://posthog.com/docs/bar",
        ]

        recalls, hits, rr, first_hit = compute_query_metrics(gold, retrieved, k_values=[1, 3, 5])

        assert recalls[1] == 0.0
        assert recalls[3] == 0.0
        assert recalls[5] == 0.0
        assert hits[1] is False
        assert hits[3] is False
        assert hits[5] is False
        assert first_hit is None
        assert rr == 0.0

    def test_empty_gold_urls(self):
        gold = []
        retrieved = ["https://posthog.com/docs/foo"]

        recalls, hits, rr, first_hit = compute_query_metrics(gold, retrieved, k_values=[1, 5])

        assert recalls[1] == 0.0
        assert hits[1] is False
        assert rr == 0.0
        assert first_hit is None

    def test_url_normalization_matching(self):
        # Gold URL with fragment and trailing slash
        gold = ["https://posthog.com/docs/events/#capture/"]
        # Retrieved URL with whitespace and different fragment
        retrieved = [
            "  https://posthog.com/docs/events#properties  ",
            "https://posthog.com/docs/other",
        ]

        recalls, hits, rr, first_hit = compute_query_metrics(gold, retrieved, k_values=[1, 5])

        assert recalls[1] == 1.0
        assert hits[1] is True
        assert first_hit == 1
        assert rr == 1.0

    def test_duplicate_retrieved_urls_from_multiple_chunks(self):
        # Multiple chunks belonging to the same document URL
        gold = ["https://posthog.com/docs/surveys"]
        retrieved = [
            "https://posthog.com/docs/surveys#section1",
            "https://posthog.com/docs/surveys#section2",
            "https://posthog.com/docs/surveys#section3",
        ]

        recalls, hits, rr, first_hit = compute_query_metrics(gold, retrieved, k_values=[1, 3, 5])

        assert recalls[1] == 1.0
        assert recalls[3] == 1.0
        assert hits[1] is True
        assert first_hit == 1
        assert rr == 1.0

    def test_empty_retrieved_urls(self):
        gold = ["https://posthog.com/docs/feature-flags"]
        recalls, hits, rr, first_hit = compute_query_metrics(gold, [], k_values=[1, 5])
        assert recalls[1] == 0.0
        assert recalls[5] == 0.0
        assert hits[1] is False
        assert hits[5] is False
        assert first_hit is None
        assert rr == 0.0

    def test_duplicate_gold_urls_same_canonical_path(self):
        # Two gold entries with different anchors that collapse to the same canonical URL
        gold = [
            "https://posthog.com/docs/events#part1",
            "https://posthog.com/docs/events#part2",
        ]
        retrieved = ["https://posthog.com/docs/events"]
        recalls, hits, rr, first_hit = compute_query_metrics(gold, retrieved, k_values=[1, 5])
        assert recalls[1] == 1.0
        assert hits[1] is True
        assert rr == 1.0
        assert first_hit == 1

    def test_mrr_cutoff_bounded(self):
        # Hit is at rank 12, but evaluated cutoffs are only up to 10
        gold = ["https://posthog.com/docs/target"]
        retrieved = [f"https://posthog.com/docs/other{i}" for i in range(1, 12)] + [
            "https://posthog.com/docs/target"
        ]
        recalls, hits, rr, first_hit = compute_query_metrics(gold, retrieved, k_values=[1, 3, 5, 10])
        assert recalls[10] == 0.0
        assert hits[10] is False
        assert rr == 0.0
        assert first_hit is None



class TestAggregateMetrics:
    """Tests for aggregating per-query evaluation results."""

    def test_aggregate_metrics_two_queries(self):
        q1 = QueryEvaluationResult(
            item_id="q1",
            question="What is X?",
            question_type="single-chunk",
            gold_urls=["https://posthog.com/docs/x"],
            retrieved_urls=["https://posthog.com/docs/x"],
            retrieved_chunk_ids=["x#0"],
            reciprocal_rank=1.0,
            first_hit_rank=1,
            hits_at_k={1: True, 3: True, 5: True, 10: True},
            recall_at_k={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0},
            latency_ms=10.0,
        )
        q2 = QueryEvaluationResult(
            item_id="q2",
            question="What is Y?",
            question_type="single-chunk",
            gold_urls=["https://posthog.com/docs/y1", "https://posthog.com/docs/y2"],
            retrieved_urls=["https://posthog.com/docs/other", "https://posthog.com/docs/y1"],
            retrieved_chunk_ids=["other#0", "y1#0"],
            reciprocal_rank=0.5,
            first_hit_rank=2,
            hits_at_k={1: False, 3: True, 5: True, 10: True},
            recall_at_k={1: 0.0, 3: 0.5, 5: 0.5, 10: 0.5},
            latency_ms=30.0,
        )

        agg = aggregate_metrics([q1, q2], k_values=[1, 3, 5, 10])

        assert agg.query_count == 2
        assert agg.hit_rate_at_1 == 0.5
        assert agg.hit_rate_at_3 == 1.0
        assert agg.hit_rate_at_5 == 1.0
        assert agg.recall_at_1 == 0.5
        assert agg.recall_at_3 == 0.75
        assert agg.recall_at_5 == 0.75
        assert agg.mrr == 0.75
        assert agg.latency_mean_ms == 20.0
        assert agg.latency_p50_ms == 20.0

    def test_aggregate_metrics_empty_list(self):
        agg = aggregate_metrics([])
        assert agg.query_count == 0
        assert agg.hit_rate_at_1 == 0.0
        assert agg.recall_at_5 == 0.0
        assert agg.mrr == 0.0
        assert agg.latency_p50_ms == 0.0


class TestFilteringAndEvaluationMode:
    """Tests for dataset filtering and mode evaluation."""

    def test_filter_evaluation_items(self):
        dataset = GoldenDataset(
            items=[
                GoldenItem(
                    id="g1",
                    question="Q1",
                    reference_answer="A1",
                    gold_urls=["https://posthog.com/docs/1"],
                    type="single-chunk",
                    answerable=True,
                    split="dev",
                ),
                GoldenItem(
                    id="g2",
                    question="Q2",
                    reference_answer="A2",
                    gold_urls=["https://posthog.com/docs/2"],
                    type="multi-chunk",
                    answerable=True,
                    split="holdout",
                ),
                GoldenItem(
                    id="g3",
                    question="Q3",
                    reference_answer="Cannot answer",
                    gold_urls=[],
                    type="unanswerable",
                    answerable=False,
                    split="dev",
                ),
            ]
        )

        dev_items = filter_evaluation_items(dataset, split="dev")
        assert len(dev_items) == 1
        assert dev_items[0].id == "g1"

        all_items = filter_evaluation_items(dataset, split="all")
        assert len(all_items) == 2
        assert {item.id for item in all_items} == {"g1", "g2"}

    def test_evaluate_mode_with_mock_retriever(self):
        items = [
            GoldenItem(
                id="item-1",
                question="How to test flags?",
                reference_answer="Answer 1",
                gold_urls=["https://posthog.com/docs/feature-flags"],
                type="single-chunk",
                answerable=True,
                split="dev",
            ),
            GoldenItem(
                id="item-2",
                question="How to configure surveys and destinations?",
                reference_answer="Answer 2",
                gold_urls=[
                    "https://posthog.com/docs/surveys",
                    "https://posthog.com/docs/destinations",
                ],
                type="multi-chunk",
                answerable=True,
                split="dev",
            ),
        ]

        def mock_retriever(query: str, mode: str, k: int, trace: bool):
            if "flags" in query:
                return [
                    RetrievalResult(
                        chunk_id="chunk_flag_1",
                        score=0.95,
                        rank=1,
                        text="Feature flags guide",
                        url="https://posthog.com/docs/feature-flags",
                        heading_path="Flags > Setup",
                        title="Feature Flags",
                    ),
                    RetrievalResult(
                        chunk_id="chunk_other_2",
                        score=0.80,
                        rank=2,
                        text="Other info",
                        url="https://posthog.com/docs/other",
                        heading_path="Other",
                        title="Other",
                    ),
                ]
            else:
                return [
                    RetrievalResult(
                        chunk_id="chunk_survey_1",
                        score=0.90,
                        rank=1,
                        text="Surveys guide",
                        url="https://posthog.com/docs/surveys",
                        heading_path="Surveys",
                        title="Surveys",
                    ),
                    RetrievalResult(
                        chunk_id="chunk_dest_2",
                        score=0.85,
                        rank=2,
                        text="Destinations guide",
                        url="https://posthog.com/docs/destinations",
                        heading_path="Destinations",
                        title="Destinations",
                    ),
                ]

        result = evaluate_mode(
            items=items,
            mode="hybrid",
            k=5,
            retriever_fn=mock_retriever,
        )

        assert isinstance(result, ModeEvaluationResult)
        assert result.mode == "hybrid"
        assert result.overall.query_count == 2
        assert result.overall.hit_rate_at_1 == 1.0
        assert result.overall.recall_at_5 == 1.0
        assert result.overall.mrr == 1.0

        # Check by_type breakdown
        assert "single-chunk" in result.by_type
        assert "multi-chunk" in result.by_type
        assert result.by_type["single-chunk"].query_count == 1
        assert result.by_type["multi-chunk"].query_count == 1


class TestReportGeneration:
    """Tests for report rendering without banned punctuation or syntax issues."""

    def test_generate_markdown_report_formatting(self):
        metrics = RetrievalMetrics(
            hit_rate_at_1=0.6111,
            hit_rate_at_3=0.7222,
            hit_rate_at_5=0.7778,
            hit_rate_at_10=0.7778,
            recall_at_1=0.5556,
            recall_at_3=0.6667,
            recall_at_5=0.7083,
            recall_at_10=0.7778,
            mrr=0.6840,
            latency_p50_ms=52.91,
            latency_p95_ms=113.11,
            latency_mean_ms=60.15,
            query_count=36,
        )
        mode_res = ModeEvaluationResult(
            mode="hybrid",
            overall=metrics,
            by_type={"single-chunk": metrics, "multi-chunk": metrics},
            queries=[],
        )
        summary = BenchmarkSummary(
            dataset_path="/tmp/eval/golden_set.json",
            split="dev",
            k=10,
            timestamp="2026-11-04T12:00:00Z",
            modes={"hybrid": mode_res},
        )

        report = generate_markdown_report(summary)

        assert "# Retrieval Evaluation Report: Deterministic Metrics (E0 - E2)" in report
        assert "| Mode | Hit@1 | Hit@5 | Recall@5 | Recall@10 | MRR | P50 (ms) | P95 (ms) |" in report
        assert "Hybrid RRF (E2)" in report
        assert "0.6840" in report

        # Verify no em dashes or en dashes per Rule 3
        assert "\u2014" not in report, "Found em dash in markdown report"
        assert "\u2013" not in report, "Found en dash in markdown report"
