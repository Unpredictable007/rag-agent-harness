"""Automated CI/test regression gate for deterministic retrieval.

Evaluates retrieval performance across the dev split of eval/golden_set.json against
the accepted production baseline (E3: Hybrid RRF + Cross-Encoder Reranker).
Enforces hard gate assertion:
    measured_recall_at_5 >= baseline_recall_at_5 - 0.03 (3 percentage points)

Includes negative regression tests verifying that deliberate retrieval degradation
(e.g., BM25 omitted or degraded candidates) trips the gate and raises AssertionError.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from eval.eval_ablation import compute_retrieval_query_metrics
from eval.eval_retrieval import filter_evaluation_items
from eval.schema import GoldenDataset, GoldenItem
from src.config import REPO_ROOT
from src.retrieval import RetrievalResult, retrieve

BASELINE_PATH = REPO_ROOT / "eval" / "baseline.json"
GOLDEN_SET_PATH = REPO_ROOT / "eval" / "golden_set.json"


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, Any]:
    """Load baseline configuration and metrics from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Baseline file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_regression_gate(
    measured_recall_at_5: float,
    baseline_recall_at_5: float,
    max_allowed_drop: float = 0.03,
) -> None:
    """Enforce the hard regression gate threshold.

    Raises:
        AssertionError: If measured recall@5 falls below (baseline - max_allowed_drop).
    """
    min_acceptable_recall = baseline_recall_at_5 - max_allowed_drop
    if measured_recall_at_5 < min_acceptable_recall:
        drop = baseline_recall_at_5 - measured_recall_at_5
        raise AssertionError(
            f"REGRESSION GATE FAILED: measured recall@5 ({measured_recall_at_5:.4f}) "
            f"fell below baseline ({baseline_recall_at_5:.4f}) minus margin ({max_allowed_drop:.4f}) "
            f"[min acceptable threshold: {min_acceptable_recall:.4f}]. "
            f"Observed regression drop: {drop:.4f} ({drop * 100:.2f} percentage points)."
        )


def evaluate_retrieval_on_dev(
    dev_items: list[GoldenItem],
    retriever_fn: Callable[..., list[RetrievalResult]] = retrieve,
    retriever_kwargs: dict[str, Any] | None = None,
    k: int = 5,
) -> dict[str, float]:
    """Run retrieval across dev split items and compute deterministic metrics."""
    kwargs = retriever_kwargs or {}
    recalls: list[float] = []
    hits: list[float] = []
    mrrs: list[float] = []

    for item in dev_items:
        results = retriever_fn(item.question, **kwargs)
        rec, hit, mrr = compute_retrieval_query_metrics(item.gold_urls, results, k=k)
        recalls.append(rec)
        hits.append(hit)
        mrrs.append(mrr)

    n = len(dev_items)
    if n == 0:
        return {"recall_at_5": 0.0, "hit_at_5": 0.0, "mrr": 0.0, "count": 0}

    return {
        "recall_at_5": float(sum(recalls) / n),
        "hit_at_5": float(sum(hits) / n),
        "mrr": float(sum(mrrs) / n),
        "count": float(n),
    }


@pytest.fixture(scope="module")
def baseline_data() -> dict[str, Any]:
    """Fixture providing loaded baseline configuration data."""
    return load_baseline(BASELINE_PATH)


@pytest.fixture(scope="module")
def dev_items() -> list[GoldenItem]:
    """Fixture providing filtered answerable dev items from golden dataset."""
    dataset = GoldenDataset.from_file(GOLDEN_SET_PATH)
    items = filter_evaluation_items(dataset, split="dev")
    assert len(items) == 36, f"Expected 36 answerable dev items, got {len(items)}"
    return items


@pytest.fixture(scope="module")
def standard_e3_eval_results(dev_items: list[GoldenItem]) -> dict[str, float]:
    """Execute standard E3 retrieval benchmark once across dev items."""
    kwargs = {
        "mode": "hybrid",
        "k": 5,
        "rerank": True,
        "candidate_pool_size": 50,
        "trace": False,
    }
    return evaluate_retrieval_on_dev(dev_items, retriever_fn=retrieve, retriever_kwargs=kwargs, k=5)


class TestRegressionGate:
    """Test suite for deterministic retrieval regression gate."""

    def test_baseline_json_integrity(self, baseline_data: dict[str, Any]) -> None:
        """Validate baseline.json schema, required fields, and mathematical consistency."""
        assert baseline_data["version"] == "1.0.0"
        assert baseline_data.get("baseline_id") == "production_e3_v1"
        assert baseline_data["config_name"] == "E3"
        assert "config" in baseline_data
        assert "metrics" in baseline_data
        assert "regression_gate" in baseline_data

        config = baseline_data["config"]
        assert config["name"] == "E3"
        assert config["mode"] == "hybrid"
        assert config["rerank"] is True
        assert config["k"] == 5

        metrics = baseline_data["metrics"]
        assert pytest.approx(metrics["recall_at_5"], rel=1e-3) == 0.7639
        assert pytest.approx(metrics["hit_at_5"], rel=1e-3) == 0.8333
        assert pytest.approx(metrics["mrr"], rel=1e-3) == 0.7787

        gate = baseline_data["regression_gate"]
        assert gate["primary_metric"] == "recall_at_5"
        assert gate["max_allowed_drop"] == 0.03
        assert pytest.approx(gate["min_acceptable_recall_at_5"], rel=1e-4) == (
            gate["baseline_recall_at_5"] - gate["max_allowed_drop"]
        )

    def test_regression_gate_passes_standard_run(
        self,
        standard_e3_eval_results: dict[str, float],
        baseline_data: dict[str, Any],
    ) -> None:
        """Standard run test: ensure live accepted config satisfies the hard gate."""
        measured_recall = standard_e3_eval_results["recall_at_5"]
        measured_hit = standard_e3_eval_results["hit_at_5"]
        measured_mrr = standard_e3_eval_results["mrr"]

        baseline_recall = baseline_data["regression_gate"]["baseline_recall_at_5"]
        max_allowed_drop = baseline_data["regression_gate"]["max_allowed_drop"]
        min_acceptable_recall = baseline_recall - max_allowed_drop

        # Hard regression gate check
        check_regression_gate(
            measured_recall_at_5=measured_recall,
            baseline_recall_at_5=baseline_recall,
            max_allowed_drop=max_allowed_drop,
        )

        assert measured_recall >= min_acceptable_recall, (
            f"Measured recall@5 ({measured_recall:.4f}) is below acceptable threshold "
            f"({min_acceptable_recall:.4f})"
        )

        # Confirm measured values match baseline numbers closely
        assert pytest.approx(measured_recall, abs=0.01) == baseline_recall
        assert pytest.approx(measured_hit, abs=0.01) == baseline_data["metrics"]["hit_at_5"]
        assert pytest.approx(measured_mrr, abs=0.01) == baseline_data["metrics"]["mrr"]

    def test_regression_gate_catches_degraded_retrieval(
        self,
        dev_items: list[GoldenItem],
        baseline_data: dict[str, Any],
    ) -> None:
        """Negative test: deliberately simulate degraded retrieval and assert gate trips.

        Evaluates dense-only retrieval (E0) where BM25 lexical matching and cross-encoder
        reranking are disabled. In E0, recall@5 drops to ~0.7222 (exceeding the 3 pp margin),
        which must trigger an AssertionError in the regression gate.
        """
        degraded_kwargs = {
            "mode": "dense",
            "k": 5,
            "rerank": False,
            "trace": False,
        }
        degraded_results = evaluate_retrieval_on_dev(
            dev_items,
            retriever_fn=retrieve,
            retriever_kwargs=degraded_kwargs,
            k=5,
        )

        measured_recall = degraded_results["recall_at_5"]
        baseline_recall = baseline_data["regression_gate"]["baseline_recall_at_5"]
        max_drop = baseline_data["regression_gate"]["max_allowed_drop"]

        # Confirm that dense-only retrieval is indeed below threshold
        assert measured_recall < (baseline_recall - max_drop), (
            f"Expected degraded recall to be below {baseline_recall - max_drop:.4f}, "
            f"got {measured_recall:.4f}"
        )

        # Assert that the regression gate trips and raises AssertionError
        with pytest.raises(AssertionError) as exc_info:
            check_regression_gate(
                measured_recall_at_5=measured_recall,
                baseline_recall_at_5=baseline_recall,
                max_allowed_drop=max_drop,
            )

        assert "REGRESSION GATE FAILED" in str(exc_info.value)
        assert "measured recall@5" in str(exc_info.value)

    def test_regression_gate_catches_synthetic_drop(
        self,
        baseline_data: dict[str, Any],
    ) -> None:
        """Negative test: verify threshold logic on synthetic retrieval drops."""
        baseline_recall = baseline_data["regression_gate"]["baseline_recall_at_5"]
        max_drop = baseline_data["regression_gate"]["max_allowed_drop"]

        # Exactly at boundary: should pass
        exact_boundary = baseline_recall - max_drop
        check_regression_gate(exact_boundary, baseline_recall, max_drop)

        # Slightly above boundary (within allowed margin): should pass
        within_margin = baseline_recall - 0.02
        check_regression_gate(within_margin, baseline_recall, max_drop)

        # Slightly below boundary (exceeds allowed margin): must fail
        exceeds_margin = baseline_recall - 0.0301
        with pytest.raises(AssertionError) as exc_info:
            check_regression_gate(exceeds_margin, baseline_recall, max_drop)
        assert "REGRESSION GATE FAILED" in str(exc_info.value)

        # Severe drop: must fail
        severe_drop = 0.50
        with pytest.raises(AssertionError) as exc_info:
            check_regression_gate(severe_drop, baseline_recall, max_drop)
        assert "REGRESSION GATE FAILED" in str(exc_info.value)

    def test_regression_gate_zero_results_fails(
        self,
        dev_items: list[GoldenItem],
        baseline_data: dict[str, Any],
    ) -> None:
        """Negative test: verify that a broken retriever returning no results fails gate."""
        def broken_retriever(query: str, **kwargs: Any) -> list[RetrievalResult]:
            return []

        broken_results = evaluate_retrieval_on_dev(
            dev_items,
            retriever_fn=broken_retriever,
            k=5,
        )

        assert broken_results["recall_at_5"] == 0.0
        assert broken_results["hit_at_5"] == 0.0
        assert broken_results["mrr"] == 0.0

        baseline_recall = baseline_data["regression_gate"]["baseline_recall_at_5"]
        max_drop = baseline_data["regression_gate"]["max_allowed_drop"]

        with pytest.raises(AssertionError) as exc_info:
            check_regression_gate(
                measured_recall_at_5=broken_results["recall_at_5"],
                baseline_recall_at_5=baseline_recall,
                max_allowed_drop=max_drop,
            )

        assert "REGRESSION GATE FAILED" in str(exc_info.value)
