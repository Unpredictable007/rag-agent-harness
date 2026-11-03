import json
import math
from datetime import datetime
from pathlib import Path

import pytest

from src.config import RETRIEVAL_TRACES_PATH
from src.retrieval import (
    RetrievalResult,
    get_bm25_and_lookup,
    get_embedding_model,
    get_lancedb_table,
    reciprocal_rank_fusion,
    retrieve,
)


class TestReciprocalRankFusion:
    """Unit tests for Reciprocal Rank Fusion (RRF) mathematical correctness."""

    def test_rrf_known_scores(self):
        ranked_lists = [
            ["chunk_a", "chunk_b", "chunk_c"],
            ["chunk_b", "chunk_a", "chunk_d"],
        ]
        k = 60
        fused = reciprocal_rank_fusion(ranked_lists, k=k)

        # Expected calculations:
        # chunk_a: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
        # chunk_b: 1/(60+2) + 1/(60+1) = 1/62 + 1/61
        # chunk_c: 1/(60+3) = 1/63
        # chunk_d: 1/(60+3) = 1/63
        expected_ab = (1.0 / 61) + (1.0 / 62)
        expected_cd = 1.0 / 63

        scores = dict(fused)
        assert math.isclose(scores["chunk_a"], expected_ab, rel_tol=1e-6)
        assert math.isclose(scores["chunk_b"], expected_ab, rel_tol=1e-6)
        assert math.isclose(scores["chunk_c"], expected_cd, rel_tol=1e-6)
        assert math.isclose(scores["chunk_d"], expected_cd, rel_tol=1e-6)

        # Verify descending order
        top_two = [fused[0][0], fused[1][0]]
        bottom_two = [fused[2][0], fused[3][0]]
        assert set(top_two) == {"chunk_a", "chunk_b"}
        assert set(bottom_two) == {"chunk_c", "chunk_d"}

    def test_rrf_single_ranked_list(self):
        ranked_lists = [["doc_1", "doc_2", "doc_3"]]
        k = 10
        fused = reciprocal_rank_fusion(ranked_lists, k=k)

        assert len(fused) == 3
        assert fused[0] == ("doc_1", 1.0 / 11)
        assert fused[1] == ("doc_2", 1.0 / 12)
        assert fused[2] == ("doc_3", 1.0 / 13)

    def test_rrf_empty_lists(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_rrf_disjoint_lists(self):
        ranked_lists = [["x1", "x2"], ["y1", "y2"]]
        fused = reciprocal_rank_fusion(ranked_lists, k=60)
        scores = dict(fused)

        assert math.isclose(scores["x1"], 1.0 / 61, rel_tol=1e-6)
        assert math.isclose(scores["y1"], 1.0 / 61, rel_tol=1e-6)
        assert math.isclose(scores["x2"], 1.0 / 62, rel_tol=1e-6)
        assert math.isclose(scores["y2"], 1.0 / 62, rel_tol=1e-6)


class TestRetrievalEngine:
    """Unit and integration tests for dense, BM25, and hybrid retrieval."""

    def test_retrieve_dense_mode(self):
        k = 5
        results = retrieve("feature flag rollout percentage", mode="dense", k=k)

        assert len(results) == k
        for rank_idx, result in enumerate(results, start=1):
            assert isinstance(result, RetrievalResult)
            assert result.rank == rank_idx
            assert isinstance(result.score, float)
            assert result.chunk_id
            assert result.text
            assert result.url.startswith("https://posthog.com/docs")
            assert result.heading_path
            assert result.title

        # Check descending scores
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_retrieve_bm25_mode(self):
        query = "posthog.capture"
        k = 5
        results = retrieve(query, mode="bm25", k=k)

        assert len(results) == k
        assert results[0].rank == 1
        assert results[0].score > 0.0

        # BM25 should rank capture documentation at the very top
        top_text = (results[0].title + " " + results[0].heading_path + " " + results[0].text).lower()
        assert "capture" in top_text

        # Verify scores are strictly non-increasing
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_retrieve_hybrid_mode(self):
        query = "how to capture custom events"
        k = 5
        results = retrieve(query, mode="hybrid", k=k)

        assert len(results) == k
        for rank_idx, result in enumerate(results, start=1):
            assert isinstance(result, RetrievalResult)
            assert result.rank == rank_idx
            assert result.score > 0.0
            assert result.chunk_id
            assert result.text
            assert result.url.startswith("https://posthog.com/docs")

        # Verify descending order of RRF scores
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_invalid_parameters(self):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            retrieve("test", mode="hybrid", k=0)

        with pytest.raises(ValueError, match="k must be a positive integer"):
            retrieve("test", mode="hybrid", k=-5)

        with pytest.raises(ValueError, match="Invalid retrieval mode"):
            retrieve("test", mode="invalid_mode", k=5)  # type: ignore


class TestTraceLogging:
    """Tests verifying structured JSONL trace logging on retrieval calls."""

    def test_trace_logging_to_custom_path(self, tmp_path: Path):
        trace_file = tmp_path / "custom_traces.jsonl"
        assert not trace_file.exists()

        results_1 = retrieve(
            "custom trace query 1",
            mode="bm25",
            k=3,
            trace=True,
            trace_path=trace_file,
        )

        assert trace_file.exists()
        lines = trace_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

        record_1 = json.loads(lines[0])
        assert record_1["query"] == "custom trace query 1"
        assert record_1["mode"] == "bm25"
        assert record_1["k"] == 3
        assert record_1["latency_ms"] >= 0.0
        assert record_1["top_chunk_ids"] == [r.chunk_id for r in results_1]
        assert len(record_1["top_scores"]) == 3
        # Timestamp must be valid ISO-8601
        assert datetime.fromisoformat(record_1["timestamp"]) is not None

        # Call again to verify append mode ("a") does not overwrite
        results_2 = retrieve(
            "custom trace query 2",
            mode="hybrid",
            k=2,
            trace=True,
            trace_path=trace_file,
        )

        lines = trace_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        record_2 = json.loads(lines[1])
        assert record_2["query"] == "custom trace query 2"
        assert record_2["mode"] == "hybrid"
        assert record_2["k"] == 2
        assert record_2["top_chunk_ids"] == [r.chunk_id for r in results_2]

    def test_default_trace_logging(self):
        results = retrieve("default trace file query", mode="dense", k=2, trace=True)
        assert RETRIEVAL_TRACES_PATH.exists()

        with open(RETRIEVAL_TRACES_PATH, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        last_entry = json.loads(lines[-1])
        assert last_entry["query"] == "default trace file query"
        assert last_entry["mode"] == "dense"
        assert last_entry["k"] == 2
        assert last_entry["top_chunk_ids"] == [r.chunk_id for r in results]


class TestCacheAndLazyLoading:
    """Tests verifying lazy loading and singleton cache instances."""

    def test_instances_cached(self):
        model1 = get_embedding_model()
        model2 = get_embedding_model()
        assert model1 is model2

        table1 = get_lancedb_table()
        table2 = get_lancedb_table()
        assert table1 is table2

        bm25_1, lookup_1 = get_bm25_and_lookup()
        bm25_2, lookup_2 = get_bm25_and_lookup()
        assert bm25_1 is bm25_2
        assert lookup_1 is lookup_2
