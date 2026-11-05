"""Unit and integration tests for Cross-Encoder Reranker (Milestone M7)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.retrieval import (
    RetrievalResult,
    get_reranker_model,
    reset_cache,
    retrieve,
)


class TestRerankerEngine:
    """Tests verifying cross-encoder integration, batched scoring, and fallback handling."""

    def test_retrieve_with_rerank_returns_sorted_results(self):
        query = "how to capture custom events"
        k = 5
        results = retrieve(query, mode="hybrid", k=k, rerank=True, candidate_pool_size=20)

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

        # Verify descending order of cross-encoder scores
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_cross_encoder_batched_scoring_without_loop(self):
        """Verify cross-encoder predict is invoked once as a batched call over all pairs."""
        mock_reranker = MagicMock()
        # Mock predict returning descending scores for 10 candidates
        mock_reranker.predict.return_value = [0.9 - (0.05 * i) for i in range(10)]

        with patch("src.retrieval.get_reranker_model", return_value=mock_reranker):
            results = retrieve(
                "session replay configuration",
                mode="hybrid",
                k=5,
                rerank=True,
                candidate_pool_size=10,
                trace=False,
            )

        assert len(results) == 5
        # Must be called exactly once (batched), not once per candidate in a loop
        assert mock_reranker.predict.call_count == 1
        call_args = mock_reranker.predict.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) == 10
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in call_args)
        assert all(pair[0] == "session replay configuration" for pair in call_args)

    def test_reranker_cache_clearing(self):
        """Verify get_reranker_model caching and reset_cache clearing."""
        m1 = get_reranker_model()
        m2 = get_reranker_model()
        assert m1 is m2

        reset_cache()
        # Cache info should reflect clear
        info = get_reranker_model.cache_info()
        assert info.hits == 0

    def test_fallback_when_candidate_pool_smaller_than_k(self):
        """Verify graceful fallback when the retrieved candidate pool is smaller than k."""
        # Mock cross-encoder and simulate a query where only 2 candidates exist
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.95, 0.42]

        # Patch lancedb table search to return only 2 rows when k=5 is requested
        with patch("src.retrieval.get_reranker_model", return_value=mock_reranker), patch(
            "src.retrieval.get_lancedb_table"
        ) as mock_table:
            mock_search = MagicMock()
            mock_search.metric.return_value.limit.return_value.to_list.return_value = [
                {
                    "chunk_id": "chunk_1",
                    "text": "text 1",
                    "url": "https://posthog.com/docs/1",
                    "heading_path": "H1",
                    "title": "Title 1",
                    "_distance": 0.1,
                },
                {
                    "chunk_id": "chunk_2",
                    "text": "text 2",
                    "url": "https://posthog.com/docs/2",
                    "heading_path": "H2",
                    "title": "Title 2",
                    "_distance": 0.3,
                },
            ]
            mock_table.return_value.search.return_value = mock_search

            results = retrieve(
                "rare query with few matches",
                mode="dense",
                k=5,
                rerank=True,
                trace=False,
            )

        assert len(results) == 2
        for rank_idx, res in enumerate(results, start=1):
            assert res.rank == rank_idx

        assert results[0].score >= results[1].score
        assert results[0].chunk_id == "chunk_1"
        assert results[1].chunk_id == "chunk_2"

    def test_empty_candidate_list_returns_empty(self):
        """Verify retrieve with rerank=True safely returns empty list when no candidates exist."""
        mock_reranker = MagicMock()

        with patch("src.retrieval.get_reranker_model", return_value=mock_reranker), patch(
            "src.retrieval.get_lancedb_table"
        ) as mock_table:
            mock_search = MagicMock()
            mock_search.metric.return_value.limit.return_value.to_list.return_value = []
            mock_table.return_value.search.return_value = mock_search

            results = retrieve(
                "query matching nothing",
                mode="dense",
                k=5,
                rerank=True,
                trace=False,
            )

        assert results == []
        mock_reranker.predict.assert_not_called()

    def test_single_candidate_reranking(self):
        """Verify reranking operates cleanly when exactly one candidate is retrieved."""
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.88]

        with patch("src.retrieval.get_reranker_model", return_value=mock_reranker), patch(
            "src.retrieval.get_lancedb_table"
        ) as mock_table:
            mock_search = MagicMock()
            mock_search.metric.return_value.limit.return_value.to_list.return_value = [
                {
                    "chunk_id": "single_chunk",
                    "text": "single text",
                    "url": "https://posthog.com/docs/single",
                    "heading_path": "H1",
                    "title": "Single Title",
                    "_distance": 0.15,
                }
            ]
            mock_table.return_value.search.return_value = mock_search

            results = retrieve(
                "single match query",
                mode="dense",
                k=5,
                rerank=True,
                trace=False,
            )

        assert len(results) == 1
        assert results[0].rank == 1
        assert results[0].score == 0.88
        assert results[0].chunk_id == "single_chunk"
        assert mock_reranker.predict.call_count == 1

    def test_trace_logging_with_rerank(self, tmp_path: Path):
        """Verify structured JSONL trace logging captures mode and rerank flag."""
        trace_file = tmp_path / "rerank_traces.jsonl"
        assert not trace_file.exists()

        results = retrieve(
            "hogql query syntax",
            mode="hybrid",
            k=3,
            rerank=True,
            candidate_pool_size=10,
            trace=True,
            trace_path=trace_file,
        )

        assert trace_file.exists()
        lines = trace_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["query"] == "hogql query syntax"
        assert record["mode"] == "hybrid_rerank"
        assert record["rerank"] is True
        assert record["k"] == 3
        assert record["latency_ms"] > 0.0
        assert record["top_chunk_ids"] == [r.chunk_id for r in results]
        assert len(record["top_scores"]) == 3
