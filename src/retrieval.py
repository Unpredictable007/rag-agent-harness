import argparse
import json
import os
import pickle
import re
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import lancedb
import numpy as np
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import (
    BM25_INDEX_PATH,
    EMBEDDING_MODEL_NAME,
    LANCEDB_DIR,
    LANCEDB_TABLE_NAME,
    RETRIEVAL_TRACES_PATH,
    RRF_K,
    TRACES_DIR,
)
from src.ingest import ChunkLookup, DocumentChunk


class RetrievalResult(BaseModel):
    chunk_id: str
    score: float
    rank: int
    text: str
    url: str
    heading_path: str
    title: str


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Return cached SentenceTransformer instance or initialize on first call."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_lancedb_table() -> Any:
    """Return cached LanceDB table connection or initialize on first call."""
    db = lancedb.connect(str(LANCEDB_DIR))
    return db.open_table(LANCEDB_TABLE_NAME)


@lru_cache(maxsize=1)
def get_bm25_and_lookup() -> tuple[BM25Okapi, ChunkLookup]:
    """Return cached BM25 index and ChunkLookup or deserialize on first call."""
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(f"BM25 index file not found at: {BM25_INDEX_PATH}")
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25, chunk_lookup = pickle.load(f)
    return bm25, chunk_lookup


def reset_cache() -> None:
    """Clear all cached model, database, and index instances."""
    get_embedding_model.cache_clear()
    get_lancedb_table.cache_clear()
    get_bm25_and_lookup.cache_clear()


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Compute Reciprocal Rank Fusion (RRF) scores across multiple ranked candidate lists.

    Formula: score(d) = sum(1.0 / (k + rank)) for 1-based ranks across all lists containing d.
    Returns list of (chunk_id, fused_score) tuples sorted in descending order of fused score.
    """
    scores: dict[str, float] = {}
    for r_list in ranked_lists:
        for rank, chunk_id in enumerate(r_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def log_retrieval_trace(
    query: str,
    mode: str,
    k: int,
    latency_ms: float,
    results: list[RetrievalResult],
    trace_path: Path | None = None,
) -> dict[str, Any]:
    """
    Append a single structured JSON line to retrieval traces file.
    Uses append mode ('a') to prevent overwriting existing audit logs.
    """
    target_path = trace_path or RETRIEVAL_TRACES_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    trace_entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "mode": mode,
        "k": k,
        "latency_ms": round(float(latency_ms), 3),
        "top_chunk_ids": [r.chunk_id for r in results],
        "top_scores": [float(r.score) for r in results],
    }

    with open(target_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace_entry) + "\n")

    return trace_entry


def retrieve(
    query: str,
    mode: Literal["dense", "bm25", "hybrid"] = "hybrid",
    k: int = 5,
    trace: bool = True,
    trace_path: Path | None = None,
) -> list[RetrievalResult]:
    """
    Execute retrieval across dense vector search, BM25 sparse search, or hybrid RRF fusion.

    Args:
        query: User search query string.
        mode: Retrieval mode ('dense', 'bm25', 'hybrid'). Defaults to 'hybrid'.
        k: Number of top ranked results to return. Defaults to 5.
        trace: Whether to log structured trace to retrieval_traces.jsonl.
        trace_path: Optional override for trace output file path.

    Returns:
        List of RetrievalResult objects ranked in descending order of relevance score.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    valid_modes = ("dense", "bm25", "hybrid")
    if mode not in valid_modes:
        raise ValueError(f"Invalid retrieval mode: '{mode}'. Must be one of: {valid_modes}")

    t_start = time.perf_counter()
    results: list[RetrievalResult] = []

    if mode == "dense":
        model = get_embedding_model()
        table = get_lancedb_table()

        query_vec = model.encode(query, normalize_embeddings=True).tolist()
        rows = table.search(query_vec).metric("cosine").limit(k).to_list()

        for rank, row in enumerate(rows, start=1):
            distance = float(row.get("_distance", 0.0))
            score = 1.0 - distance
            results.append(
                RetrievalResult(
                    chunk_id=str(row["chunk_id"]),
                    score=float(score),
                    rank=rank,
                    text=str(row["text"]),
                    url=str(row["url"]),
                    heading_path=str(row["heading_path"]),
                    title=str(row["title"]),
                )
            )

    elif mode == "bm25":
        bm25, chunk_lookup = get_bm25_and_lookup()

        tokens = re.findall(r"\w+", query.lower())
        if not tokens:
            scores = np.zeros(bm25.corpus_size, dtype=float)
        else:
            scores = bm25.get_scores(tokens)

        top_indices = np.argsort(scores)[::-1][:k]

        for rank, idx in enumerate(top_indices, start=1):
            chunk = chunk_lookup[int(idx)]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    score=float(scores[idx]),
                    rank=rank,
                    text=chunk.text,
                    url=chunk.url,
                    heading_path=chunk.heading_path,
                    title=chunk.title,
                )
            )

    elif mode == "hybrid":
        model = get_embedding_model()
        table = get_lancedb_table()
        bm25, chunk_lookup = get_bm25_and_lookup()

        candidate_pool_size = max(50, k)

        # 1. Fetch dense candidate pool
        query_vec = model.encode(query, normalize_embeddings=True).tolist()
        dense_rows = table.search(query_vec).metric("cosine").limit(candidate_pool_size).to_list()
        dense_chunk_ids = [str(r["chunk_id"]) for r in dense_rows]
        dense_map = {str(r["chunk_id"]): r for r in dense_rows}

        # 2. Fetch BM25 candidate pool
        tokens = re.findall(r"\w+", query.lower())
        if not tokens:
            bm25_scores = np.zeros(bm25.corpus_size, dtype=float)
        else:
            bm25_scores = bm25.get_scores(tokens)

        top_bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_pool_size]
        # Include chunks with positive BM25 scores in sparse candidate list
        bm25_chunk_ids = [
            chunk_lookup[int(idx)].chunk_id
            for idx in top_bm25_indices
            if bm25_scores[idx] > 0
        ]

        # 3. Fuse candidate rankings with Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion([dense_chunk_ids, bm25_chunk_ids], k=RRF_K)
        top_fused = fused[:k]

        # 4. Construct RetrievalResult objects
        for rank, (chunk_id, fused_score) in enumerate(top_fused, start=1):
            chunk = chunk_lookup.get(chunk_id)
            if chunk is not None:
                results.append(
                    RetrievalResult(
                        chunk_id=chunk.chunk_id,
                        score=float(fused_score),
                        rank=rank,
                        text=chunk.text,
                        url=chunk.url,
                        heading_path=chunk.heading_path,
                        title=chunk.title,
                    )
                )
            elif chunk_id in dense_map:
                row = dense_map[chunk_id]
                results.append(
                    RetrievalResult(
                        chunk_id=str(row["chunk_id"]),
                        score=float(fused_score),
                        rank=rank,
                        text=str(row["text"]),
                        url=str(row["url"]),
                        heading_path=str(row["heading_path"]),
                        title=str(row["title"]),
                    )
                )

    latency_ms = (time.perf_counter() - t_start) * 1000.0

    if trace:
        log_retrieval_trace(
            query=query,
            mode=mode,
            k=k,
            latency_ms=latency_ms,
            results=results,
            trace_path=trace_path,
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute retrieval query over PostHog documentation.")
    parser.add_argument(
        "query",
        nargs="?",
        default="How to capture custom events",
        help="Search query text (default: 'How to capture custom events')",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="hybrid",
        choices=["dense", "bm25", "hybrid"],
        help="Retrieval mode: dense, bm25, or hybrid (default: hybrid)",
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        default=5,
        help="Number of top chunks to return (default: 5)",
    )
    args = parser.parse_args()

    print(f"Executing retrieval query: '{args.query}' (mode={args.mode}, k={args.k})")
    results = retrieve(query=args.query, mode=args.mode, k=args.k)
    print(f"\nRetrieved {len(results)} results:")
    for res in results:
        print(f"\n[Rank {res.rank}] Score: {res.score:.6f}")
        print(f"  Chunk ID: {res.chunk_id}")
        print(f"  Title:    {res.title}")
        print(f"  Heading:  {res.heading_path}")
        print(f"  URL:      {res.url}")
        snippet = res.text.replace("\n", " ")[:150]
        print(f"  Snippet:  {snippet}...")


if __name__ == "__main__":
    main()
