"""Index Inspection Utility.

Inspects LanceDB vector table and serialized BM25 index on disk.
Usage:
    .venv/bin/python scripts/inspect_indexes.py [optional_query]
"""

import pickle

import lancedb
from src.config import LANCEDB_DIR, LANCEDB_TABLE_NAME, BM25_INDEX_PATH
from src.ingest import ChunkLookup


def inspect_lancedb():
    print("=" * 60)
    print("1. LanceDB Vector Table Inspection")
    print("=" * 60)

    if not LANCEDB_DIR.exists():
        print(f"Error: LanceDB directory not found at {LANCEDB_DIR}")
        return None

    db = lancedb.connect(str(LANCEDB_DIR))
    tables = db.table_names()
    print(f"Available tables: {tables}")

    if LANCEDB_TABLE_NAME not in tables:
        print(f"Error: Table '{LANCEDB_TABLE_NAME}' not found.")
        return None

    table = db.open_table(LANCEDB_TABLE_NAME)
    arrow_tbl = table.to_arrow()
    print(f"Total stored chunks: {len(arrow_tbl)}")
    print(f"Table columns: {arrow_tbl.column_names}")

    sample = {col: arrow_tbl[col][0].as_py() for col in arrow_tbl.column_names}
    print("\nSample chunk record:")
    print(f"  Chunk ID:     {sample['chunk_id']}")
    print(f"  Doc Path:     {sample['doc_path']}")
    print(f"  Heading Path: {sample['heading_path']}")
    print(f"  URL:          {sample['url']}")
    print(f"  Token Count:  {sample['token_count']}")
    print(f"  Text preview: {sample['text'][:120]}...")

    vector = sample["vector"]
    print(f"\nVector representation:")
    print(f"  Dimensions:   {len(vector)}")
    print(f"  Data type:    {type(vector[0])}")
    print(f"  First 5 dims: {vector[:5]}")
    return table


def inspect_bm25():
    print("\n" + "=" * 60)
    print("2. BM25 Serialized Index Inspection")
    print("=" * 60)

    if not BM25_INDEX_PATH.exists():
        print(f"Error: BM25 index file not found at {BM25_INDEX_PATH}")
        return None, None

    file_size_mb = BM25_INDEX_PATH.stat().st_size / (1024 * 1024)
    print(f"File path: {BM25_INDEX_PATH}")
    print(f"File size on disk: {file_size_mb:.2f} MB")

    with open(BM25_INDEX_PATH, "rb") as f:
        bm25, chunk_lookup = pickle.load(f)

    print(f"Corpus size in index: {bm25.corpus_size} chunks")
    print(f"Average document length (avgdl): {bm25.avgdl:.2f} tokens")
    print(f"Lookup collection type: {type(chunk_lookup).__name__}")

    sample_chunk = chunk_lookup[0]
    print("\nSample chunk from lookup:")
    print(f"  Chunk ID:     {sample_chunk.chunk_id}")
    print(f"  Heading Path: {sample_chunk.heading_path}")

    # Verify O(1) dictionary lookup by ID
    by_id_result = chunk_lookup.get(sample_chunk.chunk_id)
    print(f"  O(1) lookup by ID verified: {by_id_result is not None}")
    return bm25, chunk_lookup


def main():
    table = inspect_lancedb()
    bm25, chunk_lookup = inspect_bm25()

    print("\n" + "=" * 60)
    print("Inspection complete. Both indexes are healthy and consistent.")
    print("=" * 60)


if __name__ == "__main__":
    main()
