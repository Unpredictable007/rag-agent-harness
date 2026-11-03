import os
import re
import time
import pickle
from pathlib import Path
from typing import Any
import yaml
import numpy as np
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import lancedb

from src.config import (
    RAW_DOCS_DIR,
    LANCEDB_DIR,
    LANCEDB_TABLE_NAME,
    BM25_INDEX_PATH,
    CHUNK_MIN_TOKENS,
    CHUNK_MAX_TOKENS,
    OVERLAP_TOKENS,
    EMBEDDING_MODEL_NAME,
)


class DocumentChunk(BaseModel):
    chunk_id: str = Field(description="Deterministic ID: f'{doc_path}#{section_idx}'")
    doc_path: str = Field(description="Relative path of source file, e.g. feature-flags/rollout.mdx")
    url: str = Field(description="Canonical documentation URL")
    heading_path: str = Field(description="Breadcrumb trail, e.g. 'Feature Flags > Percentage Rollout'")
    title: str = Field(description="Document title or section heading")
    text: str = Field(description="Cleaned markdown text with preserved code blocks")
    token_count: int = Field(description="Estimated token count")


class ChunkLookup(list):
    """List of DocumentChunk supporting indexing by integer index and by chunk_id string."""

    def __init__(self, chunks: list[DocumentChunk]):
        super().__init__(chunks)
        self._by_id: dict[str, DocumentChunk] = {c.chunk_id: c for c in chunks}

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self._by_id[key]
        return super().__getitem__(key)

    def get(self, chunk_id: str, default: Any = None) -> Any:
        return self._by_id.get(chunk_id, default)


def estimate_tokens(text: str) -> int:
    """Estimate token count for English technical text using standard 4 chars/token heuristic."""
    return max(1, len(text) // 4)


def derive_doc_url(rel_path: str) -> str:
    """Derive canonical documentation URL from relative path."""
    clean_path = rel_path.replace("\\", "/")
    for ext in [".mdx", ".md"]:
        if clean_path.endswith(ext):
            clean_path = clean_path[:-len(ext)]
            break

    if clean_path.endswith("/index"):
        clean_path = clean_path[:-6]
    elif clean_path == "index":
        clean_path = ""

    if clean_path:
        return f"https://posthog.com/docs/{clean_path}"
    return "https://posthog.com/docs"


def clean_mdx(raw_content: str) -> tuple[dict, str]:
    """
    Clean MDX content:
    - Extract YAML frontmatter (title, description, etc.).
    - Preserve fenced code blocks completely.
    - Strip HTML comments and JS/TS import and export statements.
    - Remove JSX self-closing and container tags while preserving inner text.
    - Convert any non-standard dashes to standard hyphens.
    """
    # Replace em dash and en dash with standard hyphen
    content = raw_content.replace("\u2014", "-").replace("\u2013", "-")

    metadata: dict = {}
    text = content

    # 1. Extract YAML frontmatter
    fm_match = re.match(r"^---\s*\n([\s\S]*?)\n---\s*\n?", content)
    if fm_match:
        try:
            parsed = yaml.safe_load(fm_match.group(1))
            if isinstance(parsed, dict):
                metadata = parsed
        except Exception:
            metadata = {}
        text = content[fm_match.end():]

    # 2. Extract and preserve fenced code blocks
    code_blocks: list[str] = []

    def mask_code(match: re.Match) -> str:
        idx = len(code_blocks)
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{idx}__"

    text = re.sub(r"(```+[\s\S]*?```+|~~~+[\s\S]*?~~~+)", mask_code, text)

    # 3. Strip HTML comments
    text = re.sub(r"<!--[\s\S]*?-->", "", text)

    # 4. Strip import and export statements outside code blocks
    text = re.sub(r"^\s*import\b[\s\S]*?from\s+[\x27\x22][^\x27\x22]+[\x27\x22];?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*import\s+[\x27\x22][^\x27\x22]+[\x27\x22];?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*export\s+(?:const|let|var|function|default|type|interface)\s+[^;]+;\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*export\s+[\s\S]*?;\s*$", "", text, flags=re.MULTILINE)

    # 5. Remove JSX tags outside code blocks
    # Self-closing tags: <Tag ... />
    text = re.sub(r"<[A-Z][\w\d]*[^>]*/>", "", text)
    # Opening container tags: <Tag ...>
    text = re.sub(r"<[A-Z][\w\d]*[^>]*>", "", text)
    # Closing container tags: </Tag>
    text = re.sub(r"</[A-Z][\w\d]*>", "", text)

    # 6. Restore preserved code blocks
    for idx, cb in enumerate(code_blocks):
        text = text.replace(f"__CODE_BLOCK_{idx}__", cb)

    # 7. Normalize excessive whitespace and blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return metadata, text


def split_paragraphs_preserving_code(text: str) -> list[str]:
    """Split markdown text on double-newlines while keeping fenced code blocks atomic."""
    code_blocks: list[str] = []

    def mask_code(match: re.Match) -> str:
        idx = len(code_blocks)
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{idx}__"

    masked = re.sub(r"(```+[\s\S]*?```+|~~~+[\s\S]*?~~~+)", mask_code, text)
    raw_paras = [p.strip() for p in re.split(r"\n\s*\n", masked) if p.strip()]

    paragraphs: list[str] = []
    for p in raw_paras:
        restored = p
        for idx, cb in enumerate(code_blocks):
            restored = restored.replace(f"__CODE_BLOCK_{idx}__", cb)
        paragraphs.append(restored)

    return paragraphs


def split_long_section(
    section_text: str,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[str]:
    """Split section exceeding max_tokens on paragraph boundaries with sliding overlap."""
    if estimate_tokens(section_text) <= max_tokens:
        return [section_text]

    paragraphs = split_paragraphs_preserving_code(section_text)
    if len(paragraphs) <= 1:
        return [section_text]

    chunks: list[str] = []
    current_paras: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tok = estimate_tokens(para)
        if current_paras and (current_tokens + para_tok > max_tokens):
            chunks.append("\n\n".join(current_paras).strip())

            # Select overlap paragraphs from the tail of current_paras
            overlap_paras: list[str] = []
            overlap_count = 0
            for p in reversed(current_paras):
                p_tok = estimate_tokens(p)
                if overlap_count + p_tok <= overlap_tokens or not overlap_paras:
                    overlap_paras.insert(0, p)
                    overlap_count += p_tok
                else:
                    break

            # Avoid full duplication if overlap equals all current_paras
            if len(overlap_paras) >= len(current_paras):
                overlap_paras = []
                overlap_count = 0

            current_paras = overlap_paras + [para]
            current_tokens = overlap_count + para_tok
        else:
            current_paras.append(para)
            current_tokens += para_tok

    if current_paras:
        final_text = "\n\n".join(current_paras).strip()
        if not chunks or final_text != chunks[-1]:
            chunks.append(final_text)

    return chunks


def chunk_document(
    doc_path: str,
    url: str,
    cleaned_text: str,
    metadata: dict | None = None,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[DocumentChunk]:
    """
    Split a cleaned document into semantic chunks using heading hierarchy (H2, H3).
    Each chunk inherits hierarchical breadcrumbs and code blocks remain atomic.
    """
    if metadata is None:
        metadata = {}

    title = str(metadata.get("title") or "").strip()
    if not title:
        h1_match = re.search(r"^#\s+(.+)$", cleaned_text, re.MULTILINE)
        if h1_match:
            title = h1_match.group(1).strip()
        else:
            title = Path(doc_path).stem.replace("-", " ").title()

    lines = cleaned_text.splitlines()
    in_code = False
    raw_sections: list[tuple[str | None, str | None, list[str]]] = []
    current_h2: str | None = None
    current_h3: str | None = None
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code = not in_code
            current_lines.append(line)
            continue

        if not in_code and line.startswith("## "):
            if current_lines:
                raw_sections.append((current_h2, current_h3, current_lines))
            current_h2 = line[3:].strip()
            current_h3 = None
            current_lines = [line]
        elif not in_code and line.startswith("### "):
            if current_lines:
                raw_sections.append((current_h2, current_h3, current_lines))
            current_h3 = line[4:].strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        raw_sections.append((current_h2, current_h3, current_lines))

    chunks: list[DocumentChunk] = []
    section_idx = 0

    for h2, h3, s_lines in raw_sections:
        sec_text = "\n".join(s_lines).strip()
        if not sec_text:
            continue

        # Construct breadcrumb heading path
        parts = [title]
        if h2:
            parts.append(h2)
        if h3:
            parts.append(h3)
        heading_path = " > ".join(parts)
        sec_title = h3 if h3 else (h2 if h2 else title)

        sub_texts = split_long_section(sec_text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        for sub_text in sub_texts:
            clean_sub = sub_text.strip()
            if not clean_sub:
                continue
            token_count = estimate_tokens(clean_sub)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_path}#{section_idx}",
                    doc_path=doc_path,
                    url=url,
                    heading_path=heading_path,
                    title=sec_title,
                    text=clean_sub,
                    token_count=token_count,
                )
            )
            section_idx += 1

    return chunks


def build_indexes(
    chunks: list[DocumentChunk],
    lancedb_dir: Path = LANCEDB_DIR,
    bm25_path: Path = BM25_INDEX_PATH,
    table_name: str = LANCEDB_TABLE_NAME,
) -> tuple[Any, BM25Okapi, ChunkLookup]:
    """
    Build dual indexes:
    1. LanceDB dense vector index using BAAI/bge-small-en-v1.5 with normalized embeddings.
    2. BM25Okapi sparse lexical index serialized with ChunkLookup.
    """
    if not chunks:
        raise ValueError("Cannot build index on empty chunk list.")

    print(f"Building indexes for {len(chunks)} chunks...")
    t0 = time.time()

    # 1. Dense Index (LanceDB)
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [c.text for c in chunks]
    print(f"Encoding {len(texts)} texts in batches...")
    t_embed_start = time.time()
    embeddings = embed_model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"Embeddings generated in {time.time() - t_embed_start:.2f}s")

    lancedb_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(lancedb_dir))

    records = []
    for chunk, vector in zip(chunks, embeddings):
        records.append({
            "vector": vector,
            "chunk_id": chunk.chunk_id,
            "doc_path": chunk.doc_path,
            "url": chunk.url,
            "heading_path": chunk.heading_path,
            "title": chunk.title,
            "text": chunk.text,
            "token_count": chunk.token_count,
        })

    print(f"Creating LanceDB table '{table_name}'...")
    table = db.create_table(table_name, data=records, mode="overwrite")
    print(f"LanceDB table created with {len(table)} records.")

    # 2. Sparse Index (BM25Okapi)
    print("Building BM25 sparse index...")
    t_bm25_start = time.time()
    tokenized_corpus = [re.findall(r"\w+", c.text.lower()) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    chunk_lookup = ChunkLookup(chunks)

    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump((bm25, chunk_lookup), f)
    print(f"BM25 index saved to {bm25_path} in {time.time() - t_bm25_start:.2f}s")

    print(f"Index build complete in {time.time() - t0:.2f}s.")
    return table, bm25, chunk_lookup


def ingest_all(
    docs_dir: Path = RAW_DOCS_DIR,
    lancedb_dir: Path = LANCEDB_DIR,
    bm25_path: Path = BM25_INDEX_PATH,
) -> tuple[list[DocumentChunk], Any, BM25Okapi, ChunkLookup]:
    """Scan docs directory, parse, chunk, and build dual indexes."""
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documentation directory not found: {docs_dir}")

    t_start = time.time()
    all_files = [
        p for p in docs_dir.rglob("*")
        if p.suffix in [".md", ".mdx"] and not any(part.startswith("_") for part in p.parts)
    ]
    all_files.sort()
    print(f"Found {len(all_files)} markdown files in {docs_dir}.")

    all_chunks: list[DocumentChunk] = []
    for file_path in all_files:
        try:
            rel_path = str(file_path.relative_to(docs_dir))
            url = derive_doc_url(rel_path)
            raw_content = file_path.read_text(encoding="utf-8", errors="ignore")
            metadata, cleaned = clean_mdx(raw_content)
            chunks = chunk_document(
                doc_path=rel_path,
                url=url,
                cleaned_text=cleaned,
                metadata=metadata,
            )
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Warning: Failed to process {file_path}: {e}")

    t_chunked = time.time()
    print(f"Successfully generated {len(all_chunks)} chunks from {len(all_files)} files in {t_chunked - t_start:.2f}s.")

    table, bm25, chunk_lookup = build_indexes(
        all_chunks,
        lancedb_dir=lancedb_dir,
        bm25_path=bm25_path,
    )
    t_total = time.time() - t_start
    print(f"Ingestion finished in {t_total:.2f}s total.")
    return all_chunks, table, bm25, chunk_lookup


def run_sanity_checks(
    table: Any,
    bm25: BM25Okapi,
    chunk_lookup: ChunkLookup,
) -> None:
    """Run 5 sanity check queries against both BM25 and LanceDB and print top matches."""
    queries = [
        "feature flag creation",
        "distinct_id capture python",
        "session replay masking",
        "warehouse sync properties",
        "identify API",
    ]

    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("\n" + "=" * 60)
    print("SANITY CHECKS: Top retrieval match per query")
    print("=" * 60)

    for q in queries:
        print(f"\nQuery: '{q}'")

        # 1. BM25 search
        t0 = time.time()
        tokens = re.findall(r"\w+", q.lower())
        bm25_scores = bm25.get_scores(tokens)
        top_bm25_idx = int(np.argmax(bm25_scores))
        bm25_chunk = chunk_lookup[top_bm25_idx]
        bm25_lat = (time.time() - t0) * 1000
        print(f"  [BM25] ({bm25_lat:.2f}ms, score={bm25_scores[top_bm25_idx]:.2f})")
        print(f"    ID:      {bm25_chunk.chunk_id}")
        print(f"    Heading: {bm25_chunk.heading_path}")
        print(f"    URL:     {bm25_chunk.url}")
        snippet = bm25_chunk.text.replace("\n", " ")[:120]
        print(f"    Snippet: {snippet}...")

        # 2. LanceDB dense search
        t1 = time.time()
        q_vec = embed_model.encode(q, normalize_embeddings=True).tolist()
        dense_results = table.search(q_vec).limit(1).to_list()
        dense_lat = (time.time() - t1) * 1000
        if dense_results:
            top_dense = dense_results[0]
            print(f"  [LanceDB] ({dense_lat:.2f}ms, dist={top_dense.get('_distance', 0.0):.4f})")
            print(f"    ID:      {top_dense['chunk_id']}")
            print(f"    Heading: {top_dense['heading_path']}")
            print(f"    URL:     {top_dense['url']}")
            dense_snippet = top_dense["text"].replace("\n", " ")[:120]
            print(f"    Snippet: {dense_snippet}...")
        else:
            print("  [LanceDB] No result found.")

    print("\n" + "=" * 60)
    print("Sanity checks complete.")
    print("=" * 60)


if __name__ == "__main__":
    import src.ingest as self_module

    chunks, table, bm25, chunk_lookup = self_module.ingest_all()
    self_module.run_sanity_checks(table, bm25, chunk_lookup)
