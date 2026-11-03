import pytest
import re
from pathlib import Path
from pydantic import BaseModel

from src.ingest import (
    DocumentChunk,
    ChunkLookup,
    clean_mdx,
    chunk_document,
    derive_doc_url,
    estimate_tokens,
    split_paragraphs_preserving_code,
    split_long_section,
    build_indexes,
)


def test_clean_mdx_frontmatter_extraction():
    raw = """---
title: Getting Started with Flags
description: Quickstart guide to feature flags
availability:
  free: full
---

# Getting Started
Some introductory content here.
"""
    meta, cleaned = clean_mdx(raw)
    assert meta.get("title") == "Getting Started with Flags"
    assert meta.get("description") == "Quickstart guide to feature flags"
    assert "# Getting Started" in cleaned
    assert "---" not in cleaned


def test_clean_mdx_jsx_and_callout_handling():
    raw = """---
title: Test Title
---

import { CalloutBox } from "components/Docs/CalloutBox";
import { ProductScreenshot } from "components/ProductScreenshot";
export const ScreenshotUrl = "https://example.com/img.png";

<!-- HTML comment to be removed -->

<ProductScreenshot
  imageLight={ScreenshotUrl}
  alt="Sample screenshot"
/>

<CalloutBox icon="IconInfo" title="Flag Limits" type="fyi">
Each team can have a maximum of 2,000 non-deleted feature flags.
</CalloutBox>
"""
    meta, cleaned = clean_mdx(raw)
    assert meta.get("title") == "Test Title"
    assert "CalloutBox" not in cleaned
    assert "ProductScreenshot" not in cleaned
    assert "ScreenshotUrl" not in cleaned
    assert "HTML comment to be removed" not in cleaned
    assert "import" not in cleaned
    assert "export" not in cleaned
    assert "Each team can have a maximum of 2,000 non-deleted feature flags." in cleaned


def test_clean_mdx_code_block_preservation():
    raw = """---
title: Code Sample
---

Here is a code example:

```tsx
import React from 'react';
// This comment must remain intact
export function MyComponent() {
  return <CalloutBox>Hello from JSX</CalloutBox>;
}
```

Regular text after code block.
"""
    meta, cleaned = clean_mdx(raw)
    assert "```tsx" in cleaned
    assert "import React from 'react';" in cleaned
    assert "// This comment must remain intact" in cleaned
    assert "<CalloutBox>Hello from JSX</CalloutBox>" in cleaned
    assert "Regular text after code block." in cleaned


def test_chunk_document_heading_splitting_and_breadcrumbs():
    doc_path = "feature-flags/rollout.mdx"
    url = "https://posthog.com/docs/feature-flags/rollout"
    cleaned = """Introduction overview text.

## Setting Up Flags

Configure your flags in the dashboard.

### Percentage Rollout

Use percentage targeting to gradually release features.

### User Group Targeting

Filter users based on custom properties.

## Best Practices

Always clean up stale flags.
"""
    metadata = {"title": "Feature Flags"}
    chunks = chunk_document(doc_path, url, cleaned, metadata)

    assert len(chunks) == 5

    # Chunk 0: Overview preamble
    assert chunks[0].chunk_id == "feature-flags/rollout.mdx#0"
    assert chunks[0].heading_path == "Feature Flags"
    assert chunks[0].title == "Feature Flags"
    assert "Introduction overview text." in chunks[0].text

    # Chunk 1: H2 section
    assert chunks[1].chunk_id == "feature-flags/rollout.mdx#1"
    assert chunks[1].heading_path == "Feature Flags > Setting Up Flags"
    assert chunks[1].title == "Setting Up Flags"
    assert "Configure your flags in the dashboard." in chunks[1].text

    # Chunk 2: H3 subsection
    assert chunks[2].chunk_id == "feature-flags/rollout.mdx#2"
    assert chunks[2].heading_path == "Feature Flags > Setting Up Flags > Percentage Rollout"
    assert chunks[2].title == "Percentage Rollout"
    assert "Use percentage targeting" in chunks[2].text

    # Chunk 3: H3 subsection
    assert chunks[3].chunk_id == "feature-flags/rollout.mdx#3"
    assert chunks[3].heading_path == "Feature Flags > Setting Up Flags > User Group Targeting"
    assert chunks[3].title == "User Group Targeting"

    # Chunk 4: Second H2 section
    assert chunks[4].chunk_id == "feature-flags/rollout.mdx#4"
    assert chunks[4].heading_path == "Feature Flags > Best Practices"
    assert chunks[4].title == "Best Practices"


def test_chunk_document_code_block_integrity():
    doc_path = "tutorials/python-capture.mdx"
    url = "https://posthog.com/docs/tutorials/python-capture"
    cleaned = """## Python Setup

Here is how you capture events:

```python
# Note: ## not a header inside code fence
from posthog import Posthog

posthog = Posthog('API_KEY', host='https://us.i.posthog.com')
### Also not a header inside code fence
posthog.capture('distinct_id', 'test_event')
```

End of setup section.
"""
    metadata = {"title": "Python SDK"}
    chunks = chunk_document(doc_path, url, cleaned, metadata)

    # Should only create 1 chunk for the section, not split on ## inside code
    assert len(chunks) == 1
    assert chunks[0].heading_path == "Python SDK > Python Setup"
    assert "from posthog import Posthog" in chunks[0].text
    assert "posthog.capture('distinct_id', 'test_event')" in chunks[0].text
    assert "### Also not a header inside code fence" in chunks[0].text


def test_split_long_section_with_overlap():
    paragraphs = [f"Paragraph {i}: " + ("text " * 50) for i in range(8)]
    section_text = "\n\n".join(paragraphs)

    # Max tokens 200, overlap 50
    chunks = split_long_section(section_text, max_tokens=200, overlap_tokens=50)
    assert len(chunks) > 1

    # Ensure adjacent chunks overlap
    for i in range(len(chunks) - 1):
        first_chunk_paras = chunks[i].split("\n\n")
        second_chunk_paras = chunks[i + 1].split("\n\n")
        # The last paragraph of chunk i should match the first paragraph of chunk i+1
        assert first_chunk_paras[-1] == second_chunk_paras[0]


def test_derive_doc_url():
    assert derive_doc_url("feature-flags/rollout.mdx") == "https://posthog.com/docs/feature-flags/rollout"
    assert derive_doc_url("data-warehouse/index.mdx") == "https://posthog.com/docs/data-warehouse"
    assert derive_doc_url("index.mdx") == "https://posthog.com/docs"
    assert derive_doc_url("api/identify.md") == "https://posthog.com/docs/api/identify"


def test_build_indexes_and_lookup(tmp_path):
    chunks = [
        DocumentChunk(
            chunk_id="doc1.mdx#0",
            doc_path="doc1.mdx",
            url="https://posthog.com/docs/doc1",
            heading_path="Doc 1 > Section A",
            title="Section A",
            text="How to create a feature flag in PostHog application.",
            token_count=10,
        ),
        DocumentChunk(
            chunk_id="doc2.mdx#0",
            doc_path="doc2.mdx",
            url="https://posthog.com/docs/doc2",
            heading_path="Doc 2 > Section B",
            title="Section B",
            text="Setting up session replay recording and masking sensitive data.",
            token_count=10,
        ),
        DocumentChunk(
            chunk_id="doc3.mdx#0",
            doc_path="doc3.mdx",
            url="https://posthog.com/docs/doc3",
            heading_path="Doc 3 > Section C",
            title="Section C",
            text="Configuring data warehouse sources and sync schedules.",
            token_count=10,
        ),
    ]

    lancedb_dir = tmp_path / "lancedb"
    bm25_path = tmp_path / "bm25.pkl"

    table, bm25, chunk_lookup = build_indexes(
        chunks=chunks,
        lancedb_dir=lancedb_dir,
        bm25_path=bm25_path,
        table_name="test_docs",
    )

    # Verify chunk_lookup acts as both list and dict
    assert len(chunk_lookup) == 3
    assert chunk_lookup[0].chunk_id == "doc1.mdx#0"
    assert chunk_lookup["doc2.mdx#0"].title == "Section B"
    assert chunk_lookup.get("non_existent") is None

    # Test BM25 query
    query_tokens = ["feature", "flag"]
    scores = bm25.get_scores(query_tokens)
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]

    # Test LanceDB search
    import numpy as np
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    q_vec = embed_model.encode("feature flag creation", normalize_embeddings=True).tolist()
    results = table.search(q_vec).limit(1).to_list()
    assert len(results) == 1
    assert results[0]["chunk_id"] == "doc1.mdx#0"
