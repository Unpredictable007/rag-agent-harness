"""Tests for the Model Context Protocol (MCP) server and search_docs tool."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from src.mcp_server import (
    format_retrieval_results,
    main as cli_main,
    search_docs,
    server,
)
from src.retrieval import RetrievalResult, retrieve

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_search_docs_empty_query():
    """Verify that empty, whitespace, or None queries return descriptive notices without retrieval."""
    msg_empty = search_docs("")
    assert "No search query provided" in msg_empty

    msg_whitespace = search_docs("   \n\t  ")
    assert "No search query provided" in msg_whitespace

    msg_none = search_docs(None)  # type: ignore[arg-type]
    assert "No search query provided" in msg_none


def test_format_retrieval_results_empty():
    """Verify fallback message when no retrieval results are returned."""
    output = format_retrieval_results([])
    assert "No relevant documentation passages found" in output


def test_format_retrieval_results_structure():
    """Verify structured markdown output contains all expected fields and delimiters."""
    dummy_results = [
        RetrievalResult(
            chunk_id="doc1.md#0",
            score=0.9876,
            rank=1,
            text="First passage content snippet about feature flags.",
            url="https://posthog.com/docs/flags",
            heading_path="Feature Flags > Overview",
            title="Feature Flags Guide",
        ),
        RetrievalResult(
            chunk_id="doc2.md#3",
            score=0.8543,
            rank=2,
            text="Second passage content snippet about analytics ingestion.",
            url="https://posthog.com/docs/ingestion",
            heading_path="",
            title="",
        ),
    ]

    output = format_retrieval_results(dummy_results)

    # Check first passage formatting
    assert "### [1] Feature Flags Guide" in output
    assert "- **Chunk ID**: `doc1.md#0`" in output
    assert "- **URL**: https://posthog.com/docs/flags" in output
    assert "- **Heading**: Feature Flags > Overview" in output
    assert "- **Score**: 0.9876" in output
    assert "First passage content snippet about feature flags." in output

    # Check second passage formatting with default fallbacks
    assert "### [2] PostHog Documentation" in output
    assert "- **Chunk ID**: `doc2.md#3`" in output
    assert "- **URL**: https://posthog.com/docs/ingestion" in output
    assert "- **Heading**: Top Level" in output
    assert "- **Score**: 0.8543" in output
    assert "Second passage content snippet about analytics ingestion." in output

    # Check delimiter between blocks
    assert "\n\n---\n\n" in output


@pytest.mark.parametrize(
    ("input_k", "expected_effective_k"),
    [
        (50, 10),      # Upper bound clamping
        (100, 10),     # Extreme upper bound
        (10, 10),      # Exact upper boundary
        (5, 5),        # Standard middle value
        (1, 1),        # Exact lower boundary
        (0, 1),        # Lower bound clamping
        (-5, 1),       # Negative value clamping
        ("invalid", 5),  # Malformed value fallback
        (None, 5),     # None fallback
    ],
)
def test_search_docs_parameter_clamping(input_k, expected_effective_k):
    """Verify poka-yoke parameter clamping: top_k is strictly clamped between 1 and 10."""
    with patch("src.mcp_server.retrieve") as mock_retrieve:
        mock_retrieve.return_value = []
        search_docs(query="feature flags", top_k=input_k)

        mock_retrieve.assert_called_once()
        _, kwargs = mock_retrieve.call_args
        assert kwargs["k"] == expected_effective_k
        assert kwargs["mode"] == "hybrid"
        assert kwargs["rerank"] is True


def test_search_docs_end_to_end_direct():
    """Verify end-to-end direct execution of search_docs against the indexed documentation."""
    output = search_docs(query="feature flags", top_k=3)

    assert "### [1]" in output
    assert "### [2]" in output
    assert "### [3]" in output
    assert "- **Chunk ID**:" in output
    assert "- **URL**:" in output
    assert "- **Score**:" in output


@pytest.mark.anyio
async def test_mcp_server_metadata_and_in_process_call():
    """Verify tool registration, schema definition, and in-memory execution via MCPServer."""
    assert server.name == "posthog-docs-retriever"

    tools = await server.list_tools()
    tool_map = {t.name: t for t in tools}
    assert "search_docs" in tool_map

    tool = tool_map["search_docs"]
    assert tool.description is not None
    assert "Search PostHog technical documentation" in tool.description
    assert "query" in tool.input_schema["properties"]
    assert "top_k" in tool.input_schema["properties"]
    assert "query" in tool.input_schema["required"]

    result = await server.call_tool("search_docs", {"query": "feature flags", "top_k": 2})
    assert not result.is_error
    assert len(result.content) >= 1
    content_text = result.content[0].text
    assert "### [1]" in content_text
    assert "### [2]" in content_text


@pytest.mark.anyio
async def test_mcp_server_stdio_client_integration():
    """Verify complete stdio MCP protocol negotiation, tool discovery, and query execution."""
    server_script = str(REPO_ROOT / "src" / "mcp_server.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)

    params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
        cwd=str(REPO_ROOT),
        env=env,
    )

    # 1. Direct ground-truth retrieval output for parity check
    expected_results = retrieve(query="feature flags", mode="hybrid", k=2, rerank=True)
    expected_chunk_ids = [r.chunk_id for r in expected_results]

    # 2. Client interaction over stdio transport
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Tool discovery
            tools_response = await session.list_tools()
            tool_names = [t.name for t in tools_response.tools]
            assert "search_docs" in tool_names

            # Tool execution
            call_result = await session.call_tool(
                "search_docs",
                {"query": "feature flags", "top_k": 2},
            )
            assert not call_result.is_error
            assert len(call_result.content) >= 1

            result_text = call_result.content[0].text
            assert "### [1]" in result_text
            assert "### [2]" in result_text

            # Verify chunk ID parity with direct retrieval
            for chunk_id in expected_chunk_ids:
                assert f"`{chunk_id}`" in result_text


def test_cli_query_mode(capsys):
    """Verify that CLI execution with --query outputs formatted results to stdout."""
    with patch("sys.argv", ["mcp_server.py", "--query", "feature flags", "--top-k", "2"]):
        with patch("src.mcp_server.search_docs", return_value="### Mocked CLI Output") as mock_search:
            cli_main()
            mock_search.assert_called_once_with(query="feature flags", top_k=2)
            captured = capsys.readouterr()
            assert "### Mocked CLI Output" in captured.out


def test_cli_server_run_mode():
    """Verify that CLI execution without --query defaults to server.run with stdio transport."""
    with patch("sys.argv", ["mcp_server.py"]):
        with patch.object(server, "run") as mock_run:
            cli_main()
            mock_run.assert_called_once_with(transport="stdio")

