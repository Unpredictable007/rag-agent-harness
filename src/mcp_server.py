"""Model Context Protocol (MCP) server exposing PostHog documentation search.

This module implements the `posthog-docs-retriever` MCP server equipped with the
`search_docs` tool, exposing the E3 production baseline retriever (Hybrid RRF
+ Cross-Encoder reranking) for agentic and tool-calling consumption.
"""

from __future__ import annotations

import argparse
from typing import Any

try:
    from mcp.server import MCPServer
except ImportError:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP as MCPServer

from src.retrieval import RetrievalResult, retrieve

server = MCPServer("posthog-docs-retriever")


def format_retrieval_results(results: list[RetrievalResult]) -> str:
    """Format a list of RetrievalResult objects into structured Markdown passages."""
    if not results:
        return "No relevant documentation passages found. Please try refining or expanding your search query."

    passages: list[str] = []
    for res in results:
        title_display = res.title if res.title else "PostHog Documentation"
        heading_display = res.heading_path if res.heading_path else "Top Level"
        passage = (
            f"### [{res.rank}] {title_display}\n"
            f"- **Chunk ID**: `{res.chunk_id}`\n"
            f"- **URL**: {res.url}\n"
            f"- **Heading**: {heading_display}\n"
            f"- **Score**: {res.score:.4f}\n\n"
            f"{res.text.strip()}"
        )
        passages.append(passage)

    return "\n\n---\n\n".join(passages)


@server.tool()
def search_docs(query: str, top_k: int = 5) -> str:
    """Search PostHog technical documentation for relevant guides, APIs, SDKs, and concepts.

    Use this tool whenever you need verified technical information, code examples,
    configuration options, feature flag details, or troubleshooting steps from the official
    PostHog documentation.

    Args:
        query: The search query describing the technical topic, API, or question
            (e.g., 'how to set up feature flags in python', 'event ingestion rate limits').
        top_k: The number of relevant documentation passages to return. Defaults to 5.
            Values are automatically clamped between 1 and 10 to preserve model context limits.

    Returns:
        Structured Markdown containing ranked documentation passages with chunk IDs,
        canonical URLs, heading paths, relevance scores, and content snippets.
        Returns a fallback notice if no relevant passages are found or the query is empty.
    """
    query_clean = query.strip() if query else ""
    if not query_clean:
        return "No search query provided. Please specify a non-empty query string."

    try:
        parsed_k = int(top_k)
    except (ValueError, TypeError):
        parsed_k = 5

    effective_k = max(1, min(10, parsed_k))

    results = retrieve(query=query_clean, mode="hybrid", k=effective_k, rerank=True)
    return format_retrieval_results(results)


def main() -> None:
    """CLI entrypoint for running the MCP server or executing a direct query."""
    parser = argparse.ArgumentParser(
        description="PostHog Documentation MCP Server (search_docs)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Execute a one-off query test and print formatted markdown to stdout",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve for one-off CLI query (default: 5, max: 10)",
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio"],
        help="Transport protocol to use for serving (default: stdio)",
    )
    args = parser.parse_args()

    if args.query is not None:
        result = search_docs(query=args.query, top_k=args.top_k)
        print(result)
    else:
        server.run(transport=args.transport)


if __name__ == "__main__":
    main()
