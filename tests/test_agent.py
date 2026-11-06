"""Comprehensive tests for Milestone M9: Minimal ReAct Agent Loop.

All tests are completely offline and mocked to guarantee $0.00 API spend,
strict adherence to step budgets, duplicate query interception (loop guard),
citation validation, and replayable transcript persistence.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.agent import (
    AgentResult,
    arun_agent,
    convert_mcp_tools_to_anthropic,
    extract_chunk_ids_from_tool_text,
    log_trajectory_transcript,
    main as agent_cli_main,
    run_agent,
    serialize_messages,
)
from src.config import REPO_ROOT
from src.generation import GenerationOutput, extract_json_payload, parse_generation_output

TEST_TMP_DIR = REPO_ROOT / "tests" / "tmp"


@pytest.fixture(autouse=True)
def clean_test_tmp_dir():
    """Ensure project-local tmp directory exists and is cleaned up (adheres to D-043)."""
    TEST_TMP_DIR.mkdir(parents=True, exist_ok=True)
    yield TEST_TMP_DIR
    if TEST_TMP_DIR.exists():
        shutil.rmtree(TEST_TMP_DIR, ignore_errors=True)


# --- Helper Mock Classes for Anthropic Messages API ---


class MockTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class MockToolUseBlock:
    def __init__(self, tool_id: str, name: str, input_dict: dict[str, Any]):
        self.type = "tool_use"
        self.id = tool_id
        self.name = name
        self.input = input_dict


class MockUsage:
    def __init__(self, input_tokens: int = 15, output_tokens: int = 25):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockAnthropicResponse:
    def __init__(
        self,
        content: list[Any],
        stop_reason: str = "end_turn",
        usage: MockUsage | None = None,
    ):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage or MockUsage()


# --- Helper Mock Classes for MCP ClientSession ---


class MockMCPTool:
    def __init__(self, name: str, description: str, input_schema: dict[str, Any]):
        self.name = name
        self.description = description
        self.input_schema = input_schema


class MockListToolsResult:
    def __init__(self, tools: list[Any]):
        self.tools = tools


class MockClientSession:
    def __init__(self, responses_by_query: dict[str, str] | None = None):
        self.responses_by_query = responses_by_query or {}
        self.call_log: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> MockListToolsResult:
        return MockListToolsResult(
            [
                MockMCPTool(
                    name="search_docs",
                    description="Search PostHog documentation for relevant passages.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "default": 5},
                        },
                        "required": ["query"],
                    },
                )
            ]
        )

    async def call_tool(self, name: str, tool_input: dict[str, Any]) -> Any:
        self.call_log.append((name, tool_input))
        query = str(tool_input.get("query", "")).strip().lower()

        if query in self.responses_by_query:
            text = self.responses_by_query[query]
        else:
            text = (
                f"### [1] Default PostHog Doc for '{query}'\n"
                f"- **Chunk ID**: `docs/{query}.md#0`\n"
                f"- **URL**: https://posthog.com/docs/{query}\n"
                f"- **Heading**: Overview\n"
                f"- **Score**: 0.9500\n\n"
                f"Content retrieved for technical query: {query}"
            )

        mock_content = MagicMock()
        mock_content.text = text
        return MagicMock(content=[mock_content], is_error=False)


# --- 1. Tool Schema Conversion Tests ---


def test_convert_mcp_tools_to_anthropic():
    """Verify conversion of MCP tool objects to Anthropic Messages API tool schema."""
    mcp_tools = [
        MockMCPTool(
            name="search_docs",
            description="Search PostHog docs",
            input_schema={"properties": {"query": {"type": "string"}}, "required": ["query"]},
        ),
        MockMCPTool(
            name="lookup_api",
            description="",
            input_schema={"properties": {"id": {"type": "integer"}}},
        ),
    ]

    anthropic_tools = convert_mcp_tools_to_anthropic(mcp_tools)
    assert len(anthropic_tools) == 2
    assert anthropic_tools[0] == {
        "name": "search_docs",
        "description": "Search PostHog docs",
        "input_schema": {"properties": {"query": {"type": "string"}}, "required": ["query"]},
    }
    assert anthropic_tools[1]["name"] == "lookup_api"
    assert anthropic_tools[1]["description"] == ""


# --- 2. Chunk ID Extraction Tests ---


def test_extract_chunk_ids_from_tool_text():
    """Verify regex extraction of chunk IDs from structured tool markdown outputs."""
    sample_text = (
        "### [1] Feature Flags\n"
        "- **Chunk ID**: `libraries/python/index.mdx#3`\n"
        "- **URL**: https://posthog.com/docs/flags\n\n"
        "Some passage content.\n\n---\n\n"
        "### [2] Ingestion Limits\n"
        "- **Chunk ID**: `api/ingest.md#0`\n"
        "- **URL**: https://posthog.com/docs/api\n\n"
        "More content."
    )
    extracted = extract_chunk_ids_from_tool_text(sample_text)
    assert extracted == ["libraries/python/index.mdx#3", "api/ingest.md#0"]


def test_extract_chunk_ids_empty_or_malformed():
    """Verify empty list when tool text contains no chunk IDs or non-matching formats."""
    assert extract_chunk_ids_from_tool_text("No results found.") == []
    assert extract_chunk_ids_from_tool_text("- Chunk ID: missing_backticks") == []


# --- 3. JSON Extraction Resilience Tests ---


def test_json_extraction_resilience():
    """Verify JSON extraction across raw JSON, fenced markdown blocks, and surrounding chatter."""
    # Clean JSON
    raw_clean = '{"answer": "Use posthog.capture()", "citations": ["doc1.md#0"], "abstain": false}'
    parsed1 = parse_generation_output(raw_clean)
    assert parsed1.answer == "Use posthog.capture()"
    assert parsed1.citations == ["doc1.md#0"]
    assert not parsed1.abstain

    # Fenced markdown codeblock
    raw_fenced = (
        "```json\n"
        '{\n  "answer": "Feature flags can be checked with is_feature_enabled()",\n'
        '  "citations": ["flags.md#1"],\n  "abstain": false\n}\n'
        "```"
    )
    parsed2 = parse_generation_output(raw_fenced)
    assert "is_feature_enabled" in parsed2.answer
    assert parsed2.citations == ["flags.md#1"]

    # Surrounding conversational text
    raw_conversational = (
        "Based on the official documentation, here is your answer:\n\n"
        '{"answer": "PostHog recommends 500 token chunking.", "citations": [], "abstain": false}\n\n'
        "Let me know if you need more details!"
    )
    parsed3 = parse_generation_output(raw_conversational)
    assert "500 token chunking" in parsed3.answer


# --- 4. Single-Turn Agent Trajectory (Direct Answer Without Tools) ---


@pytest.mark.anyio
async def test_agent_single_turn_direct_answer(clean_test_tmp_dir):
    """Verify agent handling when model answers directly without calling tools."""
    mock_client = MagicMock()
    mock_response = MockAnthropicResponse(
        content=[
            MockTextBlock(
                '{"answer": "PostHog is an open-source product analytics suite.", '
                '"citations": [], "abstain": false}'
            )
        ],
        stop_reason="end_turn",
        usage=MockUsage(input_tokens=50, output_tokens=30),
    )
    mock_client.messages.create.return_value = mock_response

    mock_session = MockClientSession()
    transcripts_path = clean_test_tmp_dir / "transcripts.jsonl"

    result = await arun_agent(
        task="What is PostHog?",
        anthropic_client=mock_client,
        mcp_session=mock_session,
        transcripts_path=transcripts_path,
        task_id="task_single_turn",
    )

    assert result.task_id == "task_single_turn"
    assert result.steps == 1
    assert result.total_tool_calls == 0
    assert result.output.answer == "PostHog is an open-source product analytics suite."
    assert not result.output.abstain
    assert result.status == "completed"
    assert result.tokens_in == 50
    assert result.tokens_out == 30
    assert len(mock_session.call_log) == 0

    # Verify transcript file
    assert transcripts_path.exists()
    transcript_lines = transcripts_path.read_text().strip().split("\n")
    assert len(transcript_lines) == 1
    logged_data = json.loads(transcript_lines[0])
    assert logged_data["task_id"] == "task_single_turn"
    assert logged_data["total_tool_calls"] == 0


# --- 5. Multi-Turn Trajectory with Tool Calls & Citation Grounding ---


@pytest.mark.anyio
async def test_agent_multi_turn_with_tool_call_and_valid_citation(clean_test_tmp_dir):
    """Verify complete ReAct cycle: tool call, context assimilation, and valid citation output."""
    mock_client = MagicMock()

    # Step 1: Model requests tool call
    step1_resp = MockAnthropicResponse(
        content=[
            MockTextBlock("I need to search for feature flags in the Python SDK."),
            MockToolUseBlock(
                tool_id="call_123",
                name="search_docs",
                input_dict={"query": "python feature flags", "top_k": 3},
            ),
        ],
        stop_reason="tool_use",
        usage=MockUsage(input_tokens=60, output_tokens=30),
    )

    # Step 2: Model synthesizes answer using retrieved chunk ID
    step2_resp = MockAnthropicResponse(
        content=[
            MockTextBlock(
                '{"answer": "Call posthog.is_feature_enabled(key, distinct_id) to evaluate flags.", '
                '"citations": ["docs/python feature flags.md#0"], "abstain": false}'
            )
        ],
        stop_reason="end_turn",
        usage=MockUsage(input_tokens=180, output_tokens=40),
    )

    mock_client.messages.create.side_effect = [step1_resp, step2_resp]
    mock_session = MockClientSession()
    transcripts_path = clean_test_tmp_dir / "transcripts.jsonl"

    result = await arun_agent(
        task="How do I evaluate feature flags in Python?",
        anthropic_client=mock_client,
        mcp_session=mock_session,
        transcripts_path=transcripts_path,
        task_id="task_tool_call",
    )

    assert result.steps == 2
    assert result.total_tool_calls == 1
    assert len(mock_session.call_log) == 1
    assert mock_session.call_log[0] == ("search_docs", {"query": "python feature flags", "top_k": 3})

    # Citations
    assert "docs/python feature flags.md#0" in result.all_retrieved_chunk_ids
    assert result.valid_citations == ["docs/python feature flags.md#0"]
    assert result.hallucinated_citations == []
    assert result.status == "completed"

    # Verify XML prompt boundary parity (<retrieved_context>)
    second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
    tool_result_msg = second_call_kwargs["messages"][-2]
    assert tool_result_msg["role"] == "user"
    assert "<retrieved_context>" in tool_result_msg["content"][0]["content"]
    assert "</retrieved_context>" in tool_result_msg["content"][0]["content"]

    # Verify tool call history entry
    assert len(result.tool_call_history) == 1
    call_rec = result.tool_call_history[0]
    assert call_rec["tool_name"] == "search_docs"
    assert not call_rec["is_duplicate"]
    assert "docs/python feature flags.md#0" in call_rec["chunk_ids"]


# --- 6. Loop Guard: Duplicate Query Interception ---


@pytest.mark.anyio
async def test_agent_loop_guard_duplicate_query_interception(clean_test_tmp_dir):
    """Verify that repeated search queries (case-insensitive stripped) are intercepted without server call."""
    mock_client = MagicMock()

    # Turn 1: Initial query
    turn1_resp = MockAnthropicResponse(
        content=[
            MockToolUseBlock(
                tool_id="call_1",
                name="search_docs",
                input_dict={"query": "event capture rate limit"},
            )
        ],
        stop_reason="tool_use",
    )

    # Turn 2: Duplicate query with whitespace and varying casing
    turn2_resp = MockAnthropicResponse(
        content=[
            MockToolUseBlock(
                tool_id="call_2",
                name="search_docs",
                input_dict={"query": "  EVENT CAPTURE RATE LIMIT  "},
            )
        ],
        stop_reason="tool_use",
    )

    # Turn 3: Final answer
    turn3_resp = MockAnthropicResponse(
        content=[
            MockTextBlock(
                '{"answer": "Standard ingestion rate limits apply per project.", '
                '"citations": ["docs/event capture rate limit.md#0"], "abstain": false}'
            )
        ],
        stop_reason="end_turn",
    )

    mock_client.messages.create.side_effect = [turn1_resp, turn2_resp, turn3_resp]
    mock_session = MockClientSession()
    transcripts_path = clean_test_tmp_dir / "transcripts.jsonl"

    result = await arun_agent(
        task="What is the rate limit for event capture?",
        anthropic_client=mock_client,
        mcp_session=mock_session,
        transcripts_path=transcripts_path,
        task_id="task_loop_guard",
    )

    assert result.total_tool_calls == 2
    # Verify MCP session only received 1 call; second was intercepted by loop guard!
    assert len(mock_session.call_log) == 1

    # Check history flags
    assert not result.tool_call_history[0]["is_duplicate"]
    assert result.tool_call_history[1]["is_duplicate"]
    assert "Duplicate query detected" in result.tool_call_history[1]["tool_output_snippet"]


# --- 7. Step Budget Enforcement Tests ---


@pytest.mark.anyio
async def test_agent_step_budget_enforcement(clean_test_tmp_dir):
    """Verify that agent cannot exceed max_steps tool calls and is forced to conclude."""
    mock_client = MagicMock()

    # Generate repeated tool call responses
    tool_resp = MockAnthropicResponse(
        content=[
            MockToolUseBlock(
                tool_id="call_step",
                name="search_docs",
                input_dict={"query": "query"},
            )
        ],
        stop_reason="tool_use",
    )

    # Conclude response when prompted at budget exhaustion
    final_resp = MockAnthropicResponse(
        content=[
            MockTextBlock(
                '{"answer": "Could not find sufficient data within allowed budget.", '
                '"citations": [], "abstain": true}'
            )
        ],
        stop_reason="end_turn",
    )

    # Configure max_steps = 3: 3 tool calls, then 1 conclude response
    # Unique queries so loop guard does not intercept
    def make_unique_call(idx: int):
        return MockAnthropicResponse(
            content=[
                MockToolUseBlock(
                    tool_id=f"call_{idx}",
                    name="search_docs",
                    input_dict={"query": f"unique query {idx}"},
                )
            ],
            stop_reason="tool_use",
        )

    mock_client.messages.create.side_effect = [
        make_unique_call(1),
        make_unique_call(2),
        make_unique_call(3),
        final_resp,
    ]

    mock_session = MockClientSession()
    transcripts_path = clean_test_tmp_dir / "transcripts.jsonl"

    result = await arun_agent(
        task="Investigate deep architecture details",
        anthropic_client=mock_client,
        mcp_session=mock_session,
        max_steps=3,
        transcripts_path=transcripts_path,
        task_id="task_budget",
    )

    assert result.total_tool_calls == 3
    assert result.output.abstain
    assert result.status == "budget_exhausted"


# --- 8. Hallucinated Citation Detection ---


@pytest.mark.anyio
async def test_agent_hallucinated_citation_detection(clean_test_tmp_dir):
    """Verify that citations not present in retrieved chunk IDs are flagged as hallucinated."""
    mock_client = MagicMock()

    step1_resp = MockAnthropicResponse(
        content=[
            MockToolUseBlock(
                tool_id="call_1",
                name="search_docs",
                input_dict={"query": "flags"},
            )
        ],
        stop_reason="tool_use",
    )

    # Agent returns 1 valid retrieved citation (wrapped in backticks) and 1 hallucinated citation
    step2_resp = MockAnthropicResponse(
        content=[
            MockTextBlock(
                '{"answer": "Answer based on flags.", '
                '"citations": ["`docs/flags.md#0`", "hallucinated/chunk.md#99"], "abstain": false}'
            )
        ],
        stop_reason="end_turn",
    )

    mock_client.messages.create.side_effect = [step1_resp, step2_resp]
    mock_session = MockClientSession()
    transcripts_path = clean_test_tmp_dir / "transcripts.jsonl"

    result = await arun_agent(
        task="Tell me about flags",
        anthropic_client=mock_client,
        mcp_session=mock_session,
        transcripts_path=transcripts_path,
        task_id="task_hallucination",
    )

    assert result.valid_citations == ["docs/flags.md#0"]
    assert result.hallucinated_citations == ["hallucinated/chunk.md#99"]
    assert result.status == "completed_with_hallucinated_citations"


# --- 9. Synchronous Wrapper Test ---


def test_run_agent_sync_wrapper(clean_test_tmp_dir):
    """Verify synchronous run_agent wrapper completes correctly without async runner boilerplate."""
    mock_client = MagicMock()
    mock_response = MockAnthropicResponse(
        content=[
            MockTextBlock('{"answer": "Sync execution works.", "citations": [], "abstain": false}')
        ],
        stop_reason="end_turn",
    )
    mock_client.messages.create.return_value = mock_response
    mock_session = MockClientSession()
    transcripts_path = clean_test_tmp_dir / "sync_transcripts.jsonl"

    result = run_agent(
        task="Test sync wrapper",
        anthropic_client=mock_client,
        mcp_session=mock_session,
        transcripts_path=transcripts_path,
        task_id="task_sync",
    )

    assert isinstance(result, AgentResult)
    assert result.output.answer == "Sync execution works."
    assert result.task_id == "task_sync"


# --- 10. CLI Entrypoint Test ---


def test_agent_cli_entrypoint(capsys, clean_test_tmp_dir):
    """Verify CLI entrypoint argument parsing and standard output display."""
    transcripts_path = str(clean_test_tmp_dir / "cli_transcripts.jsonl")

    dummy_result = AgentResult(
        task_id="cli_test",
        question="How do I install PostHog?",
        output=GenerationOutput(
            answer="Run npm install posthog-js.",
            citations=["docs/install.md#0"],
            abstain=False,
        ),
        valid_citations=["docs/install.md#0"],
        hallucinated_citations=[],
        steps=2,
        total_tool_calls=1,
        tool_call_history=[],
        all_retrieved_chunk_ids=["docs/install.md#0"],
        latency_ms=120.5,
        tokens_in=100,
        tokens_out=50,
        status="completed",
        transcript_path=transcripts_path,
    )

    with patch(
        "sys.argv",
        [
            "agent.py",
            "How do I install PostHog?",
            "--model",
            "claude-haiku-5-5",
            "--max-steps",
            "4",
            "--transcripts-path",
            transcripts_path,
        ],
    ):
        with patch("src.agent.run_agent", return_value=dummy_result) as mock_run:
            agent_cli_main()
            mock_run.assert_called_once_with(
                task="How do I install PostHog?",
                model="claude-haiku-5-5",
                max_steps=4,
                transcripts_path=Path(transcripts_path),
            )
            captured = capsys.readouterr()
            assert "=== Agent Result ===" in captured.out
            assert "Status: completed" in captured.out
            assert "Run npm install posthog-js." in captured.out


# --- 11. Real Stdio FastMCP Server Integration Test ---


@pytest.mark.anyio
async def test_agent_real_stdio_mcp_integration(clean_test_tmp_dir):
    """Verify end-to-end integration connecting to the real FastMCP server over stdio transport."""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        MockAnthropicResponse(
            content=[
                MockToolUseBlock(
                    tool_id="call_real_1",
                    name="search_docs",
                    input_dict={"query": "feature flags", "top_k": 2},
                )
            ],
            stop_reason="tool_use",
        ),
        MockAnthropicResponse(
            content=[
                MockTextBlock(
                    '{"answer": "Feature flags enable progressive rollout.", '
                    '"citations": ["feature-flag-use-cases.md#0"], "abstain": false}'
                )
            ],
            stop_reason="end_turn",
        ),
    ]

    transcripts_path = clean_test_tmp_dir / "real_stdio_transcripts.jsonl"

    result = await arun_agent(
        task="What are feature flags used for in PostHog?",
        anthropic_client=mock_client,
        mcp_session=None,
        transcripts_path=transcripts_path,
        task_id="task_real_stdio",
    )

    assert result.steps == 2
    assert result.total_tool_calls == 1
    assert "feature-flag-use-cases.md#0" in result.all_retrieved_chunk_ids
    assert result.valid_citations == ["feature-flag-use-cases.md#0"]
    assert result.status == "completed"
    assert transcripts_path.exists()


@pytest.mark.anyio
async def test_unsupported_model_provider_raises_error():
    """Verify that specifying a non-Anthropic provider model raises a clear ValueError."""
    with pytest.raises(ValueError, match="Unsupported model provider for model 'gpt-4o'"):
        await arun_agent(
            task="Test query",
            model="gpt-4o",
        )


