"""Unit tests for Calibrated LLM-as-a-Judge (Milestone M6)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from eval.calibrate_judge import compute_cohens_kappa
from src.judge import (
    CorrectnessJudge,
    FaithfulnessJudge,
    JudgeResult,
    _call_judge_llm,
    parse_judge_json,
)
from src.retrieval import RetrievalResult


# ---------------------------------------------------------------------------
# Test Pydantic JudgeResult Schema & Validation
# ---------------------------------------------------------------------------

def test_judge_result_model():
    result = JudgeResult(
        score=1.0,
        reasoning="All claims match reference answer.",
        judge_type="correctness",
        model="claude-sonnet-5-5",
        provider="anthropic",
        latency_ms=1240.5,
    )
    assert result.score == 1.0
    assert result.reasoning == "All claims match reference answer."
    assert result.judge_type == "correctness"
    assert result.model == "claude-sonnet-5-5"
    assert result.provider == "anthropic"
    assert result.latency_ms == 1240.5


# ---------------------------------------------------------------------------
# Test JSON Extraction & Parsing
# ---------------------------------------------------------------------------

def test_parse_judge_json_clean():
    raw = '{"reasoning": "Direct match.", "score": 1.0}'
    parsed = parse_judge_json(raw)
    assert parsed["score"] == 1.0
    assert parsed["reasoning"] == "Direct match."


def test_parse_judge_json_markdown_fenced():
    raw = """```json
{
  "reasoning": "Omitted important configuration details.",
  "score": 0.5
}
```"""
    parsed = parse_judge_json(raw)
    assert parsed["score"] == 0.5
    assert parsed["reasoning"] == "Omitted important configuration details."


def test_parse_judge_json_conversational_wrapper():
    raw = """Here is my evaluation:
{
  "reasoning": "Candidate contradicts reference answer.",
  "score": 0.0
}
Let me know if you need more details."""
    parsed = parse_judge_json(raw)
    assert parsed["score"] == 0.0
    assert parsed["reasoning"] == "Candidate contradicts reference answer."


def test_parse_judge_json_malformed_raises_error():
    with pytest.raises(ValueError, match="Could not parse valid JSON"):
        parse_judge_json("This is purely conversational with no JSON whatsoever.")


# ---------------------------------------------------------------------------
# Test CorrectnessJudge Abstain Logic & Clamping
# ---------------------------------------------------------------------------

def test_correctness_judge_unanswerable_abstained():
    judge = CorrectnessJudge()
    res = judge.evaluate(
        question="How to configure CAN bus?",
        reference_answer="PostHog does not support CAN bus.",
        generated_answer="I do not have enough information to answer this question.",
        answerable=False,
        abstain=True,
    )
    assert res.score == 1.0
    assert res.judge_type == "correctness"
    assert "Correctly abstained" in res.reasoning
    assert res.provider == "internal"


def test_correctness_judge_unanswerable_failed_to_abstain():
    judge = CorrectnessJudge()
    res = judge.evaluate(
        question="How to configure CAN bus?",
        reference_answer="PostHog does not support CAN bus.",
        generated_answer="Install the can-bus plugin and configure interface.",
        answerable=False,
        abstain=False,
    )
    assert res.score == 0.0
    assert res.judge_type == "correctness"
    assert "Failed to abstain" in res.reasoning


def test_correctness_judge_answerable_false_abstain():
    judge = CorrectnessJudge()
    res = judge.evaluate(
        question="What is the default timezone?",
        reference_answer="By default UTC.",
        generated_answer=None,
        answerable=True,
        abstain=True,
    )
    assert res.score == 0.0
    assert res.judge_type == "correctness"
    assert "False abstain" in res.reasoning


def test_correctness_judge_answerable_empty_answer():
    judge = CorrectnessJudge()
    res = judge.evaluate(
        question="What is the default timezone?",
        reference_answer="By default UTC.",
        generated_answer="   ",
        answerable=True,
        abstain=False,
    )
    assert res.score == 0.0
    assert "False abstain" in res.reasoning


@patch("src.judge._call_judge_llm")
def test_correctness_judge_mocked_llm_tiers(mock_call):
    judge = CorrectnessJudge()

    # Tier 1.0 (Fully Correct)
    mock_call.return_value = (
        '{"reasoning": "Fully matches reference.", "score": 1.0}',
        "claude-sonnet-5-5",
        "anthropic",
        500.0,
    )
    res_1 = judge.evaluate(
        question="What is UTC?",
        reference_answer="UTC is coordinated universal time.",
        generated_answer="UTC stands for coordinated universal time.",
        answerable=True,
        abstain=False,
    )
    assert res_1.score == 1.0

    # Tier 0.5 (Partially Correct)
    mock_call.return_value = (
        '{"reasoning": "Missing secondary step.", "score": 0.5}',
        "claude-sonnet-5-5",
        "anthropic",
        450.0,
    )
    res_05 = judge.evaluate(
        question="What are steps A and B?",
        reference_answer="Step A and Step B.",
        generated_answer="Step A.",
        answerable=True,
        abstain=False,
    )
    assert res_05.score == 0.5

    # Tier 0.0 (Incorrect)
    mock_call.return_value = (
        '{"reasoning": "Completely wrong.", "score": 0.0}',
        "claude-sonnet-5-5",
        "anthropic",
        420.0,
    )
    res_0 = judge.evaluate(
        question="What is the capital?",
        reference_answer="Paris.",
        generated_answer="Berlin.",
        answerable=True,
        abstain=False,
    )
    assert res_0.score == 0.0


@patch("src.judge._call_judge_llm")
def test_correctness_judge_score_clamping(mock_call):
    judge = CorrectnessJudge()

    # 0.9 should round to 1.0
    mock_call.return_value = (
        '{"reasoning": "Near perfect.", "score": 0.9}',
        "claude-sonnet-5-5",
        "anthropic",
        300.0,
    )
    res_high = judge.evaluate("Q", "Ref", "Gen", answerable=True, abstain=False)
    assert res_high.score == 1.0

    # 0.4 should round to 0.5
    mock_call.return_value = (
        '{"reasoning": "Slightly below half.", "score": 0.4}',
        "claude-sonnet-5-5",
        "anthropic",
        300.0,
    )
    res_mid = judge.evaluate("Q", "Ref", "Gen", answerable=True, abstain=False)
    assert res_mid.score == 0.5

    # 0.15 should round to 0.0
    mock_call.return_value = (
        '{"reasoning": "Very weak.", "score": 0.15}',
        "claude-sonnet-5-5",
        "anthropic",
        300.0,
    )
    res_low = judge.evaluate("Q", "Ref", "Gen", answerable=True, abstain=False)
    assert res_low.score == 0.0


# ---------------------------------------------------------------------------
# Test FaithfulnessJudge Abstain Logic & Grounding
# ---------------------------------------------------------------------------

def test_faithfulness_judge_abstained():
    judge = FaithfulnessJudge()
    res = judge.evaluate(
        question="How to configure CAN bus?",
        retrieved_context="Some doc chunk",
        generated_answer="I do not have enough information.",
        abstain=True,
    )
    assert res.score == 1.0
    assert res.judge_type == "faithfulness"
    assert "Abstained" in res.reasoning


def test_faithfulness_judge_empty_answer():
    judge = FaithfulnessJudge()
    res = judge.evaluate(
        question="How to configure CAN bus?",
        retrieved_context="Some doc chunk",
        generated_answer="",
        abstain=False,
    )
    assert res.score == 0.0
    assert "Empty answer" in res.reasoning


def test_faithfulness_judge_empty_context():
    judge = FaithfulnessJudge()
    res = judge.evaluate(
        question="How to configure CAN bus?",
        retrieved_context="<retrieved_context>\n</retrieved_context>",
        generated_answer="It is easy.",
        abstain=False,
    )
    assert res.score == 0.0
    assert "No context chunks" in res.reasoning


@patch("src.judge._call_judge_llm")
def test_faithfulness_judge_with_chunk_objects(mock_call):
    judge = FaithfulnessJudge()
    mock_call.return_value = (
        '{"reasoning": "All claims directly supported by chunk 1.", "score": 1.0}',
        "claude-sonnet-5-5",
        "anthropic",
        410.0,
    )
    chunks = [
        RetrievalResult(
            chunk_id="chunk-01",
            score=0.9,
            rank=1,
            text="PostHog captures events automatically.",
            url="https://posthog.com/docs/autocapture",
            heading_path="Docs > Autocapture",
            title="Autocapture Overview",
        )
    ]
    res = judge.evaluate(
        question="Does PostHog capture events automatically?",
        retrieved_context=chunks,
        generated_answer="Yes, PostHog captures events automatically.",
        abstain=False,
    )
    assert res.score == 1.0
    assert res.judge_type == "faithfulness"


@patch("src.judge._call_judge_llm")
def test_faithfulness_judge_hallucination_penalty(mock_call):
    judge = FaithfulnessJudge()
    mock_call.return_value = (
        '{"reasoning": "Candidate asserts fictitious features not in chunks.", "score": 0.0}',
        "claude-sonnet-5-5",
        "anthropic",
        450.0,
    )
    res = judge.evaluate(
        question="What is X?",
        retrieved_context="<retrieved_context><chunk id='c1'>PostHog is an analytics tool.</chunk></retrieved_context>",
        generated_answer="PostHog also provides quantum encryption.",
        abstain=False,
    )
    assert res.score == 0.0


# ---------------------------------------------------------------------------
# Test LLM Fallback Behavior
# ---------------------------------------------------------------------------

@patch("src.judge.get_gemini_client")
@patch("src.judge.get_anthropic_client")
def test_call_judge_llm_anthropic_fallback_to_gemini(mock_anthropic, mock_gemini):
    # Mock Anthropic raising 529 Overloaded error
    mock_anthropic_inst = MagicMock()
    mock_anthropic_inst.messages.create.side_effect = RuntimeError("529 Overloaded")
    mock_anthropic.return_value = mock_anthropic_inst

    # Mock Gemini succeeding
    mock_gemini_inst = MagicMock()
    mock_gemini_resp = MagicMock()
    mock_gemini_resp.text = '{"reasoning": "Fallback match.", "score": 1.0}'
    mock_gemini_inst.models.generate_content.return_value = mock_gemini_resp
    mock_gemini.return_value = mock_gemini_inst

    text, model, provider, latency = _call_judge_llm(
        prompt="prompt",
        system_prompt="system",
        primary_model="claude-sonnet-5-5",
        primary_provider="anthropic",
    )

    assert text == '{"reasoning": "Fallback match.", "score": 1.0}'
    assert provider == "gemini"
    assert latency > 0.0


@patch("src.judge.get_anthropic_client")
def test_call_judge_llm_non_fallback_error_raises(mock_anthropic):
    mock_anthropic_inst = MagicMock()
    mock_anthropic_inst.messages.create.side_effect = ValueError("400 Bad Request")
    mock_anthropic.return_value = mock_anthropic_inst

    with pytest.raises(ValueError, match="400 Bad Request"):
        _call_judge_llm(
            prompt="prompt",
            system_prompt="system",
            primary_model="claude-sonnet-5-5",
            primary_provider="anthropic",
        )


# ---------------------------------------------------------------------------
# Test Cohen's Kappa Mathematical Correctness
# ---------------------------------------------------------------------------

def test_cohens_kappa_perfect_agreement():
    y_true = [1.0, 0.5, 0.0, 1.0, 0.5, 0.0]
    y_pred = [1.0, 0.5, 0.0, 1.0, 0.5, 0.0]
    labels = [1.0, 0.5, 0.0]
    p_o, p_e, kappa, matrix = compute_cohens_kappa(y_true, y_pred, labels)

    assert p_o == 1.0
    assert kappa == 1.0
    assert matrix == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_cohens_kappa_chance_agreement():
    # Symmetric 2x2 matrix with 50% observed and 50% chance
    y_true = [1.0, 1.0, 0.0, 0.0]
    y_pred = [1.0, 0.0, 1.0, 0.0]
    labels = [1.0, 0.0]
    p_o, p_e, kappa, matrix = compute_cohens_kappa(y_true, y_pred, labels)

    assert p_o == 0.5
    assert p_e == 0.5
    assert abs(kappa - 0.0) < 1e-9


def test_cohens_kappa_known_multi_class():
    # 3-class distribution with known values
    y_true = [1.0, 1.0, 0.5, 0.5, 0.0, 0.0]
    y_pred = [1.0, 0.5, 0.5, 0.5, 0.0, 0.0]
    labels = [1.0, 0.5, 0.0]
    p_o, p_e, kappa, matrix = compute_cohens_kappa(y_true, y_pred, labels)

    assert p_o == 5 / 6  # 5 out of 6 match
    assert p_e > 0.0
    assert 0.70 <= kappa <= 1.0
    assert matrix[0][1] == 1  # 1 instance of true=1.0, pred=0.5
