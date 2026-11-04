import pytest
from unittest.mock import MagicMock
from src.llm import generate_text, is_fallback_candidate, LLMResponse


def test_is_fallback_candidate():
    class Dummy503(Exception):
        code = 503

    class Dummy429(Exception):
        code = 429

    assert is_fallback_candidate(Dummy503("Unavailable"))
    assert is_fallback_candidate(Dummy429("Quota exceeded"))
    assert is_fallback_candidate(Exception("503 UNAVAILABLE. high demand"))
    assert is_fallback_candidate(Exception("429 RESOURCE_EXHAUSTED. limit: 20 per day"))
    assert not is_fallback_candidate(ValueError("Invalid argument"))
    assert not is_fallback_candidate(Exception("404 NOT_FOUND"))
    assert not is_fallback_candidate(Exception("401 UNAUTHORIZED"))


def test_generate_text_primary_success(monkeypatch):
    mock_gemini = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Response from Gemini Flash"
    mock_gemini.models.generate_content.return_value = mock_response

    monkeypatch.setattr("src.llm.get_gemini_client", lambda: mock_gemini)

    res = generate_text("Hello")
    assert res.provider == "gemini"
    assert res.fell_back is False
    assert res.text == "Response from Gemini Flash"
    mock_gemini.models.generate_content.assert_called_once()


def test_generate_text_fallback_on_503(monkeypatch):
    mock_gemini = MagicMock()
    mock_gemini.models.generate_content.side_effect = Exception("503 UNAVAILABLE high demand")

    mock_anthropic = MagicMock()
    mock_msg = MagicMock()
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = "Response from Claude Haiku"
    mock_msg.content = [mock_block]
    mock_anthropic.messages.create.return_value = mock_msg

    monkeypatch.setattr("src.llm.get_gemini_client", lambda: mock_gemini)
    monkeypatch.setattr("src.llm.get_anthropic_client", lambda: mock_anthropic)

    res = generate_text("Hello")
    assert res.provider == "anthropic"
    assert res.fell_back is True
    assert "Gemini unavailable/exhausted" in (res.fallback_reason or "")
    assert res.text == "Response from Claude Haiku"

    # Verify that Haiku was invoked with effort="xhigh"
    mock_anthropic.messages.create.assert_called_once()
    call_kwargs = mock_anthropic.messages.create.call_args.kwargs
    assert call_kwargs.get("output_config") == {"effort": "xhigh"}


def test_generate_text_fallback_on_429(monkeypatch):
    mock_gemini = MagicMock()
    mock_gemini.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED daily limit")

    mock_anthropic = MagicMock()
    mock_msg = MagicMock()
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = "Response from Claude Haiku"
    mock_msg.content = [mock_block]
    mock_anthropic.messages.create.return_value = mock_msg

    monkeypatch.setattr("src.llm.get_gemini_client", lambda: mock_gemini)
    monkeypatch.setattr("src.llm.get_anthropic_client", lambda: mock_anthropic)

    res = generate_text("Hello")
    assert res.provider == "anthropic"
    assert res.fell_back is True
    assert res.text == "Response from Claude Haiku"


def test_generate_text_non_fallback_error_raises_directly(monkeypatch):
    mock_gemini = MagicMock()
    mock_gemini.models.generate_content.side_effect = ValueError("Bad Request (400)")

    mock_anthropic = MagicMock()

    monkeypatch.setattr("src.llm.get_gemini_client", lambda: mock_gemini)
    monkeypatch.setattr("src.llm.get_anthropic_client", lambda: mock_anthropic)

    with pytest.raises(ValueError, match="Bad Request"):
        generate_text("Hello")

    mock_anthropic.messages.create.assert_not_called()


def test_generate_text_both_fail(monkeypatch):
    mock_gemini = MagicMock()
    mock_gemini.models.generate_content.side_effect = Exception("503 UNAVAILABLE")

    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.side_effect = RuntimeError("Anthropic connection failed")

    monkeypatch.setattr("src.llm.get_gemini_client", lambda: mock_gemini)
    monkeypatch.setattr("src.llm.get_anthropic_client", lambda: mock_anthropic)

    with pytest.raises(RuntimeError, match="Both primary model .* and fallback model .* failed"):
        generate_text("Hello")


def test_generate_text_live_execution():
    """Live smoke test verifying that generate_text completes end-to-end (either primary or fallback)."""
    res = generate_text("Respond with only one word: PONG", max_tokens=20)
    assert isinstance(res, LLMResponse)
    assert res.text is not None
    assert len(res.text.strip()) > 0
    assert res.provider in ("gemini", "anthropic")
