"""Unit tests for configurable LLM client with bidirectional fallback policy."""

import pytest
from unittest.mock import MagicMock
from src.llm import generate_text, is_fallback_candidate, LLMResponse
from src.config import GEMINI_FLASH, ANTHROPIC_HAIKU


def test_is_fallback_candidate():
    class Dummy503(Exception):
        code = 503

    class Dummy429(Exception):
        code = 429

    class Dummy529(Exception):
        code = 529

    class Dummy500(Exception):
        code = 500

    # Generic / unspecified provider
    assert is_fallback_candidate(Dummy503("Unavailable"))
    assert is_fallback_candidate(Dummy429("Quota exceeded"))
    assert is_fallback_candidate(Dummy529("Overloaded"))
    assert is_fallback_candidate(Exception("503 UNAVAILABLE. high demand"))
    assert is_fallback_candidate(Exception("429 RESOURCE_EXHAUSTED. limit: 20 per day"))
    assert not is_fallback_candidate(ValueError("Invalid argument"))
    assert not is_fallback_candidate(Exception("404 NOT_FOUND"))
    assert not is_fallback_candidate(Exception("401 UNAUTHORIZED"))

    # Anthropic specific provider
    assert is_fallback_candidate(Dummy529("Overloaded"), provider="anthropic")
    assert is_fallback_candidate(Dummy503("Service Unavailable"), provider="anthropic")
    assert is_fallback_candidate(Dummy500("Internal Server Error"), provider="anthropic")
    assert is_fallback_candidate(Dummy429("Rate limit exceeded"), provider="anthropic")
    assert is_fallback_candidate(Exception("HTTP 529 overloaded"), provider="anthropic")
    assert not is_fallback_candidate(Exception("400 Bad Request"), provider="anthropic")
    assert not is_fallback_candidate(Exception("401 Unauthorized"), provider="anthropic")

    # Gemini specific provider
    assert is_fallback_candidate(Dummy503("Unavailable"), provider="gemini")
    assert is_fallback_candidate(Dummy429("Resource exhausted"), provider="gemini")
    assert not is_fallback_candidate(Dummy500("Internal Server Error"), provider="gemini")
    assert not is_fallback_candidate(Exception("400 Bad Request"), provider="gemini")


# ---------------------------------------------------------------------------
# Primary Anthropic Tests (Default)
# ---------------------------------------------------------------------------

def test_generate_text_anthropic_primary_success(monkeypatch):
    mock_anthropic = MagicMock()
    mock_msg = MagicMock()
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = "Response from Claude Haiku"
    mock_msg.content = [mock_block]
    mock_anthropic.messages.create.return_value = mock_msg

    mock_gemini = MagicMock()

    monkeypatch.setattr("src.llm.get_anthropic_client", lambda: mock_anthropic)
    monkeypatch.setattr("src.llm.get_gemini_client", lambda: mock_gemini)

    # Provider defaults to anthropic
    res = generate_text("Hello")
    assert res.provider == "anthropic"
    assert res.model == ANTHROPIC_HAIKU
    assert res.fell_back is False
    assert res.text == "Response from Claude Haiku"

    mock_anthropic.messages.create.assert_called_once()
    call_kwargs = mock_anthropic.messages.create.call_args.kwargs
    assert call_kwargs.get("model") == ANTHROPIC_HAIKU
    assert call_kwargs.get("output_config") == {"effort": "xhigh"}
    mock_gemini.models.generate_content.assert_not_called()


def test_generate_text_anthropic_fallback_to_gemini_on_529(monkeypatch):
    class Dummy529(Exception):
        status_code = 529

    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.side_effect = Dummy529("529 Overloaded")

    mock_gemini = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Response from Gemini Flash fallback"
    mock_gemini.models.generate_content.return_value = mock_response

    monkeypatch.setattr("src.llm.get_anthropic_client", lambda: mock_anthropic)
    monkeypatch.setattr("src.llm.get_gemini_client", lambda: mock_gemini)

    res = generate_text("Hello", provider="anthropic")
    assert res.provider == "gemini"
    assert res.model == GEMINI_FLASH
    assert res.fell_back is True
    assert "Anthropic unavailable/overloaded" in (res.fallback_reason or "")
    assert res.text == "Response from Gemini Flash fallback"

    mock_anthropic.messages.create.assert_called_once()
    mock_gemini.models.generate_content.assert_called_once()


def test_generate_text_anthropic_fallback_to_gemini_on_503(monkeypatch):
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.side_effect = Exception("503 Service Unavailable")

    mock_gemini = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Response from Gemini Flash fallback"
    mock_gemini.models.generate_content.return_value = mock_response

    monkeypatch.setattr("src.llm.get_anthropic_client", lambda: mock_anthropic)
    monkeypatch.setattr("src.llm.get_gemini_client", lambda: mock_gemini)

    res = generate_text("Hello", provider="anthropic")
    assert res.provider == "gemini"
    assert res.fell_back is True
    assert res.text == "Response from Gemini Flash fallback"


def test_generate_text_anthropic_fallback_to_gemini_on_429(monkeypatch):
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.side_effect = Exception("429 Rate limit exceeded")

    mock_gemini = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Response from Gemini Flash fallback"
    mock_gemini.models.generate_content.return_value = mock_response

    monkeypatch.setattr("src.llm.get_anthropic_client", lambda: mock_anthropic)
    monkeypatch.setattr("src.llm.get_gemini_client", lambda: mock_gemini)

    res = generate_text("Hello", provider="anthropic")
    assert res.provider == "gemini"
    assert res.fell_back is True
    assert res.text == "Response from Gemini Flash fallback"


def test_generate_text_anthropic_fallback_to_gemini_on_500(monkeypatch):
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.side_effect = Exception("500 Internal Server Error")

    mock_gemini = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Response from Gemini Flash fallback"
    mock_gemini.models.generate_content.return_value = mock_response

    monkeypatch.setattr("src.llm.get_anthropic_client", lambda: mock_anthropic)
    monkeypatch.setattr("src.llm.get_gemini_client", lambda: mock_gemini)

    res = generate_text("Hello", provider="anthropic")
    assert res.provider == "gemini"
    assert res.fell_back is True
    assert res.text == "Response from Gemini Flash fallback"


def test_generate_text_anthropic_non_fallback_error_raises_directly(monkeypatch):
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.side_effect = ValueError("400 Bad Request")

    mock_gemini = MagicMock()

    monkeypatch.setattr("src.llm.get_anthropic_client", lambda: mock_anthropic)
    monkeypatch.setattr("src.llm.get_gemini_client", lambda: mock_gemini)

    with pytest.raises(ValueError, match="400 Bad Request"):
        generate_text("Hello", provider="anthropic")

    mock_gemini.models.generate_content.assert_not_called()


# ---------------------------------------------------------------------------
# Primary Gemini Tests
# ---------------------------------------------------------------------------

def test_generate_text_gemini_primary_success(monkeypatch):
    mock_gemini = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Response from Gemini Flash"
    mock_gemini.models.generate_content.return_value = mock_response

    mock_anthropic = MagicMock()

    monkeypatch.setattr("src.llm.get_gemini_client", lambda: mock_gemini)
    monkeypatch.setattr("src.llm.get_anthropic_client", lambda: mock_anthropic)

    res = generate_text("Hello", provider="gemini")
    assert res.provider == "gemini"
    assert res.fell_back is False
    assert res.text == "Response from Gemini Flash"
    mock_gemini.models.generate_content.assert_called_once()
    mock_anthropic.messages.create.assert_not_called()


def test_generate_text_gemini_fallback_on_503(monkeypatch):
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

    res = generate_text("Hello", provider="gemini")
    assert res.provider == "anthropic"
    assert res.fell_back is True
    assert "Gemini unavailable/exhausted" in (res.fallback_reason or "")
    assert res.text == "Response from Claude Haiku"

    mock_anthropic.messages.create.assert_called_once()
    call_kwargs = mock_anthropic.messages.create.call_args.kwargs
    assert call_kwargs.get("output_config") == {"effort": "xhigh"}


def test_generate_text_gemini_fallback_on_429(monkeypatch):
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

    res = generate_text("Hello", provider="gemini")
    assert res.provider == "anthropic"
    assert res.fell_back is True
    assert res.text == "Response from Claude Haiku"


def test_generate_text_gemini_non_fallback_error_raises_directly(monkeypatch):
    mock_gemini = MagicMock()
    mock_gemini.models.generate_content.side_effect = ValueError("Bad Request (400)")

    mock_anthropic = MagicMock()

    monkeypatch.setattr("src.llm.get_gemini_client", lambda: mock_gemini)
    monkeypatch.setattr("src.llm.get_anthropic_client", lambda: mock_anthropic)

    with pytest.raises(ValueError, match="Bad Request"):
        generate_text("Hello", provider="gemini")

    mock_anthropic.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# Dual Failure & Validation Tests
# ---------------------------------------------------------------------------

def test_generate_text_both_fail_anthropic_primary(monkeypatch):
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.side_effect = Exception("529 Overloaded")

    mock_gemini = MagicMock()
    mock_gemini.models.generate_content.side_effect = Exception("503 UNAVAILABLE")

    monkeypatch.setattr("src.llm.get_anthropic_client", lambda: mock_anthropic)
    monkeypatch.setattr("src.llm.get_gemini_client", lambda: mock_gemini)

    with pytest.raises(RuntimeError, match="Both primary model .* and fallback model .* failed"):
        generate_text("Hello", provider="anthropic")


def test_generate_text_both_fail_gemini_primary(monkeypatch):
    mock_gemini = MagicMock()
    mock_gemini.models.generate_content.side_effect = Exception("503 UNAVAILABLE")

    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.side_effect = RuntimeError("Anthropic connection failed")

    monkeypatch.setattr("src.llm.get_gemini_client", lambda: mock_gemini)
    monkeypatch.setattr("src.llm.get_anthropic_client", lambda: mock_anthropic)

    with pytest.raises(RuntimeError, match="Both primary model .* and fallback model .* failed"):
        generate_text("Hello", provider="gemini")


def test_generate_text_invalid_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider 'invalid_provider'"):
        generate_text("Hello", provider="invalid_provider")


def test_generate_text_live_execution():
    """Live smoke test verifying that generate_text completes end-to-end (either primary or fallback)."""
    res = generate_text("Respond with only one word: PONG", max_tokens=20)
    assert isinstance(res, LLMResponse)
    assert res.text is not None
    assert len(res.text.strip()) > 0
    assert res.provider in ("gemini", "anthropic")
