import os
import pytest
from sentence_transformers import SentenceTransformer
from src.config import (
    EMBEDDING_MODEL_NAME,
    GEMINI_FLASH,
    GEMINI_ARGON,
    ANTHROPIC_HAIKU,
    PERSONAL_GEMINI_API_KEY,
    PERSONAL_ANTHROPIC_API_KEY,
    get_gemini_client,
    get_anthropic_client,
)


def test_forbidden_keys_purged_from_environment():
    """Assert that ambient GOOGLE_API_KEY and generic keys are stripped and not used."""
    assert "GOOGLE_API_KEY" not in os.environ
    assert "GEMINI_API_KEY" not in os.environ
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert bool(PERSONAL_GEMINI_API_KEY and PERSONAL_GEMINI_API_KEY.strip())
    assert bool(PERSONAL_ANTHROPIC_API_KEY and PERSONAL_ANTHROPIC_API_KEY.strip())


def test_embeddings_generation():
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    sample_text = "PostHog ingestion API and event capture"
    vector = model.encode(sample_text)
    assert len(vector) == 384


def test_gemini_flash_connection():
    """Pure connectivity test for Gemini Flash without model substitution."""
    client = get_gemini_client()
    try:
        response = client.models.generate_content(
            model=GEMINI_FLASH,
            contents="Respond with only one word: PONG",
        )
        assert response.text is not None
        assert len(response.text.strip()) > 0
    except Exception as exc:
        if any(sig in str(exc) for sig in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED")):
            pytest.xfail(f"Upstream Google API unavailable or quota exhausted: {exc}")
        raise exc


@pytest.mark.skip(reason="Disabled: Argon has limit=0 on Free Tier (requires Cloud Billing)")
def test_gemini_argon_connection():
    client = get_gemini_client()
    response = client.models.generate_content(
        model=GEMINI_ARGON,
        contents="Respond with only one word: PONG",
    )
    assert response.text is not None
    assert len(response.text.strip()) > 0


def test_anthropic_connection():
    """Pure connectivity test for Claude Haiku."""
    client = get_anthropic_client()
    message = client.messages.create(
        model=ANTHROPIC_HAIKU,
        max_tokens=10,
        messages=[{"role": "user", "content": "Respond with only one word: PONG"}],
    )
    assert message.content is not None
    assert len(message.content[0].text.strip()) > 0
