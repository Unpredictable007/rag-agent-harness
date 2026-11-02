import pytest
from sentence_transformers import SentenceTransformer
from src.config import (
    EMBEDDING_MODEL_NAME,
    GEMINI_FLASH,
    GEMINI_ARGON,
    ANTHROPIC_HAIKU,
    GEMINI_API_KEY,
    ANTHROPIC_API_KEY,
)


def test_embeddings_generation():
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    sample_text = "PostHog ingestion API and event capture"
    vector = model.encode(sample_text)
    assert len(vector) == 384


def test_gemini_flash_connection():
    assert bool(GEMINI_API_KEY and GEMINI_API_KEY.strip()), "GEMINI_API_KEY is missing from .env"
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_FLASH,
        contents="Respond with only one word: PONG",
    )
    assert response.text is not None
    assert len(response.text.strip()) > 0


@pytest.mark.skip(reason="Disabled: Argon has limit=0 on Free Tier (requires Cloud Billing)")
def test_gemini_argon_connection():
    assert bool(GEMINI_API_KEY and GEMINI_API_KEY.strip()), "GEMINI_API_KEY is missing from .env"
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_ARGON,
        contents="Respond with only one word: PONG",
    )
    assert response.text is not None
    assert len(response.text.strip()) > 0


def test_anthropic_connection():
    assert bool(ANTHROPIC_API_KEY and ANTHROPIC_API_KEY.strip()), "ANTHROPIC_API_KEY is missing from .env"
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=ANTHROPIC_HAIKU,
        max_tokens=10,
        messages=[{"role": "user", "content": "Respond with only one word: PONG"}],
    )
    assert message.content is not None
    assert len(message.content[0].text.strip()) > 0
