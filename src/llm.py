"""LLM client and generation interface with high-reasoning effort and fallback policy.

Policy:
1. Primary: Gemini 3.8 Flash (GEMINI_FLASH) with thinking level HIGH.
2. Fallback: If and only if Gemini returns 503 / UNAVAILABLE (high demand spike) or 429 (quota exhausted),
   fall back to Claude Haiku 5.5 (ANTHROPIC_HAIKU) with reasoning effort xhigh.
3. If Claude Haiku also fails, or if Gemini raises a non-retryable error (e.g. 400 Bad Request, 401/403 Auth),
   raise the exception immediately.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from google.genai import types
from src.config import (
    GEMINI_FLASH,
    ANTHROPIC_HAIKU,
    get_gemini_client,
    get_anthropic_client,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str  # "gemini" or "anthropic"
    fell_back: bool = False
    fallback_reason: Optional[str] = None


def is_fallback_candidate(error: Exception) -> bool:
    """Check if an exception from primary model warrants fallback (503 UNAVAILABLE or 429 RESOURCE_EXHAUSTED)."""
    err_str = str(error)
    code = getattr(error, "code", None)
    status_code = getattr(error, "status_code", None)
    if code in (503, 429) or status_code in (503, 429):
        return True
    if any(sig in err_str for sig in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "high demand")):
        return True
    return False


def generate_text(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 4096,
) -> LLMResponse:
    """Generate text using primary Gemini 3.8 Flash (effort: high) with fallback to Claude Haiku 5.5 (effort: xhigh).

    Args:
        prompt: User input prompt.
        system_prompt: Optional system instruction.
        max_tokens: Maximum tokens in generated completion.

    Returns:
        LLMResponse containing the generated text, metadata, and fallback details.

    Raises:
        Exception: If primary fails with non-fallback error, or if both primary and fallback fail.
    """
    gemini_client = get_gemini_client()
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        max_output_tokens=max_tokens,
        system_instruction=system_prompt,
    )

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_FLASH,
            contents=prompt,
            config=config,
        )
        return LLMResponse(
            text=response.text or "",
            model=GEMINI_FLASH,
            provider="gemini",
            fell_back=False,
        )
    except Exception as exc:
        if not is_fallback_candidate(exc):
            logger.error("Primary model %s failed with non-fallback error: %s", GEMINI_FLASH, exc)
            raise

        logger.warning(
            "Primary model %s unavailable/exhausted (%s). Initiating fallback to Claude Haiku (%s, effort=xhigh)",
            GEMINI_FLASH,
            exc,
            ANTHROPIC_HAIKU,
        )

        anthropic_client = get_anthropic_client()
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": ANTHROPIC_HAIKU,
            "max_tokens": max_tokens,
            "output_config": {"effort": "xhigh"},
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        try:
            anthropic_response = anthropic_client.messages.create(**kwargs)
            text_parts = [
                block.text
                for block in anthropic_response.content
                if getattr(block, "type", None) == "text" or (hasattr(block, "text") and not hasattr(block, "thinking"))
            ]
            response_text = "\n".join(text_parts).strip()
            return LLMResponse(
                text=response_text,
                model=ANTHROPIC_HAIKU,
                provider="anthropic",
                fell_back=True,
                fallback_reason=f"Gemini unavailable/exhausted: {exc}",
            )
        except Exception as fallback_exc:
            logger.error(
                "Fallback model %s (effort=xhigh) also failed: %s (Original primary error: %s)",
                ANTHROPIC_HAIKU,
                fallback_exc,
                exc,
            )
            raise RuntimeError(
                f"Both primary model ({GEMINI_FLASH}) and fallback model ({ANTHROPIC_HAIKU}) failed. "
                f"Primary: {exc} | Fallback: {fallback_exc}"
            ) from fallback_exc
