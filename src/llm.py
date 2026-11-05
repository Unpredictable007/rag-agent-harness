"""LLM client and generation interface with high-reasoning effort and configurable fallback policy.

Policy:
1. Primary Provider: Configurable via src.config.DEFAULT_LLM_PROVIDER (defaults to "anthropic").
2. When primary is "anthropic":
   - Primary: Claude Haiku 5.5 (ANTHROPIC_HAIKU) with reasoning effort xhigh.
   - Fallback: If Anthropic raises retryable error (HTTP 529 Overloaded, 503 Unavailable, 500, or 429 Rate Limit),
     fall back to Gemini 3.8 Flash (GEMINI_FLASH) with thinking level HIGH.
3. When primary is "gemini":
   - Primary: Gemini 3.8 Flash (GEMINI_FLASH) with thinking level HIGH.
   - Fallback: If Gemini raises 503 UNAVAILABLE or 429 RESOURCE_EXHAUSTED,
     fall back to Claude Haiku 5.5 (ANTHROPIC_HAIKU) with reasoning effort xhigh.
4. Non-retryable errors (e.g. 400 Bad Request, 401/403 Auth) raise immediately without fallback.
5. If both primary and fallback fail, raise RuntimeError immediately.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from google.genai import types
import src.config
from src.config import (
    GEMINI_FLASH,
    ANTHROPIC_HAIKU,
    DEFAULT_LLM_PROVIDER,
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


def is_fallback_candidate(error: Exception, provider: Optional[str] = None) -> bool:
    """Check if an exception from primary model warrants fallback.

    For Anthropic: HTTP 529 Overloaded, 503 Unavailable, 500 Internal Error, or 429 Rate Limit.
    For Gemini: 503 UNAVAILABLE or 429 RESOURCE_EXHAUSTED.
    """
    err_str = str(error).lower()
    code = getattr(error, "code", None)
    status_code = getattr(error, "status_code", None)
    err_code = code if code is not None else status_code

    target_provider = provider.lower() if provider else None

    if target_provider == "anthropic":
        if err_code in (529, 503, 500, 429):
            return True
        anthropic_sigs = (
            "529",
            "503",
            "500",
            "429",
            "overloaded",
            "unavailable",
            "rate_limit",
            "rate limit",
            "internal server error",
        )
        return any(sig in err_str for sig in anthropic_sigs)

    if target_provider == "gemini":
        if err_code in (503, 429):
            return True
        gemini_sigs = (
            "503",
            "429",
            "unavailable",
            "resource_exhausted",
            "high demand",
        )
        return any(sig in err_str for sig in gemini_sigs)

    # Provider unspecified: check if retryable under either provider
    if err_code in (529, 503, 429):
        return True
    generic_sigs = (
        "529",
        "503",
        "429",
        "unavailable",
        "resource_exhausted",
        "high demand",
        "overloaded",
    )
    return any(sig in err_str for sig in generic_sigs)


def _call_anthropic(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 4096,
) -> str:
    """Execute Anthropic Claude Haiku generation with reasoning effort xhigh."""
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

    anthropic_response = anthropic_client.messages.create(**kwargs)
    text_parts = [
        block.text
        for block in anthropic_response.content
        if getattr(block, "type", None) == "text"
        or (hasattr(block, "text") and getattr(block, "type", None) != "thinking")
    ]
    return "\n".join(text_parts).strip()


def _call_gemini(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 4096,
) -> str:
    """Execute Google Gemini Flash generation with thinking level HIGH."""
    gemini_client = get_gemini_client()
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        max_output_tokens=max_tokens,
        system_instruction=system_prompt,
    )
    response = gemini_client.models.generate_content(
        model=GEMINI_FLASH,
        contents=prompt,
        config=config,
    )
    return response.text or ""


def generate_text(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 4096,
    provider: Optional[str] = None,
) -> LLMResponse:
    """Generate text using primary provider with bidirectional fallback.

    Args:
        prompt: User input prompt.
        system_prompt: Optional system instruction.
        max_tokens: Maximum tokens in generated completion.
        provider: Primary LLM provider ('anthropic' or 'gemini').
                  Defaults to src.config.DEFAULT_LLM_PROVIDER ('anthropic').

    Returns:
        LLMResponse containing the generated text, metadata, and fallback details.

    Raises:
        ValueError: If provider is unknown.
        Exception: If primary fails with non-fallback error, or if both primary and fallback fail.
    """
    selected_provider = (
        provider or getattr(src.config, "DEFAULT_LLM_PROVIDER", "anthropic")
    ).lower().strip()

    if selected_provider == "anthropic":
        primary_model = ANTHROPIC_HAIKU
        fallback_model = GEMINI_FLASH
        fallback_provider = "gemini"

        try:
            text = _call_anthropic(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
            return LLMResponse(
                text=text,
                model=primary_model,
                provider="anthropic",
                fell_back=False,
            )
        except Exception as exc:
            if not is_fallback_candidate(exc, provider="anthropic"):
                logger.error(
                    "Primary model %s failed with non-fallback error: %s",
                    primary_model,
                    exc,
                )
                raise

            logger.warning(
                "Primary model %s unavailable/overloaded (%s). Initiating fallback to Gemini Flash (%s, effort=HIGH)",
                primary_model,
                exc,
                fallback_model,
            )

            try:
                fallback_text = _call_gemini(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                )
                return LLMResponse(
                    text=fallback_text,
                    model=fallback_model,
                    provider=fallback_provider,
                    fell_back=True,
                    fallback_reason=f"Anthropic unavailable/overloaded: {exc}",
                )
            except Exception as fallback_exc:
                logger.error(
                    "Fallback model %s (effort=HIGH) also failed: %s (Original primary error: %s)",
                    fallback_model,
                    fallback_exc,
                    exc,
                )
                raise RuntimeError(
                    f"Both primary model ({primary_model}) and fallback model ({fallback_model}) failed. "
                    f"Primary: {exc} | Fallback: {fallback_exc}"
                ) from fallback_exc

    elif selected_provider == "gemini":
        primary_model = GEMINI_FLASH
        fallback_model = ANTHROPIC_HAIKU
        fallback_provider = "anthropic"

        try:
            text = _call_gemini(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
            return LLMResponse(
                text=text,
                model=primary_model,
                provider="gemini",
                fell_back=False,
            )
        except Exception as exc:
            if not is_fallback_candidate(exc, provider="gemini"):
                logger.error(
                    "Primary model %s failed with non-fallback error: %s",
                    primary_model,
                    exc,
                )
                raise

            logger.warning(
                "Primary model %s unavailable/exhausted (%s). Initiating fallback to Claude Haiku (%s, effort=xhigh)",
                primary_model,
                exc,
                fallback_model,
            )

            try:
                fallback_text = _call_anthropic(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                )
                return LLMResponse(
                    text=fallback_text,
                    model=fallback_model,
                    provider=fallback_provider,
                    fell_back=True,
                    fallback_reason=f"Gemini unavailable/exhausted: {exc}",
                )
            except Exception as fallback_exc:
                logger.error(
                    "Fallback model %s (effort=xhigh) also failed: %s (Original primary error: %s)",
                    fallback_model,
                    fallback_exc,
                    exc,
                )
                raise RuntimeError(
                    f"Both primary model ({primary_model}) and fallback model ({fallback_model}) failed. "
                    f"Primary: {exc} | Fallback: {fallback_exc}"
                ) from fallback_exc

    else:
        raise ValueError(
            f"Unsupported LLM provider '{selected_provider}'. Must be 'anthropic' or 'gemini'."
        )
