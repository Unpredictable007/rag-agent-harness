"""Calibrated LLM-as-a-Judge for Correctness and Faithfulness (Milestone M6).

Implements independent evaluation judges using Claude 5.5 Sonnet with fallback to Gemini Flash:
- CorrectnessJudge: Compares generated answer against ground truth reference answer.
- FaithfulnessJudge: Evaluates whether claims in generated answer are grounded in retrieved context.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

from google.genai import types as genai_types
from pydantic import BaseModel, Field

import src.config
from src.config import (
    ANTHROPIC_SONNET,
    GEMINI_FLASH,
    get_anthropic_client,
    get_gemini_client,
)
from src.generation import format_retrieved_context
from src.llm import is_fallback_candidate

logger = logging.getLogger(__name__)

CORRECTNESS_SYSTEM_PROMPT = """You are an expert impartial evaluation judge assessing the factual and semantic correctness of a candidate answer against a gold-standard reference answer for a technical user question about PostHog.

EVALUATION CRITERIA:
Compare the candidate answer against the reference answer strictly based on factual alignment:
- 1.0 (Fully Correct): The candidate answer captures all core factual claims, technical specifics, and directives present in the reference answer. Minor stylistic, phrasing, or ordering differences are acceptable as long as no key facts are missing or distorted.
- 0.5 (Partially Correct): 
  * For multi-part questions: Answers at least one major part of the question accurately while omitting, partially answering, or stating lack of documentation for another major part.
  * For single-part questions: Captures the essential concept or majority of claims from the reference answer, but omits important technical specifics, configuration details, or includes minor inaccuracies that do not completely invalidate the answer.
- 0.0 (Incorrect): The candidate answer fails to answer ANY core part of the question correctly, makes factually false claims, directly contradicts the reference answer, or hallucinates fictitious features.

MULTI-PART QUESTION RULE:
If a user question asks multiple distinct questions (e.g. asking both "how to verify before launch" AND "how to calculate running time", or asking both "SDK version" AND "cost impact"):
- If the candidate answer correctly answers one part but omits or fails to answer the other part, it MUST be scored 0.5 (Partially Correct), NOT 0.0.
- Only assign 0.0 if NONE of the question's core parts are answered correctly.

IMPORTANT GUIDELINES:
- Focus strictly on factual accuracy and technical completeness compared to the reference answer.
- Do NOT penalize the candidate answer for containing additional helpful context or documentation URLs as long as they are accurate.
- Do NOT reward answers that sound persuasive or verbose if they omit the actual answer or misstate key parameters.

OUTPUT FORMAT:
Provide your step-by-step reasoning first, then assign a score of exactly 1.0, 0.5, or 0.0.
Respond ONLY with a valid JSON object matching this schema:
{
  "reasoning": "<concise step-by-step factual analysis>",
  "score": <1.0 | 0.5 | 0.0>
}
Do not include any text, markdown backticks, or comments outside the JSON object."""

FAITHFULNESS_SYSTEM_PROMPT = """You are an expert impartial evaluation judge assessing the faithfulness (factual grounding) of a candidate answer against retrieved context chunks from PostHog documentation.

EVALUATION CRITERIA:
Verify whether EVERY factual claim, instruction, number, and statement in the candidate answer is directly supported by or reasonably entailed by the provided context chunks.
- 1.0 (Faithful / Grounded): Every factual statement in the candidate answer is directly stated in or reasonably entailed by the context chunks. There is zero external hallucination and no reliance on unsupported external knowledge.
- 0.0 (Unfaithful / Hallucinated): The candidate answer makes one or more factual claims, provides instructions, or asserts specifics that are NOT supported by the context chunks, or that directly contradict the context chunks.

IMPORTANT RULES:
- Do NOT judge whether the answer is "helpful" or "correct in the real world" - judge ONLY whether it is grounded in the provided context chunks.
- Standard procedural steps or reasonable common-sense entailments (e.g. saying to update systems to use a new key when rotating credentials, or navigating to project settings) are considered grounded unless they directly conflict with the text.
- If the candidate answer acknowledges limitations or states that the context does not specify something, that is considered faithful.
- If the candidate answer asserts fictitious features, unsupported URLs, or invented configuration parameters not found in the chunks, it MUST be scored 0.0.

OUTPUT FORMAT:
Provide your step-by-step reasoning first, verifying each claim against the context chunks, then assign a score of exactly 1.0 or 0.0.
Respond ONLY with a valid JSON object matching this schema:
{
  "reasoning": "<concise step-by-step claim verification against chunks>",
  "score": <1.0 | 0.0>
}
Do not include any text, markdown backticks, or comments outside the JSON object."""


class JudgeResult(BaseModel):
    """Structured evaluation output produced by an LLM-as-a-judge."""

    score: float = Field(..., description="Evaluation score: 1.0, 0.5, 0.0 for correctness; 1.0, 0.0 for faithfulness")
    reasoning: str = Field(..., description="Step-by-step justification for the assigned score")
    judge_type: str = Field(..., description="Judge type identifier: 'correctness' or 'faithfulness'")
    model: str = Field(..., description="Model identifier used for evaluation")
    provider: str = Field(..., description="LLM provider name: 'anthropic', 'gemini', or 'internal'")
    latency_ms: float = Field(..., description="Evaluation latency in milliseconds")


def parse_judge_json(raw_text: str) -> dict[str, Any]:
    """Parse JSON object from judge LLM completion with fallback extraction."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text, strict=False)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            parsed = json.loads(match.group(0), strict=False)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse valid JSON dictionary from judge output: {raw_text[:200]}")


def _call_judge_llm(
    prompt: str,
    system_prompt: str,
    primary_model: Optional[str] = None,
    primary_provider: Optional[str] = None,
    max_tokens: int = 4096,
) -> tuple[str, str, str, float]:
    """Execute LLM call for judge with primary model and automatic fallback.

    Primary: Claude 5.5 Sonnet (ANTHROPIC_SONNET).
    Fallback: Gemini 3.8 Flash (GEMINI_FLASH) on retryable / overload errors.

    Returns:
        tuple of (raw_text, model_name, provider_name, latency_ms)
    """
    model = primary_model or ANTHROPIC_SONNET
    provider = (primary_provider or "anthropic").lower().strip()

    t0 = time.perf_counter()

    if provider == "anthropic":
        try:
            anthropic_client = get_anthropic_client()
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": "medium"},
            )
            text_blocks = [
                b.text
                for b in response.content
                if getattr(b, "type", None) == "text"
                or (hasattr(b, "text") and getattr(b, "type", None) != "thinking")
            ]
            raw_text = "\n".join(text_blocks).strip()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return raw_text, model, "anthropic", latency_ms

        except Exception as exc:
            if not is_fallback_candidate(exc, provider="anthropic"):
                logger.error("Primary judge Anthropic (%s) failed with fatal error: %s", model, exc)
                raise

            logger.warning(
                "Primary judge Anthropic (%s) unavailable/overloaded (%s). Initiating fallback to Gemini Flash (%s)",
                model,
                exc,
                GEMINI_FLASH,
            )
            try:
                gemini_client = get_gemini_client()
                config = genai_types.GenerateContentConfig(
                    thinking_config=genai_types.ThinkingConfig(thinking_level="HIGH"),
                    max_output_tokens=max_tokens,
                    system_instruction=system_prompt,
                )
                gemini_response = gemini_client.models.generate_content(
                    model=GEMINI_FLASH,
                    contents=prompt,
                    config=config,
                )
                raw_text = gemini_response.text or ""
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return raw_text, GEMINI_FLASH, "gemini", latency_ms
            except Exception as fallback_exc:
                logger.error(
                    "Fallback judge Gemini (%s) also failed: %s (Original error: %s)",
                    GEMINI_FLASH,
                    fallback_exc,
                    exc,
                )
                raise RuntimeError(
                    f"Both primary judge ({model}) and fallback judge ({GEMINI_FLASH}) failed: {exc} | {fallback_exc}"
                ) from fallback_exc

    elif provider == "gemini":
        try:
            gemini_client = get_gemini_client()
            config = genai_types.GenerateContentConfig(
                thinking_config=genai_types.ThinkingConfig(thinking_level="HIGH"),
                max_output_tokens=max_tokens,
                system_instruction=system_prompt,
            )
            gemini_response = gemini_client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            raw_text = gemini_response.text or ""
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return raw_text, model, "gemini", latency_ms
        except Exception as exc:
            if not is_fallback_candidate(exc, provider="gemini"):
                raise
            logger.warning(
                "Primary judge Gemini (%s) unavailable (%s). Initiating fallback to Anthropic (%s)",
                model,
                exc,
                ANTHROPIC_SONNET,
            )
            try:
                anthropic_client = get_anthropic_client()
                response = anthropic_client.messages.create(
                    model=ANTHROPIC_SONNET,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    output_config={"effort": "medium"},
                )
                text_blocks = [
                    b.text
                    for b in response.content
                    if getattr(b, "type", None) == "text"
                    or (hasattr(b, "text") and getattr(b, "type", None) != "thinking")
                ]
                raw_text = "\n".join(text_blocks).strip()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return raw_text, ANTHROPIC_SONNET, "anthropic", latency_ms
            except Exception as fallback_exc:
                logger.error(
                    "Fallback judge Anthropic (%s) also failed: %s (Original error: %s)",
                    ANTHROPIC_SONNET,
                    fallback_exc,
                    exc,
                )
                raise RuntimeError(
                    f"Both primary judge ({model}) and fallback judge ({ANTHROPIC_SONNET}) failed: {exc} | {fallback_exc}"
                ) from fallback_exc

    else:
        raise ValueError(f"Unsupported judge provider '{provider}'. Must be 'anthropic' or 'gemini'.")


class CorrectnessJudge:
    """Evaluates semantic and factual correctness of generated answers against reference answers."""

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        max_tokens: int = 4096,
    ):
        self.model = model or ANTHROPIC_SONNET
        self.provider = provider or "anthropic"
        self.max_tokens = max_tokens

    def evaluate(
        self,
        question: str,
        reference_answer: str,
        generated_answer: Optional[str],
        answerable: bool = True,
        abstain: bool = False,
    ) -> JudgeResult:
        """Evaluate correctness according to the standardized benchmark rubric.

        Rubric:
        - If reference is unanswerable (answerable == False):
          - Model abstained (abstain == True): score = 1.0 (Correctly abstained)
          - Model answered (abstain == False): score = 0.0 (Failed to abstain)
        - If reference is answerable (answerable == True):
          - Model abstained (abstain == True): score = 0.0 (False abstain)
          - Model answered: Evaluates semantic correctness using Claude 5.5 Sonnet:
            - 1.0: Fully Correct (captures all core factual claims from reference)
            - 0.5: Partially Correct (captures some claims, misses key specifics)
            - 0.0: Incorrect (factually inaccurate or contradicts reference)
        """
        # 1. Deterministic abstain handling for unanswerable questions
        if not answerable:
            if abstain:
                return JudgeResult(
                    score=1.0,
                    reasoning="Correctly abstained on unanswerable question.",
                    judge_type="correctness",
                    model="deterministic-rule",
                    provider="internal",
                    latency_ms=0.0,
                )
            else:
                return JudgeResult(
                    score=0.0,
                    reasoning="Failed to abstain on unanswerable question.",
                    judge_type="correctness",
                    model="deterministic-rule",
                    provider="internal",
                    latency_ms=0.0,
                )

        # 2. Deterministic abstain handling for answerable questions
        if abstain or not generated_answer or not generated_answer.strip():
            return JudgeResult(
                score=0.0,
                reasoning="False abstain on answerable question.",
                judge_type="correctness",
                model="deterministic-rule",
                provider="internal",
                latency_ms=0.0,
            )

        # 3. LLM semantic correctness evaluation
        prompt = (
            f"<question>\n{question.strip()}\n</question>\n\n"
            f"<reference_answer>\n{reference_answer.strip()}\n</reference_answer>\n\n"
            f"<candidate_answer>\n{generated_answer.strip()}\n</candidate_answer>\n\n"
            "Evaluate whether the candidate answer is factually correct and complete according to the reference answer."
        )

        raw_text, used_model, used_provider, latency_ms = _call_judge_llm(
            prompt=prompt,
            system_prompt=CORRECTNESS_SYSTEM_PROMPT,
            primary_model=self.model,
            primary_provider=self.provider,
            max_tokens=self.max_tokens,
        )

        parsed = parse_judge_json(raw_text)
        raw_score = float(parsed.get("score", 0.0))
        reasoning = str(parsed.get("reasoning", "")).strip()

        # Score normalization: clamp to nearest valid rubric tier {0.0, 0.5, 1.0}
        valid_tiers = [0.0, 0.5, 1.0]
        clamped_score = min(valid_tiers, key=lambda tier: abs(tier - raw_score))

        return JudgeResult(
            score=clamped_score,
            reasoning=reasoning,
            judge_type="correctness",
            model=used_model,
            provider=used_provider,
            latency_ms=latency_ms,
        )


class FaithfulnessJudge:
    """Evaluates whether every claim in generated answer is grounded in retrieved context chunks."""

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        max_tokens: int = 4096,
    ):
        self.model = model or ANTHROPIC_SONNET
        self.provider = provider or "anthropic"
        self.max_tokens = max_tokens

    def evaluate(
        self,
        question: str,
        retrieved_context: str | list[dict | Any],
        generated_answer: Optional[str],
        abstain: bool = False,
    ) -> JudgeResult:
        """Evaluate faithfulness according to the grounding rubric.

        Rubric:
        - If model abstained (abstain == True): score = 1.0 (Abstained from unsupported claims)
        - If model answered: Evaluates whether EVERY claim is grounded in retrieved context:
          - 1.0: Faithful / Grounded (all claims supported by context chunks, zero hallucinations)
          - 0.0: Unfaithful / Hallucinated (contains claims not supported by context)
        """
        # 1. Deterministic abstain handling
        if abstain:
            return JudgeResult(
                score=1.0,
                reasoning="Abstained from asserting unsupported claims.",
                judge_type="faithfulness",
                model="deterministic-rule",
                provider="internal",
                latency_ms=0.0,
            )

        if not generated_answer or not generated_answer.strip():
            return JudgeResult(
                score=0.0,
                reasoning="Empty answer provided without abstaining.",
                judge_type="faithfulness",
                model="deterministic-rule",
                provider="internal",
                latency_ms=0.0,
            )

        # 2. Format context if provided as list
        if isinstance(retrieved_context, list):
            context_str = format_retrieved_context(retrieved_context)
        else:
            context_str = str(retrieved_context)

        if not context_str.strip() or context_str.strip() == "<retrieved_context>\n</retrieved_context>":
            return JudgeResult(
                score=0.0,
                reasoning="No context chunks provided to ground the generated answer.",
                judge_type="faithfulness",
                model="deterministic-rule",
                provider="internal",
                latency_ms=0.0,
            )

        # 3. LLM faithfulness evaluation
        prompt = (
            f"<question>\n{question.strip()}\n</question>\n\n"
            f"{context_str.strip()}\n\n"
            f"<candidate_answer>\n{generated_answer.strip()}\n</candidate_answer>\n\n"
            "Verify whether every factual claim in the candidate answer is directly supported by the context chunks."
        )

        raw_text, used_model, used_provider, latency_ms = _call_judge_llm(
            prompt=prompt,
            system_prompt=FAITHFULNESS_SYSTEM_PROMPT,
            primary_model=self.model,
            primary_provider=self.provider,
            max_tokens=self.max_tokens,
        )

        parsed = parse_judge_json(raw_text)
        raw_score = float(parsed.get("score", 0.0))
        reasoning = str(parsed.get("reasoning", "")).strip()

        # Score normalization: clamp to binary {0.0, 1.0}
        clamped_score = 1.0 if raw_score >= 0.5 else 0.0

        return JudgeResult(
            score=clamped_score,
            reasoning=reasoning,
            judge_type="faithfulness",
            model=used_model,
            provider=used_provider,
            latency_ms=latency_ms,
        )
