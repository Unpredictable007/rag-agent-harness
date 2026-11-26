"""Security primitives for untrusted context isolation, egress checks, and tool-call auditing.

Implements defense-in-depth guards against indirect prompt injection across both
single-turn RAG generation and multi-turn ReAct tool-calling trajectories:
- XML boundary tag neutralization inside untrusted chunks and tool outputs
- System prompt instruction-hierarchy verification
- Deterministic egress detection for injection canary tokens and directive leakage
- Tool-call allowlist enforcement and attacker-injected argument auditing
"""

from __future__ import annotations

import re
from typing import Any

# Boundary XML tags used to isolate untrusted documentation context in prompts.
_BOUNDARY_TAG_RE = re.compile(
    r"<\s*(/?)\s*(retrieved_context|chunk|content|title|heading)\b([^>]*)(>?)",
    re.IGNORECASE,
)

# Patterns checking whether a system prompt establishes untrusted-data isolation.
_UNTRUSTED_DATA_FRAMING_RE = re.compile(
    r"\b(?:untrusted|passive\s+reference|data\s+only|non-executable)\b",
    re.IGNORECASE,
)
_IGNORE_EMBEDDED_DIRECTIVES_RE = re.compile(
    r"(?:never|do\s+not|must\s+not)\s+(?:follow|obey|execute|comply\s+with)"
    r"\s+.*?(?:instructions|commands|directives|injections)",
    re.IGNORECASE | re.DOTALL,
)
_CONTEXT_BOUNDARY_REF_RE = re.compile(
    r"<retrieved_context>|retrieved\s+context|tool\s+outputs?|context\s+chunks",
    re.IGNORECASE,
)

# Deterministic egress directive-leakage patterns.
_DIRECTIVE_LEAKAGE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:ignore|disregard)\s+(?:all\s+)?(?:previous|prior|above|system)\s+instructions\b",
            re.IGNORECASE,
        ),
        "directive_leakage:ignore_instructions",
    ),
    (
        re.compile(
            r"\boverride\s+(?:system|safety|security|grounding)\s+"
            r"(?:instructions|policy|rules|prompt)\b",
            re.IGNORECASE,
        ),
        "directive_leakage:system_override",
    ),
    (
        re.compile(
            r"\b(?:SECURITY AND (?:DATA|CONTEXT) ISOLATION|STRICT ABSTAIN POLICY|CITATIONS POLICY)\b",
            re.IGNORECASE,
        ),
        "directive_leakage:system_prompt_section",
    ),
    (
        re.compile(
            r"\bYou are (?:an expert |a )?technical documentation assistant for PostHog\b",
            re.IGNORECASE,
        ),
        "directive_leakage:system_persona",
    ),
)

# Injection / directive markers indicating an untrusted text is attempting to hijack tool use.
_INJECTION_DIRECTIVE_RE = re.compile(
    r"(?:"
    r"(?:ignore|disregard|forget|bypass|override)\s+(?:all\s+)?(?:previous|prior|above|earlier|system|safety)\s+(?:instructions|prompts|rules|directives)"
    r"|(?:system|developer|admin|security|priority)\s+override"
    r"|(?:new|mandatory|urgent|critical|hidden|secret|system)\s+(?:directive|instruction|command|order)"
    r"|prompt\s+injection"
    r"|(?:you\s+)?must\s+(?:immediately\s+)?(?:call|invoke|run|search|execute|use)\b"
    r"|(?:call|invoke|run|execute|use)\s+(?:the\s+)?tool\b"
    r"|exfiltrat(?:e|ion)"
    r")",
    re.IGNORECASE,
)


def _escape_boundary_tag(match: re.Match[str]) -> str:
    """Replace angle brackets inside matched boundary XML tags with HTML entities."""
    return match.group(0).replace("<", "&lt;").replace(">", "&gt;")


def sanitize_untrusted_context(text: str) -> str:
    """Neutralize XML boundary tag-breakout sequences inside untrusted text.

    Escapes `<` and `>` on boundary tags (`retrieved_context`, `chunk`, `content`,
    `title`, `heading`, case-insensitive) while preserving non-boundary technical
    HTML/JSX tags (e.g., `<Button>`, `<div>`, `Array<string>`) in documentation.
    """
    if not text:
        return ""
    return _BOUNDARY_TAG_RE.sub(_escape_boundary_tag, str(text))


def has_instruction_hierarchy(system_prompt: str) -> bool:
    """Verify a system prompt enforces untrusted-data isolation and instruction hierarchy."""
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        return False
    return bool(
        _UNTRUSTED_DATA_FRAMING_RE.search(system_prompt)
        and _IGNORE_EMBEDDED_DIRECTIVES_RE.search(system_prompt)
        and _CONTEXT_BOUNDARY_REF_RE.search(system_prompt)
    )


def detect_injection_output(
    text: str,
    canary_tokens: tuple[str, ...] = ("PWNED",),
) -> list[str]:
    """Flag prompt-injection canary compliance or directive leakage in model output."""
    if not text:
        return []

    flags: list[str] = []
    for token in canary_tokens:
        cleaned = token.strip()
        if not cleaned:
            continue
        if re.match(r"^\w+$", cleaned):
            pattern = re.compile(rf"\b{re.escape(cleaned)}\b", re.IGNORECASE)
        else:
            pattern = re.compile(re.escape(cleaned), re.IGNORECASE)
        if pattern.search(text):
            flag = f"canary_token_detected:{cleaned}"
            if flag not in flags:
                flags.append(flag)

    for pattern, label in _DIRECTIVE_LEAKAGE_PATTERNS:
        if pattern.search(text) and label not in flags:
            flags.append(label)

    return flags


def _extract_string_values(obj: Any) -> list[str]:
    """Recursively collect non-empty string values from tool input payloads."""
    values: list[str] = []
    if isinstance(obj, str):
        stripped = obj.strip()
        if stripped:
            values.append(stripped)
    elif isinstance(obj, dict):
        for val in obj.values():
            values.extend(_extract_string_values(val))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            values.extend(_extract_string_values(item))
    return values


def _flexible_ws_pattern(text: str) -> str:
    """Build a case-insensitive regex pattern matching `text` across varied internal whitespace."""
    tokens = text.strip().split()
    if len(tokens) <= 1:
        return re.escape(text.strip())
    return r"\s+".join(re.escape(tok) for tok in tokens)


def _arg_matches_untrusted_directive(
    tool_name: str,
    arg_val: str,
    untrusted_text: str,
) -> bool:
    """Return True if an argument appears in an attacker-injected tool directive."""
    escaped_tool = re.escape(tool_name)
    arg_pat = _flexible_ws_pattern(arg_val)
    quote_chars = r"`'\"\u201c\u201d\u2018\u2019"

    # Pattern 1: Explicit imperative referencing the tool name and argument value.
    explicit_cmd_re = re.compile(
        rf"(?:call|invoke|run|execute|use)\s+(?:the\s+)?(?:tool\s+)?"
        rf"[{quote_chars}\[]?{escaped_tool}[{quote_chars}\]]?[\s\S]{{0,180}}{arg_pat}",
        re.IGNORECASE,
    )
    if explicit_cmd_re.search(untrusted_text):
        return True

    # Pattern 2: Function-call or JSON-style tool invocation containing the argument.
    call_syntax_re = re.compile(
        rf"{escaped_tool}\s*\([^\)]{{0,180}}{arg_pat}",
        re.IGNORECASE,
    )
    if call_syntax_re.search(untrusted_text):
        return True

    # Pattern 3: Injection/override directive or explicit tool-invocation directive
    # within the same local context window as a framed argument.
    if len(arg_val) < 4:
        return False

    tool_directive_re = re.compile(
        rf"(?:call|invoke|run|execute|use)\s+(?:the\s+)?(?:tool\s+)?"
        rf"[{quote_chars}\[]?{escaped_tool}[{quote_chars}\]]?\b",
        re.IGNORECASE,
    )
    arg_re = re.compile(arg_pat, re.IGNORECASE)
    framed_arg_re = re.compile(
        rf"(?:[{quote_chars}]{arg_pat}[{quote_chars}]"
        rf"|(?:query|input|arg|argument|search\s+for|with)[{quote_chars}]?\s*[:=]?\s*[{quote_chars}]?{arg_pat})",
        re.IGNORECASE,
    )
    for match in arg_re.finditer(untrusted_text):
        start = max(0, match.start() - 220)
        end = min(len(untrusted_text), match.end() + 220)
        window = untrusted_text[start:end]
        if (
            _INJECTION_DIRECTIVE_RE.search(window) or tool_directive_re.search(window)
        ) and framed_arg_re.search(window):
            return True

    return False


def audit_tool_call_security(
    tool_name: str,
    tool_input: dict[str, Any],
    allowed_tools: set[str],
    observed_untrusted_texts: list[str],
) -> list[str]:
    """Audit a proposed tool call for allowlist violations and injected arguments.

    Flags:
    - Any tool name not present in `allowed_tools`.
    - Any tool argument string that matches an attacker-injected directive observed
      in prior untrusted tool outputs.
    """
    flags: list[str] = []

    if not tool_name or tool_name not in allowed_tools:
        flags.append(f"unauthorized_tool:{tool_name}")

    arg_values = _extract_string_values(tool_input)
    for untrusted in observed_untrusted_texts:
        if not untrusted:
            continue
        for arg_val in arg_values:
            if len(arg_val) < 3:
                continue
            if _arg_matches_untrusted_directive(tool_name, arg_val, untrusted):
                flag = f"injected_tool_argument:{tool_name}:{arg_val}"
                if flag not in flags:
                    flags.append(flag)

    return flags
