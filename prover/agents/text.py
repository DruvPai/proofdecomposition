"""Text normalization utilities used by multiple agents."""

from __future__ import annotations

import re
from typing import Iterable

from prover.constants import (
    DEFAULT_LOCAL_CONTEXT_LIMIT,
    DEFAULT_MAX_TEXT_CHARS,
    LOCAL_CONTEXT_SNIPPET_CHARS,
)
from prover.graph import AgentOutput
from prover.schemas import (
    ExplorationQuestions,
    KBEntry,
    KBKind,
    OrchestratorStatus,
    SolutionAttempt,
    VerificationReport,
    WorkerStatus,
)


def format_local_context(
    local_context: list[AgentOutput],
    *,
    limit: int = DEFAULT_LOCAL_CONTEXT_LIMIT,
    context_item_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> str:
    """Format local context into compact Markdown for prompting.

    Args:
        local_context: Upstream agent outputs routed to this agent.
        limit: Maximum number of items to include.
        context_item_chars: Maximum number of characters per context item.

    Returns:
        A Markdown string (possibly empty).
    """

    if not local_context:
        return ""

    def _clip(text: str, limit: int) -> str:
        cleaned = text.strip()
        if len(cleaned) <= limit:
            return cleaned
        return f"{cleaned[: limit - 1].rstrip()}…"

    parts: list[str] = []
    for out in local_context[-limit:]:
        normalized = out.normalized
        if isinstance(normalized, SolutionAttempt):
            snippet = _clip(normalized.final_answer_md, LOCAL_CONTEXT_SNIPPET_CHARS)
        elif isinstance(normalized, VerificationReport):
            verdict = "accepted" if normalized.accepted else "rejected"
            snippet = f"verdict={verdict}"
        elif isinstance(normalized, WorkerStatus):
            snippet = (
                f"phase={normalized.phase}, round={normalized.round_index}, "
                f"provers_spawned={normalized.provers_spawned}"
            )
        elif isinstance(normalized, OrchestratorStatus):
            snippet = (
                f"phase={normalized.phase}, "
                f"message={_clip(normalized.message, LOCAL_CONTEXT_SNIPPET_CHARS)}"
            )
        elif isinstance(normalized, ExplorationQuestions):
            snippet = (
                f"{len(normalized.questions)} questions"
                if normalized.questions
                else "no questions"
            )
        else:
            raw = out.raw_text.strip()
            snippet = (
                raw
                if len(raw) <= context_item_chars
                else f"{raw[:context_item_chars]}…"
            )
        parts.append(f"- **{out.agent_type.value}**: {snippet}")
    return "\n".join(parts)


def clean_solution_text(text: str) -> str:
    """Remove common boilerplate scaffolding from a prover's raw Markdown output.

    Args:
        text: Raw LLM output.

    Returns:
        Cleaned Markdown text with redundant headers removed.
    """

    stripped = text.strip()
    fence_match = re.match(r"```[\w+-]*\n(?P<body>.*)\n```$", stripped, re.DOTALL)
    if fence_match:
        stripped = fence_match.group("body")

    lines = stripped.splitlines()
    cleaned_lines: list[str] = []
    skipping_prefix = True
    for line in lines:
        normalized = line.strip()
        if skipping_prefix:
            if not normalized:
                continue
            if re.match(r"output\s*type\s*:", normalized, re.IGNORECASE):
                continue
            if re.match(r"solution\s*:", normalized, re.IGNORECASE):
                continue
            skipping_prefix = False
        cleaned_lines.append(line.rstrip())

    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def looks_like_display_math(paragraph: str) -> bool:
    """Return True if the paragraph resembles a standalone math block."""

    stripped = paragraph.strip()
    return bool(
        stripped
        and (
            (stripped.startswith("\\[") and stripped.endswith("\\]"))
            or (stripped.startswith("$$") and stripped.endswith("$$"))
        )
    )


def extract_result_snippet(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> str:
    """Extract a succinct result statement from a longer solution.

    Args:
        text: Raw or cleaned solution Markdown.
        max_chars: Maximum number of characters to return.

    Returns:
        A compact snippet intended for KB storage and titles.
    """

    cleaned = clean_solution_text(text)
    if not cleaned:
        return ""

    paragraphs = [
        part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()
    ]
    if not paragraphs:
        paragraphs = [cleaned.strip()]

    snippet = paragraphs[-1]
    if looks_like_display_math(snippet) and len(paragraphs) >= 2:
        snippet = f"{paragraphs[-2]}\n{snippet}"

    snippet = snippet.strip()
    if len(snippet) > max_chars:
        snippet = f"{snippet[: max_chars - 1].rstrip()}…"
    return snippet


def derive_result_title(
    text: str,
    fallback: str,
    *,
    snippet_chars: int = DEFAULT_MAX_TEXT_CHARS,
    title_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> str:
    """Derive a short title from a solution snippet."""

    snippet = extract_result_snippet(text, max_chars=snippet_chars)
    first_line = snippet.splitlines()[0].strip() if snippet else ""
    title = first_line or fallback
    if len(title) > title_chars:
        title = f"{title[: title_chars - 1].rstrip()}…"
    return title


def make_kb_entry(node_id: int, title: str, content: str) -> KBEntry:
    """Create a KB entry with deterministic id.

    Args:
        node_id: Node id used to create a stable entry id.
        title: Human-readable title.
        content: Markdown content (raw or cleaned).

    Returns:
        `KBEntry` in `Result` kind.
    """

    return KBEntry(
        id=f"Result {node_id}",
        kind=KBKind.RESULT,
        title=derive_result_title(content, fallback=title),
        content_md=extract_result_snippet(content),
        tags=[],
        sources=[f"agent-{node_id}"],
    )


def prepare_kb_entries(
    entries: Iterable[KBEntry],
    *,
    max_result_chars: int = DEFAULT_MAX_TEXT_CHARS,
    max_other_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> list[KBEntry]:
    """Return KB entries with cleaned content and whitespace normalized.

    Args:
        entries: Incoming KB entries from a tool call or agent output.
        max_result_chars: Result snippet truncation limit.
        max_other_chars: Truncation limit for non-result entries.

    Returns:
        Cleaned list of KB entries ready for KB insertion.
    """

    prepared: list[KBEntry] = []
    for entry in entries:
        if entry.kind == KBKind.RESULT:
            content = extract_result_snippet(
                entry.content_md, max_chars=max_result_chars
            )
        else:
            content = clean_solution_text(entry.content_md).strip()
            if len(content) > max_other_chars:
                content = f"{content[: max_other_chars - 1].rstrip()}…"
        content = re.sub(r"\n{3,}", "\n\n", content)
        prepared.append(
            KBEntry(
                id=entry.id.strip(),
                kind=entry.kind,
                title=entry.title.strip(),
                content_md=content,
                tags=list(entry.tags),
                sources=list(entry.sources),
            )
        )
    return prepared
