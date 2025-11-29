"""In-memory knowledge base used by all agents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from prover.constants import DEFAULT_MAX_TEXT_CHARS, KB_PROMPT_MAX_CONTENT_CHARS
from prover.schemas import KBEntry


def _normalize_text_block(text: str, *, max_chars: int = DEFAULT_MAX_TEXT_CHARS) -> str:
    """Collapse extra whitespace and truncate long Markdown blocks."""

    cleaned = text.strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    if len(cleaned) > max_chars:
        cleaned = f"{cleaned[: max_chars - 1].rstrip()}…"
    return cleaned


@dataclass
class KnowledgeBase:
    """Simple in-memory KB store."""

    entries: dict[str, KBEntry] = field(default_factory=dict)

    def add(self, entry: KBEntry) -> None:
        """Add or replace a KB entry by id."""

        normalized = KBEntry(
            id=entry.id.strip(),
            kind=entry.kind,
            title=entry.title.strip(),
            content_md=_normalize_text_block(entry.content_md),
            tags=[tag.strip() for tag in entry.tags],
            sources=[source.strip() for source in entry.sources],
        )
        self.entries[normalized.id] = normalized

    def extend(self, new_entries: Iterable[KBEntry]) -> None:
        """Add multiple KB entries."""

        for entry in new_entries:
            self.add(entry)

    def snapshot(self) -> list[KBEntry]:
        """Return a deterministic snapshot of KB entries sorted by id."""

        return sorted(self.entries.values(), key=lambda entry: entry.id)

    def render_prompt_md(
        self, *, max_content_chars: int = KB_PROMPT_MAX_CONTENT_CHARS
    ) -> str:
        """Render the KB as compact Markdown suitable for LLM prompts.

        Args:
            max_content_chars: Maximum number of characters from the entry
                content to keep the prompt concise.

        Returns:
            Markdown bullet list summarizing the KB, or ``"None."`` if empty.
        """

        entries = self.snapshot()
        if not entries:
            return "None."

        lines: list[str] = []
        for entry in entries:
            content_raw = entry.content_md.strip()
            content = content_raw.replace("\n", " ")
            if len(content) > max_content_chars:
                content = f"{content[: max_content_chars - 1].rstrip()}…"
            header = f"- [{entry.kind.value}] {entry.id}: {entry.title}".strip()
            lines.append(header)
            if content and content != entry.title.strip():
                lines.append(f"  {content}".strip())
        return "\n".join(lines)

    def render_appendix_lines(self) -> list[str]:
        """Return formatted appendix lines for the final Markdown report."""

        return [
            f"- **{entry.id} ({entry.kind.value})**: {entry.content_md}"
            for entry in self.snapshot()
        ]
