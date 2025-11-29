"""Prover agent implementation."""

from __future__ import annotations

import json
from typing import Any
import re

from prover.agents.base import BaseAgent
from prover.agents.parser import ParserAgent
from prover.agents.text import (
    clean_solution_text,
    format_local_context,
    make_kb_entry,
    prepare_kb_entries,
)
from prover.constants import KB_SUMMARY_STATEMENT_CHARS, KB_SUMMARY_TITLE_CHARS
from prover.graph import AgentNode, AgentOutput, AgentType
from prover.kb import KnowledgeBase
from prover.schemas import KBEntry, KBKind, SolutionAttempt
from prover.trace import TraceLogger


class ProverAgent(BaseAgent):
    """Prover agent producing a candidate solution attempt."""

    async def _summarize_kb_entries(
        self,
        *,
        node_id: int,
        trace: TraceLogger,
        problem: str,
        source_text_md: str,
        entries: list[KBEntry],
    ) -> list[KBEntry]:
        """Summarize KB entries into concise statements (LLM-backed, optional).

        Args:
            node_id: Node id for trace association.
            trace: Trace logger.
            problem: The current task/problem statement.
            entries: Draft KB entries (may contain proof-like text).

        Returns:
            KB entries rewritten so that `title` and `content_md` are succinct
            statement-level summaries.
        """

        summarizer = self.config.kb_summarizer
        if summarizer.llm is None or not entries:
            return entries

        def _clip(text: str, limit: int) -> str:
            cleaned = text.strip()
            if len(cleaned) <= limit:
                return cleaned
            return f"{cleaned[: limit - 1].rstrip()}…"

        summarized: list[KBEntry] = []
        for entry in entries:
            response = await self._chat_with_trace(
                llm=summarizer.llm,
                trace=trace,
                node_id=node_id,
                system_prompt=summarizer.system_prompt,
                user_prompt=f"""
Problem:
{problem}

Source text (may be a proof/explanation):
```
{source_text_md}
```

Draft KB entry:
id: {entry.id}
kind: {entry.kind.value}
title: {entry.title}

content_md:
```
{entry.content_md}
```

Return JSON with keys "title" and "statement_md".
                """.strip(),
                response_format={"type": "json_object"},
                trace_info={"kb_entry_id": entry.id},
            )
            content = response["choices"][0]["message"].get("content", "{}")
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = {}
            title = _clip(str(parsed.get("title", entry.title)), KB_SUMMARY_TITLE_CHARS)
            statement = _clip(
                str(parsed.get("statement_md", entry.content_md)),
                KB_SUMMARY_STATEMENT_CHARS,
            )
            summarized.append(
                KBEntry(
                    id=entry.id,
                    kind=entry.kind,
                    title=title,
                    content_md=statement,
                    tags=list(entry.tags),
                    sources=list(entry.sources),
                )
            )
        return summarized

    @staticmethod
    def coerce_solution_attempt(data: dict[str, Any]) -> SolutionAttempt:
        """Convert generic JSON data into a `SolutionAttempt`."""

        kb_updates: list[KBEntry] = []
        for entry in data.get("kb_updates", []) or []:
            kind_value = entry.get("kind", KBKind.RESULT.value)
            try:
                kind = KBKind(kind_value)
            except ValueError:
                kind = KBKind.RESULT
            kb_updates.append(
                KBEntry(
                    id=str(entry.get("id", "Result")),
                    kind=kind,
                    title=str(entry.get("title", "Untitled")),
                    content_md=str(entry.get("content_md", "")),
                    tags=list(entry.get("tags", []) or []),
                    sources=list(entry.get("sources", []) or []),
                )
            )
        return SolutionAttempt(
            final_answer_md=clean_solution_text(str(data.get("final_answer_md", ""))),
            outline_steps=list(data.get("outline_steps", []) or []),
            kb_updates=prepare_kb_entries(kb_updates),
            claims_incorrect_conclusion=bool(
                data.get("claims_incorrect_conclusion", False)
            ),
        )

    @staticmethod
    def parse_solution_attempt(text: str) -> SolutionAttempt:
        """Parse a prover-style Markdown document into a `SolutionAttempt`.

        Args:
            text: Raw Markdown-ish output.

        Returns:
            A best-effort `SolutionAttempt` preserving `final_answer_md` verbatim.
        """

        output_type_match = re.search(
            r"output type:\s*(plan|solution|error)", text, re.IGNORECASE
        )
        output_type = (
            output_type_match.group(1).lower() if output_type_match else "solution"
        )

        def _section_after(header: str) -> str | None:
            match = re.search(header, text, re.IGNORECASE)
            if not match:
                return None
            return text[match.end() :].strip()

        outline_steps: list[str] = []
        claims_incorrect = output_type == "error"

        if output_type == "plan":
            plan_text = (
                _section_after(r"#\s*plan\b") or _section_after(r"\bplan\b") or text
            )
            for line in plan_text.splitlines():
                if re.match(r"\s*[-*]\s+", line) or re.match(r"\s*\d+[\).\s]", line):
                    cleaned = re.sub(r"^\s*[-*\d\).\s]+", "", line).strip()
                    if cleaned:
                        outline_steps.append(cleaned)
        return SolutionAttempt(
            final_answer_md=clean_solution_text(text),
            outline_steps=outline_steps,
            kb_updates=[],
            claims_incorrect_conclusion=claims_incorrect,
        )

    @staticmethod
    def _solution_schema() -> dict[str, Any]:
        """Return the JSON schema used to parse solution attempts."""

        kb_entry_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "kind": {"type": "string"},
                "title": {"type": "string"},
                "content_md": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}, "default": []},
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
            },
            "required": ["id", "kind", "title", "content_md"],
        }
        return {
            "type": "object",
            "properties": {
                "final_answer_md": {"type": "string"},
                "outline_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
                "kb_updates": {
                    "type": "array",
                    "items": kb_entry_schema,
                    "default": [],
                },
                "claims_incorrect_conclusion": {"type": "boolean", "default": False},
            },
            "required": ["final_answer_md"],
        }

    async def run(
        self, node: AgentNode, kb: KnowledgeBase, trace: TraceLogger
    ) -> AgentOutput:
        """Generate a proof attempt using an LLM and return a normalized solution."""

        problem = node.inputs["task"].get("problem", node.inputs["problem"])
        feedback_md = node.inputs["task"].get("feedback_md")
        local_context: list[AgentOutput] = node.inputs.get("local_context", [])

        prover_llm = self.config.prover.llm
        if prover_llm is None:
            raise RuntimeError("Prover LLM configuration must be provided.")

        response = await self._chat_with_trace(
            llm=prover_llm,
            trace=trace,
            node_id=node.id,
            system_prompt=self.config.prover.system_prompt,
            user_prompt=f"""
Problem:
{problem}

Context hierarchy:
{node.inputs.get("context_hierarchy_md") or "None."}

Local context:
{format_local_context(local_context) or "None."}

Verifier feedback (if any):
{feedback_md or "None."}

KB:
{kb.render_prompt_md()}""".strip(),
            tools=self.tool_defs(),
        )

        choice = response["choices"][0]["message"]
        tool_calls = choice.get("tool_calls", [])
        spawns, kb_writes, finish_text = self.parse_tool_calls(tool_calls)
        final_md = (
            finish_text or choice.get("content", "") or f"Sketch solution for {problem}"
        )

        parsed = await ParserAgent.llm_parse_text(
            self.config,
            trace=trace,
            node_id=node.id,
            target="solution_attempt",
            text=final_md,
            schema=self._solution_schema(),
        )
        normalized = (
            self.coerce_solution_attempt(parsed)
            if parsed is not None
            else self.parse_solution_attempt(final_md)
        )

        if kb_writes:
            normalized.kb_updates = prepare_kb_entries(kb_writes)
        elif not normalized.kb_updates:
            normalized.kb_updates = prepare_kb_entries(
                [make_kb_entry(node.id, "Auto-generated fact", final_md)]
            )
        normalized.kb_updates = await self._summarize_kb_entries(
            node_id=node.id,
            trace=trace,
            problem=str(problem),
            source_text_md=str(final_md),
            entries=normalized.kb_updates,
        )

        return AgentOutput(
            agent_type=AgentType.PROVER,
            raw_text=normalized.final_answer_md,
            normalized=normalized,
            kb_writes=[],
            spawn_requests=spawns,
        )
