"""Parser agent implementation.

The parser agent optionally delegates to an LLM with a supplied JSON Schema to
convert Markdown-ish text into typed JSON payloads. In tests/offline runs, the
same interface can be mocked.
"""

from __future__ import annotations

import json
from typing import Any

from prover.agents.base import BaseAgent
from prover.constants import DEFAULT_MAX_QUESTIONS
from prover.config import RunConfig
from prover.graph import AgentNode, AgentOutput, AgentType
from prover.kb import KnowledgeBase
from prover.trace import TraceLogger


class ParserAgent(BaseAgent):
    """Parser agent converting raw text into typed normalized schemas."""

    @classmethod
    async def llm_parse_text(
        cls,
        config: RunConfig,
        trace: TraceLogger,
        node_id: int,
        target: str,
        text: str,
        schema: dict[str, Any] | None,
        *,
        max_questions: int = DEFAULT_MAX_QUESTIONS,
    ) -> dict[str, Any] | None:
        """Parse Markdown into a JSON object using the parser LLM.

        Args:
            config: Run configuration containing parser settings.
            trace: Trace logger for auditability.
            node_id: Node id for logging context.
            target: Semantic target (e.g., "solution_attempt").
            text: Raw Markdown to parse.
            schema: JSON schema provided by the caller (required for LLM path).
            max_questions: Cap for exploration question extraction.

        Returns:
            Parsed JSON data on success, else None.
        """

        agent = cls(config)
        return await agent._llm_parse(
            target=target,
            text=text,
            schema=schema,
            max_questions=max_questions,
            trace=trace,
            node_id=node_id,
        )

    async def _llm_parse(
        self,
        *,
        target: str,
        text: str,
        schema: dict[str, Any] | None,
        max_questions: int,
        trace: TraceLogger,
        node_id: int,
    ) -> dict[str, Any] | None:
        """Invoke an LLM with a provided JSON schema to parse Markdown."""

        if not schema:
            return None

        parser_llm = self.config.parser.llm
        if parser_llm is None:
            raise RuntimeError("Parser LLM configuration must be provided.")

        response = await self._chat_with_trace(
            llm=parser_llm,
            trace=trace,
            node_id=node_id,
            system_prompt=self.config.parser.system_prompt.strip(),
            user_prompt=f"""
JSON Schema:
{json.dumps(schema, indent=2)}

Markdown input:
```
{text}
```
                    """.strip(),
            response_format={"type": "json_object"},
            trace_info={"schema_target": target},
        )

        content = response["choices"][0]["message"].get("content", "{}")
        try:
            parsed = json.loads(content)
        except Exception:
            return None
        return parsed

    async def run(
        self, node: AgentNode, kb: KnowledgeBase, trace: TraceLogger
    ) -> AgentOutput:
        """Parse raw text into a requested schema.

        Expected task payload:
            - target: str, descriptive label used for logging.
            - text: str, raw text to parse.
            - schema: optional JSON schema dict supplied by the caller.

        Args:
            node: Graph node wrapper.
            kb: Knowledge base (unused for deterministic parsing).
            trace: Trace logger.

        Returns:
            A parser agent output with `normalized` set to the parsed JSON data
            (when available) or the original text.
        """

        task = node.inputs.get("task", {})
        target = task.get("target", "identity")
        text = task.get("text", "")
        raw_max_questions = task.get("max_questions", DEFAULT_MAX_QUESTIONS)
        try:
            max_questions = int(raw_max_questions)
        except (TypeError, ValueError):
            max_questions = DEFAULT_MAX_QUESTIONS

        schema = task.get("schema")
        parsed = await self._llm_parse(
            target=target,
            text=text,
            schema=schema,
            max_questions=max_questions,
            trace=trace,
            node_id=node.id,
        )
        normalized: Any = parsed if parsed is not None else text
        return AgentOutput(
            agent_type=AgentType.PARSER,
            raw_text=text,
            normalized=normalized,
            kb_writes=[],
            spawn_requests=[],
        )
