"""Shared agent base class and tool-call helpers."""

from __future__ import annotations

import json
import os
from typing import Any

from prover.config import LLMConfig, RunConfig
from prover.graph import AgentType, SpawnRequest
from prover.llm import LLMClient, OpenRouterClient
from prover.schemas import KBEntry, KBKind
from prover.trace import TraceLogger


Message = dict[str, Any]


class BaseAgent:
    """Base agent with shared helpers.

    Attributes:
        config: Run configuration for the current run.
    """

    def __init__(self, config: RunConfig) -> None:
        """Initialize the agent with run-level configuration.

        Args:
            config: Run configuration for the current run.
        """

        self.config = config

    def _client_for(self, llm_config: LLMConfig) -> LLMClient:
        """Instantiate an OpenRouter client using the provided configuration.

        Args:
            llm_config: LLM connection + sampling configuration.

        Returns:
            A concrete `LLMClient` implementation.

        Raises:
            RuntimeError: If the API key environment variable is missing.
        """

        api_key = os.getenv(llm_config.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"Environment variable '{llm_config.api_key_env}' must be set "
                "to run LLM-dependent agents. Provide a mock client in tests "
                "or configure the API key."
            )
        return OpenRouterClient(llm_config)

    async def _chat_with_trace(
        self,
        *,
        llm: LLMConfig,
        trace: TraceLogger,
        node_id: int,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        trace_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call an LLM and record request/response in the trace.

        Args:
            llm: LLM configuration to use.
            trace: Trace logger.
            node_id: Node id for trace association.
            system_prompt: System message content.
            user_prompt: User message content.
            tools: Optional OpenAI-compatible tool definitions.
            response_format: Optional OpenAI-compatible response_format.
            trace_info: Optional extra key/value data to include in the request trace.

        Returns:
            Raw OpenAI-style response dict.
        """

        messages: list[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        client = self._client_for(llm)
        try:
            response = await client.chat(
                messages,
                tools=tools,
                response_format=response_format,
            )
        finally:
            await client.aclose()

        request_payload: dict[str, Any] = {"messages": messages}
        if trace_info:
            request_payload.update(trace_info)
        trace.llm_request(node_id, request_payload)
        trace.llm_response(node_id, response)
        return response

    @staticmethod
    def tool_defs() -> list[dict[str, Any]]:
        """Return OpenAI function-call tool definitions supported by the runtime.

        Returns:
            List of OpenAI-compatible tool definitions.
        """

        return [
            {
                "type": "function",
                "function": {
                    "name": "spawn_agent",
                    "description": "Spawn a sub-agent to handle a task.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent_type": {
                                "type": "string",
                                "enum": [t.value for t in AgentType],
                            },
                            "task": {"type": "object"},
                            "edge_from_parent": {
                                "type": "boolean",
                                "default": True,
                            },
                        },
                        "required": ["agent_type", "task"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "kb_write",
                    "description": "Write entries to the knowledge base.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entries": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "kind": {"type": "string"},
                                        "title": {"type": "string"},
                                        "content_md": {"type": "string"},
                                        "tags": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "sources": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": [
                                        "id",
                                        "kind",
                                        "title",
                                        "content_md",
                                    ],
                                },
                            }
                        },
                        "required": ["entries"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "finish",
                    "description": "Signal that the answer is complete.",
                    "parameters": {
                        "type": "object",
                        "properties": {"output_text": {"type": "string"}},
                        "required": ["output_text"],
                    },
                },
            },
        ]

    @staticmethod
    def parse_tool_calls(
        tool_calls: list[dict[str, Any]],
    ) -> tuple[list[SpawnRequest], list[KBEntry], str | None]:
        """Parse OpenAI tool calls into typed runtime actions.

        Args:
            tool_calls: Raw `tool_calls` list from an OpenAI-compatible response.

        Returns:
            A tuple of `(spawn_requests, kb_entries, finish_text)`.
        """

        spawns: list[SpawnRequest] = []
        kb_entries: list[KBEntry] = []
        finish_text: str | None = None

        valid_agent_types = {t.value for t in AgentType}
        for tool_call in tool_calls:
            fn = tool_call.get("function", {})
            name = fn.get("name")
            arguments = fn.get("arguments", "{}")
            try:
                args = json.loads(arguments)
            except Exception:
                args = {}

            if name == "spawn_agent":
                agent_type_val = args.get("agent_type")
                if agent_type_val in valid_agent_types:
                    spawns.append(
                        SpawnRequest(
                            agent_type=AgentType(agent_type_val),
                            task=args.get("task", {}),
                            edge_from_parent=args.get("edge_from_parent", True),
                        )
                    )
            elif name == "kb_write":
                for entry in args.get("entries", []):
                    kind_value = entry.get("kind", KBKind.RESULT.value)
                    try:
                        kind_enum = KBKind(kind_value)
                    except ValueError:
                        kind_enum = KBKind.RESULT
                    kb_entries.append(
                        KBEntry(
                            id=str(entry.get("id", "Result")),
                            kind=kind_enum,
                            title=str(entry.get("title", "Untitled")),
                            content_md=str(entry.get("content_md", "")),
                            tags=list(entry.get("tags", []) or []),
                            sources=list(entry.get("sources", []) or []),
                        )
                    )
            elif name == "finish":
                finish_text = args.get("output_text")

        return spawns, kb_entries, finish_text
