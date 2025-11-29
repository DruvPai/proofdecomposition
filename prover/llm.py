"""LLM client interface with an OpenRouter httpx implementation."""

from __future__ import annotations

import os
from typing import Any, Protocol

import httpx

from prover.config import LLMConfig
from prover.constants import HTTP_TIMEOUT_SECONDS


Message = dict[str, Any]
ToolDef = dict[str, Any]


class LLMClient(Protocol):
    """Interface for chat-completion clients."""

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send chat completion request.

        Args:
            messages: Conversation history in OpenAI chat format.
            tools: Optional list of OpenAI function-call tool definitions.
            response_format: Optional structured response format, e.g. {"type": "json_object"}.

        Returns:
            Raw OpenAI-style completion response.
        """
        ...

    async def aclose(self) -> None:
        """Close underlying resources."""
        ...


class OpenRouterClient:
    """OpenRouter client using an OpenAI-compatible chat API via httpx."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.api_key = os.getenv(config.api_key_env, "")
        self.base_url = config.base_url or "https://openrouter.ai/api/v1"
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=HTTP_TIMEOUT_SECONDS
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = await self._client.post(
            "/chat/completions", json=payload, headers=headers
        )
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()
